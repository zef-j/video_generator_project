"""
Main entry point for the automated 60‑second vertical video generator.

This script launches a simple GUI application built with PySide6.  The user
interface exposes a single button which, when pressed, processes the next
available script in the ``scripts/`` folder into a finished, vertically
formatted video.  Status messages are displayed in a scrolling text box to
keep the user informed of progress.  Once a video is created it can be found
in the ``output/`` directory and the processed script will be moved to
``used_scripts/``.

The heavy lifting happens in ``video_generator.pipeline`` which implements
all steps of the video creation pipeline: reading a script, generating
text‑to‑speech narration, selecting and trimming background media, creating
synchronized subtitles, mixing audio and finally exporting the composite
video.  See that module for more details.

Before running this script please ensure your system has FFmpeg installed and
accessible via your PATH.  MoviePy relies on FFmpeg to perform the video
encoding.  On macOS you can install it via Homebrew (``brew install ffmpeg``),
on Linux via your package manager (e.g. ``sudo apt install ffmpeg``), and on
Windows by downloading a binary from the official FFmpeg project and adding it
to your PATH.

Example usage:

    python main.py

This will open a small window.  Place a text file into ``scripts/`` and click
“Generate Video”.  The application will produce a vertical video and save it
into ``output/``.
"""

import os
import sys
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QMessageBox,
)

# Import our pipeline logic
from video_generator.pipeline import run_video_generation


class VideoGeneratorThread(QThread):
    """Worker thread that runs the video generation pipeline.

    Signals
    -------
    log (str)
        Emitted whenever a new status message is available.
    progress (int)
        Emitted to update the progress bar.  Values should be between 0–100.
    finished (bool)
        Emitted when the pipeline completes.  A boolean indicates success
        (True) or failure (False).
    """

    log = Signal(str)
    progress = Signal(int)
    finished = Signal(bool)

    def __init__(self, project_root: str):
        super().__init__()
        self.project_root = project_root

    def run(self) -> None:
        """Execute the video generation pipeline.

        This method is executed in a separate thread when ``start()`` is called.
        It delegates to ``run_video_generation`` defined in
        ``video_generator.pipeline``, passing callbacks to update the GUI
        asynchronously.  Any exception is caught and reported back via the
        ``finished`` signal.
        """
        try:
            def log_cb(msg: str) -> None:
                # Emit a log message to the GUI
                self.log.emit(msg)

            def progress_cb(percent: int) -> None:
                # Emit a progress update (0–100)
                self.progress.emit(percent)

            run_video_generation(
                project_root=self.project_root,
                log_callback=log_cb,
                progress_callback=progress_cb,
            )
        except Exception as exc:
            # Emit a traceback to the log and signal failure
            tb = ''.join(traceback.format_exception(exc))
            self.log.emit(f"Error:\n{tb}")
            self.finished.emit(False)
        else:
            # Signal success on completion
            self.finished.emit(True)


class MainWindow(QWidget):
    """Main application window containing the GUI elements."""

    def __init__(self, project_root: str) -> None:
        super().__init__()
        self.project_root = project_root
        self.setWindowTitle("Automated Video Generator")
        self.resize(500, 350)

        # Create layout and widgets
        layout = QVBoxLayout()
        self.generate_btn = QPushButton("Generate Video")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout.addWidget(self.generate_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_box)
        self.setLayout(layout)

        # Connect button click
        self.generate_btn.clicked.connect(self.on_generate_clicked)

        # Worker thread instance
        self.worker: VideoGeneratorThread | None = None

    def append_log(self, message: str) -> None:
        """Append a log message to the text box and scroll to bottom."""
        self.log_box.append(message)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def update_progress(self, percent: int) -> None:
        """Update progress bar value.  Show the bar if not visible."""
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percent)

    def on_thread_finished(self, success: bool) -> None:
        """Handle worker thread completion.

        Re‑enable the button, hide the progress bar and optionally display a
        message box on error.
        """
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if success:
            self.append_log("Finished generating video.")
        else:
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred during video generation. See log for details.",
            )

    def on_generate_clicked(self) -> None:
        """Start the video generation in a worker thread."""
        # Disable the button to prevent multiple concurrent runs
        self.generate_btn.setEnabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Create and start the worker thread
        self.worker = VideoGeneratorThread(project_root=self.project_root)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_thread_finished)
        self.worker.start()


def ensure_project_structure(project_root: str) -> None:
    """Ensure that required directories exist within the project root.

    Parameters
    ----------
    project_root : str
        Path to the root of the project.  The following subdirectories
        will be created if they do not already exist: ``scripts``,
        ``used_scripts``, ``brainrot_videos``, ``background_music`` and
        ``output``.
    """
    subdirs = [
        "scripts",
        "used_scripts",
        "brainrot_videos",
        "background_music",
        "output",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(project_root, sub), exist_ok=True)


def main() -> int:
    """Entry point for running the GUI application."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    ensure_project_structure(project_root)
    app = QApplication(sys.argv)
    window = MainWindow(project_root=project_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())