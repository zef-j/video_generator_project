# Automated 60‑Second Vertical Video Generator

This project is a self‑contained Python application that turns a text
script into a fully edited, captioned and mixed vertical video.  It was
designed to streamline the creation of short‑form content for
platforms like TikTok, Instagram Reels and YouTube Shorts.  Drop a
script and media files into the provided folders, click a button, and
within a minute you will have a polished 1080×1920 MP4 ready for
upload.

## Features

* **One‑click automation** – Users simply add their input files and
  press “Generate Video” in a minimal GUI.  The rest happens
  automatically.
* **Vertical video output** – Videos are produced in a 9:16 Full HD
  format (1080×1920), ideal for modern social media feeds.
* **Offline text‑to‑speech** – Narration is generated using
  [pyttsx3](https://pypi.org/project/pyttsx3/), which leverages the
  system speech engine.  No internet connection or API keys are
  required.
* **Randomised media selection** – A random background video and
  background music track are chosen from your media pools, ensuring
  variation across runs.
* **Timed subtitles** – The script is automatically segmented into
  subtitles, synchronised to the narration, and rendered onto the
  video with randomised styling (font, colour and outline) for visual
  interest.
* **Audio mixing** – Narration and background music are mixed
  together, with music volume reduced so speech remains clear.
* **Robust GUI** – A simple PySide6 interface displays a status log
  and optional progress bar.  Errors are handled gracefully and
  reported to the user.

## Folder Structure

On first run the application ensures the following directory layout
exists relative to the project root:

```
video_generator_project/
├── main.py               # GUI entry point
├── video_generator/      # Pipeline implementation
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── scripts/              # Place your .txt scripts here
├── used_scripts/         # Scripts moved here after processing
├── brainrot_videos/      # Pool of background videos (.mp4, .mov, .mkv)
├── background_music/     # Pool of background audio (.mp3, .wav, .aac)
└── output/               # Generated videos (.mp4)
```

Populate the ``scripts/``, ``brainrot_videos/`` and ``background_music/``
folders with your content before generating a video.  Each run will
process one script at a time (the first file found) and move it into
``used_scripts/`` afterwards to prevent re‑use.

## Installation

1. **Install Python** – Ensure Python 3.9 or later is installed on your
   computer.  You can download it from <https://www.python.org/downloads/>.

2. **Clone or copy this repository** – Place it somewhere on your
   machine.  Change into the project directory in your terminal.

3. **Create a virtual environment (recommended)** – This keeps
   dependencies isolated:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install dependencies** – Use pip to install required Python
   packages:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Install FFmpeg** – MoviePy relies on the FFmpeg executable for
   encoding and decoding media.  It is *not* installed by pip.  Follow
   the instructions below for your platform:

   * **macOS** – Install via [Homebrew](https://brew.sh/):

     ```bash
     brew install ffmpeg
     ```

   * **Linux (Debian/Ubuntu)** – Install via apt:

     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```

   * **Windows** – Download a pre‑compiled build from
     <https://ffmpeg.org/download.html> (e.g. via
     [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).  Unzip the
     archive and add the ``bin/`` directory containing ``ffmpeg.exe``
     and ``ffprobe.exe`` to your system **PATH**.

6. (Optional) **Install ImageMagick** – If you wish to use MoviePy’s
   native TextClip renderer instead of the built‑in Pillow subtitle
   generator, install ImageMagick and ensure the ``magick`` command is
   in your PATH.  On macOS: ``brew install imagemagick``.  This is not
   strictly necessary because this project uses Pillow to render
   subtitles.

## Usage

1. **Prepare your script** – Write your narration text in a UTF‑8
   encoded ``.txt`` file and save it into the ``scripts/`` folder.
   Keep the text to approximately 60 seconds of speech (~150–170
   words) for best results.  The application will read the first
   available script in that folder on each run.

2. **Add background media** – Copy a few video clips into
   ``brainrot_videos/``.  They should be sufficiently long (at least a
   minute) or will be looped if shorter.  These can be any aspect
   ratio; they will be resized and centre‑cropped to 1080×1920.  Also
   add some music files to ``background_music/``.  Supported formats
   include MP3, WAV and AAC.

3. **Launch the application** – From within the project directory,
   run:

   ```bash
   python main.py
   ```

   A window titled “Automated Video Generator” will appear.

4. **Generate your video** – Click the *Generate Video* button.  The
   status log will update as each step completes: reading the
   script, generating narration, selecting media, rendering
   subtitles, compositing and exporting the final video.  This may
   take a minute or two depending on your hardware.

5. **Collect your output** – When the process finishes a message
   appears in the log.  The finished MP4 file will be in the
   ``output/`` directory with a name based on your script and a
   timestamp (e.g. ``my_script_20250101_123456.mp4``).  Your script file
   will have been moved to ``used_scripts/``.

6. **Repeat as needed** – Add another script to ``scripts/`` and click
   the button again.  Each run processes one script at a time.  You
   can queue multiple scripts by placing several files into
   ``scripts/``; just click the button after each run to process the
   next one.

## Customisation

* **Fonts and colours** – Subtitle fonts and colours are chosen
  randomly from a set of DejaVu Sans variants and a palette of
  contrasting colours.  You can modify the lists in
  ``video_generator/pipeline.py`` to add your own fonts (ensure the
  corresponding ``.ttf`` files are present on your system) or colour
  tuples.
* **Voice settings** – The narration uses pyttsx3 with default
  settings.  You can adjust the speaking rate or select a different
  voice in ``run_video_generation`` by setting properties on the
  ``engine`` instance.  For example:

  ```python
  engine.setProperty('rate', 180)  # Increase speech speed
  voices = engine.getProperty('voices')
  engine.setProperty('voice', voices[1].id)  # Choose a different voice
  ```

* **Duration enforcement** – If your script results in narration
  longer than 60 seconds, the program will still process it but
  platform guidelines recommend keeping videos around a minute.  You
  may manually trim your scripts or modify the pipeline to truncate
  narration and subtitles beyond the 60‑second mark.

## Troubleshooting

* **No audio or video produced** – Check that you installed FFmpeg and
  that it is available on your PATH.  Try running ``ffmpeg -version``
  from your terminal.  If it fails, install or re‑configure FFmpeg.

* **pyttsx3 errors** – On some platforms pyttsx3 requires additional
  packages.  On macOS you may need ``pyobjc`` (installable via pip).
  On Linux ensure ``espeak`` or ``festival`` is installed.  See
  pyttsx3’s documentation for platform‑specific notes.

* **Missing fonts** – If you see warnings about fonts or if subtitles
  render with a very basic font, ensure that the DejaVu fonts listed
  in ``pipeline.py`` exist on your system.  You can modify the
  ``font_paths`` list to point at fonts present on your machine.

* **Video encoding errors** – MoviePy passes most errors straight from
  FFmpeg.  Check the log for details.  Ensure your input media files
  are not corrupt and that you have write permissions to the ``output/``
  directory.

## License

This project is provided under the MIT License.  See the ``LICENSE``
file for details.
