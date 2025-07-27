"""
Core pipeline for transforming a text script into a 60‑second vertical video.

The primary entrypoint is :func:`run_video_generation`, which orchestrates
reading a script from the ``scripts/`` directory, producing a narration
audio file with pyttsx3, choosing random background video and music,
preparing timed subtitles, assembling the final composite video with
MoviePy, and writing the output MP4 to ``output/``.  Logging and progress
callbacks allow a GUI to respond to status updates.
"""

from __future__ import annotations

import os
import random
import shutil
import string
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import subprocess
import sys
import shutil
"""
The `aifc` module was removed in Python 3.13.  In earlier versions it
provided support for reading/writing AIFF/AIFF-C files but it is no
longer part of the standard library.  Our pipeline previously
imported it by default even though we never used it directly; this
caused a ModuleNotFoundError under Python 3.13.  Since the
ffmpeg-based conversion covers our needs and we don't manipulate
AIFF/CAF files directly, we can safely drop this import.  The `wave`
module remains available to handle WAV files if necessary.
"""
import wave
from moviepy.editor import (
    CompositeAudioClip,
    CompositeVideoClip,
    VideoFileClip,
    ImageClip,
)
from moviepy.audio.AudioClip import AudioArrayClip
# Import AudioFileClip directly from its submodule rather than via moviepy.editor.
# moviepy.editor is pinned to v1.x in this project, but to minimise
# dependencies on the monolithic import we explicitly import
# AudioFileClip here.  This module provides a way to load external
# audio files (e.g. background music) using ffmpeg.
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.fx import audio_loop
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.fx import loop as vfx_loop
from PIL import Image, ImageDraw, ImageFont
import pyttsx3


# Type aliases for callbacks
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]

# Configuration constants for subtitle appearance.  Adjust these values to
# customise subtitle size, line wrapping and position.  The font height is
# computed as SUBTITLE_FONT_HEIGHT_RATIO × video height.  The subtitle
# image occupies SUBTITLE_MAX_WIDTH_RATIO × video width.  Position is
# specified as a tuple accepted by MoviePy (e.g., ('center','center'),
# ('center','bottom'), etc.).
SUBTITLE_FONT_HEIGHT_RATIO: float = 0.8  # 8% of video height
SUBTITLE_MAX_WIDTH_RATIO: float = 0.6     # 70% of video width
SUBTITLE_POSITION: Tuple[str, str] = ("center", "center")


def run_video_generation(
    project_root: str,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Execute the full video generation pipeline.

    Parameters
    ----------
    project_root : str
        Root directory of the project containing ``scripts``, ``used_scripts``,
        ``brainrot_videos``, ``background_music`` and ``output`` subfolders.
    log_callback : callable, optional
        Function invoked with a message to log progress.  If not provided
        messages are printed to stdout.
    progress_callback : callable, optional
        Function invoked with an integer from 0 to 100 representing
        approximate progress through the pipeline.  If not provided no
        progress updates are made.

    Raises
    ------
    RuntimeError
        If required inputs (script, video, music) are missing.  Errors
        propagate to the caller and should be handled by a GUI thread.
    """
    # Helper functions for logging and progress
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(p: int) -> None:
        if progress_callback:
            progress_callback(p)

    # Paths setup
    root = Path(project_root)
    scripts_dir = root / "scripts"
    used_dir = root / "used_scripts"
    videos_dir = root / "brainrot_videos"
    music_dir = root / "background_music"
    output_dir = root / "output"

    # Step 1: Find a script to process
    progress(0)
    scripts = [f for f in scripts_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    if not scripts:
        log("No script available in scripts/. Please add a .txt file and try again.")
        # Nothing to process – finish gracefully
        progress(100)
        return
    # Always select the first script alphabetically to maintain a deterministic order
    scripts.sort(key=lambda p: p.name)
    script_path = scripts[0]
    log(f"Selected script: {script_path.name}")
    with script_path.open("r", encoding="utf-8") as f:
        script_text = f.read().strip()
    # If the script appears to be in Rich Text Format (RTF) convert it to plain
    # text.  RTF uses a leading "{\rtf" and contains control words (\par,
    # \b, etc.) which confuse both the speech synthesis and subtitle splitter.
    # We attempt to import striprtf, but if unavailable we leave the text
    # unchanged and rely on the user to provide plain text.
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore
        if script_text.lstrip().startswith("{\\rtf"):
            script_text = rtf_to_text(script_text)
    except Exception:
        pass
    if not script_text:
        log("The selected script is empty. Skipping.")
        # Do not move the script; leave it in scripts/ so the user can fix it
        raise RuntimeError("Empty script")
    # We defer moving the script to used_scripts until the video has been successfully written.
    used_path = used_dir / script_path.name
    progress(5)

    # Step 2: Generate TTS audio from script
    log("Generating narration audio...")
    tts_start = time.time()
    # Determine the best available text‑to‑speech engine.  On macOS the built‑in
    # `say` command produces AIFF files with proper headers.  If available,
    # prefer it over pyttsx3 to avoid issues with AIFF disguised as WAV.
    use_say = False
    if sys.platform == "darwin" and shutil.which("say") is not None:
        use_say = True

    # Generate the TTS file (tts_file).  We choose an AIFF suffix when using
    # `say` because it writes AIFF/CAF by default.  When falling back to
    # pyttsx3 we still use `.wav` as the suffix but will transcode below.
    if use_say:
        # Write the audio directly to a temporary WAV file.  Using
        # `--file-format=WAVE` along with `--data-format` instructs `say` to
        # output a proper WAV container with a valid header and 16‑bit little‑
        # endian samples.  We avoid the problematic AIFF/Caf outputs altogether.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tts_file = tmp_audio.name
        # Write the script text to a temporary file because passing large
        # multi‑paragraph strings as command line arguments can cause the
        # `say` command to fail.  The `-f` option reads from a file.
        tmp_txt_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_txt:
                tmp_txt.write(script_text)
                tmp_txt_path = tmp_txt.name
            cmd_tts = [
                "say",
                "-o",
                tts_file,
                "--file-format=WAVE",
                "--data-format=LEI16@44100",
                "-f",
                tmp_txt_path,
            ]
            subprocess.run(cmd_tts, check=True)
        except Exception as e:
            log(f"`say` command failed: {e}. Falling back to pyttsx3.")
            use_say = False
        finally:
            # Remove the temporary text file regardless of success
            if tmp_txt_path and os.path.exists(tmp_txt_path):
                try:
                    os.remove(tmp_txt_path)
                except Exception:
                    pass

    if not use_say:
        # Use pyttsx3 as a fallback.  Note: on macOS this may produce AIFF data
        # even if the suffix is `.wav`, so we'll normalise it below.
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tts_file = tmp_audio.name
        engine.save_to_file(script_text, tts_file)
        engine.runAndWait()
        engine.stop()

    # Decide whether to bypass ffmpeg conversion.  If we successfully generated
    # a WAV via `say` (use_say is True), then tts_file already points to a
    # proper PCM WAV with little-endian 16‑bit data.  In that case we can use
    # it directly without additional transcoding.  Otherwise we need to
    # normalise the output using ffmpeg.
    converted_tts = None
    narration_clip = None
    skip_transcode = use_say  # If use_say succeeded, skip ffmpeg
    try:
        if skip_transcode:
            converted_tts = tts_file
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
                converted_tts = tmpwav.name
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tts_file,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-f",
                "wav",
                converted_tts,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                log("ffmpeg conversion failed. STDERR:\n" + result.stderr.decode())
                raise RuntimeError("ffmpeg failed to convert TTS audio")

        # Read the WAV (converted or original) into a numpy array using the wave module.
        with wave.open(converted_tts, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
        if n_frames == 0:
            narration_clip = AudioFileClip(converted_tts)
        else:
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise RuntimeError(f"Unsupported sample width: {sample_width}")
            audio = np.frombuffer(frames, dtype=dtype)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            else:
                audio = audio.reshape(-1, 1)
            if np.issubdtype(dtype, np.unsignedinteger):
                audio = (audio.astype(np.float32) - 128.0) / 128.0
            else:
                max_val = float(np.iinfo(dtype).max)
                audio = audio.astype(np.float32) / max_val
            narration_clip = AudioArrayClip(audio, fps=sample_rate)
    except FileNotFoundError:
        log("ffmpeg not found. Attempting to load TTS audio directly using wave.")
        try:
            with wave.open(tts_file, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise RuntimeError(f"Unsupported sample width: {sample_width}")
            audio = np.frombuffer(frames, dtype=dtype)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            else:
                audio = audio.reshape(-1, 1)
            if np.issubdtype(dtype, np.unsignedinteger):
                audio = (audio.astype(np.float32) - 128.0) / 128.0
            else:
                max_val = float(np.iinfo(dtype).max)
                audio = audio.astype(np.float32) / max_val
            narration_clip = AudioArrayClip(audio, fps=sample_rate)
        except Exception as ex:
            log(f"Direct wave read failed: {ex}. Falling back to AudioFileClip (may fail).")
            narration_clip = AudioFileClip(tts_file)

    # Measure duration using the audio clip itself
    narration_duration = narration_clip.duration
    log(f"TTS complete (duration: {narration_duration:.2f}s)")
    tts_end = time.time()
    progress(15)

    # Step 3: Select random background video and music
    log("Selecting random background video and music...")
    videos = [f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".mkv"}]
    if not videos:
        log("No background videos found in brainrot_videos/. Please add .mp4 files.")
        raise RuntimeError("No background videos")
    music_files = [f for f in music_dir.iterdir() if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".aac"}]
    if not music_files:
        log("No background music found in background_music/. Please add audio files.")
        raise RuntimeError("No background music")
    video_path = random.choice(videos)
    music_path = random.choice(music_files)
    log(f"Selected video: {video_path.name}")
    log(f"Selected music: {music_path.name}")
    progress(25)

    # Step 4: Load and prepare video clip
    log("Loading and preparing video clip...")
    video_clip = VideoFileClip(str(video_path))
    # Resize and crop to vertical 1080x1920
    video_clip = _prepare_video_clip(video_clip, narration_duration)
    progress(35)

    # Step 5: Load and prepare music clip
    log("Loading and preparing music clip...")
    music_clip = AudioFileClip(str(music_path))
    # Loop or trim to match narration duration
    if music_clip.duration < narration_duration:
        music_clip = audio_loop(music_clip, duration=narration_duration)
    else:
        music_clip = music_clip.subclip(0, narration_duration)
    # Reduce music volume
    music_clip = music_clip.volumex(0.1)
    progress(40)

    # Step 6: Prepare subtitles
    log("Splitting script into subtitles and computing timings...")
    subtitle_lines = _split_script(script_text)
    # Normalize each subtitle line to avoid Unicode characters that may not be
    # supported by the font.  Replace curly quotes, dashes, ellipses, etc.
    subtitle_lines = [_normalize_subtitle_text(line) for line in subtitle_lines]
    timed_subs = _allocate_subtitle_times(subtitle_lines, narration_duration)
    log(f"Generated {len(timed_subs)} subtitle segments")
    progress(50)

    # Step 7: Create subtitles clip
    log("Rendering subtitle images...")
    # Determine video size from prepared video
    vid_w, vid_h = video_clip.size
    # Choose a single random font and colour for all subtitles in this video
    style_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
    ]
    style_font_paths = [f for f in style_font_paths if os.path.exists(f)] or [None]
    style_colours = [
        (255, 255, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 165, 0),
    ]
    chosen_font = random.choice(style_font_paths)
    chosen_colour = random.choice(style_colours)
    generator = _make_subtitle_generator(video_size=(vid_w, vid_h), chosen_style=(chosen_font, chosen_colour))
    # Convert our list of triples (start, end, text) into the format expected by
    # SubtitlesClip: a list of ((start, end), text) pairs.
    subtitle_items = [((start, end), text) for (start, end, text) in timed_subs]
    subtitles_clip = SubtitlesClip(subtitle_items, generator)
    # Position subtitles at the configured location
    subtitles_clip = subtitles_clip.set_position(SUBTITLE_POSITION)
    progress(60)

    # Step 8: Compose final video
    log("Compositing video, audio and subtitles...")
    # Set audio: combine narration and music
    final_audio = CompositeAudioClip([narration_clip, music_clip])
    # Replace audio on video clip
    video_with_audio = video_clip.set_audio(final_audio)
    # Overlay subtitles
    composite = CompositeVideoClip([video_with_audio, subtitles_clip])
    # Ensure duration matches exactly
    composite = composite.set_duration(narration_duration)
    progress(75)

    # Step 9: Write to output file
    # Construct output filename based on script name and timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Use the original script filename (without extension) as the base for
    # the output filename.  We reference `script_path` instead of `used_path`
    # because the file hasn't been moved yet.
    base_name = Path(script_path).stem
    output_filename = f"{base_name}_{timestamp}.mp4"
    output_path = output_dir / output_filename
    log(f"Writing video to {output_path} (this may take a while)...")
    progress(85)
    # Use a temporary directory for writing to avoid incomplete files on error
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = os.path.join(tmpdir, output_filename)
        composite.write_videofile(
            tmp_output,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            threads=4,
            temp_audiofile=os.path.join(tmpdir, "temp-audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
        # Move finished file into output directory
        shutil.move(tmp_output, output_path)
    progress(100)
    log(f"Video saved to {output_path.relative_to(root)}")

    # Now that the video has been successfully written, move the script
    # into the used_scripts directory to prevent it from being processed
    # again.  If a file with the same name already exists in used_scripts,
    # append a timestamp to avoid overwriting it.
    try:
        if script_path.exists():
            target = used_dir / script_path.name
            if target.exists():
                # Append timestamp to avoid conflict
                ts = time.strftime("%Y%m%d_%H%M%S")
                target = used_dir / f"{script_path.stem}_{ts}{script_path.suffix}"
            shutil.move(str(script_path), target)
            log(f"Moved script to used_scripts/: {target.name}")
    except Exception as ex:
        log(f"Warning: failed to move script to used_scripts: {ex}")

    # Clean up temporary TTS audio file(s)
    narration_clip.close()
    music_clip.close()
    video_clip.close()
    subtitles_clip.close()
    composite.close()
    # Remove original and converted TTS files
    for f in [tts_file, converted_tts]:
        if f:
            try:
                os.remove(f)
            except Exception:
                pass


def _prepare_video_clip(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Resize, crop and trim the video clip to produce a 9:16 vertical frame.

    Parameters
    ----------
    clip : VideoFileClip
        The original video clip to prepare.
    duration : float
        Duration in seconds to which the video should be trimmed (looped if
        shorter).

    Returns
    -------
    VideoFileClip
        A new clip resized to 1080×1920 and trimmed to the specified duration.
    """
    # Target size
    target_w, target_h = 1080, 1920
    # Resize video while preserving aspect ratio so that one dimension fits
    if clip.w / clip.h > target_w / target_h:
        # Wider than target ratio: fit height, crop width
        clip_resized = clip.resize(height=target_h)
        # Now crop horizontally to target width
        x_center = clip_resized.w / 2
        clip_cropped = clip_resized.crop(
            x_center=x_center,
            width=target_w,
            height=target_h,
        )
    else:
        # Taller than target ratio: fit width, crop height
        clip_resized = clip.resize(width=target_w)
        y_center = clip_resized.h / 2
        clip_cropped = clip_resized.crop(
            y_center=y_center,
            width=target_w,
            height=target_h,
        )
    # Trim or loop to match duration
    if clip_cropped.duration < duration:
        # Loop the clip to the desired duration
        looped = clip_cropped.fx(vfx_loop, duration=duration)
        return looped
    else:
        return clip_cropped.subclip(0, duration)


def _split_script(text: str) -> List[str]:
    """Split a script into subtitle lines.

    The function breaks the script at sentence boundaries (periods,
    question marks, exclamation marks) or newline characters.  Long
    sentences are left as-is; MoviePy will wrap text in the subtitle
    generator.

    Parameters
    ----------
    text : str
        The raw script text.

    Returns
    -------
    list of str
        A list of subtitle lines.
    """
    import re

    # Replace multiple whitespace with single spaces
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Split on punctuation followed by a space or newline
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    # Filter out empty parts
    segments = [p.strip() for p in parts if p.strip()]
    return segments


def _allocate_subtitle_times(
    subtitles: Sequence[str], total_duration: float
) -> List[Tuple[float, float, str]]:
    """Allocate start and end times to each subtitle.

    The algorithm assigns time proportional to the number of characters in
    each subtitle relative to the total number of characters.  If the sum
    of characters is zero (unlikely), equal time slices are assigned.

    Parameters
    ----------
    subtitles : sequence of str
        List of subtitle strings.
    total_duration : float
        Total duration available for all subtitles.

    Returns
    -------
    list of tuple
        Each tuple is (start_time, end_time, subtitle_text).
    """
    if not subtitles:
        return []
    lengths = [len(s) for s in subtitles]
    total_chars = sum(lengths)
    if total_chars == 0:
        # Fallback: equal segments
        per = total_duration / len(subtitles)
        times = []
        start = 0.0
        for s in subtitles:
            end = start + per
            times.append((start, end, s))
            start = end
        return times
    times: List[Tuple[float, float, str]] = []
    start = 0.0
    for s, l in zip(subtitles, lengths):
        duration = (l / total_chars) * total_duration
        end = start + duration
        times.append((start, end, s))
        start = end
    # Adjust final end time to exactly match total_duration (avoid rounding errors)
    if times:
        last_start, _, last_text = times[-1]
        times[-1] = (last_start, total_duration, last_text)
    return times


# Normalize subtitle text by replacing typographic Unicode characters with
# simpler ASCII equivalents.  This prevents encoding errors when measuring
# text sizes with PIL's ImageFont, which may not support all Unicode
# punctuation.  Only characters relevant to common scripts are replaced.
def _normalize_subtitle_text(s: str) -> str:
    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark / apostrophe
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "\u2028": " ",   # line separator
        "\u2029": " ",   # paragraph separator
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _make_subtitle_generator(
    video_size: Tuple[int, int],
    chosen_style: Optional[Tuple[Optional[str], Tuple[int, int, int]]] = None,
) -> Callable[[str], "moviepy.editor.ImageClip"]:
    """Create a generator function for SubtitleClip using Pillow.

    Parameters
    ----------
    video_size : tuple of int
        The (width, height) of the video onto which subtitles will be placed.

    Returns
    -------
    function
        A function mapping a subtitle string to a MoviePy ImageClip.
    """
    vid_w, vid_h = video_size
    # Predefine list of font paths available on most Linux distributions
    available_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
    ]
    available_fonts = [f for f in available_fonts if os.path.exists(f)] or [None]
    colour_palette = [
        (255, 255, 255),  # white
        (255, 255, 0),    # yellow
        (0, 255, 255),    # cyan
        (0, 255, 0),      # green
        (255, 165, 0),    # orange
    ]
    # Choose a single style (font, colour) for the entire video if provided.
    # Otherwise, pick one randomly for each subtitle line.
    fixed_font, fixed_colour = (None, None)
    if chosen_style is not None:
        fixed_font, fixed_colour = chosen_style

    def generator(txt: str):
        # Determine style: either the fixed style or random per subtitle
        font_path = fixed_font if fixed_font is not None else random.choice(available_fonts)
        colour = fixed_colour if fixed_colour is not None else random.choice(colour_palette)
        # Outline colour: use black unless text is dark; test by brightness
        brightness = sum(colour) / 3
        outline_colour = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        # Determine font size relative to video height
        # target proportion of height for baseline
        base_size = int(vid_h * SUBTITLE_FONT_HEIGHT_RATIO)
        # Create a font object
        try:
            font = ImageFont.truetype(font_path, base_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        # Wrap text into multiple lines if needed; use a simple width heuristic
        max_width = int(vid_w * SUBTITLE_MAX_WIDTH_RATIO)
        words = txt.split()
        lines: List[str] = []
        current_line = []
        for word in words:
            current_line.append(word)
            # measure width of current line
            w, h = font.getsize(" ".join(current_line))
            if w > max_width:
                # remove last word, commit line
                current_line.pop()
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        # Recompute height based on number of lines
        text_heights = [font.getsize(line)[1] for line in lines]
        total_text_height = sum(text_heights) + (len(lines) - 1) * int(base_size * 0.3)
        # Create image with some margin
        margin_x = int(vid_w * 0.05)
        margin_y = int(base_size * 0.5)
        img_w = vid_w
        img_h = total_text_height + margin_y * 2
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Starting y coordinate
        y = margin_y
        for line, line_height in zip(lines, text_heights):
            # Compute x for center alignment
            text_w, _ = font.getsize(line)
            x = (img_w - text_w) // 2
            # Draw outline by drawing text shifted around original position
            outline_range = 2  # thickness
            for dx in range(-outline_range, outline_range + 1):
                for dy in range(-outline_range, outline_range + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=outline_colour)
            # Draw main text
            draw.text((x, y), line, font=font, fill=colour)
            y += line_height + int(base_size * 0.3)
        # Convert PIL image to MoviePy ImageClip
        arr = np.array(img)
        clip = ImageClip(arr, ismask=False)
        return clip

    return generator
