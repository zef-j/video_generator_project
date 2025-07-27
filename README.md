# Automated 60‑Second Vertical Video Generator

A compact Python application that turns a plain‑text script into a fully edited 9:16 (1080×1920) short video with speech narration, animated background, and bold on‑screen captions. It provides a one‑button GUI and a programmatic API. The pipeline stitches together text‑to‑speech (TTS), optional forced alignment via **aeneas** (run in Docker), subtitle rendering with Pillow, and compositing/encoding with MoviePy/FFmpeg.

> **What this README is**  
> I walked through every code file in this trimmed repository and simulated the runtime path. Below you’ll find a precise description of the folders, the control flow, important functions/parameters, and how to run/debug/extend the project. Large media and some legacy scripts are omitted from this zip; anything referenced but not present is called out explicitly.

---

## Quick start

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install Python deps
pip install -r requirements.txt

# 3) Ensure FFmpeg is installed and on PATH
ffmpeg -version   # should print a version

# 4) (Optional but recommended) Build the Docker image for aeneas
#    This enables high‑quality word timing alignment.
docker build -f tools/Dockerfile.aeneas -t aeneas-ng:latest .

# 5) Put a script (.txt) into scripts/ and media into the two media folders.

# 6) Run the GUI
python main.py
```

A window with a single **Generate Video** button appears. Clicking it will:
1. Read the next script in `scripts/`.
2. Synthesize narration audio (macOS `say`; otherwise falls back to `pyttsx3`).
3. Optionally align words with audio using the Dockerized **aeneas** image to obtain precise timings.
4. Choose a random background video from `brainrot_videos/`, crop/resize to 9:16, and cut/loop to narration length.
5. Add background music from `background_music/`, ducking beneath narration.
6. Render bold captions (centered) as raster text overlays and place them in time using alignment or a robust fallback splitter.
7. Encode to an H.264 MP4 at 30 FPS and write into `output/`.
8. Move the processed script into `used_scripts/`.

---

## Repository layout

```
.
├── background_music/          # Drop .mp3/.wav/.aac tracks here
├── brainrot_videos/           # Drop .mp4/.mov/.mkv background clips here
├── output/                    # Final rendered MP4s
├── scripts/                   # Input scripts (.txt or .rtf)
│   ├── test copy.txt
│   └── test.txt
├── tmp_work/                  # Ephemeral work dirs (kept for debugging)
│   └── vg_xv06r6ly/
│       ├── out_words.json     # Example aeneas output (word timings)
│       ├── script_plain.txt   # Normalized script
│       └── script_words.txt   # Token list used for alignment
├── tools/
│   ├── Dockerfile.aeneas      # Multi‑arch image producing the `aeneas-ng` tool
│   └── docker_entrypoint.sh   # Wrapper that calls aeneas’ execute_task
├── used_scripts/              # Scripts moved here after successful render
├── video_generator/
│   ├── __init__.py
│   └── pipeline.py            # Core pipeline implementation
├── main.py                    # PySide6 GUI shell around the pipeline
├── requirements.txt           # Python deps (MoviePy 1.x, Pillow <10, etc.)
├── tidy_unused.sh             # Utility to stash/restore large or unused dirs
├── README.md                  # Original project README (kept)
└── project_structure.txt      # Snapshot of a fuller tree on the author’s machine
```

> Files such as large videos, WAVs, and some legacy scripts are intentionally omitted. The pipeline gracefully handles their absence as long as the required folders exist and contain at least one usable media file each.

---

## Runtime flow (simulated)

### Entry point: `main.py`
A minimal PySide6 GUI. Key pieces:

- **`VideoGeneratorThread(QThread)`**  
  Worker thread that owns the long‑running job. Signals:
  - `log(str)` – append message to GUI log box.
  - `progress(int)` – update progress bar (0–100).
  - `finished(bool)` – success/failure.

  In `run()`, it calls `video_generator.pipeline.run_video_generation(project_root, log, progress)` and catches exceptions to emit a failure.

- **`MainWindow(QWidget)`**  
  Shows a **Generate Video** button, a progress bar (hidden until work starts), and a read‑only log text box. On click:
  1. Disables the button.
  2. Starts the `VideoGeneratorThread`.
  3. Appends log lines and progress updates as they arrive.
  4. Re‑enables the button and shows a message box when complete.

- **`ensure_project_structure(project_root)`**  
  Creates the expected directories on startup: `scripts/`, `used_scripts/`, `brainrot_videos/`, `background_music/`, `output/`, `tmp_work/`.

### Core pipeline: `video_generator/pipeline.py`
This module performs all heavy lifting. Important constants (with defaults found in the code):

- `FPS = 30`
- `MAX_CHARS_PER_CAPTION = 15`  – soft cap per on‑screen chunk.
- `SAFE_WIDTH_RATIO = 0.75` – keeps captions within safe center region.
- `BASE_FONT_RATIO = 0.055` – base font size as a fraction of video height.
- `TEXT_FILL = (208, 255, 0, 255)` – bright lime text color.
- `STROKE_FILL = (0, 0, 0, 255)` – black stroke.
- `STROKE_W = 4` – outline thickness.
- `CAPTION_POS = ("center", "center")` – centered captions.
- `MIN_CAP_DUR = 0.28`, `MAX_CAP_DUR = 1.20` – clamp per‑caption display time.
- `AENEAS_IMAGE = "aeneas-ng:latest"` – Docker image tag the code expects.
- `CANDIDATE_FONTS = [...]` – a cross‑platform list; first existing TTF is used, else Pillow’s default bitmap font.

Key types and helpers:

- `@dataclass class Caption(start: float, end: float, text: str)` – normalized display chunks.
- `_log(cb, msg)` / `_progress(cb, pct)` – safe signal wrappers.
- `_ffprobe_duration(path) -> float|None` – uses `ffprobe` to read media duration.
- `_normalize_text(s)` – unicode NFC + collapse whitespace; keeps punctuation for alignment.
- `_clean_caption_text(s)` – removes punctuation/underscores, collapses spaces; used for display text.
- `_pick_font(size)` – choose best available font from `CANDIDATE_FONTS`.
- `_text_bbox(draw, text, font) -> (w, h)` – measure text with stroke.
- `_render_caption_image(text, vw, vh) -> PIL.Image` – rasterize a caption line centered within safe bounds. Uses `BASE_FONT_RATIO` scaled to viewport height and grows/shrinks to fit.

#### TTS

- `_tts_say_stdin(script_text, work, log)` – macOS fast path using the `say` CLI. Generates a 44.1 kHz WAV for mixdown and a resampled 16 kHz WAV for alignment; returns both paths and duration.
- `_tts_pyttsx3(script_text, work, log)` – portable fallback using `pyttsx3`; same outputs.
- `_generate_tts(script_text, work, log)` – tries `say` first; on failure falls back to `pyttsx3` and logs the reason. Returns `(wav441_path, wav16k_path, duration_seconds)`.

#### Alignment (aeneas via Docker)

- `_tokens_from_text(txt)` – tokenizes words but **keeps punctuation**, which increases alignment robustness.
- `_read_any_alignment_json(work, project_root, log)` – looks for an existing `out_words.json`/`alignment.json` (in `work/` or project root) to allow offline or previously generated alignments.
- `_align_with_aeneas(wav16k, script_text, work, project_root, log) -> list[Caption] | None` –
  - Writes `script_plain.txt` and `script_words.txt`.
  - Runs a Docker container using `AENEAS_IMAGE` mounting `work` at `/work` and executes `aeneas.tools.execute_task` via the provided entrypoint.
  - Parses the resulting JSON, merges word fragments into displayable chunks, clamps durations to `[MIN_CAP_DUR, MAX_CAP_DUR]`, and returns a list of `Caption` objects.

If alignment fails or Docker is unavailable, the pipeline falls back to heuristic timings:

- `_split_to_small_chunks(txt) -> list[str]` – sentence‑aware split, then pack to <= `MAX_CHARS_PER_CAPTION`, applying `_clean_caption_text`.
- `_proportional_timings(chunks, total_duration)` – distributes chunk durations proportionally to characters while respecting min/max caps.

#### Video & Audio

- `_prepare_video_clip(vclip, tts_dur)` – ensures the visual clip covers the narration duration:
  - Detect duration via MoviePy/FFprobe.
  - If shorter than narration, loops/repeats; if longer, selects a random start and trims.
  - Crops/letterboxes to 1080×1920 keeping center focus; resizes to target height.

- Final mix:
  - Narration (44.1 kHz WAV) placed at full volume.
  - Music track normalized and ducked under narration.
  - Captions rendered as PIL images and composited with `CompositeVideoClip`.
  - Export: H.264 video at `FPS`, AAC audio, preset `medium`, threading auto.

#### Public API

```python
from video_generator.pipeline import run_video_generation
run_video_generation(project_root="/path/to/project", log_callback=print, progress_callback=lambda p: None)
```

The function ensures directories exist, picks the next script in `scripts/`, and executes the steps outlined above. On success it moves the script to `used_scripts/`.

---

## Building the aeneas Docker image

If you want precise word timing:

```bash
# Build a multi‑arch image named aeneas-ng:latest
docker build -f tools/Dockerfile.aeneas -t aeneas-ng:latest .

# Sanity check: run the tool without arguments to see help
docker run --rm aeneas-ng:latest python -c "import aeneas, sys; print('ok')"
```

At runtime the pipeline executes (conceptually):

```text
docker run --rm \
  -e PYTHONIOENCODING=UTF-8 -e LANG=en_US.UTF-8 -e LC_ALL=en_US.UTF-8 \
  -v /absolute/path/to/tmp_work/vg_xxxxxx:/work \
  aeneas-ng:latest \
  /work/narration_16k.wav /work/script_words.txt \
  "task_language=eng|is_text_type=plain|os_task_file_format=json|..." \
  /work/out_words.json
```

If Docker is not installed or the container returns a non‑zero status, the code logs a message and falls back to heuristic timings.

---

## Requirements

- **Python** 3.10+ (tested with 3.10 in Dockerfile)
- **FFmpeg** on PATH (MoviePy requires it)
- **Python packages** (installed via `requirements.txt`):
  - `PySide6` (GUI)
  - `moviepy<2.0` (1.x API)
  - `pyttsx3` (portable TTS fallback)
  - `numpy`
  - `Pillow>=8,<10` (subtitle rendering; MoviePy 1.x expects pre‑10 constants)
  - `striprtf` (to accept `.rtf` scripts)
- **Docker** (optional) to run `aeneas` without compiling native deps locally

> If you prefer a local aeneas installation, you can adapt `_align_with_aeneas` to call `python -m aeneas.tools.execute_task` directly and skip Docker.

---

## Usage details

### GUI
```bash
python main.py
```
- Click **Generate Video**. Logs stream in the text box; a progress bar updates major stages.
- Outputs appear in `output/` named like `<script_name>_YYYYMMDD_HHMMSS.mp4`.

### Headless / programmatic
Create your own driver that calls `run_video_generation(project_root, log_callback, progress_callback)` and run it on a schedule or in a queue worker.

### Input expectations
- **Scripts**: UTF‑8 `.txt` preferred; `.rtf` supported via `striprtf`. Aim for ~150–170 words (~60 s). Avoid excessive punctuation; it is normalized.
- **Background video**: Any common container; longer than the narration is fine, shorter will be looped. Visual should be safe for crop to 9:16.
- **Background music**: Distinct from narration; the pipeline selects one at random. Provide several tracks to reduce repetition.

---

## Customization knobs

Adjust these constants in `video_generator/pipeline.py`:

- `MAX_CHARS_PER_CAPTION` – change to 10–18 to control chunk size.
- `BASE_FONT_RATIO` – larger or smaller default font sizing.
- `TEXT_FILL`, `STROKE_FILL`, `STROKE_W` – change colors/outline. Use RGBA tuples.
- `CAPTION_POS` – e.g., `("center", "bottom")` to move captions low.
- `MIN_CAP_DUR`, `MAX_CAP_DUR` – lower/raise to speed up/slow down caption flashes.
- `FPS` – frame rate. 30 is a good default for short‑form platforms.
- `CANDIDATE_FONTS` – point at fonts available on your OS.
- `AENEAS_IMAGE` – name/tag of the alignment container you built.

---

## Troubleshooting

- **No videos/music found** – You’ll get `RuntimeError("No videos in brainrot_videos/")` or similar. Add at least one usable file to each folder.
- **FFmpeg not found** – MoviePy/ffprobe calls will fail. Ensure `ffmpeg` and `ffprobe` are on PATH.
- **Docker/aeneas errors** – The pipeline logs the command and continues with fallback timings. Build the image with the provided Dockerfile and confirm you can run containers.
- **Pillow / MoviePy errors about `ANTIALIAS`** – Use `Pillow < 10` as pinned in `requirements.txt`.
- **Fonts look wrong** – Provide a suitable TTF and add it to `CANDIDATE_FONTS` or install DejaVu fonts.
- **Audio device/TTS failures** – `pyttsx3` may need platform extras (e.g., `pyobjc` on macOS; speech engines like `espeak` on Linux). The code already prefers macOS `say` when present.

---

## Development notes

- Work directories under `tmp_work/vg_*` are intentionally **kept** after runs to aid debugging (alignment JSON, normalized text, etc.). You may uncomment the `shutil.rmtree(work_dir)` line near the end of `run_video_generation()` to clean up automatically.
- `tidy_unused.sh` can “stash” bulky or unused project areas into `unused/` and later restore them. The manifest in that script lists paths that were large in the author’s full environment.
- `project_structure.txt` documents a richer original layout (e.g., `scripts/legacy`, `piper_tts.py`, `aeneas_align.py`) which are **not** present in this trimmed zip. The current pipeline does not depend on them.

---

## License

The original project README declares MIT. If you intend to redistribute modified binaries or Docker images, keep the license file alongside your distribution.

---

## Roadmap ideas

- Multi‑voice / external TTS providers (ElevenLabs, Azure, Edge‑TTS) with phoneme timings.
- Beat‑synced caption animations; word‑by‑word karaoke highlighting.
- Automatic smart‑crop based on person/subject detection.
- Simple batch mode and a queue UI (drag several scripts, render all).
- Unit tests for text normalization, chunking, and timing distribution.

