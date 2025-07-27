# video_generator/pipeline.py
from __future__ import annotations

import json
import os
import random
import re
import shutil
import string
import subprocess
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pyttsx3
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    ImageClip,
    CompositeAudioClip,
)
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx import audio_loop
from moviepy.video.fx.loop import loop as vfx_loop

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TARGET_W, TARGET_H = 1080, 1920
FPS = 30

# Caption rendering
MAX_CHARS_PER_CAPTION = 15          # increased from 10 → 15
SAFE_WIDTH_RATIO = 0.75
BASE_FONT_RATIO = 0.055
TEXT_FILL = (208, 255, 0, 255)      # lime
STROKE_FILL = (0, 0, 0, 255)
STROKE_W = 4
CAPTION_POS = ("center", "center")

# Caption timing clamps
MIN_CAP_DUR = 0.28
MAX_CAP_DUR = 1.20

# Docker image
AENEAS_IMAGE = "aeneas-ng:latest"

# Fonts to try
CANDIDATE_FONTS = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

LogCb = Callable[[str], None]
ProgCb = Callable[[int], None]


@dataclass
class Caption:
    start: float
    end: float
    text: str


# -------------------------------------------------------------------
# Small utils
# -------------------------------------------------------------------

def _log(cb: LogCb | None, msg: str) -> None:
    if cb:
        cb(msg)
    else:
        print(msg)


def _progress(cb: ProgCb | None, v: int) -> None:
    if cb:
        cb(v)


def _run(cmd: Sequence[str], cb: LogCb | None = None, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    _log(cb, "Running: " + " ".join(cmd))
    return subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _ffprobe_duration(p: Path) -> float | None:
    try:
        cp = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        if cp.returncode == 0:
            s = cp.stdout.decode("utf-8", "ignore").strip()
            if s:
                return float(s)
    except Exception:
        pass
    try:
        with wave.open(str(p), "rb") as w:
            n = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return n / float(rate)
    except Exception:
        pass
    return None


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", " ")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _clean_caption_text(s: str) -> str:
    """
    Remove ALL punctuation for display. Also collapse underscores and extra spaces.
    Examples:
      "don't stop!" -> "dont stop"
      "sun—set, wow." -> "sunset wow"
    """
    s = re.sub(r"[^\w\s]", "", s)     # remove punctuation (keeps letters, digits, underscore, spaces)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_font(size: int) -> ImageFont.ImageFont:
    for fp in CANDIDATE_FONTS:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=STROKE_W)
    return right - left, bottom - top


def _render_caption_image(text: str, vw: int, vh: int) -> Image.Image:
    safe_w = int(vw * SAFE_WIDTH_RATIO)
    font_size = max(12, int(vh * BASE_FONT_RATIO))
    font = _pick_font(font_size)

    tmp = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    while True:
        w, h = _text_bbox(draw, text, font)
        if w <= safe_w or font_size <= 10:
            break
        font_size = max(10, font_size - 2)
        font = _pick_font(font_size)

    mx = int(font_size * 0.6)
    my = int(font_size * 0.35)
    img_w = min(vw, w + mx * 2)
    img_h = h + my * 2
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(img)

    x = (img_w - w) // 2
    y = my
    for dx in range(-STROKE_W, STROKE_W + 1):
        for dy in range(-STROKE_W, STROKE_W + 1):
            if dx == 0 and dy == 0:
                continue
            d2.text((x + dx, y + dy), text, font=font, fill=STROKE_FILL)
    d2.text((x, y), text, font=font, fill=TEXT_FILL)
    return img


# -------------------------------------------------------------------
# TTS
# -------------------------------------------------------------------

def _tts_say_stdin(text: str, work: Path, log: LogCb | None) -> tuple[Path, Path, float]:
    aiff = work / "narration_raw.aiff"
    wav441 = work / "narration.wav"
    wav16k = work / "narration_16k.wav"

    cp = _run(["/usr/bin/say", "-o", str(aiff)], log, input_bytes=text.encode("utf-8"))
    if cp.returncode != 0:
        raise RuntimeError("say failed")

    _run(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "44100", str(wav441)], log)
    _run(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav16k)], log)

    dur = _ffprobe_duration(wav441)
    if not dur or dur <= 0:
        raise RuntimeError("say produced zero/unknown duration")
    return wav441, wav16k, float(dur)


def _tts_pyttsx3(text: str, work: Path, log: LogCb | None) -> tuple[Path, Path, float]:
    aiff = work / "narration_raw.aiff"
    wav441 = work / "narration.wav"
    wav16k = work / "narration_16k.wav"

    eng = pyttsx3.init()
    try:
        rate = eng.getProperty("rate")
        if isinstance(rate, int):
            eng.setProperty("rate", max(120, int(rate * 0.95)))
    except Exception:
        pass
    eng.save_to_file(text, str(aiff))
    eng.runAndWait()

    _run(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "44100", str(wav441)], log)
    _run(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav16k)], log)

    dur = _ffprobe_duration(wav441)
    if not dur or dur <= 0:
        raise RuntimeError("pyttsx3 produced zero/unknown duration")
    return wav441, wav16k, float(dur)


def _generate_tts(script_text: str, work: Path, log: LogCb | None) -> tuple[Path, Path, float]:
    _log(log, "Generating narration audio...")
    try:
        wav441, wav16k, d = _tts_say_stdin(script_text, work, log)
        _log(log, f"TTS complete (duration: {d:.2f}s)")
        return wav441, wav16k, d
    except Exception as e:
        _log(log, f"`say` failed (falling back to pyttsx3): {e}")
    wav441, wav16k, d = _tts_pyttsx3(script_text, work, log)
    _log(log, f"TTS complete (duration: {d:.2f}s)")
    return wav441, wav16k, d


# -------------------------------------------------------------------
# Aeneas
# -------------------------------------------------------------------

def _tokens_from_text(txt: str) -> List[str]:
    # keep punctuation for alignment robustness
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:—-]", txt)


def _read_any_alignment_json(work: Path, project_root: Path, log: LogCb | None) -> dict | None:
    for p in (
        work / "out_words.json",
        project_root / "out_words.json",
        work / "alignment.json",
        project_root / "alignment.json",
    ):
        if p.exists() and p.stat().st_size > 0:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as ex:
                _log(log, f"Failed to parse {p}: {ex}")
    return None


def _align_with_aeneas(wav16k: Path, script_text: str, work: Path, project_root: Path, log: LogCb | None) -> List[Caption] | None:
    txt = _normalize_text(script_text)
    (work / "script_plain.txt").write_text(txt, encoding="utf-8")
    tokens = _tokens_from_text(txt)
    (work / "script_words.txt").write_text("\n".join(tokens), encoding="utf-8")

    audio_in_work = work / "narration_16k.wav"
    if wav16k.resolve() != audio_in_work.resolve():
        shutil.copy2(wav16k, audio_in_work)

    cfg = (
        "task_language=eng|is_text_type=plain|"
        "os_task_file_format=json|os_task_file_levels=1|"
        "task_adjust_boundary_algorithm=beforenext|"
        "task_adjust_boundary_beforenext_value=0.30|"
        "task_adjust_boundary_nonspeech_min=0.05|"
        "task_adjust_boundary_no_zero=True"
    )

    cmd = [
        "docker", "run", "--rm",
        "-e", "PYTHONIOENCODING=UTF-8",
        "-e", "LANG=en_US.UTF-8",
        "-e", "LC_ALL=en_US.UTF-8",
        "-v", f"{work}:/work",
        AENEAS_IMAGE,
        "/work/narration_16k.wav",
        "/work/script_words.txt",
        cfg,
        "/work/out_words.json",
    ]
    cp = _run(cmd, log)
    if cp.returncode != 0:
        _log(log, "Aeneas docker call failed.")

    data = _read_any_alignment_json(work, project_root, log)
    if not data:
        return None

    frags = data.get("fragments", [])
    if not frags:
        return None

    captions: List[Caption] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end: float | None = None

    def flush():
        nonlocal cur_text, cur_start, cur_end
        disp = _clean_caption_text(cur_text)
        if disp and cur_start is not None and cur_end is not None:
            dur = cur_end - cur_start
            if dur < MIN_CAP_DUR:
                cur_end = cur_start + MIN_CAP_DUR
            if dur > MAX_CAP_DUR:
                cur_end = cur_start + MAX_CAP_DUR
            captions.append(Caption(cur_start, cur_end, disp))
        cur_text = ""
        cur_start = None
        cur_end = None

    for f in frags:
        try:
            b = float(f.get("begin", "0") or 0.0)
            e = float(f.get("end", "0") or 0.0)
        except Exception:
            continue
        lines = f.get("lines", [])
        raw_word = (lines[0] if lines else "").strip()
        disp_word = _clean_caption_text(raw_word)
        if not disp_word:
            continue

        candidate = (cur_text + " " + disp_word).strip() if cur_text else disp_word
        if len(candidate) > MAX_CHARS_PER_CAPTION and cur_text:
            flush()
            candidate = disp_word

        if cur_start is None:
            cur_start = b
        cur_text = candidate
        cur_end = e if cur_end is None else max(cur_end, e)

        if len(disp_word) > MAX_CHARS_PER_CAPTION:
            flush()

    flush()
    captions.sort(key=lambda c: c.start)
    return captions


# -------------------------------------------------------------------
# Fallback timings
# -------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")

def _split_to_small_chunks(txt: str) -> List[str]:
    txt = _normalize_text(txt)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(txt) if p.strip()]
    tiny: List[str] = []
    for p in parts:
        words = p.split()
        cur = ""
        for w in words:
            w_clean = _clean_caption_text(w)
            if not w_clean:
                continue
            cand = (cur + " " + w_clean).strip() if cur else w_clean
            if len(cand) <= MAX_CHARS_PER_CAPTION:
                cur = cand
            else:
                if cur:
                    disp = _clean_caption_text(cur)
                    if disp:
                        tiny.append(disp)
                if len(w_clean) > MAX_CHARS_PER_CAPTION:
                    tiny.append(w_clean)
                    cur = ""
                else:
                    cur = w_clean
        if cur:
            disp = _clean_caption_text(cur)
            if disp:
                tiny.append(disp)
    return tiny


def _proportional_timings(lines: List[str], total: float) -> List[Caption]:
    if total <= 0 or not lines:
        return []
    lens = [max(1, len(s)) for s in lines]
    L = float(sum(lens))
    t = 0.0
    caps: List[Caption] = []
    for s, ln in zip(lines, lens):
        dur = total * (ln / L)
        dur = min(max(dur, MIN_CAP_DUR), MAX_CAP_DUR)
        start = t
        end = min(t + dur, total)
        caps.append(Caption(start, end, _clean_caption_text(s)))
        t = end
        if t >= total:
            break
    if caps:
        caps[-1].end = total
    return caps


# -------------------------------------------------------------------
# Video preparation
# -------------------------------------------------------------------

def _prepare_video_clip(vclip: VideoFileClip, duration: float) -> VideoFileClip:
    w, h = vclip.size
    target_aspect = TARGET_W / TARGET_H
    in_aspect = w / h if h else target_aspect

    if in_aspect > target_aspect:
        vclip = vclip.resize(height=TARGET_H)
        w2, _ = vclip.size
        x1 = int((w2 - TARGET_W) / 2)
        vclip = vclip.crop(x1=x1, y1=0, x2=x1 + TARGET_W, y2=TARGET_H)
    else:
        vclip = vclip.resize(width=TARGET_W)
        _, h2 = vclip.size
        y1 = int((h2 - TARGET_H) / 2)
        vclip = vclip.crop(x1=0, y1=y1, x2=TARGET_W, y2=y1 + TARGET_H)

    if vclip.duration < duration:
        vclip = vfx_loop(vclip, duration=duration)
    else:
        vclip = vclip.subclip(0, duration)

    return vclip.set_fps(FPS)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_video_generation(
    project_root: str,
    log_callback: LogCb | None = None,
    progress_callback: ProgCb | None = None,
) -> None:
    root = Path(project_root)
    scripts_dir = root / "scripts"
    used_dir = root / "used_scripts"
    videos_dir = root / "brainrot_videos"
    music_dir = root / "background_music"
    output_dir = root / "output"
    tmp_root = root / "tmp_work"

    for d in (scripts_dir, used_dir, videos_dir, music_dir, output_dir, tmp_root):
        d.mkdir(parents=True, exist_ok=True)

    scripts = sorted([p for p in scripts_dir.iterdir() if p.suffix.lower() == ".txt"])
    if not scripts:
        _log(log_callback, "No script available in scripts/.")
        _progress(progress_callback, 100)
        return

    script_path = scripts[0]
    _log(log_callback, f"Selected script: {script_path.name}")

    txt = script_path.read_text(encoding="utf-8", errors="ignore")
    if txt.lstrip().startswith("{\\rtf"):
        try:
            from striprtf.striprtf import rtf_to_text  # type: ignore
            txt = rtf_to_text(txt)
        except Exception:
            pass
    script_text = _normalize_text(txt)

    work_dir = tmp_root / ("vg_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8)))
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(log_callback, f"Work dir: {work_dir}")

    wav441, wav16k, tts_dur = _generate_tts(script_text, work_dir, log_callback)
    _progress(progress_callback, 15)

    _log(log_callback, "Selecting random background video and music...")
    videos = [p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"}]
    musics = [p for p in music_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".aac"}]
    if not videos:
        raise RuntimeError("No videos in brainrot_videos/")
    if not musics:
        raise RuntimeError("No music in background_music/")
    video_path = random.choice(videos)
    music_path = random.choice(musics)
    _log(log_callback, f"Selected video: {video_path.name}")
    _log(log_callback, f"Selected music: {music_path.name}")

    _log(log_callback, "Loading and preparing video clip...")
    vclip = VideoFileClip(str(video_path))
    vclip = _prepare_video_clip(vclip, tts_dur)
    _progress(progress_callback, 30)

    _log(log_callback, "Loading and preparing music clip...")
    mclip = AudioFileClip(str(music_path))
    mclip = audio_loop(mclip, duration=tts_dur) if (mclip.duration or 0) < tts_dur else mclip.subclip(0, tts_dur)
    mclip = mclip.volumex(0.10)

    _log(log_callback, "Loading narration clip...")
    try:
        with wave.open(str(wav441), "rb") as w:
            n = w.getnframes()
            rate = w.getframerate()
            arr = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() == 2:
                arr = arr.reshape((-1, 2))
            else:
                arr = np.stack([arr, arr], axis=1)
        nclip = AudioArrayClip(arr, fps=rate).set_duration(tts_dur)
    except Exception:
        nclip = AudioFileClip(str(wav441)).subclip(0, tts_dur).set_duration(tts_dur)

    _progress(progress_callback, 40)

    _log(log_callback, "Synchronizing subtitles with narration audio...")
    caps = _align_with_aeneas(wav16k, script_text, work_dir, root, log_callback)
    if not caps:
        _log(log_callback, "Subtitle alignment failed (No alignment fragments). Using proportional timings.")
        tiny = _split_to_small_chunks(script_text)
        caps = _proportional_timings(tiny, tts_dur)
    _log(log_callback, f"Generated {len(caps)} subtitle segments")

    _log(log_callback, "Rendering subtitle images...")
    subclips: List[ImageClip] = []
    for c in caps:
        start = max(0.0, min(c.start, tts_dur))
        end = max(start + MIN_CAP_DUR, min(c.end, tts_dur))
        disp_text = _clean_caption_text(c.text)
        if not disp_text:
            continue
        img = _render_caption_image(disp_text, TARGET_W, TARGET_H)
        arr = np.array(img)
        sub = ImageClip(arr).set_start(start).set_end(end).set_position(CAPTION_POS)
        subclips.append(sub)

    _log(log_callback, "Compositing video, audio and subtitles...")
    base = vclip.set_duration(tts_dur)
    comp = CompositeVideoClip([base, *subclips], size=(TARGET_W, TARGET_H)).set_duration(tts_dur)
    acomp = CompositeAudioClip([nclip.set_duration(tts_dur), mclip.set_duration(tts_dur)]).set_duration(tts_dur)
    comp = comp.set_audio(acomp)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = (root / "output") / f"{script_path.stem}_{ts}.mp4"

    comp.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        preset="medium",
        threads=0,
        verbose=False,
        logger=None,
    )

    (root / "used_scripts").mkdir(parents=True, exist_ok=True)
    shutil.move(str(script_path), str((root / "used_scripts" / script_path.name)))

    # shutil.rmtree(work_dir, ignore_errors=True)  # keep for debugging if you want

    _log(log_callback, "Finished generating video.")
    _progress(progress_callback, 100)
