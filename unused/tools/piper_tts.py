# tools/piper_tts.py
from __future__ import annotations
import os
import shlex
import shutil
import subprocess as sp
from pathlib import Path
from typing import Tuple, Optional

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

def _run(cmd: list[str], text_stdin: Optional[str] = None) -> tuple[int, str, str]:
    proc = sp.Popen(cmd, stdin=sp.PIPE if text_stdin is not None else None,
                    stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    out, err = proc.communicate(text_stdin)
    return proc.returncode, out or "", err or ""

def _ffprobe_duration(wav: Path) -> float:
    if not wav.exists() or wav.stat().st_size == 0:
        return 0.0
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(wav)]
    code, out, _ = _run(cmd)
    try:
        return float(out.strip()) if code == 0 else 0.0
    except Exception:
        return 0.0

def _ffmpeg_resample(src: Path, dst: Path, rate: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-i", str(src), "-ac", "1", "-ar", str(rate), str(dst)]
    code, _, _ = _run(cmd)
    return code == 0 and dst.exists() and dst.stat().st_size > 0

def piper_tts(
    text: str,
    work_dir: Path,
    *,
    # Path to piper executable if installed via pip/homebrew. If None, we try shutil.which("piper").
    piper_bin: Optional[str] = None,
    # If using Docker, image name to run.
    docker_image: str = "ghcr.io/rhasspy/piper:latest",
    # Voice model files. You must provide matching .onnx and .json.
    model_onnx: Path | None = None,
    model_json: Path | None = None,
    # Desired output sample rates for downstream:
    out_rate_wav: int = 44100,
    out_rate_align: int = 16000,
    log = print,
) -> Tuple[Path, Path, float] | None:
    """
    Generate narration using Piper.
    Returns (wav_44k1, wav_16k, duration_seconds) or None if Piper unavailable/fails.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_out = work_dir / "piper_raw.wav"
    wav_44k1 = work_dir / "piper_44k1.wav"
    wav_16k  = work_dir / "piper_16k.wav"

    # Sanity: need model files
    if model_onnx is None or model_json is None or not model_onnx.exists() or not model_json.exists():
        log("[Piper] Model files not found; skipping Piper.")
        return None

    # Prefer local binary if present
    if piper_bin is None:
        piper_bin = shutil.which("piper")

    if piper_bin:
        log(f"[Piper] Using local binary: {piper_bin}")
        cmd = [
            piper_bin,
            "--model", str(model_onnx),
            "--config", str(model_json),
            "--output_file", str(raw_out),
        ]
        code, _, err = _run(cmd, text_stdin=text)
        if code != 0:
            log(f"[Piper] Local binary failed (code {code}). stderr tail:\n{err[-400:]}")
            # try docker fallback below
        else:
            dur = _ffprobe_duration(raw_out)
            if dur > 0:
                # resample
                ok1 = _ffmpeg_resample(raw_out, wav_44k1, out_rate_wav)
                ok2 = _ffmpeg_resample(raw_out, wav_16k,  out_rate_align)
                if ok1 and ok2:
                    return wav_44k1, wav_16k, dur
                log("[Piper] Resample failed after local run.")
            else:
                log("[Piper] Local piper produced zero duration.")
            # fallthrough to docker

    # Docker fallback
    docker = shutil.which("docker")
    if not docker:
        log("[Piper] Docker not available; giving up Piper.")
        return None

    log(f"[Piper] Trying Docker image: {docker_image}")
    # We pass text via stdin to Piper inside container
    cmd = [
        docker, "run", "--rm",
        "-i",  # stdin
        "-v", f"{work_dir}:/work",
        docker_image,
        "piper",
        "--model", f"/work/{model_onnx.name}",
        "--config", f"/work/{model_json.name}",
        "--output_file", f"/work/{raw_out.name}",
    ]
    code, _, err = _run(cmd, text_stdin=text)
    if code != 0:
        log(f"[Piper] Docker failed (code {code}). stderr tail:\n{err[-400:]}")
        return None

    dur = _ffprobe_duration(raw_out)
    if dur <= 0:
        log("[Piper] Docker produced zero duration.")
        return None

    ok1 = _ffmpeg_resample(raw_out, wav_44k1, out_rate_wav)
    ok2 = _ffmpeg_resample(raw_out, wav_16k,  out_rate_align)
    if not (ok1 and ok2):
        log("[Piper] Resample failed after Docker run.")
        return None

    return wav_44k1, wav_16k, dur
