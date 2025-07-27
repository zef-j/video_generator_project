#!/usr/bin/env python3
# aeneas_test.py  —  word-level alignment via aeneas (Docker)
import os, re, json, shlex, subprocess, sys, tempfile, pathlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build_aeneas"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
DOCKER = "/usr/local/bin/docker"
AENEAS_IMAGE = "aeneas-ng:latest"   # the image you built

def run(cmd, check=True, capture=False, env=None):
    print("Running:", cmd if isinstance(cmd, str) else " ".join(cmd))
    if isinstance(cmd, str):
        shell=True
    else:
        shell=False
    if capture:
        res = subprocess.run(cmd, shell=shell, check=check, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        print(res.stdout)
        return res.stdout
    else:
        subprocess.run(cmd, shell=shell, check=check, env=env)

def load_first_script(scripts_dir: Path) -> tuple[Path, str]:
    txts = sorted(scripts_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError("No .txt found in scripts/")
    src = txts[0]
    text = src.read_text(encoding="utf-8", errors="ignore")
    print(f"Using script: {src.name}")
    return src, text

def normalize_to_ascii_words(text: str) -> list[str]:
    # lower, replace dashes, remove punctuation except apostrophes inside words
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2019", "'")  # smart apostrophe -> straight
    text = text.replace("\u2018", "'")
    # keep letters, digits, apostrophes; space separation
    text = re.sub(r"[^A-Za-z0-9' \n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if not text:
        return []
    tokens = text.split(" ")
    # drop empty and lone apostrophes
    tokens = [t for t in tokens if t and t != "'"]
    return tokens

def write_words_file(words: list[str], path: Path):
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"Wrote {len(words)} tokens to {path}")

def tts_with_pyttsx3_to_aiff(text: str, out_aiff: Path):
    import pyttsx3
    engine = pyttsx3.init()
    # slightly slower voice improves alignment
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.9))
    engine.save_to_file(text, str(out_aiff))
    engine.runAndWait()

def ensure_tts_audio(text_utf8: str, raw_aiff: Path, wav_16k: Path) -> float:
    # pyttsx3 (reliable offline) -> AIFF -> ffmpeg -> mono 16 kHz WAV
    print("Generating TTS with pyttsx3...")
    tts_with_pyttsx3_to_aiff(text_utf8, raw_aiff)
    run([FFMPEG, "-y", "-i", str(raw_aiff), "-ac", "1", "-ar", "16000", str(wav_16k)])
    dur = float(run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(wav_16k)],
                    capture=True).strip())
    print(f"TTS audio ready: {wav_16k}  duration={dur:.2f}s")
    return dur

def docker_aeneas_align(audio_wav: Path, words_txt: Path, out_json: Path):
    # Each line is a fragment => levels=1. Use UTF-8 env to silence warnings.
    cmd = [
        DOCKER, "run", "--rm",
        "-e", "PYTHONIOENCODING=UTF-8",
        "-v", f"{BUILD_DIR}:/work",
        AENEAS_IMAGE,
        "/work/narration_16k.wav",
        "/work/words.txt",
        "task_language=eng|is_text_type=plain|os_task_file_format=json|os_task_file_levels=1",
        "/work/alignment_words.json"
    ]
    run(cmd)

def parse_alignment(out_json: Path) -> list[dict]:
    if not out_json.exists() or out_json.stat().st_size == 0:
        return []
    data = json.loads(out_json.read_text(encoding="utf-8", errors="ignore"))
    # Expected structure: {"fragments":[{"begin":"0.00","end":"0.10","lines":["word"]}, ...]}
    frags = data.get("fragments") or []
    items = []
    for f in frags:
        lines = f.get("lines", [])
        word = lines[0] if lines else ""
        try:
            b = float(f.get("begin", "0"))
            e = float(f.get("end", "0"))
        except ValueError:
            continue
        if word:
            items.append({"word": word, "begin": b, "end": e})
    return items

def write_preview(items: list[dict], path: Path):
    preview = items[:50]
    path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote word timings JSON: {path}")

def naive_sentences_srt(full_text: str, timings: list[dict], path: Path):
    # Group words back into coarse sentences by punctuation in original text.
    # Map words in order; if counts mismatch, just chunk into ~10 words.
    sents = re.split(r"([.!?])", full_text)
    rebuilt = []
    buf = ""
    for i in range(0, len(sents), 2):
        core = sents[i].strip()
        punct = sents[i+1] if i+1 < len(sents) else ""
        if core:
            rebuilt.append((core + punct).strip())
    if not timings:
        return
    # Assign words sequentially to sentences by proportional word count
    tokens = normalize_to_ascii_words(full_text)
    if not tokens:
        return
    # Build index of word boundaries by count
    idx = 0
    cursor = 0
    srt_lines = []
    ti = 0
    for si, s in enumerate(rebuilt, 1):
        wcount = len(normalize_to_ascii_words(s))
        if wcount == 0:
            continue
        start_idx = ti
        end_idx = min(ti + wcount, len(timings)) - 1
        if end_idx < start_idx:
            break
        t0 = timings[start_idx]["begin"]
        t1 = timings[end_idx]["end"]
        srt_lines.append(f"{si}")
        srt_lines.append(f"{sec_to_ts(t0)} --> {sec_to_ts(t1)}")
        srt_lines.append(s)
        srt_lines.append("")
        ti = end_idx + 1
        if ti >= len(timings):
            break
    path.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"Wrote SRT: {path}")

def sec_to_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    scripts_dir = PROJECT_ROOT / "scripts"
    _, script_text = load_first_script(scripts_dir)

    # Prepare inputs
    (BUILD_DIR / "tmp").mkdir(exist_ok=True)
    words = normalize_to_ascii_words(script_text)
    words_txt = BUILD_DIR / "words.txt"
    write_words_file(words, words_txt)

    raw_aiff = BUILD_DIR / "narration_raw.aiff"
    wav_16k  = BUILD_DIR / "narration_16k.wav"
    ensure_tts_audio(script_text, raw_aiff, wav_16k)

    # Ensure docker and image exist
    try:
        run([DOCKER, "version"], check=True, capture=True)
    except Exception as e:
        print("Docker not available. Please install and keep it running.")
        sys.exit(1)

    out_json = BUILD_DIR / "alignment_words.json"
    if out_json.exists():
        out_json.unlink(missing_ok=True)

    docker_aeneas_align(wav_16k, words_txt, out_json)

    items = parse_alignment(out_json)
    preview_path = BUILD_DIR / "words_preview.json"
    write_preview(items, preview_path)

    # Also make a coarse sentence SRT from these word timings (for visual check)
    srt_path = BUILD_DIR / "alignment_sentences.srt"
    naive_sentences_srt(script_text, items, srt_path)

    print("First 15 words with timings:")
    for it in items[:15]:
        print(f"{it['begin']:7.3f}–{it['end']:7.3f}  {it['word']}")

if __name__ == "__main__":
    main()
