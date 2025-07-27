#!/usr/bin/env python3
# tools/aeneas_align.py
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

AENEAS_IMAGE = "aeneas-ng:latest"

AENEAS_CFG_WORDS = (
    "language=eng"
    "|i_t_format=plain"
    "|o_format=json"
    "|o_levels=1"
    "|aba_algorithm=beforenext"
    "|aba_beforenext_value=0.30"
    "|aba_nonspeech_min=0.05"
    "|aba_no_zero=True"
)

AENEAS_CFG_SENTENCES = AENEAS_CFG_WORDS  # same, just text is sentences

AENEAS_ENV = {
    "PYTHONIOENCODING": "UTF-8",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
}

def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def docker_env_args(env: dict[str, str]) -> list[str]:
    out = []
    for k, v in env.items():
        out += ["-e", f"{k}={v}"]
    return out

def call_aeneas(work_dir: Path, wav: Path, txt: Path, cfg: str, out_json: Path) -> tuple[bool, dict, str, str]:
    base = [
        "docker", "run", "--rm",
        *docker_env_args(AENEAS_ENV),
        "-v", f"{str(work_dir)}:/work",
        AENEAS_IMAGE,
    ]
    audio_in = "/work/" + wav.name
    text_in = "/work/" + txt.name
    out_in = "/work/" + out_json.name

    attempts = [
        ["entrypoint", base + [audio_in, text_in, cfg, out_in]],
        ["py_module", base + ["python3", "-m", "aeneas.tools.execute_task", audio_in, text_in, cfg, out_in]],
    ]
    for label, cmd in attempts:
        code, out, err = run(cmd)
        if code == 0 and out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                frags = (data.get("sync_map", {}) or {}).get("fragments") or []
                if frags:
                    return True, data, out, err
            except Exception:
                pass
        # continue to next attempt
    return False, {}, out, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--wav16k", required=True, help="16 kHz mono WAV")
    ap.add_argument("--text", required=True, help="plain text file")
    args = ap.parse_args()

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    wav = Path(args.wav16k)
    text_path = Path(args.text)

    # Build words and sentences files
    text = text_path.read_text(encoding="utf-8")
    words = re.findall(r"\b[\w']+\b", text)
    sents = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+|\n+", text) if s.strip()]

    words_txt = work / "script_words.txt"
    sents_txt = work / "script_sentences.txt"
    words_txt.write_text("\n".join(words), encoding="utf-8")
    sents_txt.write_text("\n".join(sents), encoding="utf-8")

    # Ensure canonical audio name
    audio = work / "audio.wav"
    if wav.resolve() != audio.resolve():
        code, out, err = run(["/opt/homebrew/bin/ffmpeg", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", str(audio)])
        if code != 0:
            print("ffmpeg failed:", err)
            return
    else:
        audio = wav

    # Words first
    out_json = work / "alignment.json"
    ok, data, out, err = call_aeneas(work, audio, words_txt, AENEAS_CFG_WORDS, out_json)
    if not ok:
        ok, data, out, err = call_aeneas(work, audio, sents_txt, AENEAS_CFG_SENTENCES, out_json)

    frags = []
    if ok:
        frags = (data.get("sync_map", {}) or {}).get("fragments") or []
    # Save simplified
    (work / "words.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    # SRT sentences if available
    srt_path = work / "sentences.srt"
    if frags:
        lines = []
        for i, f in enumerate(frags, 1):
            b = float(f.get("begin", "0") or 0.0)
            e = float(f.get("end", "0") or 0.0)
            txt = (f.get("lines") or [""])[0]
            lines.append(f"{i}\n{_fmt_srt_time(b)} --> {_fmt_srt_time(e)}\n{txt}\n")
        srt_path.write_text("\n".join(lines), encoding="utf-8")

    # Diag
    (work / "diag_stdout.txt").write_text(out or "", encoding="utf-8")
    (work / "diag_stderr.txt").write_text(err or "", encoding="utf-8")

    print(f"Wrote {len(frags)} fragments")
    print(f"JSON : {str((work / 'words.json'))}")
    print(f"SRT  : {str(srt_path)}")
    print(f"Diag : {str(work / 'diag_stdout.txt')}, {str(work / 'diag_stderr.txt')}")
    if not frags:
        print("WARNING: No fragments parsed. Inspect diag files for errors.")

def _fmt_srt_time(t: float) -> str:
    if t < 0: t = 0.0
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int((t - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    main()
