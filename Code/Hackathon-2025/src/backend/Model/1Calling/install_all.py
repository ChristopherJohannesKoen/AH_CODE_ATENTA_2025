#!/usr/bin/env python3
"""
Bootstrap installer for all Python deps used across the project.

Usage:
  python install_all.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

REQ_CONTENT = """# Web server
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.8

# LLM + OpenAI client
openai>=1.30

# ASR (Whisper) + deps
openai-whisper>=20231117
torch>=2.1
torchaudio>=2.1
tqdm>=4.66
regex>=2023.10.3
numpy>=1.24

# Audio / media
ffmpeg-python>=0.2.0

# (Optional) Speaker diarization
pyannote.audio>=3.1

# NLP / JSON utils
rapidfuzz>=3.5
jsonschema>=4.19
"""

REQ_FILE = Path("requirements.txt")


def run(cmd):
    print(f"→ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)


def ensure_requirements_file():
    if not REQ_FILE.exists():
        print(
            f"[info] requirements.txt not found. Creating one at: {REQ_FILE.resolve()}"
        )
        REQ_FILE.write_text(REQ_CONTENT.strip() + "\n", encoding="utf-8")
    else:
        print(f"[info] Using existing requirements.txt at: {REQ_FILE.resolve()}")


def pip_install_requirements():
    print("\n[step] Installing Python dependencies from requirements.txt …")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])


def check_ffmpeg():
    print("\n[step] Checking FFmpeg availability …")
    exe = shutil.which("ffmpeg")
    if exe:
        print(f"[ok] ffmpeg found at: {exe}")
        return
    print(
        "[warn] ffmpeg not found on PATH. The code uses ffmpeg-python which shells out to the ffmpeg binary."
    )
    print("       Please install it and ensure it's on PATH:")
    print("       - Windows:   choco install ffmpeg")
    print("       - macOS:     brew install ffmpeg")
    print("       - Ubuntu/Debian: sudo apt-get install ffmpeg")


def quick_import_smoke_test():
    print("\n[step] Verifying imports …")
    modules = [
        ("fastapi", "FastAPI web server"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "data models"),
        ("openai", "OpenAI client"),
        ("whisper", "OpenAI Whisper"),
        ("torch", "PyTorch"),
        ("torchaudio", "PyTorch audio"),
        ("ffmpeg", "ffmpeg-python"),
        ("rapidfuzz", "string matching"),
        ("jsonschema", "JSON schema"),
    ]
    failed = []
    for mod, desc in modules:
        try:
            __import__(mod)
            print(f"[ok] {mod:12s} — {desc}")
        except Exception as e:
            failed.append((mod, str(e)))
            print(f"[fail] {mod:12s} — import error: {e}")

    # pyannote.audio is optional (only if diarization enabled)
    try:
        __import__("pyannote.audio")
        print(f"[ok] {'pyannote.audio':12s} — diarization (optional)")
    except Exception as e:
        print(f"[warn] {'pyannote.audio':12s} — optional; import error: {e}")

    if failed:
        print(
            "\n[warn] Some imports failed. If errors involve torch/torchaudio wheels on your platform,"
        )
        print(
            "       you may need platform-specific install instructions from https://pytorch.org/"
        )
        print("       (e.g., CUDA-enabled wheels).")
    else:
        print("\n[ok] All core imports succeeded.")


def main():
    ensure_requirements_file()
    pip_install_requirements()
    check_ffmpeg()
    quick_import_smoke_test()
    print("\n[done] Environment is ready. Remember to set:")
    print("       - OPENAI_API_KEY      (required for Brain/LLM features)")
    print("       - HUGGINGFACE_TOKEN   (required if you enable diarization)")


if __name__ == "__main__":
    main()
