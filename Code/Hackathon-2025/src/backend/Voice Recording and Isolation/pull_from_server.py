# pull_from_server.py  (Python 3.9 compatible)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Optional
import os
import subprocess
from datetime import datetime
from threading import Thread
from Run import (
    denoise,
)  # denoise(input_file: str, output_dir: Optional[str] = None) -> str

app = Flask(__name__)

# --- Paths & Startup ----------------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(base_dir)
UPLOAD_FOLDER = os.path.join(backend_dir, "Voice Recording and Isolation")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Startup: upload folder =", UPLOAD_FOLDER)


# --- Helpers ------------------------------------------------------------------


def is_wav_like(filename: str, mimetype: Optional[str]) -> bool:
    """
    Light check: only treats file as WAV if both extension and mimetype suggest WAV.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext in (".wav", ".wave"):
        mt = (mimetype or "").lower()
        return mt in ("audio/wav", "audio/x-wav", "audio/wave")
    return False


def transcode_to_wav(input_path: str, output_path: str, target_sr: int = 48000) -> None:
    """
    Convert any audio to WAV PCM s16, mono, target_sr using ffmpeg.
    Requires ffmpeg on PATH. Verify with `ffmpeg -version`.
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i",
        input_path,  # input file
        "-ac",
        "1",  # mono
        "-ar",
        str(target_sr),  # sample rate
        "-sample_fmt",
        "s16",  # 16-bit PCM
        output_path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0 or (not os.path.exists(output_path)):
        raise RuntimeError("ffmpeg failed to transcode.\n" + (proc.stderr or ""))


def _run_denoise_async(input_path: str, mimetype: Optional[str]) -> None:
    """
    Background job: ensure proper WAV, then run DeepFilterNet denoise.
    Always transcodes to a canonical 48kHz mono WAV to avoid format surprises.
    """
    try:
        # You can skip forced transcode if strict WAV detection passes:
        # if is_wav_like(os.path.basename(input_path), mimetype):
        #     wav_path = input_path
        # else:
        safe_name = os.path.splitext(os.path.basename(input_path))[0]
        wav_path = os.path.join(UPLOAD_FOLDER, f"{safe_name}_48k_mono.wav")
        transcode_to_wav(input_path, wav_path, target_sr=48000)

        out_path = denoise(wav_path)
        print(f"[denoise] completed: {out_path}")
    except Exception as e:
        print(f"[denoise] error for {input_path}: {e}")


# --- Routes -------------------------------------------------------------------


@app.route("/save-recording", methods=["POST"])
def save_recording():
    """
    Accepts multipart/form-data with field 'audio' (preferred) or 'file'.
    Saves upload, transcodes to WAV (48k mono), and denoises in a background thread.
    Responds immediately with the saved original file path.
    """
    # Accept 'audio' (your client) or 'file' (generic)
    file_key = (
        "audio"
        if "audio" in request.files
        else ("file" if "file" in request.files else None)
    )
    if not file_key:
        return (
            jsonify({"error": "No audio file. Expect form field 'audio' or 'file'."}),
            400,
        )

    audio_file = request.files[file_key]
    if (audio_file is None) or (audio_file.filename == ""):
        return jsonify({"error": "No selected file"}), 400

    # Prepare a safe file name; keep original extension (webm/ogg/m4a/mp3/etc.)
    original_name = secure_filename(audio_file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".webm"  # sensible default if client didn’t provide one
    saved_name = f"noisy_recording_{timestamp}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, saved_name)

    try:
        audio_file.save(save_path)
        print(f"Saved upload to {save_path}")
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {e}"}), 500

    # Kick off background processing (transcode + denoise)
    Thread(
        target=_run_denoise_async,
        args=(save_path, getattr(audio_file, "mimetype", None)),
        daemon=True,
    ).start()

    return (
        jsonify(
            {"status": "success", "message": "Recording saved", "file_path": save_path}
        ),
        200,
    )


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Flask app starting on http://127.0.0.1:8000 ...")
    app.run(port=8000, debug=True)
