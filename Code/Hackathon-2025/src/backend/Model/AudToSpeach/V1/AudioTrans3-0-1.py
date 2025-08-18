import os
from pathlib import Path
import numpy as np
import torch
import ffmpeg
from tqdm import tqdm

# IMPORTANT: make sure you installed OpenAI's whisper package: 'openai-whisper'
import whisper

# pyannote for diarization
from pyannote.audio import Pipeline


# --------- Config ---------
# Prefer reading your HF token from an env var for safety:
#   set HUGGINGFACE_TOKEN=hf_********************************
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "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"

SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # change to "base", "medium", etc. if needed
# --------------------------


def load_audio_ffmpeg(audio_file: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio via ffmpeg -> mono 16 kHz float32 [-1, 1]."""
    try:
        out, err = (
            ffmpeg
            .input(str(audio_file), threads=0)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=f"{sample_rate}")
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        return audio
    except ffmpeg.Error as e:
        msg = e.stderr.decode(errors="ignore") if e.stderr else "Unknown error"
        print(f"FFmpeg error: {msg}")
        raise


def diarize_audio(audio_file: Path):
    """Run pyannote speaker diarization."""
    try:
        if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
            raise RuntimeError(
                "Missing/invalid Hugging Face token. "
                "Set HUGGINGFACE_TOKEN env var or edit HF_TOKEN."
            )

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HF_TOKEN
        )
        return pipeline(str(audio_file))
    except Exception as e:
        print(f"Error during diarization: {e}")
        raise


def transcribe_audio_with_diarization(audio_file: Path, output_file: Path):
    # Load Whisper and choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL).to(device)

    # Diarize
    diarization_result = diarize_audio(audio_file)

    # Load audio samples
    audio = load_audio_ffmpeg(audio_file, sample_rate=SAMPLE_RATE)

    full_transcription = []

    # itertracks returns (segment, track, label) when yield_label=True
    for segment, _, speaker_label in tqdm(
        diarization_result.itertracks(yield_label=True),
        desc="Transcribing with speakers"
    ):
        start_sample = max(0, int(segment.start * SAMPLE_RATE))
        end_sample = min(len(audio), int(segment.end * SAMPLE_RATE))
        if end_sample <= start_sample:
            continue

        # Whisper expects 16 kHz mono float32 array length ~30s; we pad/trim per chunk
        audio_chunk = audio[start_sample:end_sample]
        audio_chunk = whisper.pad_or_trim(np.array(audio_chunk))
        mel = whisper.log_mel_spectrogram(audio_chunk).to(device)

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        text = (result.text or "").strip()
        if text:
            full_transcription.append(f"{speaker_label}: {text}")

    transcription_text = "\n".join(full_transcription)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    print(f"Transcription with speaker labels saved to: {output_file}")


def convert_mp3_to_wav(input_file: Path, output_file: Path):
    """Convert mp3 -> wav (mono, 16k) to be consistent with processing."""
    try:
        (
            ffmpeg
            .input(str(input_file))
            .output(str(output_file), ac=1, ar=SAMPLE_RATE)
            .overwrite_output()
            .run()
        )
        print(f"Converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        msg = e.stderr.decode(errors="ignore") if e.stderr else "Unknown error"
        print(f"FFmpeg error during conversion: {msg}")
        raise


def main(audio_path: str, output_path: str):
    # Robust, Windows-safe paths
    audio_file = Path(audio_path)
    output_file = Path(output_path)

    if not audio_file.exists():
        print(f"Error: File '{audio_file}' not found.")
        return

    # Convert mp3 if needed
    if audio_file.suffix.lower() == ".mp3":
        wav_file = audio_file.with_suffix(".wav")
        convert_mp3_to_wav(audio_file, wav_file)
        transcribe_audio_with_diarization(wav_file, output_file)
    else:
        transcribe_audio_with_diarization(audio_file, output_file)


if __name__ == "__main__":
    # Use raw string or forward slashes on Windows!
    # Example with raw string:
    main(
        r"resources\audio\consultation_x1_combined_dialogue.mp3",
        r"src\Model\AudToSpeach\V1\transcription_output-M1.txt"
    )
