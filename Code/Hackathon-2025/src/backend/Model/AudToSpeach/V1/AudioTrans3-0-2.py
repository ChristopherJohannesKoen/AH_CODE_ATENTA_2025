import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import ffmpeg
from tqdm import tqdm

# Transcription
import whisper

# Speaker embeddings (no token required; SpeechBrain >=1.0 import path)
try:
    from speechbrain.inference import EncoderClassifier  # new path
except Exception:
    from speechbrain.pretrained import EncoderClassifier  # fallback for older SB

# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ---------------- Config ----------------
SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # "base" | "small" | "medium" | "large-v3"
TARGET_NUM_SPEAKERS: Optional[int] = 2  # set None to auto-select (1..4)
FRAME_SEC = 0.5  # analysis window
HOP_SEC = 0.25  # hop between windows
MIN_SPEECH_SEC = 0.8  # drop ultra-short segments
SILENCE_PAD_SEC = 0.2  # pad around boundaries
# ----------------------------------------


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):  # macOS only
        return "mps"
    return "cpu"


def load_audio_ffmpeg(audio_file: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio via ffmpeg -> mono 16 kHz float32 [-1, 1]."""
    try:
        out, err = (
            ffmpeg.input(str(audio_file), threads=0)
            .output(
                "pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=f"{sample_rate}"
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        return audio
    except ffmpeg.Error as e:
        msg = e.stderr.decode(errors="ignore") if e.stderr else "Unknown error"
        print(f"FFmpeg error: {msg}")
        raise


def energy_vad(wave: np.ndarray, sr: int, frame_sec: float, hop_sec: float):
    """Simple energy-based VAD. Returns (speech_mask, starts, frame_len, hop_len)."""
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("Invalid FRAME_SEC or HOP_SEC.")

    energies, starts = [], []
    for start in range(0, max(1, len(wave) - frame_len + 1), hop_len):
        chunk = wave[start : start + frame_len]
        energy = float(np.mean(chunk**2)) if len(chunk) else 0.0
        energies.append(energy)
        starts.append(start)

    energies = np.array(energies)
    thr = max(1e-8, np.median(energies) * 2.0)  # adaptive threshold
    speech_mask = energies > thr
    return speech_mask, np.array(starts), frame_len, hop_len


def merge_mask_to_segments(
    speech_mask, starts, frame_len, hop_len, sr, min_speech_sec, pad_sec
):
    """Convert frame-level mask to merged [start,end] segments in seconds."""
    segments = []
    in_speech = False
    seg_start = None

    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            in_speech = True
            seg_start = starts[i]
        elif not is_speech and in_speech:
            in_speech = False
            seg_end = starts[i] + frame_len
            segments.append([seg_start, seg_end])

    if in_speech:
        segments.append([seg_start, starts[len(speech_mask) - 1] + frame_len])

    # merge close neighbors
    merged = []
    for s, e in segments:
        if not merged:
            merged.append([s, e])
        else:
            if (s - merged[-1][1]) / sr < 0.3:
                merged[-1][1] = e
            else:
                merged.append([s, e])

    # pad + filter short
    pad = int(pad_sec * sr)
    min_len = int(min_speech_sec * sr)
    final = []
    for s, e in merged:
        s2 = max(0, s - pad)
        e2 = e + pad
        if (e2 - s2) >= min_len:
            final.append((s2 / sr, e2 / sr))

    return final


def extract_embeddings(wave: np.ndarray, sr: int, segments, device: str):
    """One ECAPA embedding per segment (avg over subwindows for long segments)."""
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
    )
    embeddings = []
    for s, e in segments:
        seg = wave[int(s * sr) : int(e * sr)]
        if len(seg) == 0:
            embeddings.append(np.zeros((192,), dtype=np.float32))
            continue

        win = int(1.5 * sr)
        hop = int(0.75 * sr)
        chunk_embs = []
        if len(seg) <= win:
            wav_t = torch.from_numpy(seg).float().unsqueeze(0).to(device)
            with torch.no_grad():
                emb_t = classifier.encode_batch(wav_t).squeeze(0).mean(dim=0)
            chunk_embs.append(emb_t.detach().cpu().numpy())
        else:
            for st in range(0, len(seg) - win + 1, hop):
                chunk = seg[st : st + win]
                wav_t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb_t = classifier.encode_batch(wav_t).squeeze(0).mean(dim=0)
                chunk_embs.append(emb_t.detach().cpu().numpy())

        emb = (
            np.mean(np.stack(chunk_embs, axis=0), axis=0)
            if chunk_embs
            else np.zeros((192,), dtype=np.float32)
        )
        embeddings.append(emb.astype(np.float32))
    return (
        np.stack(embeddings, axis=0)
        if embeddings
        else np.zeros((0, 192), dtype=np.float32)
    )


def _cluster_labels(embeddings: np.ndarray, n_clusters: int):
    """Compat for sklearn old/new APIs (metric vs affinity)."""
    try:
        # sklearn >= 1.2
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", metric="euclidean"
        )
    except TypeError:
        # older sklearn
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", affinity="euclidean"
        )
    return model.fit_predict(embeddings)


def choose_num_speakers(embeddings: np.ndarray, target: Optional[int]) -> int:
    """Use target if provided; otherwise search k=1..4 with silhouette score."""
    if target is not None:
        return max(1, int(target))
    if embeddings.shape[0] <= 1:
        return 1

    best_k, best_score = 1, -1.0
    for k in range(1, min(4, embeddings.shape[0]) + 1):
        if k == 1:
            score = -1.0
        else:
            labels = _cluster_labels(embeddings, k)
            if len(set(labels)) < 2:
                score = -1.0
            else:
                try:
                    score = silhouette_score(embeddings, labels)
                except Exception:
                    score = -1.0
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_speakers(embeddings: np.ndarray, n_speakers: int):
    if n_speakers <= 1 or embeddings.shape[0] == 0:
        return np.zeros((embeddings.shape[0],), dtype=int)
    return _cluster_labels(embeddings, n_speakers)


def diarize_local(audio_file: Path, target_speakers: Optional[int], device: str):
    """
    Token-free diarization:
      1) Energy VAD -> speech segments
      2) Speaker embeddings (SpeechBrain ECAPA) on GPU/CPU
      3) Agglomerative clustering -> speaker labels
      4) Merge adjacent same-speaker segments
    Returns: list of dicts with start, end, label (e.g., "SPEAKER_00").
    """
    wave = load_audio_ffmpeg(audio_file, sample_rate=SAMPLE_RATE)

    speech_mask, starts, frame_len, hop_len = energy_vad(
        wave, SAMPLE_RATE, FRAME_SEC, HOP_SEC
    )
    segments = merge_mask_to_segments(
        speech_mask,
        starts,
        frame_len,
        hop_len,
        SAMPLE_RATE,
        MIN_SPEECH_SEC,
        SILENCE_PAD_SEC,
    )
    if not segments:
        segments = [(0.0, len(wave) / SAMPLE_RATE)]

    embeddings = extract_embeddings(wave, SAMPLE_RATE, segments, device=device)

    k = choose_num_speakers(embeddings, target_speakers)
    labels = cluster_speakers(embeddings, k)

    diar = []
    last_lab = None
    cur_s, cur_e = None, None
    for (s, e), lab in zip(segments, labels):
        if last_lab is None:
            last_lab = lab
            cur_s, cur_e = s, e
            continue
        if lab == last_lab and (s - cur_e) < 0.3:
            cur_e = e
        else:
            diar.append(
                {"start": cur_s, "end": cur_e, "label": f"SPEAKER_{last_lab:02d}"}
            )
            last_lab = lab
            cur_s, cur_e = s, e
    if cur_s is not None:
        diar.append({"start": cur_s, "end": cur_e, "label": f"SPEAKER_{last_lab:02d}"})

    return diar


def transcribe_audio_with_diarization(audio_file: Path, output_file: Path):
    device = get_device()
    print(f"[INFO] Using device: {device}")

    model = whisper.load_model(WHISPER_MODEL, device=device)

    diar_segments = diarize_local(audio_file, TARGET_NUM_SPEAKERS, device=device)

    audio = load_audio_ffmpeg(audio_file, sample_rate=SAMPLE_RATE)

    full_transcription = []
    for seg in tqdm(diar_segments, desc="Transcribing with speakers"):
        start_sample = max(0, int(seg["start"] * SAMPLE_RATE))
        end_sample = min(len(audio), int(seg["end"] * SAMPLE_RATE))
        if end_sample <= start_sample:
            continue

        chunk = audio[start_sample:end_sample]
        chunk = whisper.pad_or_trim(np.array(chunk))
        mel = whisper.log_mel_spectrogram(chunk).to(device, non_blocking=True)

        use_fp16 = device == "cuda"
        options = whisper.DecodingOptions(fp16=use_fp16)
        result = whisper.decode(model, mel, options)

        text = (result.text or "").strip()
        if text:
            full_transcription.append(f"{seg['label']}: {text}")

    transcription_text = "\n".join(full_transcription)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    print(f"Transcription with speaker labels saved to: {output_file}")


def convert_mp3_to_wav(input_file: Path, output_file: Path):
    """Convert mp3 -> wav (mono, 16k) to be consistent with processing."""
    try:
        (
            ffmpeg.input(str(input_file))
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
    audio_file = Path(audio_path)
    output_file = Path(output_path)

    if not audio_file.exists():
        print(f"Error: File '{audio_file}' not found.")
        return

    if audio_file.suffix.lower() == ".mp3":
        wav_file = audio_file.with_suffix(".wav")
        convert_mp3_to_wav(audio_file, wav_file)
        transcribe_audio_with_diarization(wav_file, output_file)
    else:
        transcribe_audio_with_diarization(audio_file, output_file)


if __name__ == "__main__":
    main(
        r"resources\audio\consultation_x1_combined_dialogue.mp3",
        r"src\Model\AudToSpeach\V1\transcription_output-M1.txt",
    )
