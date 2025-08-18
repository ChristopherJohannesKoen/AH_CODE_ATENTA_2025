import os
import subprocess
import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
START_IDX = 11
END_IDX = 19

SCRIPT = r"src\backend\Model\AudToSpeach\V2\universal_convo_to_json-1.2.0.py"
EVAL_SCRIPT = r"src\backend\Model\AudToSpeach\V2\eval_quality.py"
TEMPLATE = r"src\backend\Model\Templates\clinical_note_template.json"

# Trained style profile name to use (Mode 3)
STYLE_PROFILE_NAME = "myclinic_v1"
STYLE_PROFILE_PATH = (
    Path(r"src\backend\Model\Trained\style_profiles")
    / f"{STYLE_PROFILE_NAME}.style.txt"
)

# Output dirs (run-specific to avoid mixing with baseline)
RUN_TAG = f"trained_{STYLE_PROFILE_NAME}_s{START_IDX:02d}_s{END_IDX:02d}"
OUTPUT_ROOT = Path(r"src\backend\Model\Output\batch_runs")
RUN_DIR = OUTPUT_ROOT / RUN_TAG
RUN_DIR.mkdir(parents=True, exist_ok=True)

EVAL_OUT_DIR = RUN_DIR / "eval_outputs"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gold-standard (ground truth) dirs
AUDIO_DIR = Path(r"resources\audio")
REDUCED_DIR = Path(r"resources\reduced_clinical_notes")
TRANSCRIPT_DIR = Path(r"resources\transcript")

# -----------------------------
# Sanity checks
# -----------------------------
if not STYLE_PROFILE_PATH.exists():
    print(
        f"[WARN] Trained style profile not found at {STYLE_PROFILE_PATH}. "
        f"Ensure you ran Mode 2 training and saved as '{STYLE_PROFILE_NAME}'. Continuing anyway..."
    )

# -----------------------------
# Batch inference (Mode 3)
# -----------------------------
all_records = []

for i in range(START_IDX, END_IDX + 1):
    audio_file = AUDIO_DIR / f"consultation_x{i}_combined_dialogue.mp3"
    reduced_note = REDUCED_DIR / f"reduced_note_x{i}.txt"
    transcript = TRANSCRIPT_DIR / f"consultation_x{i}.txt"

    # Skip if any GT files missing (avoid crashing mid-batch)
    missing = [p for p in [audio_file, reduced_note, transcript] if not p.exists()]
    if missing:
        print(f"[{i:02d}] Skipping: missing files -> {[str(m) for m in missing]}")
        continue

    # Model outputs (JSON + transcript written alongside)
    output_json = RUN_DIR / f"clinical_note_x{i}_filled.json"
    output_tx = (
        RUN_DIR / f"clinical_note_x{i}_filled_transcript.txt"
    )  # writer will create this

    # Build the exact command you specified, switching to Mode 3 + load-model
    cmd = [
        "python",
        SCRIPT,
        "--mode",
        "3",
        "--load-model",
        STYLE_PROFILE_NAME,
        "--audio",
        str(audio_file),
        "--template",
        TEMPLATE,
        "--output",
        str(output_json),
        "--whisper-model",
        "base",
        "--diarize",
        "--role-map",
        "SPEAKER_00=Doctor",
        "SPEAKER_01=Patient",
        "--use-brain",
        "--brain-provider",
        "openai",
        "--brain-model",
        "gpt-4o-mini",
        "--use-llm",
        "--llm-provider",
        "openai",
        "--llm-model",
        "gpt-4o-mini",
    ]

    print(
        f"[{i:02d}] Running model for {audio_file.name} with trained profile '{STYLE_PROFILE_NAME}'..."
    )
    subprocess.run(cmd, check=True)

    # Record for mapping
    all_records.append(
        {
            "session": i,
            "audio": str(audio_file),
            "reduced_note_gold": str(reduced_note),
            "transcript_gold": str(transcript),
            "model_output_json": str(output_json),
            "model_output_transcript": str(output_tx),
            "style_profile": STYLE_PROFILE_NAME,
            "run_dir": str(RUN_DIR),
        }
    )

# Save mapping for analysis
mapping_file = RUN_DIR / "comparison_index.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=2)
print(f"\n[OK] Batch completed. Index saved -> {mapping_file}")

# -----------------------------
# Evaluation (only this run)
# -----------------------------
eval_cmd = [
    "python",
    EVAL_SCRIPT,
    "--gt-notes-dir",
    str(REDUCED_DIR),
    "--gt-transcripts-dir",
    str(TRANSCRIPT_DIR),
    "--model-json-dir",
    str(RUN_DIR),
    "--model-transcripts-dir",
    str(RUN_DIR),
    "--out-dir",
    str(EVAL_OUT_DIR),
]

print("\n[QA] Running quality evaluation on trained run...")
subprocess.run(eval_cmd, check=True)
print(
    f"[QA] Done. See:\n  - {EVAL_OUT_DIR}\\per_pair_scores.csv\n  - {EVAL_OUT_DIR}\\per_pair_section_matches.csv\n  - {EVAL_OUT_DIR}\\summary.json"
)
