import os
import subprocess
import json
from pathlib import Path

# CONFIG
NUM_PAIRS = 19
SCRIPT = r"src\backend\Model\AudToSpeach\V2\universal_convo_to_json-1.2.0.py"
EVAL_SCRIPT = r"src\backend\Model\AudToSpeach\V2\eval_quality.py"
TEMPLATE = r"src\backend\Model\Templates\clinical_note_template.json"

OUTPUT_DIR = Path(r"src\backend\Model\Output\batch_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_OUT_DIR = OUTPUT_DIR / "eval_outputs"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# GOLD-STANDARD DIRECTORIES
AUDIO_DIR = Path(r"resources\audio")
REDUCED_DIR = Path(r"resources\reduced_clinical_notes")
TRANSCRIPT_DIR = Path(r"resources\transcript")

# COMPARISON RECORD
all_records = []

for i in range(1, NUM_PAIRS + 1):
    audio_file = AUDIO_DIR / f"consultation_x{i}_combined_dialogue.mp3"
    reduced_note = REDUCED_DIR / f"reduced_note_x{i}.txt"
    transcript = TRANSCRIPT_DIR / f"consultation_x{i}.txt"

    # Model outputs
    output_json = OUTPUT_DIR / f"clinical_note_x{i}_filled.json"
    # universal_convo_to_json writes transcript alongside JSON as "<base>_transcript.txt"
    output_tx = OUTPUT_DIR / f"clinical_note_x{i}_filled_transcript.txt"

    # Build command exactly in your required format
    cmd = [
        "python", SCRIPT,
        "--mode", "1",
        "--audio", str(audio_file),
        "--template", TEMPLATE,
        "--output", str(output_json),
        "--whisper-model", "base",
        "--diarize",
        "--role-map", "SPEAKER_00=Doctor", "SPEAKER_01=Patient",
        "--use-brain",
        "--brain-provider", "openai",
        "--brain-model", "gpt-4o-mini",
        "--use-llm",
        "--llm-provider", "openai",
        "--llm-model", "gpt-4o-mini"
    ]

    print(f"[{i:02d}] Running model for {audio_file.name}...")
    subprocess.run(cmd, check=True)

    # Store record for later analysis
    all_records.append({
        "session": i,
        "audio": str(audio_file),
        "reduced_note_gold": str(reduced_note),
        "transcript_gold": str(transcript),
        "model_output_json": str(output_json),
        "model_output_transcript": str(output_tx),
    })

# Save mapping for analysis (ordered)
mapping_file = OUTPUT_DIR / "comparison_index.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=2)

print(f"\n[OK] Batch completed. Index saved -> {mapping_file}")

# ---------------------------
# Run the quality evaluation
# ---------------------------
eval_cmd = [
    "python", EVAL_SCRIPT,
    "--gt-notes-dir", str(REDUCED_DIR),
    "--gt-transcripts-dir", str(TRANSCRIPT_DIR),
    "--model-json-dir", str(OUTPUT_DIR),
    "--model-transcripts-dir", str(OUTPUT_DIR),
    "--out-dir", str(EVAL_OUT_DIR),
]

print("\n[QA] Running quality evaluation...")
subprocess.run(eval_cmd, check=True)
print(f"[QA] Done. See:\n  - {EVAL_OUT_DIR}\\per_pair_scores.csv\n  - {EVAL_OUT_DIR}\\per_pair_section_matches.csv\n  - {EVAL_OUT_DIR}\\summary.json")
