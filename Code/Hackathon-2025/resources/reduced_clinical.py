import os
import time
from pathlib import Path
from typing import Tuple
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# ---------- CONFIG ----------
MODEL = "gpt-4.1"  # high accuracy for careful redaction; you may switch to a cheaper model if needed
NUM_PAIRS = 20

TRANSCRIPTS_DIR = Path(r"resources\transcript")
CLINICAL_DIR = Path(r"resources\clinical_note")
OUTPUT_DIR = Path(r"resources\reduced_clinical_notes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: max characters from each file to send (safety for very large notes).
# Set to None to disable truncation.
MAX_CHARS_TRANSCRIPT = None
MAX_CHARS_NOTE = None

SYSTEM_PROMPT = """You are a careful clinical scribe-checker.

TASK
Given:
(1) a verbatim patient–clinician conversation transcript (spoken words only),
(2) a full clinical note from the same session,

produce a REDUCED CLINICAL NOTE that:
- Preserves the structure, headings, and formatting of the original clinical note as much as reasonably possible.
- Retains ONLY information that is explicitly present in the transcript (spoken).
- For any field/line/section/value that is NOT clearly supported by the transcript, replace its value with the literal string: null
- Do NOT invent or infer. If in doubt, use null.
- Keep units and formatting for retained values exactly as in the clinical note where possible.
- If a whole section is unsupported, keep the section header but set its content to null.
- Dates, vitals, demographics, exam findings, lab/imaging orders or results, medications, diagnoses, past history, and plan items must all be backed by words in the transcript; otherwise set to null.
- If the clinical note contains templated defaults not spoken, set them to null.

OUTPUT
Return ONLY the reduced clinical note as plain text (no extra commentary).
"""

USER_PROMPT_TEMPLATE = """TRANSCRIPT (spoken only)
-----------------------
{transcript}

FULL CLINICAL NOTE (original)
-----------------------------
{clinical_note}

INSTRUCTIONS (apply strictly):
- Rewrite the FULL CLINICAL NOTE into a reduced version that retains the original structure but sets any content not explicitly spoken in the TRANSCRIPT to null.
- Do not remove headings; replace unsupported values with null.
- Do not add analysis or explanation; return only the reduced clinical note text.
"""


def read_pair(idx: int) -> Tuple[str, str, str]:
    t_path = TRANSCRIPTS_DIR / f"consultation_x{idx}.txt"
    c_path = CLINICAL_DIR / f"clinical_note_x{idx}.txt"
    if not t_path.exists():
        raise FileNotFoundError(f"Missing transcript: {t_path}")
    if not c_path.exists():
        raise FileNotFoundError(f"Missing clinical note: {c_path}")

    t_txt = t_path.read_text(encoding="utf-8", errors="replace")
    c_txt = c_path.read_text(encoding="utf-8", errors="replace")

    if MAX_CHARS_TRANSCRIPT is not None and len(t_txt) > MAX_CHARS_TRANSCRIPT:
        t_txt = t_txt[:MAX_CHARS_TRANSCRIPT]
    if MAX_CHARS_NOTE is not None and len(c_txt) > MAX_CHARS_NOTE:
        c_txt = c_txt[:MAX_CHARS_NOTE]

    return str(t_path), t_txt, c_txt


def write_reduced(idx: int, content: str):
    out_path = OUTPUT_DIR / f"reduced_note_x{idx}.txt"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def call_openai(client: OpenAI, transcript: str, clinical_note: str) -> str:
    """
    Calls the Responses API with careful instructions.
    Returns the reduced clinical note text.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript=transcript.strip(), clinical_note=clinical_note.strip()
    )

    # Using the Responses API (official client)
    # Docs: https://platform.openai.com/docs/api-reference/responses  (cited above)
    rsp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # We explicitly ask for text output only.
        temperature=0,
    )

    # The text is typically at output_text; this helper safely extracts it.
    # openai-python exposes a convenience property .output_text for Responses.
    # If unavailable in your installed version, fall back to walking the structure.
    text = getattr(rsp, "output_text", None)
    if not text:
        # Fallback: concatenate text parts
        text_parts = []
        for item in getattr(rsp, "output", []) or []:
            if getattr(item, "type", "") == "message" and getattr(
                item, "content", None
            ):
                for block in item.content:
                    if block.type == "output_text" and getattr(block, "text", None):
                        text_parts.append(block.text)
        text = "\n".join(text_parts).strip()

    if not text or not text.strip():
        raise RuntimeError("Empty response from model.")
    return text.strip()


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    successes, failures = 0, 0
    for i in range(1, NUM_PAIRS + 1):
        try:
            t_path, transcript, clinical_note = read_pair(i)
            print(f"[{i:02d}] Processing {t_path} ...")

            # Retry wrapper for transient errors
            backoff = 2
            for attempt in range(5):
                try:
                    reduced = call_openai(client, transcript, clinical_note)
                    break
                except (RateLimitError, APITimeoutError):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                except APIError as e:
                    # Retry 5xx; fail on 4xx (likely prompt/file issues)
                    if getattr(e, "status_code", 500) >= 500:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                    else:
                        raise
            else:
                raise RuntimeError("Exceeded retries for API call.")

            out_path = write_reduced(i, reduced)
            print(f"    -> Wrote {out_path}")
            successes += 1

        except Exception as e:
            print(f"    !! Failed pair {i}: {e}")
            failures += 1

    print(f"\nDone. Successes: {successes} | Failures: {failures}")


if __name__ == "__main__":
    main()
