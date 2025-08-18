#!/usr/bin/env python3
# universal_convo_to_json-1.1.2.py
r"""
Audio + Template -> Structured JSON (industry-agnostic, healthcare-ready)
with optional diarization and LLM contextualization.

HOW TO RUN (PowerShell example from project root):
  python src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.1.2.py `
    --audio "resources/audio/consultation_x1_combined_dialogue.mp3" `
    --template "src/backend/Model/Templates/clinical_note_template.json" `
    --output "src/backend/Model/Output/clinical_note_filled.json" `
    --whisper-model base `
    --diarize `
    --role-map SPEAKER_00=Doctor SPEAKER_01=Patient `
    --use-llm `
    --llm-provider openai `
    --llm-model gpt-4o-mini

Notes:
- If diarization isn't configured, omit --diarize.
- Set OPENAI key before using --use-llm:
    PowerShell: $env:OPENAI_API_KEY = "sk-xxxx"
"""


import os
import re
import json
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import whisper
import ffmpeg

# Optional diarization (requires HuggingFace token + model access)
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False

from rapidfuzz import fuzz

# =========================
# Robust audio loader (kept for future pre-processing needs)
# =========================
def load_audio_ffmpeg(audio_file: str, sample_rate: int = 16000) -> np.ndarray:
    try:
        out, err = (
            ffmpeg
            .input(audio_file, threads=0)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=f"{sample_rate}")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        msg = e.stderr.decode() if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg error loading audio: {msg}") from e

@dataclass
class Utterance:
    speaker: str
    role: str
    start: float
    end: float
    text: str

# =========================
# Whisper Transcription
# =========================
class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            condition_on_previous_text=True,
            beam_size=5
        )
        return result

# =========================
# Diarization
# =========================
class Diarizer:
    def __init__(self, enable: bool = True, hf_token: Optional[str] = None):
        self.enable = enable and _HAS_PYANNOTE
        self.pipeline = None
        if self.enable:
            token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                self.enable = False
            else:
                self.pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token
                )

    def diarize(self, audio_path: str):
        if not self.enable or self.pipeline is None:
            return None
        return self.pipeline(audio_path)

# =========================
# Assign speakers to ASR segments
# =========================
def assign_speakers(asr_segments: List[Dict[str, Any]], diarization_obj) -> List[Tuple[str, float, float, str]]:
    if diarization_obj is None:
        return [("SPEAKER_00", seg["start"], seg["end"], seg["text"].strip()) for seg in asr_segments]

    turns = []
    for turn, _, speaker in diarization_obj.itertracks(yield_label=True):
        turns.append((speaker, float(turn.start), float(turn.end)))

    assigned = []
    for seg in asr_segments:
        s, e = float(seg["start"]), float(seg["end"])
        best_label, best_overlap = "SPEAKER_00", 0.0
        for spk, ts, te in turns:
            overlap = max(0.0, min(e, te) - max(s, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = spk
        assigned.append((best_label, s, e, seg["text"].strip()))
    return assigned

# =========================
# Role mapping & transcript lines
# =========================
def build_role_map(speaker_labels: List[str], user_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    role_map = {}
    for i, spk in enumerate(sorted(set(speaker_labels))):
        if user_map and spk in user_map:
            role_map[spk] = user_map[spk]
        else:
            role_map[spk] = f"Speaker {i+1}"
    return role_map

def build_transcript_lines(assigned: List[Tuple[str, float, float, str]], role_map: Dict[str, str]) -> List[str]:
    # Merge adjacent same-role lines with <1s gap for readability
    utterances: List[Utterance] = []
    for spk, s, e, txt in assigned:
        utterances.append(Utterance(speaker=spk, role=role_map.get(spk, spk), start=s, end=e, text=txt))

    merged: List[Utterance] = []
    for u in utterances:
        if merged and merged[-1].role == u.role and (u.start - merged[-1].end) < 1.0:
            merged[-1].text = (merged[-1].text + " " + u.text).strip()
            merged[-1].end = u.end
        else:
            merged.append(u)

    return [f"{u.role}: {u.text}" for u in merged]

# =========================
# Template utils (flatten & tiny setter)
# =========================
def flatten_template(template: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(template, dict):
        for k, v in template.items():
            path = f"{prefix}.{k}" if prefix else k
            items.extend(flatten_template(v, path))
    elif isinstance(template, list):
        # If list has predefined elements, include them; empty lists produce no leaves.
        for idx, v in enumerate(template):
            path = f"{prefix}[{idx}]"
            items.extend(flatten_template(v, path))
    else:
        items.append((prefix, template))
    return items

def _set_path(root: Dict[str, Any], path: str, value: Any) -> None:
    """Set a dot.path in dict-only structures (no list indexing)."""
    parts = path.split(".")
    cur = root
    for i, p in enumerate(parts):
        last = (i == len(parts) - 1)
        if last:
            cur[p] = value
        else:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]

# =========================
# Rule-based extraction (fallback)
# =========================
DATE_PAT = re.compile(
    r"\b(?:(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?))\b",
    flags=re.IGNORECASE
)
EMAIL_PAT = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
PHONE_PAT = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d\b")
AGE_PAT = re.compile(r"\b(\d{1,3})\s*(?:years? old|yo|yrs?)\b", flags=re.IGNORECASE)
NAME_ANCHOR_PAT = re.compile(r"\b(my name is|i am|this is|full name|name and(?:\s+date of birth)?|confirm your name)\b", flags=re.IGNORECASE)

def extract_by_rules(text_lines: List[str]) -> Dict[str, List[Tuple[str, float, int]]]:
    out: Dict[str, List[Tuple[str, float, int]]] = {"date": [], "email": [], "phone": [], "age": [], "name": []}
    for i, line in enumerate(text_lines):
        for m in DATE_PAT.finditer(line):
            out["date"].append((m.group(0), 0.7, i))
        for m in EMAIL_PAT.finditer(line):
            out["email"].append((m.group(0), 0.9, i))
        for m in PHONE_PAT.finditer(line):
            out["phone"].append((m.group(0), 0.6, i))
        for m in AGE_PAT.finditer(line):
            out["age"].append((m.group(1), 0.7, i))
        if NAME_ANCHOR_PAT.search(line):
            name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", line)
            if name_match:
                out["name"].append((name_match.group(1), 0.6, i))
    return out

def best_span_for_field(field_name: str, text_lines: List[str]) -> Optional[Tuple[str, float, int]]:
    tokens = re.findall(r"[A-Za-z]+", field_name)
    key = " ".join(tokens).lower().strip()
    if not key:
        return None
    best = None
    best_score = -1
    for idx, line in enumerate(text_lines):
        score = fuzz.token_set_ratio(key, line.lower())
        if score > best_score:
            best_score = score
            best = (line, idx)
    if best is None:
        return None
    line, idx = best
    conf = 0.4 + 0.35 * (best_score / 100.0)
    val = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
    return (val, conf, idx)

def fill_from_conversation(template: Dict[str, Any], convo_lines: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    filled: Dict[str, Any] = {}
    evidence: Dict[str, Dict[str, Any]] = {}
    candidates = extract_by_rules(convo_lines)
    flat = flatten_template(template)

    for path, _placeholder in flat:
        key_tail = path.split(".")[-1].lower()

        val_conf_idx = None
        if any(k in key_tail for k in ["dob", "dateofbirth", "birthdate", "date_of_birth"]):
            if candidates["date"]:
                val_conf_idx = max(candidates["date"], key=lambda x: x[1])
        elif "email" in key_tail:
            if candidates["email"]:
                val_conf_idx = max(candidates["email"], key=lambda x: x[1])
        elif any(k in key_tail for k in ["phone", "mobile", "tel", "contact_number"]):
            if candidates["phone"]:
                val_conf_idx = max(candidates["phone"], key=lambda x: x[1])
        elif any(k in key_tail for k in ["age", "years"]):
            if candidates["age"]:
                val_conf_idx = max(candidates["age"], key=lambda x: x[1])
        elif any(k in key_tail for k in ["name", "fullname", "full_name"]):
            if candidates["name"]:
                val_conf_idx = max(candidates["name"], key=lambda x: x[1])

        if val_conf_idx is None:
            maybe = best_span_for_field(key_tail, convo_lines)
            if maybe:
                val_conf_idx = maybe

        if val_conf_idx is None:
            value, conf, idx = None, 0.0, -1
            line_text = ""
        else:
            value, conf, idx = val_conf_idx
            line_text = convo_lines[idx] if 0 <= idx < len(convo_lines) else ""

        _set_path(filled, path, value)
        evidence[path] = {
            "value": value,
            "confidence": round(conf, 3),
            "line_index": idx,
            "line_text": line_text
        }

    return filled, evidence

# =========================
# LLM contextualization + validation
# =========================
from jsonschema import validate as jsonschema_validate, Draft7Validator, ValidationError

def make_json_schema_from_template(template):
    def node(t):
        if isinstance(t, dict):
            props = {k: node(v) for k, v in t.items()}
            required = list(props.keys())
            return {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False
            }
        elif isinstance(t, list):
            return {"type": "array"}
        else:
            return {"type": ["string", "number", "boolean", "null"]}
    return node(template)

def strip_to_template_keys(template, data):
    if isinstance(template, dict):
        out = {}
        for k, v in template.items():
            if isinstance(data, dict) and k in data:
                out[k] = strip_to_template_keys(v, data[k])
            else:
                out[k] = [] if isinstance(v, list) else ({} if isinstance(v, dict) else None)
        return out
    elif isinstance(template, list):
        return data if isinstance(data, list) else []
    else:
        return data if data is not None else None

class LLMExtractor:
    """
    Provider-agnostic contextualizer. Default provider: OpenAI (JSON mode).
    Requires environment variable: OPENAI_API_KEY
    """
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("openai package missing. `pip install openai`") from e
            self._client = OpenAI()
        else:
            raise NotImplementedError(f"Provider {provider} not implemented yet.")

    def _prompt(self, convo_text: str, template_obj: dict) -> Tuple[str, str]:
        system = (
            "You are a structured information extractor. "
            "Given a role-labelled conversation transcript and a JSON template, "
            "return a SINGLE JSON object with the exact same keys and nesting as the template. "
            "Do not invent keys. If absent, use null (or [] for arrays). "
            "Prefer concise strings (e.g., 'Female', '1985-08-14', '120/78 mmHg')."
        )
        user = f"""TEMPLATE (keys to preserve exactly):
{json.dumps(template_obj, ensure_ascii=False, indent=2)}

TRANSCRIPT:
{convo_text}

INSTRUCTIONS:
- Fill the TEMPLATE with values inferred from the TRANSCRIPT.
- Use ISO dates if available (YYYY-MM-DD).
- Keep units compact (e.g., "98.6 F", "120/78 mmHg", "72 bpm").
- Concise sentences/phrases for HPI/ROS/Assessment/Plan.
- ONLY return JSON. No extra text.
"""
        return system, user

    def extract(self, convo_lines: List[str], template: Dict[str, Any]) -> Dict[str, Any]:
        system, user = self._prompt("\n".join(convo_lines), template)

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        else:
            raise NotImplementedError

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM returned non-JSON: {content[:200]}...") from e

        # Keep only template keys
        data = strip_to_template_keys(template, data)

        # Validate structure
        schema = make_json_schema_from_template(template)
        try:
            jsonschema_validate(instance=data, schema=schema, cls=Draft7Validator)
        except ValidationError:
            # defensive prune + re-validate
            data = strip_to_template_keys(template, data)
            jsonschema_validate(instance=data, schema=schema, cls=Draft7Validator)

        return data

# =========================
# Main pipeline
# =========================
def parse_role_map(kvs: List[str]) -> Dict[str, str]:
    out = {}
    for kv in kvs:
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def process_audio_to_template(
    audio_path: str,
    template_path: str,
    output_json_path: str,
    output_transcript_path: Optional[str] = None,
    whisper_model: str = "base",
    language: Optional[str] = None,
    use_diarization: bool = True,
    role_map_overrides: Optional[Dict[str, str]] = None,
    hf_token: Optional[str] = None,
    use_llm: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> None:

    # 1) Transcribe
    transcriber = WhisperTranscriber(model_name=whisper_model)
    asr = transcriber.transcribe(audio_path, language=language)
    segments = asr.get("segments", [])
    if not segments:
        raise RuntimeError("No transcription segments produced by Whisper.")

    # 2) Diarize (optional)
    diarizer = Diarizer(enable=use_diarization, hf_token=hf_token)
    diarization_obj = diarizer.diarize(audio_path)

    # 3) Assign speakers
    assigned = assign_speakers(segments, diarization_obj)
    speaker_labels = [spk for (spk, _, _, _) in assigned]
    role_map = build_role_map(speaker_labels, role_map_overrides)

    # 4) Build transcript lines & save
    convo_lines = build_transcript_lines(assigned, role_map)
    if output_transcript_path is None:
        base, _ = os.path.splitext(output_json_path)
        output_transcript_path = base + "_transcript.txt"
    os.makedirs(os.path.dirname(output_transcript_path), exist_ok=True)
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(convo_lines))

    # 5) Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    # 6) Fill via LLM or rule-based fallback
    if use_llm:
        llm = LLMExtractor(provider=llm_provider, model=llm_model)
        filled = llm.extract(convo_lines, template)
        # Minimal evidence scaffold (keeps API stable)
        evidence = {}
        for path, _ in flatten_template(template):
            evidence[path] = {"value": None, "confidence": 1.0, "line_index": -1, "line_text": ""}
    else:
        filled, evidence = fill_from_conversation(template, convo_lines)

    # 7) Package & save
    payload = {
        "filled": filled,
        "evidence": evidence,
        "meta": {
            "audio_path": audio_path,
            "template_path": template_path,
            "transcript_path": output_transcript_path,
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "whisper_model": whisper_model,
            "diarization_enabled": bool(diarization_obj is not None),
            "roles": role_map,
            "llm": {
                "used": use_llm,
                "provider": llm_provider if use_llm else None,
                "model": llm_model if use_llm else None
            }
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote filled JSON -> {output_json_path}")
    print(f"[OK] Wrote transcript -> {output_transcript_path}")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Audio + Template -> Filled JSON (with optional LLM contextualization)")
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--template", required=True, help="Path to template JSON")
    ap.add_argument("--output", required=True, help="Path to output filled JSON")
    ap.add_argument("--whisper-model", default="base", help="Whisper model name (tiny/base/small/medium/large-v3)")
    ap.add_argument("--language", default=None, help="Force language code (e.g., 'en')")
    ap.add_argument("--diarize", action="store_true", help="Enable pyannote diarization (requires HF token)")
    ap.add_argument("--hf-token", default=None, help="HuggingFace token (or set env HUGGINGFACE_TOKEN)")
    ap.add_argument("--role-map", nargs="*", default=[], help='Override mapping, e.g. SPEAKER_00=Doctor SPEAKER_01=Patient')
    # LLM options
    ap.add_argument("--use-llm", action="store_true", help="Use LLM contextualizer to fill the template.")
    ap.add_argument("--llm-provider", default="openai", help="LLM provider (default: openai).")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini).")
    args = ap.parse_args()

    role_map = parse_role_map(args.role_map) if args.role_map else None

    process_audio_to_template(
        audio_path=args.audio,
        template_path=args.template,
        output_json_path=args.output,
        whisper_model=args.whisper_model,
        language=args.language,
        use_diarization=args.diarize,
        role_map_overrides=role_map,
        hf_token=args.hf_token,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

if __name__ == "__main__":
    main()
