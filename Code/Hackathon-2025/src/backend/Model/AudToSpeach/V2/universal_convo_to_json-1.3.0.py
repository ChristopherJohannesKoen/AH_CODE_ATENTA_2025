#!/usr/bin/env python3
# universal_convo_to_json-1.2.0.py
r"""
Audio + Template -> Structured JSON (industry-agnostic, healthcare-ready)
with optional diarization, Brain (doctor-style reasoning), LLM contextualization,
and trainable, reusable STYLE PROFILES (numeric modes), including text-only bootstrap.

HOW TO RUN (PowerShell example from project root):

# MODE 1: Run as-is (no training), optional Brain + LLM
  python src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.2.0.py `
    --mode 1 `
    --audio "resources/audio/consultation_x1_combined_dialogue.mp3" `
    --template "src/backend/Model/Templates/clinical_note_template.json" `
    --output "src/backend/Model/Output/clinical_note_filled.json" `
    --whisper-model base `
    --diarize `
    --role-map SPEAKER_00=Doctor SPEAKER_01=Patient `
    --use-brain `
    --brain-provider openai `
    --brain-model gpt-4o-mini `
    --use-llm `
    --llm-provider openai `
    --llm-model gpt-4o-mini `
    --output-format note

# Windows short form:
python src\backend\Model\AudToSpeach\V2\universal_convo_to_json-1.2.0.py --mode 1 --audio "resources\audio\consultation_x1_combined_dialogue.mp3" --template "src\backend\Model\Templates\clinical_note_template.json" --output "src\backend\Model\Output\clinical_note_filled.json" --whisper-model base --diarize --role-map SPEAKER_00=Doctor SPEAKER_01=Patient --use-brain --brain-provider openai --brain-model gpt-4o-mini --use-llm --llm-provider openai --llm-model gpt-4o-mini --output-format note

# MODE 2: Train a style profile; supports text-only bootstrap
  python src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.2.0.py `
    --mode 2 `
    --train-dir "src/backend/Model/Training/text_only" `
    --model-name "myclinic_textonly_v1" `
    --bootstrap `
    --bootstrap-save `
    --audio "resources/audio/consultation_x1_combined_dialogue.mp3" `
    --template "src/backend/Model/Templates/clinical_note_template.json" `
    --output "src/backend/Model/Output/clinical_note_filled.json" `
    --use-brain `
    --use-llm `
    --output-format note

# MODE 3: Use a previously saved style profile by name
  python src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.2.0.py `
    --mode 3 `
    --load-model "myclinic_textonly_v1" `
    --audio "resources/audio/consultation_x1_combined_dialogue.mp3" `
    --template "src/backend/Model/Templates/clinical_note_template.json" `
    --output "src/backend/Model/Output/clinical_note_filled.json" `
    --use-brain `
    --use-llm `
    --output-format note

Notes:
- If diarization isn't configured, omit --diarize (or skip HF token).
- Set keys before using LLM features:
    PowerShell: $env:OPENAI_API_KEY = "sk-xxxx"
    (and if diarizing) $env:HUGGINGFACE_TOKEN = "hf_xxxx"
- Training examples:
    • Preferred: each example folder (or top-level) contains one transcript .txt + one desired filled .json
    • Text-only: pass --bootstrap to auto-generate the missing .json from the .txt (use --bootstrap-save to write *.boot.json files)
"""

import os
import re
import json
import glob
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
from jsonschema import validate as jsonschema_validate, Draft7Validator, ValidationError


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
# LLM Brain (doctor-style reasoning)
# =========================
class BrainGenerator:
    """
    Generates a concise clinician-style reasoning note (no JSON).
    The output is a monologue with clear sections and explicit confidence markers.
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

    def _prompt(self, convo_text: str, persona: str = "doctor", template_keys: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        system = (
            "You are an experienced clinician writing a private reasoning note. "
            "Analyze the role-labelled transcript and produce a concise monologue (200-350 words) "
            "to help a downstream model fill a clinical note template. "
            "You may infer plausible details; if inferred, mark them clearly and include confidence 0–1 and rationale. "
            "Do NOT output JSON; use readable sections and bullet points where helpful."
        )
        keys_hint = f"\nTEMPLATE KEYS OF INTEREST:\n{json.dumps(list(template_keys.keys()), ensure_ascii=False)}\n" if isinstance(template_keys, dict) else ""
        user = f"""PERSONA: {persona}

TRANSCRIPT (role-tagged):
{convo_text}

{keys_hint}
WRITE A NOTE WITH THESE SECTIONS (non-binding but recommended):
1) Patient snapshot (demographics if inferable)
2) Key symptoms & timeline (salient positives/negatives)
3) Provisional assessment & differentials (brief rationale)
4) Suggested field inferences (non-binding):
   - Format per line: FieldName → Value  (confidence=0.x; rationale: …)
5) Missing info / follow-ups
6) Risk flags or red flags (if any)

Constraints:
- Keep it concise and clinically useful.
- If the transcript contradicts a prior inference, prefer the transcript.
- No JSON. Plain text only.
"""
        return system, user

    def generate(self, convo_lines: List[str], persona: str = "doctor", template: Optional[Dict[str, Any]] = None) -> Optional[str]:
        system, user = self._prompt("\n".join(convo_lines), persona, template)
        if self.provider == "openai":
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                return None
        else:
            return None


# =========================
# STYLE TRAINER (distills a reusable style profile from examples)
# =========================
class StyleTrainer:
    """
    Distills a STYLE_GUIDE (plain text) from example pairs:
    - transcript .txt
    - desired filled .json (or bootstrap-generated if --bootstrap used)
    Saves the guide to {model_store}/{model_name}.style.txt (+ meta.json).
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

    def _distill_prompt(self, examples: List[Tuple[str, Dict[str, Any]]], template: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "You are a data-to-JSON transformation auditor. "
            "From provided training pairs (TRANSCRIPT -> JSON OUTPUT) and the target TEMPLATE, "
            "infer the user's preferred formatting, phrasing, defaults, and per-field mapping rules. "
            "Return a STYLE_GUIDE as plain text with explicit, actionable rules. No JSON."
        )
        chunks = []
        for i, (tx, js) in enumerate(examples, start=1):
            tx_trim = tx.strip()
            if len(tx_trim) > 6000:
                tx_trim = tx_trim[:6000] + "\n[...trimmed...]"
            chunks.append(
                f"EXAMPLE {i}\nTRANSCRIPT:\n{tx_trim}\nOUTPUT JSON:\n{json.dumps(js, ensure_ascii=False, indent=2)}\n"
            )

        pairs_block = "\n".join(chunks)
        template_block = json.dumps(template, ensure_ascii=False, indent=2)

        user = f"""TEMPLATE KEYS (preserve exactly):
{template_block}

TRAINING PAIRS:
{pairs_block}

INSTRUCTIONS:
Write a STYLE_GUIDE (plain text) that captures:
- Global principles (tone, brevity, tense, units, date formats, list formatting).
- Per-field mapping rules (how values are derived, normalized, summarized).
- Defaults and fallbacks when absent in transcript.
- Examples of acceptable synonyms/aliases per field (if applicable).
- Strict constraints to avoid hallucination; if missing, advise null.
- Tie-breakers when multiple values appear.

Constraints:
- Be specific and concise.
- Plain text only. No JSON in the response.
"""
        return system, user

    def distill(self, examples: List[Tuple[str, Dict[str, Any]]], template: Dict[str, Any]) -> str:
        system, user = self._distill_prompt(examples, template)
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2,
            )
        else:
            raise NotImplementedError
        return resp.choices[0].message.content.strip()


# ---------- TEXT-ONLY BOOTSTRAP HELPERS ----------
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

    def _prompt(self, convo_text: str, template_obj: dict, brain_text: Optional[str] = None, style_guide: Optional[str] = None) -> Tuple[str, str]:
        system = (
            "You are a structured information extractor. "
            "Given a STYLE_GUIDE, an optional doctor-style reasoning note (BRAIN), "
            "a role-labelled transcript, and a JSON template, "
            "return a SINGLE JSON object with the exact same keys and nesting as the template. "
            "Never invent keys. If a value is unavailable, use null (or [] for arrays). "
            "Prefer explicit facts in TRANSCRIPT over BRAIN if they conflict. "
            "Follow STYLE_GUIDE for formatting, phrasing, and defaults."
        )
        style_block = f"\nSTYLE_GUIDE (authoritative formatting & mapping rules):\n{style_guide}\n" if style_guide else ""
        brain_block = f"\nBRAIN (doctor-style reasoning; may include inferred values):\n{brain_text}\n" if brain_text else ""
        user = f"""TEMPLATE (keys to preserve exactly):
{json.dumps(template_obj, ensure_ascii=False, indent=2)}
{style_block}{brain_block}
TRANSCRIPT (role-tagged):
{convo_text}

INSTRUCTIONS:
- Fill the TEMPLATE using STYLE_GUIDE + (BRAIN) + TRANSCRIPT.
- If conflict: prefer TRANSCRIPT, then STYLE_GUIDE, then BRAIN.
- Use ISO dates where applicable (YYYY-MM-DD).
- Keep units compact (e.g., "98.6 F", "120/78 mmHg", "72 bpm").
- Concise phrases for HPI/ROS/Assessment/Plan.
- ONLY return JSON. No extra text.
"""
        return system, user

    def extract(self, convo_lines: List[str], template: Dict[str, Any], brain_text: Optional[str] = None, style_guide: Optional[str] = None) -> Dict[str, Any]:
        system, user = self._prompt("\n".join(convo_lines), template, brain_text, style_guide)

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
            data = strip_to_template_keys(template, data)
            jsonschema_validate(instance=data, schema=schema, cls=Draft7Validator)

        return data


def _bootstrap_label_for_transcript(
    transcript_text: str,
    template: Dict[str, Any],
    use_llm: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Produce a best-effort JSON label from a plain text transcript using:
      - LLMExtractor (preferred) OR
      - rule-based fallback if LLM unavailable
    """
    lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
    if use_llm:
        try:
            extractor = LLMExtractor(provider=provider, model=model)
            return extractor.extract(lines, template, brain_text=None, style_guide=None)
        except Exception:
            # fall through to rule-based
            pass
    filled, _ev = fill_from_conversation(template, lines)
    return filled


def collect_training_examples(
    train_dir: str,
    max_examples: int = 12,
    template: Optional[Dict[str, Any]] = None,
    bootstrap: bool = False,
    bootstrap_save: bool = False,
    bootstrap_provider: str = "openai",
    bootstrap_model: str = "gpt-4o-mini",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Collect (transcript_text, filled_json) examples.
    If .json is missing and bootstrap=True, auto-generate from the .txt.
    Saving bootstrapped labels to disk requires bootstrap_save=True.
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training dir not found: {train_dir}")
    examples: List[Tuple[str, Dict[str, Any]]] = []

    def load_pair(txt_path: str, json_path: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        if json_path and os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                try:
                    js = json.load(jf)
                except json.JSONDecodeError:
                    jf.seek(0)
                    raw = jf.read().strip()
                    js = json.loads(raw)
            return (transcript, js)
        # Bootstrap if allowed
        if bootstrap and template is not None:
            js = _bootstrap_label_for_transcript(
                transcript_text=transcript,
                template=template,
                use_llm=True,
                provider=bootstrap_provider,
                model=bootstrap_model,
            )
            if bootstrap_save:
                stem, _ = os.path.splitext(txt_path)
                outp = stem + ".boot.json"
                try:
                    with open(outp, "w", encoding="utf-8") as wf:
                        json.dump(js, wf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            return (transcript, js)
        return None

    # Strategy:
    # 1) If subdirectories exist, treat each subdir as one example (first .txt + matching .json or bootstrap)
    # 2) Otherwise, treat each .txt in the root as a separate example, pairing with same-stem .json if present
    subdirs = [p for p in glob.glob(os.path.join(train_dir, "*")) if os.path.isdir(p)]
    if subdirs:
        for d in sorted(subdirs):
            txts = sorted(glob.glob(os.path.join(d, "*.txt")))
            # Prefer same-stem pairing if exists
            candidate = None
            if txts:
                txt_path = txts[0]
                stem = os.path.splitext(os.path.basename(txt_path))[0]
                # look for same-stem .json OR any .json
                same = os.path.join(d, stem + ".json")
                jsons = sorted(glob.glob(os.path.join(d, "*.json")))
                json_path = same if os.path.isfile(same) else (jsons[0] if jsons else None)
                candidate = load_pair(txt_path, json_path)
            if candidate:
                examples.append(candidate)
            if len(examples) >= max_examples:
                break
    else:
        txts = sorted(glob.glob(os.path.join(train_dir, "*.txt")))
        for txt_path in txts:
            stem = os.path.splitext(os.path.basename(txt_path))[0]
            same = os.path.join(train_dir, stem + ".json")
            json_path = same if os.path.isfile(same) else None
            candidate = load_pair(txt_path, json_path)
            if candidate:
                examples.append(candidate)
            if len(examples) >= max_examples:
                break

    if not examples:
        raise RuntimeError(
            f"No valid training examples found in {train_dir}. "
            f"Need .txt + .json per example or pass --bootstrap for text-only."
        )
    return examples


def save_style_profile(model_store: str, model_name: str, style_text: str, template: Dict[str, Any]) -> str:
    os.makedirs(model_store, exist_ok=True)
    style_path = os.path.join(model_store, f"{model_name}.style.txt")
    meta_path = os.path.join(model_store, f"{model_name}.meta.json")
    with open(style_path, "w", encoding="utf-8") as f:
        f.write(style_text.strip() + "\n")
    meta = {
        "name": model_name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "template_keys": list(template.keys()) if isinstance(template, dict) else [],
        "version": "1.0",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return style_path


def load_style_profile(model_store: str, model_name: str) -> Optional[str]:
    style_path = os.path.join(model_store, f"{model_name}.style.txt")
    if not os.path.isfile(style_path):
        return None
    with open(style_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# =========================
# LLM contextualization + validation
# =========================
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
    def coerce_scalar(v):
        # If the template expects a scalar but the model gave a list/dict,
        # flatten to a compact string to satisfy schema validation.
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items() if val is not None)
        return v

    if isinstance(template, dict):
        out = {}
        for k, v in template.items():
            if isinstance(data, dict) and k in data:
                out[k] = strip_to_template_keys(v, data[k])
            else:
                out[k] = [] if isinstance(v, list) else ({} if isinstance(v, dict) else None)
        return out
    elif isinstance(template, list):
        # Template wants an array
        return data if isinstance(data, list) else []
    else:
        # Template wants a scalar (string/number/boolean/null)
        return coerce_scalar(data) if data is not None else None


# =========================
# PLAIN-TEXT NOTE RENDERING (NEW)
# =========================
def _as_text(v):
    """Convert scalars/lists/dicts to a compact, human-readable string."""
    if v is None:
        return "null"
    if isinstance(v, list):
        vals = [_as_text(x) for x in v if x not in (None, "")]
        return ", ".join(vals) if vals else "null"
    if isinstance(v, dict):
        parts = []
        for k, val in v.items():
            if val in (None, ""):
                continue
            parts.append(f"{k}: {_as_text(val)}")
        return "; ".join(parts) if parts else "null"
    return str(v)


def render_clinical_note(filled: dict) -> str:
    """Render the 'filled' JSON into the plain-text clinical note."""
    patient = filled.get("patient", {}) or {}
    enc = filled.get("encounter", {}) or {}
    note = filled.get("clinical_note", {}) or {}
    prev = note.get("previous_history", {}) or {}
    vitals = note.get("vital_signs", {}) or {}

    lines = []
    lines.append("Clinical Note:\n")
    lines.append(f"Patient Name: {_as_text(patient.get('name'))}\n")
    lines.append(f"Date of Birth: {_as_text(patient.get('date_of_birth'))}\n")
    lines.append(f"Age: {_as_text(patient.get('age'))}\n")
    lines.append(f"Sex: {_as_text(patient.get('sex'))}\n")
    lines.append(f"Medical Record #: {_as_text(patient.get('medical_record_number'))}\n")
    lines.append(f"Date of clinic visit: {_as_text(enc.get('date_of_visit'))}\n")
    lines.append(f"Primary care provider: {_as_text(enc.get('primary_care_provider'))}\n")
    lines.append(f"Personal note: {_as_text(enc.get('personal_note'))}\n")

    lines.append(f"History of Present Illness: {_as_text(note.get('history_of_present_illness'))}\n")
    lines.append(f"Allergies: {_as_text(note.get('allergies'))}\n")
    lines.append(f"Medications: {_as_text(note.get('medications'))}\n")

    lines.append("Previous History: ")
    lines.append(f"Past Medical History: {_as_text(prev.get('past_medical_history'))}")
    lines.append(f"Past Surgical History: {_as_text(prev.get('past_surgical_history'))}")
    lines.append(f"Family History: {_as_text(prev.get('family_history'))}")
    lines.append(f"Social History: {_as_text(prev.get('social_history'))}\n")

    lines.append(f"Review of Systems: {_as_text(note.get('review_of_systems'))}\n")
    lines.append(f"Physical Exam: {_as_text(note.get('physical_exam'))}\n")

    vs = []
    if vitals.get("temperature") is not None:      vs.append(f"Temp: {_as_text(vitals.get('temperature'))}")
    if vitals.get("blood_pressure") is not None:   vs.append(f"BP: {_as_text(vitals.get('blood_pressure'))}")
    if vitals.get("heart_rate") is not None:       vs.append(f"HR: {_as_text(vitals.get('heart_rate'))}")
    if vitals.get("respiratory_rate") is not None: vs.append(f"RR: {_as_text(vitals.get('respiratory_rate'))}")
    if vitals.get("oxygen_saturation") is not None:vs.append(f"SpO2: {_as_text(vitals.get('oxygen_saturation'))}")
    lines.append("Vital Signs: " + (", ".join(vs) if vs else "null") + "\n")

    assess = note.get("assessment") or {}
    assess_text = _as_text(assess.get("summary"))
    icd = assess.get("icd10_code")
    if icd:
        assess_text = f"{assess_text} (ICD-10: {icd})"
    lines.append(f"Assessment: {assess_text}\n")

    lines.append(f"Plan: {_as_text(note.get('plan'))}\n")
    lines.append(f"Medical Decision Making: {_as_text(note.get('medical_decision_making'))}")

    return "\n".join(lines).rstrip() + "\n"


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
    use_brain: bool = False,
    brain_provider: str = "openai",
    brain_model: str = "gpt-4o-mini",
    brain_persona: str = "doctor",
    # Style profile (training/usage)
    mode: int = 1,
    train_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    load_model: Optional[str] = None,
    model_store: str = "src/backend/Model/Trained/style_profiles",
    max_train_examples: int = 12,
    # Bootstrap flags (Mode 2)
    bootstrap: bool = False,
    bootstrap_save: bool = False,
    bootstrap_provider: str = "openai",
    bootstrap_model: str = "gpt-4o-mini",
    # Output selection (NEW)
    output_format: str = "note",   # default to plain text clinical note
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

    # 4) Build transcript lines
    convo_lines = build_transcript_lines(assigned, role_map)

    # 5) Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    # 6) Generate Brain (optional)
    brain_text: Optional[str] = None
    if use_brain:
        brain = BrainGenerator(provider=brain_provider, model=brain_model)
        try:
            brain_text = brain.generate(convo_lines, persona=brain_persona, template=template)
        except Exception:
            brain_text = None  # Fail-safe: proceed without brain

    # 7) Train / Load STYLE_GUIDE depending on mode
    style_guide_text: Optional[str] = None
    used_style_profile_name: Optional[str] = None

    if mode == 2:
        if not train_dir or not model_name:
            raise ValueError("Mode 2 requires both --train-dir and --model-name.")
        trainer = StyleTrainer(provider=llm_provider, model=llm_model)
        examples = collect_training_examples(
            train_dir,
            max_examples=max_train_examples,
            template=template,
            bootstrap=bootstrap,
            bootstrap_save=bootstrap_save,
            bootstrap_provider=bootstrap_provider,
            bootstrap_model=bootstrap_model,
        )
        try:
            style_guide_text = trainer.distill(examples, template)
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
        save_style_profile(model_store, model_name, style_guide_text, template)
        used_style_profile_name = model_name

    elif mode == 3:
        if not load_model:
            raise ValueError("Mode 3 requires --load-model (the saved profile name).")
        style_guide_text = load_style_profile(model_store, load_model)
        if not style_guide_text:
            raise FileNotFoundError(f"Saved style profile '{load_model}' not found in {model_store}.")
        used_style_profile_name = load_model

    # 8) Save transcript file with Brain (and keep transcript below)
    if output_transcript_path is None or output_transcript_path == '':
        base, _ = os.path.splitext(output_json_path)
        output_transcript_path = base + "_transcript.txt"
    os.makedirs(os.path.dirname(output_transcript_path), exist_ok=True)
    lines_to_write: List[str] = []
    if brain_text:
        lines_to_write.append("### Brain (doctor-style reasoning)")
        lines_to_write.append(brain_text.strip())
        lines_to_write.append("")  # spacer
    lines_to_write.append("### Transcript (role-tagged)")
    lines_to_write.extend(convo_lines)
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_to_write))

    # 9) Fill via LLM (with STYLE_GUIDE + Brain + Transcript) or rule-based fallback
    if use_llm:
        llm = LLMExtractor(provider=llm_provider, model=llm_model)
        filled = llm.extract(convo_lines, template, brain_text=brain_text, style_guide=style_guide_text)
        evidence = {}
        for path, _ in flatten_template(template):
            evidence[path] = {"value": None, "confidence": 1.0, "line_index": -1, "line_text": ""}
    else:
        filled, evidence = fill_from_conversation(template, convo_lines)

    # 10) Package & save (NOTE vs JSON)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    if output_format.lower() == "note":
        # Write only the plain-text Clinical Note to --output (even if the extension is .json)
        note_text = render_clinical_note(filled)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(note_text)
        print(f"[OK] Wrote Clinical Note (text) -> {output_json_path}")
    else:
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
                },
                "brain": {
                    "used": use_brain,
                    "provider": brain_provider if use_brain else None,
                    "model": brain_model if use_brain else None,
                    "persona": brain_persona if use_brain else None
                },
                "style_profile": {
                    "mode": mode,
                    "used": bool(style_guide_text is not None),
                    "name": used_style_profile_name,
                    "store": model_store if style_guide_text is not None else None
                },
                "bootstrap": {
                    "enabled": bool(bootstrap),
                    "saved_labels": bool(bootstrap_save) if mode == 2 else False,
                    "provider": bootstrap_provider if bootstrap else None,
                    "model": bootstrap_model if bootstrap else None
                }
            }
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote filled JSON -> {output_json_path}")

    print(f"[OK] Wrote transcript -> {output_transcript_path}")
    if style_guide_text and used_style_profile_name:
        print(f"[OK] Style profile applied -> {used_style_profile_name} ({model_store})")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Audio + Template -> Filled JSON or Plain-Text Clinical Note (Brain + optional LLM + trainable STYLE profiles + text-only bootstrap)")
    # Numeric mode
    ap.add_argument("--mode", type=int, choices=[1, 2, 3], default=1,
                    help="1=Run as-is; 2=Train style profile from examples (and apply); 3=Use previously saved style profile.")

    # Core I/O
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--template", required=True, help="Path to template JSON")
    ap.add_argument("--output", required=True, help="Path to output (text note by default; see --output-format)")
    ap.add_argument("--whisper-model", default="base", help="Whisper model name (tiny/base/small/medium/large-v3)")
    ap.add_argument("--language", default=None, help="Force language code (e.g., 'en')")

    # Diarization
    ap.add_argument("--diarize", action="store_true", help="Enable pyannote diarization (requires HF token)")
    ap.add_argument("--hf-token", default=None, help="HuggingFace token (or set env HUGGINGFACE_TOKEN)")

    # Role mapping
    ap.add_argument("--role-map", nargs="*", default=[], help='Override mapping, e.g. SPEAKER_00=Doctor SPEAKER_01=Patient')

    # Brain
    ap.add_argument("--use-brain", action="store_true", help="Generate a doctor-style 'Brain' reasoning note and prepend it above the transcript.")
    ap.add_argument("--brain-provider", default="openai", help="Brain LLM provider (default: openai).")
    ap.add_argument("--brain-model", default="gpt-4o-mini", help="Brain LLM model (default: gpt-4o-mini).")
    ap.add_argument("--brain-persona", default="doctor", help="Persona for Brain (default: doctor).")

    # LLM contextualizer
    ap.add_argument("--use-llm", action="store_true", help="Use LLM contextualizer to fill the template.")
    ap.add_argument("--llm-provider", default="openai", help="LLM provider (default: openai).")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini).")

    # Style profile training/usage
    ap.add_argument("--train-dir", default=None, help="(Mode 2) Directory containing training examples (.txt + .json); with --bootstrap, .txt only is allowed.")
    ap.add_argument("--model-name", default=None, help="(Mode 2) Name to save the trained style profile under.")
    ap.add_argument("--load-model", default=None, help="(Mode 3) Name of previously saved style profile to load.")
    ap.add_argument("--model-store", default="src/backend/Model/Trained/style_profiles", help="Directory to store/load style profiles.")
    ap.add_argument("--max-train-examples", type=int, default=12, help="Maximum number of training examples to use.")

    # Text-only bootstrap flags (for Mode 2)
    ap.add_argument("--bootstrap", action="store_true", help="(Mode 2) Allow training from .txt-only examples by auto-generating JSON labels.")
    ap.add_argument("--bootstrap-save", action="store_true", help="(Mode 2) Save generated labels as *.boot.json alongside the .txt files.")
    ap.add_argument("--bootstrap-provider", default="openai", help="(Mode 2) Provider to use for bootstrap labeling (default: openai).")
    ap.add_argument("--bootstrap-model", default="gpt-4o-mini", help="(Mode 2) Model to use for bootstrap labeling (default: gpt-4o-mini).")

    # Output selection
    ap.add_argument(
        "--output-format",
        choices=["json", "note"],
        default="note",
        help="Write either the original JSON payload ('json') or a plain-text clinical note ('note') to --output."
    )

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
        use_brain=args.use_brain,
        brain_provider=args.brain_provider,
        brain_model=args.brain_model,
        brain_persona=args.brain_persona,
        mode=args.mode,
        train_dir=args.train_dir,
        model_name=args.model_name,
        load_model=args.load_model,
        model_store=args.model_store,
        max_train_examples=args.max_train_examples,
        bootstrap=args.bootstrap,
        bootstrap_save=args.bootstrap_save,
        bootstrap_provider=args.bootstrap_provider,
        bootstrap_model=args.bootstrap_model,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
