#!/usr/bin/env python3
"""
Evaluate model outputs against ground-truth across up to 20 paired sessions.

Inputs (directories):
- --gt-notes-dir            : contains reduced_note_x#.txt
- --gt-transcripts-dir      : contains consultation_x#.txt
- --model-json-dir          : contains clinical_note_filled_x#.json  (or any *.json with x#)
- --model-transcripts-dir   : contains clinical_note_filled_transcript_x#.txt (or any *.txt with x#)

Outputs:
- per_pair_scores.csv       : per-stem transcript & note metrics (+ coverage)
- per_pair_section_matches.csv : section-level ROUGE-L matches (GT section → best model section)
- summary.json              : macro-averages across all stems & a quick missing-files report

Metrics:
- Transcript: WER (↓), BLEU-4 (↑), ROUGE-L F1 (↑), token Jaccard (↑)
- Notes (JSON→text vs GT note): BLEU-4 (↑), ROUGE-L F1 (↑), token Jaccard (↑)
- Coverage: fraction of non-empty JSON leaves (↑)

No external deps required.
"""

import os
import re
import json
import csv
import math
import argparse
import collections
from typing import List, Dict, Tuple, Any

# ---------------- Text normalization & tokenization ----------------

def normalize_text(t: str) -> str:
    t = t.lower()
    # remove bracketed tags like [noise], [inaudible], etc.
    t = re.sub(r"\[[^\]]+\]", " ", t)
    # keep alnum, whitespace, / ' -
    t = re.sub(r"[^\w\s/'-]+", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(t: str) -> List[str]:
    return normalize_text(t).split()

# ---------------- Core metrics: WER, BLEU-4, ROUGE-L, Jaccard ----------------

def wer(ref: str, hyp: str) -> float:
    r, h = tokenize(ref), tokenize(hyp)
    n, m = len(r), len(h)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # replace
            )
    return dp[n][m] / n

def ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    return collections.Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def bleu(ref: str, hyp: str, max_n: int = 4) -> Dict[str, Any]:
    r, h = tokenize(ref), tokenize(hyp)
    if not h:
        return {"BLEU": 0.0, "BP": 0.0, "precisions": [0.0]*max_n}
    precisions = []
    for n in range(1, max_n+1):
        rc = ngram_counts(r, n)
        hc = ngram_counts(h, n)
        overlap = sum(min(c, rc.get(ng, 0)) for ng, c in hc.items())
        total = sum(hc.values())
        precisions.append(overlap/total if total else 0.0)
    # brevity penalty
    BP = 1.0 if len(h) > len(r) else math.exp(1 - len(r)/max(1, len(h)))
    # geometric mean of precisions
    gm = math.exp(sum(math.log(p if p > 0 else 1e-9) for p in precisions)/max_n)
    return {"BLEU": BP * gm, "BP": BP, "precisions": precisions}

def lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        ai = a[i]
        row = dp[i]
        nxt = dp[i+1]
        for j in range(n):
            if ai == b[j]:
                nxt[j+1] = row[j] + 1
            else:
                nxt[j+1] = max(row[j+1], nxt[j])
    return dp[m][n]

def rouge_l(ref: str, hyp: str) -> Dict[str, float]:
    r, h = tokenize(ref), tokenize(hyp)
    if not r or not h:
        return {"R": 0.0, "P": 0.0, "F1": 0.0}
    L = lcs_len(r, h)
    R, P = L/len(r), L/len(h)
    F1 = (2*R*P)/(R+P) if (R+P) else 0.0
    return {"R": R, "P": P, "F1": F1}

def jaccard(ref: str, hyp: str) -> float:
    a, b = set(tokenize(ref)), set(tokenize(hyp))
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

# ---------------- Helpers for extracting/evaluating content ----------------

def extract_transcript_from_model_file(txt: str) -> str:
    """
    Model transcripts typically include:
      ### Brain (doctor-style reasoning)
      ...
      ### Transcript (role-tagged)
      Doctor: ...
      Patient: ...
    We return only the transcript block if that header exists; otherwise full text.
    """
    lines = txt.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("### transcript"):
            start = i + 1
            break
    return "\n".join(lines[start:]).strip() if start is not None else txt

def flatten_strings(data: Any) -> str:
    out = []
    def _walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, (str, int, float)) and str(k).strip():
                    out.append(str(k))
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        else:
            if x is not None and str(x).strip():
                out.append(str(x))
    _walk(data)
    return " ".join(out)

def split_sections_from_note(note_text: str) -> Dict[str, str]:
    """
    Splits ground-truth free-text note by headers shaped like:
      'Physical Exam:'  (header alone on line)
    Returns {normalized_header: body_text}
    """
    lines = note_text.splitlines()
    sections: Dict[str, str] = {}
    cur = "document"
    buf: List[str] = []

    def flush():
        nonlocal buf, cur
        if buf:
            sections[cur] = "\n".join(buf).strip()
            buf = []

    for ln in lines:
        m = re.match(r"^\s*([A-Za-z/ \-\(\)]+):\s*$", ln)
        if m:
            flush()
            cur = m.group(1).strip().lower()
        else:
            buf.append(ln)
    flush()
    return sections

def json_sections(filled: Dict[str, Any]) -> Dict[str, str]:
    """
    Project model JSON to canonical section-like text buckets for matching:
    """
    out: Dict[str, str] = {}
    cn = (filled or {}).get("clinical_note", {}) if isinstance(filled, dict) else {}
    def add(name, val):
        if val is None: return
        if isinstance(val, dict):
            buf = []
            for k, v in val.items():
                if v is None: continue
                if isinstance(v, (str, int, float)): buf.append(str(v))
                elif isinstance(v, list): buf.extend(str(x) for x in v if x is not None)
            if buf:
                out[name] = " ".join(buf)
        elif isinstance(val, list):
            out[name] = "; ".join(str(x) for x in val if x is not None)
        elif isinstance(val, str):
            if val.strip():
                out[name] = val

    add("history of present illness (hpi)", cn.get("hpi"))
    add("review of systems", cn.get("ros"))
    add("past medical history", cn.get("past_medical_history") or (filled or {}).get("past_medical_history"))
    add("medications", cn.get("medications") or (filled or {}).get("medications"))
    add("allergies", cn.get("allergies") or (filled or {}).get("allergies"))
    add("physical exam", cn.get("physical_exam"))
    if isinstance(cn.get("vitals"), dict):
        out["vitals"] = " ".join(f"{k}: {v}" for k, v in cn["vitals"].items() if v is not None)
    elif isinstance(cn.get("vitals"), str):
        out["vitals"] = cn["vitals"]
    add("assessment", cn.get("assessment"))
    add("plan", cn.get("plan"))
    # catch-all for anything else
    if "free_text" in cn:
        add("clinical note", cn.get("free_text"))
    return out

def leaf_stats(data: Any) -> Tuple[int, int]:
    """
    Count (total_leaves, non_empty_leaves) in a nested JSON-like structure.
    A leaf is any non-dict/non-list entry (including None).
    Non-empty: not None, not "", and not empty list/dict after basic normalization.
    """
    total, non_empty = 0, 0
    def _walk(x):
        nonlocal total, non_empty
        if isinstance(x, dict):
            if not x:
                total += 1  # empty dict as a leaf-like
            for v in x.values(): _walk(v)
        elif isinstance(x, list):
            if not x:
                total += 1  # empty list as a leaf-like
            for v in x: _walk(v)
        else:
            total += 1
            if x is None: return
            s = str(x).strip()
            if s != "": non_empty += 1
    _walk(data)
    return total, non_empty

# ---------------- Pairwise evaluation ----------------

def evaluate_pair(gt_note_path: str, gt_trans_path: str, model_json_path: str, model_transcript_path: str) -> Dict[str, Any]:
    # Load files
    with open(gt_note_path, "r", encoding="utf-8") as f: gt_note = f.read()
    with open(gt_trans_path, "r", encoding="utf-8") as f: gt_trans = f.read()
    with open(model_transcript_path, "r", encoding="utf-8") as f: model_tx_raw = f.read()
    with open(model_json_path, "r", encoding="utf-8") as f: model_json = json.load(f)

    model_tx = extract_transcript_from_model_file(model_tx_raw)
    filled = model_json.get("filled", {})

    # Transcript metrics
    wer_val = wer(gt_trans, model_tx)
    bleu_tx = bleu(gt_trans, model_tx, max_n=4)
    rouge_tx = rouge_l(gt_trans, model_tx)
    jacc_tx = jaccard(gt_trans, model_tx)

    # Note metrics (flattened JSON vs GT text note)
    model_note_text = flatten_strings(filled)
    bleu_nt = bleu(gt_note, model_note_text, max_n=4)
    rouge_nt = rouge_l(gt_note, model_note_text)
    jacc_nt = jaccard(gt_note, model_note_text)

    # Coverage
    leaves_total, leaves_non_empty = leaf_stats(filled)
    coverage = (leaves_non_empty / leaves_total) if leaves_total else 0.0

    # Section-level matching for explainability
    gt_secs = split_sections_from_note(gt_note)
    mdl_secs = json_sections(filled)
    sec_rows = []
    for sec_name, gt_text in gt_secs.items():
        best_key, best = None, {"F1": 0.0, "R": 0.0, "P": 0.0}
        for mk, mv in mdl_secs.items():
            r = rouge_l(gt_text, mv)
            if r["F1"] > best["F1"]:
                best_key, best = mk, r
        sec_rows.append({
            "gt_section": sec_name,
            "matched_model_section": (best_key or ""),
            "rougeL_R": best["R"], "rougeL_P": best["P"], "rougeL_F1": best["F1"]
        })

    return {
        "transcript": {
            "wer": wer_val,
            "bleu": bleu_tx["BLEU"], "bp": bleu_tx["BP"], "precisions": bleu_tx["precisions"],
            "rougeL": rouge_tx, "jaccard": jacc_tx
        },
        "note": {
            "bleu": bleu_nt["BLEU"], "bp": bleu_nt["BP"], "precisions": bleu_nt["precisions"],
            "rougeL": rouge_nt, "jaccard": jacc_nt
        },
        "coverage": coverage,
        "note_section_matches": sec_rows
    }

# ---------------- Dataset scanning & I/O ----------------

def find_stem(path: str) -> str:
    """
    Returns the numeric 'x#' part (without the 'x'), e.g. '1' for reduced_note_x1.txt
    """
    m = re.search(r"x(\d+)", os.path.basename(path).lower())
    return m.group(1) if m else None

def map_by_stem(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    out = {}
    for fn in os.listdir(root):
        if not fn.lower().endswith(exts): continue
        s = find_stem(fn)
        if s: out[s] = os.path.join(root, fn)
    return out

def mean(vals: List[float]) -> float:
    return sum(vals)/len(vals) if vals else 0.0

def run_eval(gt_notes_dir: str, gt_trans_dir: str, model_json_dir: str, model_tx_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    gt_notes = map_by_stem(gt_notes_dir, (".txt",))
    gt_tx    = map_by_stem(gt_trans_dir, (".txt",))
    mdl_json = map_by_stem(model_json_dir, (".json",))
    mdl_tx   = map_by_stem(model_tx_dir, (".txt",))

    common = sorted(set(gt_notes) & set(gt_tx) & set(mdl_json) & set(mdl_tx), key=lambda x: int(x))
    missing = {
        "in_gt_notes_only": sorted(set(gt_notes) - set(common)),
        "in_gt_tx_only":    sorted(set(gt_tx) - set(common)),
        "in_model_json_only": sorted(set(mdl_json) - set(common)),
        "in_model_tx_only": sorted(set(mdl_tx) - set(common)),
    }

    # Per-pair results
    pair_rows = []
    section_rows = []

    for s in common:
        res = evaluate_pair(gt_notes[s], gt_tx[s], mdl_json[s], mdl_tx[s])

        # per-pair flat row
        pair_rows.append({
            "stem": s,
            # transcript
            "WER": round(res["transcript"]["wer"], 6),
            "BLEU4_tx": round(res["transcript"]["bleu"], 6),
            "ROUGE_L_F1_tx": round(res["transcript"]["rougeL"]["F1"], 6),
            "Jaccard_tx": round(res["transcript"]["jaccard"], 6),
            # note
            "BLEU4_note": round(res["note"]["bleu"], 6),
            "ROUGE_L_F1_note": round(res["note"]["rougeL"]["F1"], 6),
            "Jaccard_note": round(res["note"]["jaccard"], 6),
            # coverage
            "JSON_coverage": round(res["coverage"], 6),
        })

        # section matches
        for r in res["note_section_matches"]:
            section_rows.append({
                "stem": s,
                "gt_section": r["gt_section"],
                "matched_model_section": r["matched_model_section"],
                "ROUGE_L_R": round(r["rougeL_R"], 6),
                "ROUGE_L_P": round(r["rougeL_P"], 6),
                "ROUGE_L_F1": round(r["rougeL_F1"], 6),
            })

    # Write CSVs
    pair_csv = os.path.join(out_dir, "per_pair_scores.csv")
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "stem",
            "WER", "BLEU4_tx", "ROUGE_L_F1_tx", "Jaccard_tx",
            "BLEU4_note", "ROUGE_L_F1_note", "Jaccard_note",
            "JSON_coverage",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in pair_rows: w.writerow(r)

    sec_csv = os.path.join(out_dir, "per_pair_section_matches.csv")
    with open(sec_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["stem", "gt_section", "matched_model_section", "ROUGE_L_R", "ROUGE_L_P", "ROUGE_L_F1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in section_rows: w.writerow(r)

    # Summary stats
    def col(vals, key): 
        return [v[key] for v in vals]

    summary = {
        "pairs_evaluated": len(common),
        "stems": common,
        "missing_files": missing,
        "macro_averages": {
            "WER": mean(col(pair_rows, "WER")),
            "BLEU4_tx": mean(col(pair_rows, "BLEU4_tx")),
            "ROUGE_L_F1_tx": mean(col(pair_rows, "ROUGE_L_F1_tx")),
            "Jaccard_tx": mean(col(pair_rows, "Jaccard_tx")),
            "BLEU4_note": mean(col(pair_rows, "BLEU4_note")),
            "ROUGE_L_F1_note": mean(col(pair_rows, "ROUGE_L_F1_note")),
            "Jaccard_note": mean(col(pair_rows, "Jaccard_note")),
            "JSON_coverage": mean(col(pair_rows, "JSON_coverage")),
        },
        "artifacts": {
            "per_pair_scores_csv": pair_csv,
            "per_pair_section_matches_csv": sec_csv
        }
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"[OK] Evaluated {len(common)} pairs.")
    if common:
        print(" Averages:")
        for k, v in summary["macro_averages"].items():
            print(f"  - {k:>17}: {v:.4f}")
    else:
        print(" No complete pairs found. Check your directories and file stems (x1..x20).")
    print(f" CSV: {pair_csv}")
    print(f" CSV: {sec_csv}")
    print(f" JSON summary: {os.path.join(out_dir, 'summary.json')}")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Objective evaluation of transcripts & clinical notes across paired sessions.")
    ap.add_argument("--gt-notes-dir", required=True, help="Directory with reduced_note_x#.txt")
    ap.add_argument("--gt-transcripts-dir", required=True, help="Directory with consultation_x#.txt")
    ap.add_argument("--model-json-dir", required=True, help="Directory with clinical_note_filled_x#.json (or any *.json with x#)")
    ap.add_argument("--model-transcripts-dir", required=True, help="Directory with clinical_note_filled_transcript_x#.txt (or any *.txt with x#)")
    ap.add_argument("--out-dir", default="eval_outputs", help="Directory to write results")
    args = ap.parse_args()

    run_eval(
        gt_notes_dir=args.gt_notes_dir,
        gt_trans_dir=args.gt_transcripts_dir,
        model_json_dir=args.model_json_dir,
        model_tx_dir=args.model_transcripts_dir,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
