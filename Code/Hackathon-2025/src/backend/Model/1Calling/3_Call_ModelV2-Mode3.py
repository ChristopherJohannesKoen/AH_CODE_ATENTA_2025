#!/usr/bin/env python3
# server_mode3.py
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="universal_convo_to_json - Mode 3 Server")

SCRIPT = r"src\backend\Model\AudToSpeach\V2\universal_convo_to_json-1.2.0.py"


class RunRequest(BaseModel):
    # style profile to load
    load_model: str = Field(
        ..., description='Saved style profile name, e.g. "myclinic_v1"'
    )
    model_store: str = Field("src/backend/Model/Trained/style_profiles")

    # run inputs
    audio: str = Field(..., description="Path to audio")
    template: str = Field(..., description="Path to template JSON")
    output: str = Field(..., description="Path to output JSON")

    whisper_model: str = Field("base")
    language: Optional[str] = None
    diarize: bool = Field(False)
    hf_token: Optional[str] = None
    role_map: Optional[Dict[str, str]] = None

    # brain/llm
    use_brain: bool = Field(False)
    brain_provider: str = Field("openai")
    brain_model: str = Field("gpt-4o-mini")
    brain_persona: str = Field("doctor")

    use_llm: bool = Field(False)
    llm_provider: str = Field("openai")
    llm_model: str = Field("gpt-4o-mini")

    timeout_seconds: int = Field(0)


def _ensure_parent_dirs(path_str: str) -> None:
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _role_map_args(role_map: Optional[Dict[str, str]]) -> List[str]:
    if not role_map:
        return []
    flat = [f"{k}={v}" for k, v in role_map.items()]
    return ["--role-map", *flat] if flat else []


@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong (mode 3)"}


@app.post("/run")
def run(req: RunRequest):
    if not Path(req.audio).exists():
        raise HTTPException(400, f"audio not found: {req.audio}")
    if not Path(req.template).exists():
        raise HTTPException(400, f"template not found: {req.template}")
    _ensure_parent_dirs(req.output)

    cmd = [
        sys.executable,
        SCRIPT,
        "--mode",
        "3",
        "--load-model",
        req.load_model,
        "--model-store",
        req.model_store,
        "--audio",
        req.audio,
        "--template",
        req.template,
        "--output",
        req.output,
        "--whisper-model",
        req.whisper_model,
    ]

    if req.language:
        cmd += ["--language", req.language]
    if req.diarize:
        cmd += ["--diarize"]
    if req.hf_token:
        cmd += ["--hf-token", req.hf_token]
    cmd += _role_map_args(req.role_map)

    if req.use_brain:
        cmd += [
            "--use-brain",
            "--brain-provider",
            req.brain_provider,
            "--brain-model",
            req.brain_model,
            "--brain-persona",
            req.brain_persona,
        ]
    if req.use_llm:
        cmd += [
            "--use-llm",
            "--llm-provider",
            req.llm_provider,
            "--llm-model",
            req.llm_model,
        ]

    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=None if req.timeout_seconds == 0 else req.timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            504, f"process timed out after {req.timeout_seconds}s"
        ) from e

    return {
        "executed_cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-10000:],
        "stderr": completed.stderr[-10000:],
        "output_json": req.output,
        "transcript_txt": str(Path(req.output).with_suffix("").as_posix())
        + "_transcript.txt",
        "load_model": req.load_model,
        "model_store": req.model_store,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_mode3:app", host="0.0.0.0", port=8013, reload=False)
