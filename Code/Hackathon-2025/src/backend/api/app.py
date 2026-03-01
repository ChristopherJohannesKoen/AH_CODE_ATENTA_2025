try:
    from . import db_manager as manager
    from . import db_init
except ImportError:
    import db_manager as manager
    import db_init
import subprocess
import os
import sys
import uvicorn
import json
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_AUDIO_PATH = PROJECT_ROOT / "input.wav"
MODEL_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "Model"
    / "AudToSpeach"
    / "V2"
    / "universal_convo_to_json-1.2.0.py"
)
TEMPLATE_PATH = (
    PROJECT_ROOT / "src" / "backend" / "Model" / "Templates" / "clinical_note_template.json"
)
MODEL_OUTPUT_PATH = (
    PROJECT_ROOT / "src" / "backend" / "Model" / "Output" / "clinical_note_filled.json"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your front-end dev URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TemplatePayload(BaseModel):
    name: str
    data: dict


class UpdatePayload(BaseModel):
    data: dict
    session_id: Optional[int] = None


@app.get("/get_template")
def get_template(name: str):
    return manager.get_template(name)


@app.get("/get_data_set")
def get_data_set(patient_number: str):
    return manager.get_data_set(patient_number)


@app.post("/post_json")
def post_json(payload: TemplatePayload):
    manager.save_template_json(payload.name, payload.data)
    return {"status": "saved", "template_name": payload.name}


def start(name: str, patient_number: str, data: dict = Body(...)):
    manager.save_template_json(name, data)
    session_id = manager.save_session(patient_number, data)
    return {"status": "saved", "session_id": session_id}


def start_from_template():
    if not MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"Model script not found: {MODEL_SCRIPT}")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")

    command = [
        sys.executable,
        str(MODEL_SCRIPT),
        "--mode",
        "1",
        "--audio",
        str(INPUT_AUDIO_PATH),
        "--template",
        str(TEMPLATE_PATH),
        "--output",
        str(MODEL_OUTPUT_PATH),
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
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


@app.get("/generate_txt")
def generate_txt(id: int):
    data = manager.get_session_data(id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session {id} not found.")
    structured = json.loads(data)
    data2 = {}
    for x in structured:
        data2[x] = manager.generate_txt(structured[x])

    return manager.generate_txt(data2)


@app.post("/update-json")
def update_json(payload: UpdatePayload):
    if payload.session_id is not None:
        manager.overwrite_data(payload.session_id, payload.data)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_PATH.write_text(
        json.dumps(payload.data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "updated"}


@app.post("/save-recording")
async def save_recording(file: UploadFile = File(...)):
    contents = await file.read()
    with open(INPUT_AUDIO_PATH, "wb") as f:
        f.write(contents)
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Configure environment variables first.",
        )
    try:
        start_from_template()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed with code {exc.returncode}: {exc.stderr}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not MODEL_OUTPUT_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Expected output file not found: {MODEL_OUTPUT_PATH}",
        )

    with open(MODEL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    db_init.db_init(False)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
