from pydantic import BaseModel
import db_manager as manager
import db_init
import subprocess
import os
import sys
import uvicorn
import ModelV2_Mode1 as mv2

install = False
if install:
    package = [
        "fastapi",
        "uvicorn",
        "fpdf",
        "json2pdf_converter",
        "flask",
        "python-multipart",
        "torch",
        "torchaudio",
        "deepfilternet",
    ]
    for x in package:
        subprocess.check_call([sys.executable, "-m", "pip", "install", x])

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File
from fastapi import UploadFile
from pathlib import Path
import json
from fpdf import FPDF
from typing import Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your front-end dev URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_template")
def get_template(name):
    ret = manager.get_template(name)
    return ret


@app.get("/get_data_set")
def get_data_set(patient_number):
    ret = manager.query_fetch(
        "SELECT date_time, data FROM session WHERE patient_number='"
        + patient_number
        + "' ORDER BY date_time DESC;"
    )
    return ret


@app.post("/post_json")
def post_json(data: dict = Body(...)):
    manager.add_template()


# @app.post("/start")
def start(name, patient_number, data: dict = Body(...)):
    manager.save_template_json(name, data)
    manager.save_session(patient_number, data)


# @app.post("/start_from_template")
def start_from_template(name, patient_number, aud):
    # data = manager.get_template(name)
    # manager.save_session(patient_number, data)
    import subprocess

    subprocess.run(
        r"python src\backend\Model\AudToSpeach\V2\universal_convo_to_json-1.2.0.py --mode 1  --audio input.wav --template src\backend\Model\Templates\clinical_note_template.json --output 'src\backend\Model\Output\clinical_note_filled.json' --whisper-model base --diarize --role-map SPEAKER_00=Doctor SPEAKER_01=Patient --use-brain --brain-provider openai --brain-model gpt-4o-mini --use-llm --llm-provider openai --llm-model gpt-4o-mini"
    )


@app.get("/generate_txt")
def generate_txt(id):
    data = manager.query_fetch_one("SELECT data FROM session WHERE id = " + id + ";")

    data = json.loads(data)

    data2 = {}
    for x in data:
        data2[x] = manager.generate_txt(data[x])

    ret = manager.generate_txt(data2)
    print(ret)
    return ret


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

    print("name name mark")

# Ensure upload directory exists
UPLOAD_FOLDER = "recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/save-recording")
async def save_recording(file: UploadFile = File(...)):
    contents = await file.read()
    with open("input.wav", "wb") as f:
        f.write(contents)
    os.environ["OPENAI_API_KEY"] = (
        "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"
    )
    start_from_template("clinical_note_template.json", "cameron", "input.wav")

    with open(r"'src\backend\Model\Output\clinical_note_filled.json'", "r") as f:
        data = json.load(f)
        return data
    print("could not find file")
    return ""
