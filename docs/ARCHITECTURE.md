# Architecture

## Overview

The repository contains a prototype stack with:

- React frontend (`Code/Hackathon-2025/src/frontend`)
- FastAPI backend (`Code/Hackathon-2025/src/backend/dao/app.py`)
- Audio-to-structured-note pipeline (`Code/Hackathon-2025/src/backend/Model/AudToSpeach/V2`)
- Supporting training/evaluation resources (`Code/Hackathon-2025/resources`)

## High-Level Flow

```text
User (browser)
  -> React UI (record/edit/approve)
  -> POST /save-recording (FastAPI, localhost:8000)
  -> audio saved as input.wav
  -> model pipeline script execution (subprocess)
  -> JSON output file generated
  -> JSON returned to frontend
  -> user reviews/edits/approves
```

## Main Components

## 1) Frontend

- Tech: React + Vite + Axios
- Primary file: `src/frontend/src/App.jsx`
- Responsibilities:
  - Capture audio with `MediaRecorder`
  - Upload recording to backend
  - Show returned JSON
  - Allow manual JSON edits

## 2) API Backend (FastAPI)

- Primary file: `src/backend/dao/app.py`
- Responsibilities:
  - Expose endpoints used by frontend
  - Save uploaded audio
  - Trigger model processing
  - Return structured output

## 3) Model Pipeline

- Primary script family:
  - `src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.2.0.py`
  - `src/backend/Model/AudToSpeach/V2/universal_convo_to_json-1.3.0.py`
- Responsibilities:
  - Whisper transcription
  - Optional diarization (pyannote)
  - Optional LLM-assisted JSON fill
  - Optional style profile training/inference modes

## 4) Data and Artifacts

- Training/evaluation resources: `Code/Hackathon-2025/resources`
- Generated model outputs: `Code/Hackathon-2025/src/backend/Model/Output`
- Recordings and intermediate audio appear in multiple locations

## Alternate Service Path (Legacy/Parallel)

- There is also a Flask service:
  - `src/backend/Voice Recording and Isolation/pull_from_server.py`
- It has overlapping endpoint naming (`/save-recording`) but different behavior.

## Current Architectural Risks

- Multiple backend entrypoints and mixed service styles (FastAPI + Flask)
- Hardcoded credentials in older scripts
- Endpoint mismatch between frontend and backend for JSON update action
- Generated artifacts tracked in repository

