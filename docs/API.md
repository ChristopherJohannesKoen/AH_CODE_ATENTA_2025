# API Reference

This document describes the currently exposed API surfaces found in the repository.

## Primary Backend (FastAPI)

Source: `Code/Hackathon-2025/src/backend/dao/app.py`

Base URL (local): `http://127.0.0.1:8000`

## Endpoints

### `GET /get_template`

- Query params:
  - `name` (string)
- Purpose: fetch a template by name from SQLite
- Response: template JSON object

### `GET /get_data_set`

- Query params:
  - `patient_number` (string)
- Purpose: fetch session records for a patient
- Response: list of `(date_time, data)` tuples

### `POST /post_json`

- Body: JSON object
- Purpose: template insertion path (currently incomplete in implementation)
- Response: implementation-dependent

### `GET /generate_txt`

- Query params:
  - `id` (string/integer)
- Purpose: generate text representation from a stored session JSON
- Response: plain text content

### `POST /save-recording`

- Content type: `multipart/form-data`
- Form field:
  - `file` (`UploadFile`)
- Purpose:
  - Save uploaded audio to `input.wav`
  - Trigger model generation command
  - Return generated JSON file
- Response: JSON payload from generated output file

## Example: Upload Recording

```bash
curl -X POST "http://127.0.0.1:8000/save-recording" \
  -F "file=@input.wav"
```

## Frontend Contract Notes

Frontend currently calls:

- `POST /save-recording` (implemented)
- `POST /update-json` (not currently implemented in FastAPI backend)

## Secondary Backend (Flask)

Source: `Code/Hackathon-2025/src/backend/Voice Recording and Isolation/pull_from_server.py`

Exposes:

- `POST /save-recording`

This service performs upload + transcode + background denoise and returns quickly with file path info.

## OpenAPI Docs

When FastAPI service is running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

