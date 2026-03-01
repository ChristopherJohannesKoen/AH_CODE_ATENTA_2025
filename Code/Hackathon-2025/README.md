# Hackathon-2025

Clinical documentation hackathon prototype.

## Structure

- `src/frontend` React + Vite UI
- `src/backend/api` FastAPI backend and SQLite access layer
- `src/backend/Model` transcription/model pipelines
- `src/backend/voice_recording_isolation` Flask denoise/upload path
- `resources` sample transcript/audio/note data
- `scripts` local helper scripts
- `tests` ad hoc test assets

## Run Locally

### Frontend

```powershell
cd "src/frontend"
npm install
npm run dev
```

### Backend API

```powershell
cd "src/backend/api"
python app.py
```

## Environment Variables

Set provider keys before running model features:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:HUGGINGFACE_TOKEN = "your-token"
```

## Notes

- This is still a prototype and not production-ready.
- Generated artifacts and local runtime files are now git-ignored by default.

## Contributors

- Hanru Visser
- Luhandre Olivier
- Cameron Hatch
