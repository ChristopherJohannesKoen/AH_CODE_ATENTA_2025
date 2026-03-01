# Setup Guide

This guide covers a working local development setup for the current prototype.

## Prerequisites

- Windows PowerShell (project currently includes Windows-oriented scripts and paths)
- Python 3.9+
- Node.js 18+ and npm
- FFmpeg available on `PATH`

Check tools:

```powershell
python --version
node --version
npm --version
ffmpeg -version
```

## Frontend Setup

```powershell
cd "Code/Hackathon-2025/src/frontend"
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Backend Setup (FastAPI route used by frontend)

```powershell
cd "Code/Hackathon-2025/src/backend/api"
python app.py
```

Backend runs at `http://127.0.0.1:8000`.

## Optional Model Environment Bootstrap

The project includes an installer script for model dependencies:

```powershell
cd "Code/Hackathon-2025/src/backend/Model/1Calling"
python install_all.py
```

## Required Environment Variables

Set these in your shell session before using provider-backed model functionality:

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
$env:HUGGINGFACE_TOKEN = "your-huggingface-token"
```

## Typical Run Order

1. Start backend server in `src/backend/api`
2. Start frontend in `src/frontend`
3. Open the frontend URL and test recording flow

## Troubleshooting

### Frontend can load but save/edit fails

- Confirm backend is running on port `8000`.
- Confirm CORS settings in backend match frontend URL.
- The current frontend includes a call to `/update-json`, which is not implemented yet in the FastAPI backend.

### Database errors (`no such table`)

- Run DB initialization script from `src/backend/api`:

```powershell
python db_init.py
```

### Diarization errors

- Ensure `HUGGINGFACE_TOKEN` is valid.
- Ensure the account has access to required pyannote models.

### FFmpeg errors

- Install FFmpeg and confirm `ffmpeg -version` works in the same shell session.
