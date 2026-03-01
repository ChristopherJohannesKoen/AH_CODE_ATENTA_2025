# Atenta Hackathon 2025

Prototype repository for an AI-assisted clinical documentation workflow built during Hackathon 2025.

## Project Status

This repository is currently in **prototype** state. It contains active experiments, generated artifacts, and multiple backend paths (FastAPI and Flask variants).

## Quick Links

- Setup guide: [docs/SETUP.md](docs/SETUP.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API reference: [docs/API.md](docs/API.md)
- Repository map: [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)
- Known issues: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Quick Start

### 1) Frontend (React + Vite)

```powershell
cd "Code/Hackathon-2025/src/frontend"
npm install
npm run dev
```

Frontend default dev URL: `http://localhost:5173`

### 2) Backend (FastAPI path currently used by frontend)

```powershell
cd "Code/Hackathon-2025/src/backend/dao"
python app.py
```

Backend URL: `http://127.0.0.1:8000`

## Environment Variables

Set these before running model features that require external providers:

- `OPENAI_API_KEY` for OpenAI features
- `HUGGINGFACE_TOKEN` for diarization features

## Repository Layout (high level)

```text
.
|- Code/
|  |- Hackathon-2025/
|  |  |- src/
|  |  |  |- frontend/
|  |  |  |- backend/
|  |  |- resources/
|  |  |- scripts/
|  |  |- testing/
|- recordings/
```

## Important Notes

- This repository contains large binary artifacts (audio, generated outputs). See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
- Before publishing publicly, run a secret scan and remove or rotate any exposed credentials.
- No explicit license file is currently included.

