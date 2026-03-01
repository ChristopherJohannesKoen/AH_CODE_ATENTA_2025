# Repository Structure

## Top Level

```text
.
|- Code/
|- recordings/
|- input.wav
|- README.md
|- CONTRIBUTING.md
|- SECURITY.md
|- CODE_OF_CONDUCT.md
|- CHANGELOG.md
|- docs/
|- .github/
```

## Core Project Folder

Main project lives in:

`Code/Hackathon-2025`

Key subfolders:

- `src/frontend` React/Vite UI
- `src/backend/api` FastAPI and database manager
- `src/backend/Model` transcription and note generation pipelines
- `src/backend/voice_recording_isolation` Flask upload/denoise path
- `resources` training, transcript, and audio datasets
- `scripts` helper scripts and local utility setup
- `tests` ad hoc testing assets

## Notes on Special Paths

- There is both `src` and a quoted `'src` directory in `Code/Hackathon-2025`; the quoted tree appears to be accidental artifact output.
- Generated model outputs are stored under `src/backend/Model/Output`.

## Documentation Map

- Setup: [SETUP.md](SETUP.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- API: [API.md](API.md)
- Known issues: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
