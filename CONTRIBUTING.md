# Contributing Guide

Thanks for contributing.

## Before You Start

- Read [README.md](README.md)
- Read [docs/SETUP.md](docs/SETUP.md)
- Check [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)

## Workflow

1. Create a branch from your default branch.
2. Make focused changes.
3. Run relevant checks locally.
4. Open a pull request with clear scope and test notes.

## Local Checks

Frontend:

```powershell
cd "Code/Hackathon-2025/src/frontend"
npm run lint
npm run build
```

Backend (basic sanity):

```powershell
cd "Code/Hackathon-2025/src/backend/api"
python -m compileall .
```

## Pull Request Expectations

- Describe the problem and the fix.
- Include repro and validation steps.
- Keep PRs small where possible.
- Add or update docs when behavior changes.

## Commit Hygiene

- Use clear, imperative commit messages.
- Avoid mixing unrelated refactors and feature changes.
- Do not commit secrets, tokens, or private credentials.

## Data and Artifact Policy

- Avoid adding large binary artifacts unless required.
- Prefer links or generated-on-demand scripts for heavy outputs.
- If sample data is needed, use synthetic or scrubbed data only.

## Need Help

- Open an issue using the issue templates.
- Link the issue in your PR when applicable.
