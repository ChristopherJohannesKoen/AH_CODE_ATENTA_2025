# Frontend

React + Vite frontend for the Atenta clinical-note workflow.

## Scripts

```powershell
npm install
npm run dev
npm run lint
npm run build
npm run preview
```

## Environment Variables

- `VITE_API_BASE_URL` (optional, default: `http://localhost:8000`)

Example:

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

## Current UI Flow

1. Record audio in browser
2. Upload recording to backend
3. Display returned JSON
4. Edit JSON and save with `POST /update-json`
5. Approve final state
