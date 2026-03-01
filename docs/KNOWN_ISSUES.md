# Known Issues

This repository is a prototype. The following items are known and should be treated as active cleanup work.

## Critical

- Credentials and tokens appear in committed files/scripts.
- SQL queries are assembled via string concatenation in DAO/API paths.

## High

- Frontend calls `POST /update-json`, but the FastAPI backend does not expose this route.
- Several mode wrapper scripts call `uvicorn.run("server_modeX:app", ...)` with module names that do not exist as files.
- `POST /post_json` calls `manager.add_template()` without required arguments.
- Quoted output paths in one backend path can write/read from an accidental `'src` directory tree.

## Medium

- DB test script assumes initialized schema (`session` table may not exist until `db_init.py` runs).
- Repository tracks large binary artifacts and `scripts/node_modules`.
- Multiple backend implementations overlap (`FastAPI` and `Flask`) with similarly named endpoints.

## Operational Notes

- Frontend build and lint pass in current state.
- Backend behavior is partially functional but not production-ready.

