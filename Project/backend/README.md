# Backend

FastAPI backend service.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
cd Project/backend
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI (interactive docs) |
| http://localhost:8000/redoc | ReDoc |
| http://localhost:8000/api/v1/status | Status / health check |

## Environment Variables

Create a `.env` file in `Project/backend/`:

```env
APP_NAME=FastAPI App
VERSION=0.1.0
ENVIRONMENT=development
```

## Project Structure

```
app/
├── main.py              # FastAPI app instance
├── api/
│   ├── router.py        # Central API router (imported by main.py)
│   └── v1/
│       ├── router.py    # v1 route aggregator
│       └── endpoints/
│           └── status.py
├── core/
│   └── config.py        # App settings (pydantic-settings)
└── utils/
    └── helpers.py       # Shared utility functions
```
