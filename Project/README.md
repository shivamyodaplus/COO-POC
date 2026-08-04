# Project

Full-stack application — FastAPI backend + Streamlit frontend + Milvus vector database + vLLM-powered vision/text models, orchestrated from a single entry point.

---

## Project Structure

```
Project/
├── start.sh                      ← single entry point (vLLM + Docker, GPU-aware)
├── .env                          ← local config (created from .env.example on first run)
├── .env.example                  ← config template (commit this, not .env)
│
├── docker-compose.yml            ← application service (app container)
├── docker-compose.infra.yml      ← infrastructure services (postgres, milvus, minio, etcd)
├── docker-compose.gpu.yml        ← GPU overlay (merged on top when GPU is detected)
│
├── scripts/
│   └── vllm/
│       ├── vllm_preflight.sh     ← env validation + model download
│       ├── serve_all.sh          ← start qwen-text + qwen-vl vLLM servers
│       └── stop_all.sh           ← stop vLLM servers
│
├── backend/
│   ├── pyproject.toml            ← Python dependencies (managed by uv)
│   ├── supervisord.conf          ← runs FastAPI + Streamlit in one container
│   ├── Dockerfile
│   └── app/
│       ├── main.py               ← FastAPI entry point
│       ├── api/v1/endpoints/     ← REST endpoints
│       ├── core/
│       │   ├── config.py         ← app settings (reads .env)
│       │   └── startup.py        ← lifespan: DB connect, model load, vLLM health
│       ├── models/               ← embedding + OCR model loaders
│       ├── services/
│       │   ├── vllm_service.py   ← async client for qwen-text and qwen-vl
│       │   ├── document_service.py
│       │   ├── milvus_service.py
│       │   └── postgres_service.py
│       └── utils/
└── frontend/
    └── app.py                    ← Streamlit UI
```

---

## Services

### Infrastructure (`docker-compose.infra.yml`)

| Service  | Port(s)     | Description                     |
| -------- | ----------- | ------------------------------- |
| Postgres | 5432        | Relational store + outbox       |
| Milvus   | 19530       | Vector database                 |
| MinIO    | 9000 / 9001 | Object storage (Milvus backend) |
| etcd     | —          | Milvus metadata store           |

### Application (`docker-compose.yml`)

| Service   | URL                        | Description          |
| --------- | -------------------------- | -------------------- |
| FastAPI   | http://localhost:8000      | REST API             |
| Swagger   | http://localhost:8000/docs | Interactive API docs |
| Streamlit | http://localhost:8501      | Frontend dashboard   |

### vLLM model servers (host process, not Docker)

| Model        | Port | Served name  | Purpose            |
| ------------ | ---- | ------------ | ------------------ |
| Qwen3-4B-AWQ | 8001 | qwen3-4b-awq | Text chat          |
| Qwen3-VL-4B  | 8002 | qwen3-vl-4b  | Vision + text chat |

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+) and [Docker Compose](https://docs.docker.com/compose/) (v2.20+)
- Python venv with [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) installed
- `curl`, `jq`, `ss` available on the host (`sudo apt install curl jq iproute2`)
- *(Optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Getting Started

### Step 1 — Make `start.sh` executable *(first time only)*

```bash
chmod +x start.sh
```

### Step 2 — Start everything

```bash
./start.sh
```

On **first run** `start.sh` will:

1. Detect that `.env` is missing and create it from `.env.example`
2. Prompt for `VLLM_VENV` — the path to the Python venv with vLLM installed (e.g. `/home/user/model-expose/.venv`)
3. Check whether the Qwen model weights are cached locally; if not, offer to download them (~4 GB each)
4. Start the vLLM servers (`qwen-text` on :8001, `qwen-vl` on :8002) and wait for their `/health` checks to pass
5. GPU-detect and launch Docker Compose (infrastructure + app)

On **subsequent runs** the `.env` is already in place, model weights are cached, and the vLLM servers may already be running — startup is much faster.

### Step 3 — Open the app

| Interface | URL                        |
| --------- | -------------------------- |
| Frontend  | http://localhost:8501      |
| API docs  | http://localhost:8000/docs |
| MinIO UI  | http://localhost:9001      |

---

## Configuration

All tunables live in `.env` at the project root. Copy `.env.example` as a starting point:

```bash
cp .env.example .env
```

Key variables:

| Variable           | Required      | Default                      | Description                         |
| ------------------ | ------------- | ---------------------------- | ----------------------------------- |
| `VLLM_VENV`      | **yes** | —                           | Path to venv with vLLM installed    |
| `QWEN_TEXT_PORT` | no            | `8001`                     | Port for the text model server      |
| `QWEN_VL_PORT`   | no            | `8002`                     | Port for the vision-language server |
| `QWEN_TEXT_GPU`  | no            | `0.14`                     | GPU memory fraction for qwen-text   |
| `QWEN_VL_GPU`    | no            | `0.20`                     | GPU memory fraction for qwen-vl     |
| `READY_TIMEOUT`  | no            | `900`                      | Seconds to wait for vLLM /health    |
| `VLLM_HF_HOME`   | no            | `$HOME/.cache/huggingface` | Host-side HuggingFace cache root    |

---

## Stopping the Stack

```bash
# Ctrl-C in the start.sh terminal — prompts whether to also stop vLLM servers

# Stop only the Docker services (vLLM servers keep running)
docker compose -f docker-compose.infra.yml -f docker-compose.yml down

# Stop vLLM servers separately
bash scripts/vllm/stop_all.sh

# Stop Docker services and delete all data volumes
docker compose -f docker-compose.infra.yml -f docker-compose.yml down -v
```

> If the vLLM servers were already running before `start.sh` was launched (e.g. from a previous session), `start.sh` reuses them and **never stops them automatically** on exit.

---

## Manual Docker Compose Commands

Run infrastructure and app together in the foreground:

```bash
# CPU
docker compose -f docker-compose.infra.yml -f docker-compose.yml up

# GPU
docker compose -f docker-compose.infra.yml -f docker-compose.yml -f docker-compose.gpu.yml up
```

Run only the infrastructure in the background (useful during app development):

```bash
docker compose -f docker-compose.infra.yml up -d
```

Rebuild and restart only the app container:

```bash
docker compose -f docker-compose.infra.yml -f docker-compose.yml up --build app
```

---

## Logs

Infrastructure services (`etcd`, `minio`, `milvus`) run with `attach: false` and do not clutter the terminal. Inspect them when needed:

```bash
docker compose -f docker-compose.infra.yml -f docker-compose.yml logs -f app
docker compose -f docker-compose.infra.yml logs milvus
docker compose -f docker-compose.infra.yml logs minio
```

vLLM server logs are written to `scripts/vllm/logs/`:

```bash
tail -f scripts/vllm/logs/qwen-text.log
tail -f scripts/vllm/logs/qwen-vl.log
```

---

## GPU Configuration

To change which GPU is assigned to Milvus, edit `docker-compose.gpu.yml`:

```yaml
device_ids: ["0"]        # single GPU
device_ids: ["0", "1"]   # multiple GPUs
```

To adjust how much GPU memory each vLLM model uses, edit `.env`:

```env
QWEN_TEXT_GPU=0.14   # fraction of the unified memory pool
QWEN_VL_GPU=0.20
```

Run `nvidia-smi` to find device IDs and available memory.

---

## Using the vLLM Models from the Backend

Both model endpoints are available via `app.services.vllm_service`:

```python
from app.services.vllm_service import chat_text, chat_vision

# Text
response = await chat_text([{"role": "user", "content": "Summarise this document."}])

# Vision
with open("page.png", "rb") as f:
    response = await chat_vision(f.read(), "Extract all text from this image.")
```

The backend logs a warning (not an error) at startup if the vLLM servers are unreachable, so the app still starts for non-vision workflows.
