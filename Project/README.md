# Project

Full-stack application — FastAPI backend + Streamlit frontend + Milvus vector database, all running in a single Docker setup.

---

## Project Structure

```
Project/
├── start.sh                  ← recommended entry point (auto-detects GPU)
├── docker-compose.yml        ← base stack (CPU)
├── docker-compose.gpu.yml    ← GPU overlay for Milvus
├── backend/
│   ├── pyproject.toml        ← Python dependencies (managed by uv)
│   ├── supervisord.conf      ← runs FastAPI + Streamlit in one container
│   ├── Dockerfile
│   └── app/
│       ├── main.py           ← FastAPI app entry point
│       ├── api/
│       │   ├── router.py     ← central API router
│       │   └── v1/
│       │       ├── router.py
│       │       └── endpoints/
│       │           └── status.py
│       ├── core/
│       │   └── config.py     ← app settings (.env support)
│       └── utils/
│           └── helpers.py
└── frontend/
    └── app.py                ← Streamlit UI
```

---

## Services

| Service       | URL                        | Description          |
| ------------- | -------------------------- | -------------------- |
| FastAPI       | http://localhost:8000      | REST API             |
| Swagger UI    | http://localhost:8000/docs | Interactive API docs |
| Streamlit     | http://localhost:8501      | Frontend dashboard   |
| Milvus        | localhost:19530            | Vector database      |
| MinIO Console | http://localhost:9001      | Object storage UI    |

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.0+)
- *(Optional)* NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU-accelerated Milvus

---

## Getting Started

### Step 1 — Clone / navigate to the project

```bash
cd Project
```

### Step 2 — (Optional) Configure environment variables

Create a `.env` file inside `backend/` to override defaults:

```env
APP_NAME=My App
VERSION=0.1.0
ENVIRONMENT=development
```

### Step 3 — Make `start.sh` executable *(first time only)*

```bash
chmod +x start.sh
```

> This is a one-time step. Once set, you never need to run it again.

### Step 4 — Start the stack

**Recommended — auto-detects GPU:**

```bash
./start.sh --build
```

**CPU only (explicit):**

```bash
docker compose up --build
```

**GPU only (explicit):**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

> On first run, Docker will pull the Milvus, MinIO, and etcd images from the internet (~2 GB total). Ensure your machine has an active internet connection.

### Step 5 — Wait for all services to be healthy

Milvus takes ~60–90 seconds to initialise. The `app` container starts automatically once Milvus passes its health check. Monitor progress with:

```bash
docker compose ps
docker compose logs -f
```

### Step 6 — Open the app

- Frontend → http://localhost:8501
- API docs → http://localhost:8000/docs

---

## Stopping the Stack

```bash
# Stop containers (preserves volumes / data)
docker compose down

# Stop and delete all data volumes
docker compose down -v
```

---

## Viewing Logs

Milvus, MinIO, and etcd run silently in the background (`attach: false`) so their output does not clutter the terminal. Only the `app` logs are streamed by default.

To inspect infrastructure logs when needed:

```bash
# All services
docker compose logs -f

# App only (default when using docker compose up)
docker compose logs -f app

# Milvus / MinIO / etcd individually
docker compose logs milvus
docker compose logs minio
docker compose logs etcd
```

---

## Connecting to Milvus from the App

The `MILVUS_URI` environment variable is pre-configured inside the app container:

```
MILVUS_URI=http://milvus:19530
```

Install the client in `backend/pyproject.toml` and connect:

```python
from pymilvus import MilvusClient
import os

client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
```

---

## GPU Configuration

To change which GPU is assigned to Milvus, edit `docker-compose.gpu.yml`:

```yaml
device_ids: ["0"]        # single GPU
device_ids: ["0", "1"]   # multiple GPUs
```

Run `nvidia-smi` to find your GPU device IDs.
