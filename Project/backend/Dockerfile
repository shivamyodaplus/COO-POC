FROM python:3.11-slim

WORKDIR /app

# Install supervisor to manage both processes
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor gcc libc6-dev git \
        libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 0 = CPU torch (default), 1 = CUDA 12.4 torch (set via docker-compose.gpu.yml)
ARG CUDA_TORCH=0

# Install deps first for layer caching (build context is project root)
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Reinstall torch/torchvision with CUDA 12.8 wheels when CUDA_TORCH=1
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$CUDA_TORCH" = "1" ]; then \
        /bin/uv pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu128 \
            --upgrade; \
        echo "[torch] CUDA 12.8 build installed"; \
    else \
        echo "[torch] CPU build (default)"; \
    fi

# PaddlePaddle has no PyPI wheel for linux/aarch64 — install manually from
# PaddlePaddle's own package index. The || true ensures build continues even
# if this fails (OCR will be gracefully disabled at runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv pip install paddlepaddle paddleocr \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    || echo "[warn] PaddlePaddle install failed — OCR disabled"

# LightGlue has no PyPI release — install from GitHub. The || true ensures
# the build continues even if GitHub is unreachable (visual matching will be
# gracefully disabled at runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv pip install git+https://github.com/cvg/LightGlue.git \
    || echo "[warn] LightGlue install failed — visual template matching disabled"

# Copy backend and frontend sources
COPY backend/app/ ./app/
COPY frontend/ ./frontend/
COPY backend/supervisord.conf ./supervisord.conf

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["supervisord", "-c", "/app/supervisord.conf"]
