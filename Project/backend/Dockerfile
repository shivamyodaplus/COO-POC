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

# Install deps first for layer caching (build context is project root)
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# 0 = CPU torch (default), 1 = CUDA 12.8 torch (set via docker-compose.gpu.yml)
# Declared here (after uv sync) so switching GPU/CPU does not bust the uv sync cache layer.
ARG CUDA_TORCH=0

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

# PaddlePaddle + PaddleOCR — required for hybrid search text component.
# uv does not bundle pip in the venv; install pip first, then use it because
# uv's strict platform-marker resolver rejects paddleocr on aarch64.
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv pip install pip \
    && /app/.venv/bin/python -m pip install --no-cache-dir paddlepaddle paddleocr \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

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
