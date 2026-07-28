FROM python:3.11-slim

WORKDIR /app

# Install supervisor to manage both processes
RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install deps first for layer caching (build context is project root)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen

# Copy backend and frontend sources
COPY backend/app/ ./app/
COPY frontend/app.py ./frontend/app.py
COPY backend/supervisord.conf ./supervisord.conf

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["supervisord", "-c", "/app/supervisord.conf"]
