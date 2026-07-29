#!/usr/bin/env bash
set -euo pipefail

# Detect NVIDIA GPU: requires nvidia-smi and a responsive driver
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
    echo "[start] GPU detected — launching with GPU support (docker-compose.gpu.yml overlay)"
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up "$@"
else
    echo "[start] No GPU detected — launching CPU-only"
    docker compose up "$@"
fi
