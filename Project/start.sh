#!/usr/bin/env bash
#
# start.sh — single entry point for the full application stack.
#
# Flow:
#   1. Bootstrap / validate .env (create from .env.example if missing)
#   2. Validate vLLM venv + model cache (prompt to download if absent)
#   3. Start qwen-text + qwen-vl vLLM servers on the host (skip if already running)
#   4. Wait for both /health endpoints to pass
#   5. Start Docker Compose (GPU-aware)
#   6. On exit: ask whether to stop vLLM servers (they're expensive to restart)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/vllm"

C_RESET=$'\033[0m'
C_YEL=$'\033[33m'
C_BLU=$'\033[36m'
C_GRN=$'\033[32m'
C_RED=$'\033[31m'

log()  { printf '%s[start]%s %s\n'             "$C_BLU" "$C_RESET" "$*"; }
warn() { printf '%s[start]%s %s%s%s\n'         "$C_BLU" "$C_RESET" "$C_YEL" "$*" "$C_RESET"; }
die()  { printf '%s[start]%s %sERROR:%s %s\n'  "$C_BLU" "$C_RESET" "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# ── Pre-load .env if it already exists so port vars are available early ────────
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

QWEN_TEXT_PORT="${QWEN_TEXT_PORT:-8001}"
QWEN_VL_PORT="${QWEN_VL_PORT:-8002}"

# ── Phase 0 & 1: Preflight — env bootstrap + vLLM binary + model cache ────────
log "Running preflight checks..."
bash "$SCRIPTS_DIR/vllm_preflight.sh"

# Re-source .env after preflight (it may have just been created with VLLM_VENV)
set -a
source "$PROJECT_ROOT/.env"
set +a
QWEN_TEXT_PORT="${QWEN_TEXT_PORT:-8001}"
QWEN_VL_PORT="${QWEN_VL_PORT:-8002}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

# ── Phase 2: Start vLLM servers if not already listening ──────────────────────
port_listening() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

VLLM_SERVERS_WERE_RUNNING=0
if port_listening "$QWEN_TEXT_PORT" && port_listening "$QWEN_VL_PORT"; then
    log "vLLM servers already running on :${QWEN_TEXT_PORT} (qwen-text) and :${QWEN_VL_PORT} (qwen-vl) — reusing."
    VLLM_SERVERS_WERE_RUNNING=1
else
    log "Starting vLLM servers (qwen-text :${QWEN_TEXT_PORT}, qwen-vl :${QWEN_VL_PORT}) ..."
    bash "$SCRIPTS_DIR/serve_all.sh" --only qwen-text,qwen-vl --detach --no-smoke

    # Phase 3: Poll until both /health endpoints pass
    deadline=$(( SECONDS + READY_TIMEOUT ))
    spin='|/-\'
    for port in "$QWEN_TEXT_PORT" "$QWEN_VL_PORT"; do
        i=0
        while ! curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )); then
                echo
                die "vLLM on :$port did not become ready within ${READY_TIMEOUT}s.
  Check logs at $SCRIPTS_DIR/logs/ for details."
            fi
            printf '\r%s[start]%s waiting for vLLM on :%s  %s  (%ss)   ' \
                "$C_BLU" "$C_RESET" "$port" "${spin:i++%4:1}" "$SECONDS"
            sleep 2
        done
        printf '\r%s[start]%s %svLLM on :%s ready%s (%ss)                   \n' \
            "$C_BLU" "$C_RESET" "$C_GRN" "$port" "$C_RESET" "$SECONDS"
    done
fi

# ── Cleanup handler (idempotent) ──────────────────────────────────────────────
_CLEANUP_DONE=0
cleanup() {
    [[ $_CLEANUP_DONE -eq 1 ]] && return
    _CLEANUP_DONE=1

    printf '\n'
    log "Stopping Docker Compose..."
    docker compose \
        -f "$PROJECT_ROOT/docker-compose.infra.yml" \
        -f "$PROJECT_ROOT/docker-compose.yml" \
        down 2>/dev/null || true

    if [[ $VLLM_SERVERS_WERE_RUNNING -eq 1 ]]; then
        log "vLLM servers were already running before this session — leaving them up."
        log "Stop manually with:  bash $SCRIPTS_DIR/stop_all.sh"
        return
    fi

    printf '%s[start]%s Stop vLLM servers? [y/N] ' "$C_BLU" "$C_RESET"
    read -r ans </dev/tty 2>/dev/null || ans="N"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        log "Stopping vLLM servers..."
        bash "$SCRIPTS_DIR/stop_all.sh"
        log "vLLM servers stopped."
    else
        log "vLLM servers left running on :${QWEN_TEXT_PORT} and :${QWEN_VL_PORT}."
        log "Stop them later with:  bash $SCRIPTS_DIR/stop_all.sh"
    fi
}

trap 'cleanup; exit 130' INT TERM

# ── Phase 4: Launch Docker Compose (GPU-aware) ────────────────────────────────
if command -v nvidia-smi &>/dev/null \
        && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
    log "GPU detected — launching with GPU support (docker-compose.gpu.yml overlay)"
    docker compose \
        -f "$PROJECT_ROOT/docker-compose.infra.yml" \
        -f "$PROJECT_ROOT/docker-compose.yml" \
        -f "$PROJECT_ROOT/docker-compose.gpu.yml" \
        up "$@"
else
    log "No GPU detected — launching CPU-only"
    docker compose \
        -f "$PROJECT_ROOT/docker-compose.infra.yml" \
        -f "$PROJECT_ROOT/docker-compose.yml" \
        up "$@"
fi

# Normal exit (docker compose up finished on its own)
cleanup
