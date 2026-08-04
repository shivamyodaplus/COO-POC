#!/usr/bin/env bash
# stop_all.sh — stop every vLLM server started by serve_all.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VLLM_VENV:-$ROOT/.venv}"
RUN_DIR="$ROOT/run"
PORTS=("${QWEN_TEXT_PORT:-8001}" "${QWEN_VL_PORT:-8002}")

TARGETS=()

add_target() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    for t in ${TARGETS[@]+"${TARGETS[@]}"}; do [[ "$t" == "$pid" ]] && return 0; done
    TARGETS+=("$pid")
}

# 1. PIDs recorded by serve_all.sh
for f in "$RUN_DIR"/*.pid; do
    [[ -e "$f" ]] || continue
    pid="$(cat "$f" 2>/dev/null || true)"
    add_target "$pid"
    rm -f "$f"
done

# 2. Anything holding our ports (e.g. a server started by hand)
for port in "${PORTS[@]}"; do
    for pid in $(ss -ltnp "sport = :$port" 2>/dev/null \
                 | grep -oP 'pid=\K[0-9]+' | sort -u || true); do
        add_target "$pid"
    done
done

# 3. Any stray `vllm serve` processes from this venv that the above missed
for pid in $(pgrep -f "$VENV/bin/vllm serve" 2>/dev/null || true); do
    add_target "$pid"
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "nothing running."
    exit 0
fi

echo "stopping ${#TARGETS[@]} process(es): ${TARGETS[*]}"
for pid in "${TARGETS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done

for _ in $(seq 1 30); do
    alive=0
    for pid in "${TARGETS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ $alive -eq 0 ]] && break
    sleep 1
done

for pid in "${TARGETS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "force-killing pid $pid"
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

# vLLM spawns EngineCore children that can outlive a killed parent.
sleep 2
for pid in $(pgrep -f "$VENV/bin/vllm serve" 2>/dev/null || true); do
    echo "force-killing leftover engine pid $pid"
    kill -KILL "$pid" 2>/dev/null || true
done

echo "done."
