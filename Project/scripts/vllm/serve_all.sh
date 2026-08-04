#!/usr/bin/env bash
#
# serve_all.sh — Launch qwen-text and qwen-vl vLLM servers.
# Trimmed from the full model-expose/serve_all.sh; nemotron removed.
#
#   ./serve_all.sh                  start both, health-check, smoke-test, stay attached
#   ./serve_all.sh --detach         same, but exit and leave the servers running
#   ./serve_all.sh --only qwen-text start only one server
#   ./serve_all.sh --parallel       load both at once instead of one after another
#   ./serve_all.sh --no-smoke       skip the POST smoke tests
#   ../vllm/stop_all.sh             shut everything down
#
# Tunables (env / .env):
#   VLLM_VENV       path to the Python venv with vLLM  (set by vllm_preflight.sh)
#   QWEN_TEXT_PORT  QWEN_VL_PORT
#   QWEN_TEXT_GPU   QWEN_VL_GPU   (gpu-memory-utilization fractions)
#   READY_TIMEOUT   (seconds to wait per server, default 900)
#   VLLM_HF_HOME    host-side HuggingFace cache root (default: $HOME/.cache/huggingface)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# VLLM_VENV is injected by start.sh sourcing .env; fall back to adjacent .venv.
VENV="${VLLM_VENV:-$ROOT/.venv}"
VLLM="$VENV/bin/vllm"
PY="$VENV/bin/python"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"
# Use the host-side HF cache, not the Docker container's HF_HOME.
HF_HUB="${VLLM_HF_HOME:-$HOME/.cache/huggingface}/hub"

QWEN_TEXT_PORT="${QWEN_TEXT_PORT:-8001}"
QWEN_VL_PORT="${QWEN_VL_PORT:-8002}"

QWEN_TEXT_GPU="${QWEN_TEXT_GPU:-0.14}"
QWEN_VL_GPU="${QWEN_VL_GPU:-0.20}"

READY_TIMEOUT="${READY_TIMEOUT:-900}"

DETACH=0
PARALLEL=0
SMOKE=1
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach)   DETACH=1 ;;
        --parallel) PARALLEL=1 ;;
        --no-smoke) SMOKE=0 ;;
        --only)     ONLY="${2:-}"; shift ;;
        --only=*)   ONLY="${1#*=}" ;;
        -h|--help)  sed -n '3,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
C_GRN=$'\033[32m';  C_YEL=$'\033[33m'; C_BLU=$'\033[36m'; C_BLD=$'\033[1m'

log()  { printf '%s[serve-all]%s %s\n' "$C_BLU" "$C_RESET" "$*"; }
warn() { printf '%s[serve-all]%s %s%s%s\n' "$C_BLU" "$C_RESET" "$C_YEL" "$*" "$C_RESET"; }
die()  { printf '%s[serve-all]%s %sERROR:%s %s\n' "$C_BLU" "$C_RESET" "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# ── server selection ──────────────────────────────────────────────────────────
ALL_NAMES=(qwen-text qwen-vl)
SELECTED=()
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra want <<< "$ONLY"
    for w in "${want[@]}"; do
        w="$(echo "$w" | tr -d '[:space:]')"
        found=0
        for n in "${ALL_NAMES[@]}"; do [[ "$n" == "$w" ]] && found=1; done
        [[ $found -eq 1 ]] || die "unknown server '$w' (valid: ${ALL_NAMES[*]})"
        SELECTED+=("$w")
    done
else
    SELECTED=("${ALL_NAMES[@]}")
fi

selected() { for n in "${SELECTED[@]}"; do [[ "$n" == "$1" ]] && return 0; done; return 1; }

# ── preflight ─────────────────────────────────────────────────────────────────
[[ -x "$VLLM" ]] || die "vllm CLI not found at $VLLM
  Set VLLM_VENV in .env to the venv that has vLLM installed."
command -v curl >/dev/null || die "curl is required"
command -v jq   >/dev/null || die "jq is required"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# Resolve a model to its local HF snapshot dir so we never hit the network;
# fall back to the repo id if the cache is not populated.
resolve_model() {
    local repo_id="$1" cache_name="$2" snap
    snap="$(ls -d "$HF_HUB/$cache_name"/snapshots/*/ 2>/dev/null | head -1 || true)"
    if [[ -n "$snap" && -f "${snap}config.json" ]]; then
        echo "${snap%/}"
    else
        echo "$repo_id"
    fi
}

QWEN_TEXT_MODEL="$(resolve_model Qwen/Qwen3-4B-AWQ        models--Qwen--Qwen3-4B-AWQ)"
QWEN_VL_MODEL="$(resolve_model   Qwen/Qwen3-VL-4B-Instruct models--Qwen--Qwen3-VL-4B-Instruct)"

port_busy() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

# ── launching ─────────────────────────────────────────────────────────────────
declare -A PIDS=() PORTS=() STATUS=()

spawn() {
    local name="$1" port="$2"; shift 2
    local logf="$LOG_DIR/$name.log"

    if port_busy "$port"; then
        warn "port $port already in use — skipping $name (stop it first: ./stop_all.sh)"
        STATUS[$name]="skipped"
        PORTS[$name]="$port"
        return 0
    fi

    log "starting ${C_BLD}$name${C_RESET} on port $port  ${C_DIM}(log: logs/$name.log)${C_RESET}"
    : > "$logf"
    "$@" >>"$logf" 2>&1 &
    local pid=$!
    PIDS[$name]=$pid
    PORTS[$name]=$port
    STATUS[$name]="starting"
    echo "$pid" > "$RUN_DIR/$name.pid"
}

start_qwen_text() {
    spawn qwen-text "$QWEN_TEXT_PORT" \
        "$VLLM" serve "$QWEN_TEXT_MODEL" \
        --served-model-name qwen3-4b-awq \
        --port "$QWEN_TEXT_PORT" \
        --dtype float16 \
        --max-model-len 40960 \
        --gpu-memory-utilization "$QWEN_TEXT_GPU" \
        --max-num-seqs 32 \
        --attention-backend FLASH_ATTN \
        --enable-prefix-caching
}

start_qwen_vl() {
    spawn qwen-vl "$QWEN_VL_PORT" \
        "$VLLM" serve "$QWEN_VL_MODEL" \
        --served-model-name qwen3-vl-4b \
        --port "$QWEN_VL_PORT" \
        --dtype bfloat16 \
        --max-model-len 65536 \
        --gpu-memory-utilization "$QWEN_VL_GPU" \
        --max-num-seqs 16 \
        --limit-mm-per-prompt '{"image":2,"video":0}' \
        --attention-backend FLASH_ATTN \
        --enable-prefix-caching
}

# ── readiness ─────────────────────────────────────────────────────────────────
wait_ready() {
    local name="$1"
    local port="${PORTS[$name]:-}"
    local pid="${PIDS[$name]:-}"
    local logf="$LOG_DIR/$name.log"
    local deadline=$(( SECONDS + READY_TIMEOUT )) spin='|/-\' i=0

    while (( SECONDS < deadline )); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            STATUS[$name]="dead"
            printf '\r%s[serve-all]%s %s crashed during startup%s\n' \
                "$C_BLU" "$C_RESET" "$name" "$C_RESET"
            printf '%s---- last 25 lines of logs/%s.log ----%s\n' "$C_DIM" "$name" "$C_RESET"
            tail -n 25 "$logf"
            return 1
        fi
        if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            STATUS[$name]="ready"
            printf '\r%s[serve-all]%s %s%s ready%s on :%s  (%ss)          \n' \
                "$C_BLU" "$C_RESET" "$C_GRN" "$name" "$C_RESET" "$port" "$SECONDS"
            return 0
        fi
        printf '\r%s[serve-all]%s waiting for %s %s %s(%ss)%s   ' \
            "$C_BLU" "$C_RESET" "$name" "${spin:i++%4:1}" "$C_DIM" "$SECONDS" "$C_RESET"
        sleep 2
    done

    STATUS[$name]="timeout"
    printf '\r%s[serve-all]%s %s did not become ready in %ss%s\n' \
        "$C_BLU" "$C_RESET" "$name" "$READY_TIMEOUT" "$C_RESET"
    tail -n 25 "$logf"
    return 1
}

# ── smoke tests ───────────────────────────────────────────────────────────────
smoke_text() {
    local port="$1" model="$2" label="$3"
    local body resp
    body="$(jq -n --arg m "$model" '{
        model: $m,
        messages: [{role:"user", content:"Reply with exactly: OK"}],
        max_tokens: 64,
        temperature: 0,
        chat_template_kwargs: {enable_thinking: false}
    }')"
    resp="$(curl -fsS --max-time 120 -H 'Content-Type: application/json' \
            -d "$body" "http://127.0.0.1:$port/v1/chat/completions" 2>&1)" || {
        printf '  %sPOST %s -> FAILED%s\n' "$C_RED" "$label" "$C_RESET"; return 1; }
    printf '  %sPOST %s%s -> %s\n' "$C_BLD" "$label" "$C_RESET" \
        "$(jq -r '.choices[0].message.content // "(empty)"' <<< "$resp" \
           | tr '\n' ' ' | head -c 200)"
}

smoke_vision() {
    local port="$1" model="$2" label="$3" img="$4" prompt="$5" maxtok="$6"
    local tmp resp
    [[ -f "$img" ]] || { printf '  %sPOST %s -> no test image at %s%s\n' \
        "$C_YEL" "$label" "$img" "$C_RESET"; return 1; }

    tmp="$(mktemp -d)"
    base64 -w0 "$img" > "$tmp/img.b64"
    jq -n --arg m "$model" --arg p "$prompt" --argjson mt "$maxtok" \
          --rawfile b "$tmp/img.b64" '{
        model: $m,
        messages: [{role:"user", content:[
            {type:"text", text:$p},
            {type:"image_url", image_url:{url:("data:image/png;base64," + $b)}}
        ]}],
        max_tokens: $mt,
        temperature: 0,
        top_k: 1,
        repetition_penalty: 1.1,
        skip_special_tokens: false
    }' > "$tmp/body.json"

    resp="$(curl -fsS --max-time 600 -H 'Content-Type: application/json' \
            -d "@$tmp/body.json" "http://127.0.0.1:$port/v1/chat/completions" 2>&1)"
    local rc=$?
    rm -rf "$tmp"
    if (( rc != 0 )); then
        printf '  %sPOST %s -> FAILED: %s%s\n' "$C_RED" "$label" \
            "$(head -c 200 <<< "$resp")" "$C_RESET"
        return 1
    fi
    local out
    out="$(jq -r '.choices[0].message.content // ""' <<< "$resp" | tr '\n' ' ')"
    if [[ -z "${out// /}" ]]; then
        printf '  %sPOST %s -> empty content%s (finish_reason=%s, completion_tokens=%s)\n' \
            "$C_YEL" "$label" "$C_RESET" \
            "$(jq -r '.choices[0].finish_reason // "?"' <<< "$resp")" \
            "$(jq -r '.usage.completion_tokens // "?"' <<< "$resp")"
        return 1
    fi
    printf '  %sPOST %s%s -> %s\n' "$C_BLD" "$label" "$C_RESET" "$(head -c 300 <<< "$out")"
}

run_smoke() {
    printf '\n%s=== smoke tests (live POSTs) ===%s\n' "$C_BLD" "$C_RESET"
    [[ "${STATUS[qwen-text]:-}" == "ready" ]] && \
        smoke_text   "$QWEN_TEXT_PORT" qwen3-4b-awq "qwen-text  /v1/chat/completions"
    [[ "${STATUS[qwen-vl]:-}"   == "ready" ]] && \
        smoke_vision "$QWEN_VL_PORT"   qwen3-vl-4b  "qwen-vl    /v1/chat/completions" \
                     "${SMOKE_IMAGE:-/dev/null}" "Describe this image in one short sentence." 64
    return 0
}

# ── summary ───────────────────────────────────────────────────────────────────
summary() {
    printf '\n%s=== endpoints ===%s\n' "$C_BLD" "$C_RESET"
    printf '%-12s %-18s %-7s %-8s %-9s %s\n' NAME MODEL-ID PORT PID STATUS BASE-URL
    printf '%s\n' "-----------------------------------------------------------------------"
    local -A ids=( [qwen-text]=qwen3-4b-awq [qwen-vl]=qwen3-vl-4b )
    local ok=0 bad=0
    for n in "${SELECTED[@]}"; do
        local st="${STATUS[$n]:-not-started}" col="$C_RED"
        [[ "$st" == "ready" ]] && { col="$C_GRN"; ok=$((ok+1)); } || bad=$((bad+1))
        printf '%-12s %-18s %-7s %-8s %s%-9s%s http://127.0.0.1:%s/v1\n' \
            "$n" "${ids[$n]}" "${PORTS[$n]:--}" "${PIDS[$n]:--}" \
            "$col" "$st" "$C_RESET" "${PORTS[$n]:--}"
    done
    printf '\n%d ready, %d not ready.  Logs: %s/  |  Stop: %s/stop_all.sh\n' \
        "$ok" "$bad" "${LOG_DIR#$ROOT/}" "$ROOT"
    [[ $bad -eq 0 ]]
}

shutdown_all() {
    printf '\n'
    log "shutting down..."
    for n in "${!PIDS[@]}"; do
        local pid="${PIDS[$n]}"
        kill -0 "$pid" 2>/dev/null || continue
        log "  stopping $n (pid $pid)"
        kill -TERM "$pid" 2>/dev/null
    done
    for _ in $(seq 1 30); do
        local alive=0
        for n in "${!PIDS[@]}"; do kill -0 "${PIDS[$n]}" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
    done
    for n in "${!PIDS[@]}"; do kill -KILL "${PIDS[$n]}" 2>/dev/null; done
    sleep 2
    for pid in $(pgrep -f "$VLLM serve" 2>/dev/null || true); do kill -KILL "$pid" 2>/dev/null; done
    rm -f "$RUN_DIR"/*.pid 2>/dev/null
    log "all stopped."
}

# ── main ──────────────────────────────────────────────────────────────────────
printf '%s' "$C_BLD"
cat <<'BANNER'
  vLLM launcher — qwen-text + qwen-vl
BANNER
printf '%s' "$C_RESET"
log "serving: ${SELECTED[*]}"

trap 'shutdown_all; exit 130' INT TERM

if [[ $PARALLEL -eq 1 ]]; then
    selected qwen-text && start_qwen_text
    selected qwen-vl   && start_qwen_vl
    for n in "${SELECTED[@]}"; do
        [[ "${STATUS[$n]:-}" == "skipped" ]] || wait_ready "$n"
    done
else
    for n in "${SELECTED[@]}"; do
        case "$n" in
            qwen-text) start_qwen_text ;;
            qwen-vl)   start_qwen_vl   ;;
        esac
        [[ "${STATUS[$n]:-}" == "skipped" ]] || wait_ready "$n"
    done
fi

[[ $SMOKE -eq 1 ]] && run_smoke
summary
all_ok=$?

if [[ $DETACH -eq 1 ]]; then
    trap - INT TERM
    printf '\n'
    log "detached — servers keep running.  Stop with: $ROOT/stop_all.sh"
    exit $all_ok
fi

printf '\n'
log "attached. Ctrl-C stops all servers."
while true; do
    for n in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$n]}" 2>/dev/null; then
            warn "$n (pid ${PIDS[$n]}) exited — see logs/$n.log"
            unset 'PIDS[$n]'
        fi
    done
    [[ ${#PIDS[@]} -eq 0 ]] && { warn "no servers left running."; exit 1; }
    sleep 5
done
