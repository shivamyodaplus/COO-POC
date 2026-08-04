#!/usr/bin/env bash
#
# vllm_preflight.sh — validate env, vLLM installation, and model cache.
# Called by start.sh before launching any vLLM server.
#
# If .env is missing it is auto-created from .env.example; VLLM_VENV is then
# prompted interactively (it has no sensible default).
# If a required model is not cached the user is offered an inline download.

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

C_RESET=$'\033[0m'
C_RED=$'\033[31m'
C_GRN=$'\033[32m'
C_YEL=$'\033[33m'
C_BLU=$'\033[36m'
C_BLD=$'\033[1m'

info() { printf '%s[preflight]%s %s\n'              "$C_BLU" "$C_RESET" "$*"; }
ok()   { printf '%s[preflight]%s %s%s%s\n'          "$C_BLU" "$C_RESET" "$C_GRN" "$*" "$C_RESET"; }
warn() { printf '%s[preflight]%s %s%s%s\n'          "$C_BLU" "$C_RESET" "$C_YEL" "$*" "$C_RESET"; }
die()  { printf '%s[preflight]%s %sERROR:%s %s\n'   "$C_BLU" "$C_RESET" "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ensure .env exists — create from .env.example, prompt for VLLM_VENV
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    warn ".env not found — creating from .env.example"
    [[ -f "$ENV_EXAMPLE" ]] || die ".env.example not found at $PROJECT_ROOT"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    info "Created $ENV_FILE"

    printf '\n%s=== vLLM first-time setup ===%s\n' "$C_BLD" "$C_RESET"
    printf 'VLLM_VENV is required: path to a Python venv with vLLM installed.\n'
    printf 'Example: /home/%s/Documents/model-expose/.venv\n\n' "$(whoami)"

    while true; do
        printf 'Enter VLLM_VENV path: '
        read -r venv_input </dev/tty || die "Cannot read from terminal. Set VLLM_VENV in .env manually."
        venv_input="${venv_input%/}"   # strip trailing slash

        if [[ -z "$venv_input" ]] || [[ "$venv_input" == "." ]] || [[ "$venv_input" == ".." ]]; then
            warn "Please enter an absolute path to the venv, e.g. /home/$(whoami)/Documents/model-expose/.venv"
            continue
        fi

        if [[ ! -x "$venv_input/bin/vllm" ]]; then
            warn "No vllm binary at $venv_input/bin/vllm"
            printf '  Install vLLM there first, or enter a different path.\n'
            printf '  Accept this path anyway and continue? [y/N] '
            read -r force </dev/tty || force="N"
            [[ "$force" =~ ^[Yy]$ ]] || continue
        fi

        # Write VLLM_VENV into the newly created .env
        sed -i "s|^VLLM_VENV=.*|VLLM_VENV=$venv_input|" "$ENV_FILE"
        ok "VLLM_VENV saved to .env: $venv_input"
        break
    done
    printf '\n'
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Source .env (idempotent — caller may have already done this)
# ─────────────────────────────────────────────────────────────────────────────
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# ─────────────────────────────────────────────────────────────────────────────
# 3. Validate VLLM_VENV
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "${VLLM_VENV:-}" ]]; then
    die "VLLM_VENV is not set in $ENV_FILE
  Edit .env and set it to the Python venv that has vLLM installed, e.g.:
    VLLM_VENV=/home/$(whoami)/Documents/model-expose/.venv"
fi

VLLM_BIN="$VLLM_VENV/bin/vllm"
if [[ ! -x "$VLLM_BIN" ]]; then
    die "vllm binary not found at $VLLM_BIN
  Install vLLM inside that venv:
    $VLLM_VENV/bin/pip install vllm"
fi

ok "vLLM binary: $VLLM_BIN"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Validate required system tools
# ─────────────────────────────────────────────────────────────────────────────
missing_tools=()
for tool in curl jq ss; do
    command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
done
if (( ${#missing_tools[@]} > 0 )); then
    die "Required system tools not found: ${missing_tools[*]}
  Install them:  sudo apt install ${missing_tools[*]}"
fi
ok "System tools: curl, jq, ss"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check / download models
# ─────────────────────────────────────────────────────────────────────────────
PY="$VLLM_VENV/bin/python"
# Use VLLM_HF_HOME for the host cache; fall back to $HOME/.cache/huggingface.
# This is deliberately separate from the Docker container's HF_HOME.
HF_HUB="${VLLM_HF_HOME:-$HOME/.cache/huggingface}/hub"

declare -A MODELS
MODELS["qwen-text"]="Qwen/Qwen3-4B-AWQ|models--Qwen--Qwen3-4B-AWQ"
MODELS["qwen-vl"]="Qwen/Qwen3-VL-4B-Instruct|models--Qwen--Qwen3-VL-4B-Instruct"

for name in "qwen-text" "qwen-vl"; do
    IFS='|' read -r repo_id cache_name <<< "${MODELS[$name]}"
    snap="$(ls -d "$HF_HUB/$cache_name/snapshots"/*/ 2>/dev/null | head -1 || true)"

    if [[ -n "$snap" && -f "${snap}config.json" ]]; then
        ok "Model cached: $name  ($cache_name)"
        continue
    fi

    warn "Model not found in local cache: $name ($repo_id)"
    printf '  Expected at: %s/%s\n' "$HF_HUB" "$cache_name"
    printf '  Download now? (~4 GB) [y/N] '
    read -r ans </dev/tty 2>/dev/null || ans="N"

    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        printf '\n  To download manually:\n'
        printf '    %s -m huggingface_hub.commands.huggingface_cli download %s\n\n' "$PY" "$repo_id"
        die "Required model not available: $name. Aborting startup."
    fi

    info "Downloading $repo_id (this may take several minutes) ..."
    if ! "$PY" -m huggingface_hub.commands.huggingface_cli download "$repo_id"; then
        die "Download failed for $repo_id
  Possible causes: no internet access, missing HuggingFace token, or disk space."
    fi
    ok "Downloaded: $name"
done

ok "Preflight complete — all requirements satisfied."
