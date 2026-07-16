#!/usr/bin/env bash
# Full paired run for RQ2: iterative loop (175 seeds) + budget-matched
# baseline, using this machine's local diffusers/ollama setup (see
# setup_cloud_gpu.sh). Explicit backend/flags below on purpose — don't rely
# on setup_cloud_gpu.sh's local config.py patch alone, so this still runs
# correctly even on an unpatched checkout.
#
# Usage:
#   scripts/run_full_cloud.sh            # start a fresh run
#   scripts/run_full_cloud.sh <RUN_ID>    # resume an interrupted run
#
# This runs for hours — launch it inside tmux/screen so an SSH disconnect
# doesn't kill it:
#   tmux new -s ouroboros
#   scripts/run_full_cloud.sh
#   [detach: Ctrl-b d — reattach later with: tmux attach -t ouroboros]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

echo "=== Pre-flight checks ==="
command -v ouroboros >/dev/null || { echo "ouroboros not found — did setup_cloud_gpu.sh run?" >&2; exit 1; }
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "Ollama not responding on localhost:11434" >&2; exit 1; }
ollama list | grep -q "dolphin-llama3" || { echo "dolphin-llama3 not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
ollama list | grep -q "qwen3-vl" || { echo "qwen3-vl:8b not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "nvidia-smi failed — no GPU visible" >&2; exit 1; }

mkdir -p logs
LOGFILE="logs/run_full_$(date +%Y%m%d_%H%M%S).log"

RESUME_ARGS=()
if [ $# -ge 1 ]; then
  RESUME_ARGS=(--resume "$1")
  echo "Resuming run $1"
fi

echo "=== Starting run — log: $LOGFILE ==="
ouroboros run \
  --mode full \
  --baseline \
  --target-backend diffusers \
  --judge-backend ollama \
  --judge-model qwen3-vl:8b \
  --no-aggressive-unload \
  --flux-quantize 16 \
  --flux-size 1024 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOGFILE"

echo
echo "=== Done — log written to $LOGFILE ==="
echo "Next: ouroboros report <run_id> --bls"
