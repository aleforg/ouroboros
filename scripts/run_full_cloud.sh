#!/usr/bin/env bash
# Full paired run for RQ2: iterative loop (175 seeds) + budget-matched
# baseline, using this machine's local diffusers/ollama setup (see
# setup_cloud_gpu.sh). Explicit backend/flags below on purpose — don't rely
# on setup_cloud_gpu.sh's local config.py patch alone, so this still runs
# correctly even on an unpatched checkout.
#
# Usage:
#   scripts/run_full_cloud.sh                            # FLUX.2-klein (default)
#   scripts/run_full_cloud.sh --backend qwen-image       # Qwen-Image-2.1
#   scripts/run_full_cloud.sh <RUN_ID>                   # resume an interrupted run
#   scripts/run_full_cloud.sh --backend qwen-image <RUN_ID>
#
# The two backends are two different MODELS, not two platforms: running both
# over the same 175 seeds is what separates "this model is skewed" from "the
# FLUX family is skewed". Sampling params differ accordingly — see TARGET_ARGS.
#
# This runs for hours — launch it inside tmux/screen so an SSH disconnect
# doesn't kill it:
#   tmux new -s ouroboros
#   scripts/run_full_cloud.sh
#   [detach: Ctrl-b d — reattach later with: tmux attach -t ouroboros]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

BACKEND=diffusers
if [ "${1:-}" = "--backend" ]; then
  [ $# -ge 2 ] || { echo "--backend needs a value: diffusers | qwen-image" >&2; exit 1; }
  BACKEND="$2"
  shift 2
fi

# Qwen-Image-2.1 has a 7B visual transformer + an 8B Qwen3-VL text encoder.
# Use NF4 on both components: conservative, unmeasured VRAM estimates are
# 12 GB (4-bit), 20 GB (8-bit), 36 GB (bf16), plus activation headroom.
# These are not a guarantee of fitting at 1024 px with Ollama also resident.
# Steps are left to the per-backend default (config.TARGET_DEFAULTS): 4 for the
# distilled klein, 40 for Qwen-Image-2.1 (no CFG).
case "$BACKEND" in
  diffusers)  TARGET_ARGS=(--target-quantize 16 --target-size 1024) ;;
  qwen-image) TARGET_ARGS=(--target-quantize 4  --target-size 1024) ;;
  *) echo "Unknown backend '$BACKEND' — use 'diffusers' or 'qwen-image'." >&2; exit 1 ;;
esac

echo "=== Pre-flight checks ==="
command -v ouroboros >/dev/null || { echo "ouroboros not found — did setup_cloud_gpu.sh run?" >&2; exit 1; }
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "Ollama not responding on localhost:11434" >&2; exit 1; }
ollama list | grep -q "dolphin-llama3" || { echo "dolphin-llama3 not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
ollama list | grep -q "qwen3-vl:8b-instruct" || { echo "qwen3-vl:8b-instruct not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "nvidia-smi failed — no GPU visible" >&2; exit 1; }

mkdir -p logs
LOGFILE="logs/run_full_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"

RESUME_ARGS=()
if [ $# -ge 1 ]; then
  RESUME_ARGS=(--resume "$1")
  echo "Resuming run $1"
fi

echo "=== Starting run — backend: $BACKEND — log: $LOGFILE ==="
ouroboros run \
  --mode full \
  --baseline \
  --target-backend "$BACKEND" \
  --judge-backend ollama \
  --judge-model qwen3-vl:8b-instruct \
  --no-aggressive-unload \
  "${TARGET_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOGFILE"

echo
echo "=== Done — log written to $LOGFILE ==="
echo "Next: ouroboros report <run_id> --bls"
