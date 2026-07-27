#!/usr/bin/env bash
# Full paired run for RQ2: iterative loop (175 seeds) + budget-matched
# baseline, using this machine's local diffusers/ollama setup (see
# setup_cloud_gpu.sh). Explicit backend/flags below on purpose — don't rely
# on setup_cloud_gpu.sh's local config.py patch alone, so this still runs
# correctly even on an unpatched checkout.
#
# Usage:
#   scripts/run_full_cloud.sh                            # FLUX.2-klein (default)
#   scripts/run_full_cloud.sh --backend qwen-image       # Qwen-Image 20B
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

# Quantization is not shared between the two: klein-4B fits in bf16 with room to
# spare, while Qwen-Image is 20B + a 7B text encoder — bf16 would be ~60 GB and
# does not fit a 48 GB A6000, so it runs at NF4 on both components (~18 GB).
# Steps are left to the per-backend default (config.TARGET_DEFAULTS): 4 for the
# distilled klein, 50 for the undistilled Qwen-Image.
case "$BACKEND" in
  diffusers)  TARGET_ARGS=(--target-quantize 16 --target-size 1024) ;;
  qwen-image) TARGET_ARGS=(--target-quantize 4  --target-size 1024) ;;
  *) echo "Unknown backend '$BACKEND' — use 'diffusers' or 'qwen-image'." >&2; exit 1 ;;
esac

echo "=== Pre-flight checks ==="
command -v ouroboros >/dev/null || { echo "ouroboros not found — did setup_cloud_gpu.sh run?" >&2; exit 1; }
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "Ollama not responding on localhost:11434" >&2; exit 1; }
ollama list | grep -q "dolphin-llama3" || { echo "dolphin-llama3 not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
ollama list | grep -q "qwen3-vl" || { echo "qwen3-vl:8b not pulled — run setup_cloud_gpu.sh" >&2; exit 1; }
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
  --judge-model qwen3-vl:8b \
  --no-aggressive-unload \
  "${TARGET_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOGFILE"

echo
echo "=== Done — log written to $LOGFILE ==="
echo "Next: ouroboros report <run_id> --bls"
