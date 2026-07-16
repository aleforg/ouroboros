#!/usr/bin/env bash
# One-time environment setup for running Ouroboros on a rented NVIDIA GPU box
# (e.g. RunPod A6000). Run this from the repo root, right after cloning.
#
# What it configures for THIS machine only (not committed to the repo):
#   - target backend: diffusers (FLUX.2-klein-4B/CUDA) — mflux/MLX is Apple-only
#     and cannot run here at all, so this is a hard requirement, not a choice.
#   - judge backend: ollama / qwen3-vl:8b — replaces the repo's Mac-oriented
#     "gemini" default so `ouroboros run` needs no --judge-backend flag here.
#
# Recommended flags NOT baked in here (pass them explicitly per run — see the
# summary printed at the end): --no-aggressive-unload, --flux-quantize 16.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/5] GPU sanity check ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found — this script targets an NVIDIA CUDA box. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [2/5] Python env + package install ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,diffusers]"
pip install bitsandbytes   # needed for --flux-quantize 4/8 (NF4 / int8)

echo "=== [3/5] Ollama: install, configure, pull models ==="
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# OLLAMA_MAX_LOADED_MODELS must be >=2 on this box: the attacker
# (dolphin-llama3) and the judge (qwen3-vl:8b) are both served by the same
# Ollama daemon. With the default of 1 they would evict each other every
# iteration even with --no-aggressive-unload, defeating the point of keeping
# models resident.
export OLLAMA_MAX_LOADED_MODELS=2

if pgrep -x ollama &>/dev/null; then
  echo "NOTE: an ollama daemon is already running (e.g. as a systemd service)."
  echo "      OLLAMA_MAX_LOADED_MODELS=2 only applies to processes started"
  echo "      after this export — if it's a service, set the same variable in"
  echo "      its unit/environment file and restart it, or run:"
  echo "        sudo systemctl stop ollama"
  echo "      then re-run this script so it can start ollama itself."
else
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

ollama pull dolphin-llama3:latest
ollama pull qwen3-vl:8b

echo "=== [4/5] .env ==="
if [ ! -f .env ]; then
  cat > .env <<'ENV'
# No Vertex/GOOGLE_* vars needed: judge defaults to ollama on this machine.
OLLAMA_HOST=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_NUM_PARALLEL=1
ENV
  echo ".env written."
else
  echo ".env already exists — leaving it as-is (check OLLAMA_MAX_LOADED_MODELS=2 manually)."
fi

echo "=== [5/5] Local defaults: target=diffusers, judge=ollama ==="
CONFIG=src/config.py

if grep -q '^TARGET_BACKEND_DEFAULT: Literal\["flux", "diffusers"\] = "flux"' "$CONFIG"; then
  sed -i 's/^TARGET_BACKEND_DEFAULT: Literal\["flux", "diffusers"\] = "flux"/TARGET_BACKEND_DEFAULT: Literal["flux", "diffusers"] = "diffusers"/' "$CONFIG"
  echo "  TARGET_BACKEND_DEFAULT -> diffusers"
else
  echo "  TARGET_BACKEND_DEFAULT already patched or line not found — skipped."
fi

if grep -q '^JUDGE_BACKEND_DEFAULT: Literal\["mlx", "ollama"\] = "mlx"' "$CONFIG"; then
  sed -i 's/^JUDGE_BACKEND_DEFAULT: Literal\["mlx", "ollama"\] = "mlx"/JUDGE_BACKEND_DEFAULT: Literal["mlx", "ollama"] = "ollama"/' "$CONFIG"
  echo "  JUDGE_BACKEND_DEFAULT -> ollama"
else
  echo "  JUDGE_BACKEND_DEFAULT already patched or line not found — skipped."
fi

cat <<'SUMMARY'

=== Setup complete ===

target and judge now default to diffusers/ollama (qwen3-vl:8b) on this
machine only — this local patch to src/config.py is NOT meant to be
committed/pushed (it would flip the default for the Mac dev setup too).

Still pass these explicitly per run (not baked in as defaults):
  --no-aggressive-unload   keeps attacker/target/judge resident (48GB VRAM
                           is plenty; keeps models in GPU memory without swaps)
  --flux-quantize 16       bfloat16 unquantized FLUX.2-klein-4B (best quality,
                           ~11GB VRAM, leaving ~37GB for Ollama models resident)
  --flux-size 1024         1024x1024 resolution (FLUX native scale for best quality)

Sanity check before the full run:
  source .venv/bin/activate
  ouroboros validate-judge --judge-backend ollama --sample 100 ...
  ouroboros run --mode test --no-aggressive-unload --flux-quantize 16 --flux-size 1024
SUMMARY
