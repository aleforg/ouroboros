#!/usr/bin/env python3
"""Smoke test for the qwen-image target backend. NVIDIA CUDA only.

Exercises the real path — build_target() → _load() → pipeline call → PNG bytes —
without the attacker or the judge, so a failure here is unambiguously the target.

Run it once with cheap settings to check the plumbing, then with the defaults:

    python smoke_qwen.py --steps 4 --size 512   # fast, ugly images, proves it loads
    python smoke_qwen.py                        # real config: 50 steps, 1024 px

Set HF_HOME to a big volume first — the checkpoint is ~57 GB (bitsandbytes
quantizes at load time, so the full bf16 weights are downloaded regardless):

    export HF_HOME=/workspace/hf
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import time

from ouroboros.targets import build_target

PROMPT = "Photo portrait of a software engineer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--quantize", type=int, default=4, help="4 = NF4, 8 = int8, other = bf16")
    ap.add_argument("--m", type=int, default=2, help="images to generate")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="smoke_out")
    ap.add_argument(
        "--offload", choices=["auto", "on", "off"], default="auto",
        help="CPU-offload the weights. 'auto' offloads only when VRAM is tight. "
             "'on' is ~10x slower on a big card — use it to reproduce the "
             "small-card path, not for real runs.",
    )
    args = ap.parse_args()

    if args.offload != "auto":
        os.environ["OUROBOROS_QWEN_CPU_OFFLOAD"] = "1" if args.offload == "on" else "0"

    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. This backend is NVIDIA-only.", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    target = build_target(
        "qwen-image",
        target_steps=args.steps,
        target_width=args.size,
        target_height=args.size,
        target_quantize=args.quantize,
    )
    print(f"backend={target.name}  estimated VRAM={target.estimated_peak_ram_gb} GB")

    # Load explicitly, outside the generation timer. _load() is private, but
    # folding it into the per-image average is exactly the mistake that makes a
    # run-cost estimate wrong: loading + NF4-quantizing a 20B pipeline is a
    # one-off cost per process, while the loop pays the per-image cost 28k times.
    print("loading pipeline (one-off: download on first run, then quantization) …")
    t0 = time.monotonic()
    target._load()
    load_s = time.monotonic() - t0

    print(f"generating {args.m} × {args.size}px @ {args.steps} steps — this is not fast")
    t0 = time.monotonic()
    results = asyncio.run(target.generate_m(args.prompt, args.m))
    elapsed = time.monotonic() - t0

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, r in enumerate(results):
        if r.outcome == "image" and r.image_bytes:
            path = out_dir / f"sample_{i}.png"
            path.write_bytes(r.image_bytes)
            print(f"  [{i}] {r.outcome:7s} {len(r.image_bytes):>9,} bytes → {path}")
            ok += 1
        else:
            print(f"  [{i}] {r.outcome:7s} {r.error}")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    per_image = elapsed / max(len(results), 1)
    print(f"\nload:      {load_s:6.0f}s  (one-off per process)")
    print(f"generate:  {elapsed:6.0f}s for {len(results)} images → {per_image:.0f}s each")
    print(f"peak VRAM: {peak:6.1f} GB")

    # The per-image figure is the one that decides whether a full run is
    # feasible: worst case is 175 seeds × M=8 × max_iter=20 = 28k images, and
    # --baseline matched roughly doubles it.
    print(f"\nAt {per_image:.0f}s/image a worst-case paired full run "
          f"(~56k images) would take {56_000 * per_image / 3600:.0f}h "
          f"({56_000 * per_image / 86_400:.1f} days) of target time alone, "
          "judge excluded.")
    if args.steps != 50 or args.size != 1024:
        print("NOTE: measured at non-default settings — re-run without flags "
              "for the number that actually applies to a real run.")

    asyncio.run(target.aclose())
    after = torch.cuda.memory_allocated() / 1024**3
    print(f"after aclose(): {after:.1f} GB still allocated")

    if ok != len(results):
        print("\nFAIL: not every sample produced an image.", file=sys.stderr)
        return 1
    print("\nOK — now open the PNGs and check they actually show a person.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
