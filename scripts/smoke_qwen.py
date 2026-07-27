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
    args = ap.parse_args()

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
    print(f"\n{ok}/{len(results)} images in {elapsed:.0f}s "
          f"({elapsed / max(len(results), 1):.0f}s each), peak VRAM {peak:.1f} GB")

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
