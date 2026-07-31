from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ouroboros.config import (
    TARGET_BACKOFF_BASE,
    TARGET_BACKOFF_MAX,
)

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    outcome: Literal["image", "refused", "error"]
    image_bytes: bytes | None = None
    error: str | None = None


# Distance between the noise-seed blocks of two consecutive generate_m calls.
# Large enough that a block can never reach into the next one: the per-sample
# step is at most 1000 and M is a handful.
CALL_SEED_STRIDE = 1_000_000


def call_seeds(
    seed_base: int,
    call_idx: int,
    m: int,
    *,
    sample_step: int = 1000,
    stride: int = CALL_SEED_STRIDE,
) -> list[int]:
    """Noise seeds for one ``generate_m`` call.

    Seeding only by position inside the batch (``seed_base + i * sample_step``)
    makes two calls with the same prompt return bit-identical images. That is
    invisible in the iterative loop, whose prompt changes every iteration, but it
    silently voids the budget-matched comparator in ``baseline.py``, which calls
    ``generate_m`` repeatedly with the *same* neutral prompt: every extra batch is
    a copy, so the max-over-batches selection can climb on the attacker's side and
    not on the control's. Including ``call_idx`` gives each call its own disjoint
    block of seeds.

    ``stride=0`` restores the legacy behaviour exactly — needed to byte-replay
    runs generated before this fix, and nothing else.
    """
    return [seed_base + call_idx * stride + i * sample_step for i in range(m)]


class RateLimiter:
    """Simple token-bucket rate limiter for async use."""

    def __init__(self, per_min: int) -> None:
        self._interval = 60.0 / max(per_min, 1)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


async def backoff_wait(attempt: int) -> None:
    delay = min(TARGET_BACKOFF_BASE**attempt, TARGET_BACKOFF_MAX)
    logger.debug("Backing off %.1f s (attempt %d)", delay, attempt)
    await asyncio.sleep(delay)


@runtime_checkable
class TargetBackend(Protocol):
    name: str
    estimated_peak_ram_gb: float

    async def generate_m(self, prompt: str, m: int) -> list[SampleResult]:
        ...

    async def aclose(self) -> None:
        """Release model from RAM. No-op for cloud targets."""
        ...


def build_target(
    backend: Literal["flux", "diffusers", "qwen-image"] = "flux",
    *,
    target_quantize: int = 4,
    target_steps: int = 4,
    target_width: int = 512,
    target_height: int = 512,
    target_seed_base: int = 42,
    target_call_seed_stride: int = CALL_SEED_STRIDE,
) -> TargetBackend:
    """Factory for target backends.

    ``flux``       — FLUX.2-klein-4B via mflux (Apple Silicon only).
    ``diffusers``  — FLUX.2-klein-4B via HuggingFace diffusers + NVIDIA CUDA.
    ``qwen-image`` — Qwen-Image 20B via HuggingFace diffusers + NVIDIA CUDA.

    Both CUDA backends require ``pip install -e '.[diffusers]'``. Their heavy
    imports live inside the backend's ``_load()``, so constructing any target
    here is safe on a machine where the extra is not installed.

    The sampling params are not interchangeable across backends — see
    ``config.TARGET_DEFAULTS`` / ``config.resolve_target_params``, which is
    what callers should use to fill them in.

    ``target_call_seed_stride=0`` reverts to seeding by position inside the batch
    only, which makes repeated calls with the same prompt return identical images.
    It exists for one purpose: byte-replaying runs generated before that was
    fixed. Do not use it for a new run.
    """
    if backend == "flux":
        from ouroboros.targets.flux import FluxLocalTarget

        return FluxLocalTarget(
            quantize=target_quantize,
            steps=target_steps,
            width=target_width,
            height=target_height,
            seed_base=target_seed_base,
            call_seed_stride=target_call_seed_stride,
        )
    if backend == "diffusers":
        from ouroboros.targets.diffusers_flux import FluxDiffusersTarget

        return FluxDiffusersTarget(
            steps=target_steps,
            width=target_width,
            height=target_height,
            quantize_bits=target_quantize,
            seed_base=target_seed_base,
            call_seed_stride=target_call_seed_stride,
        )
    if backend == "qwen-image":
        from ouroboros.targets.qwen_image import QwenImageTarget

        return QwenImageTarget(
            steps=target_steps,
            width=target_width,
            height=target_height,
            quantize_bits=target_quantize,
            seed_base=target_seed_base,
            call_seed_stride=target_call_seed_stride,
        )
    raise ValueError(
        f"Unknown target backend {backend!r}. Supported: "
        "'flux' (FLUX.2-klein, Apple Silicon/mflux), "
        "'diffusers' (FLUX.2-klein, NVIDIA CUDA), "
        "'qwen-image' (Qwen-Image 20B, NVIDIA CUDA)."
    )
