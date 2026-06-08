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
    backend: Literal["flux", "diffusers"] = "flux",
    *,
    flux_quantize: int = 4,
    flux_steps: int = 4,
    flux_width: int = 512,
    flux_height: int = 512,
    flux_seed_base: int = 42,
) -> TargetBackend:
    """Factory for target backends.

    ``flux``      — FLUX.2-klein-4B via mflux (Apple Silicon only).
    ``diffusers`` — FLUX.1-schnell via HuggingFace diffusers + NVIDIA CUDA.
                    Requires ``pip install -e '.[diffusers]'``.
    """
    if backend == "flux":
        from ouroboros.targets.flux import FluxLocalTarget

        return FluxLocalTarget(
            quantize=flux_quantize,
            steps=flux_steps,
            width=flux_width,
            height=flux_height,
            seed_base=flux_seed_base,
        )
    if backend == "diffusers":
        from ouroboros.targets.diffusers_flux import FluxDiffusersTarget

        return FluxDiffusersTarget(
            steps=flux_steps,
            width=flux_width,
            height=flux_height,
            quantize_bits=flux_quantize,
            seed_base=flux_seed_base,
        )
    raise ValueError(
        f"Unknown target backend {backend!r}. "
        "Supported: 'flux' (Apple Silicon/mflux), 'diffusers' (NVIDIA CUDA)."
    )
