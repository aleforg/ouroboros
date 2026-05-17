"""Target backends for the T2I model under test.

Each backend implements the `TargetBackend` Protocol (see base.py) so the PAIR
loop can swap implementations without changes. Currently only FLUX.2-klein-4B
(local, mflux) is shipped; cloud backends (DALL-E, SDXL Vertex, Imagen) can be
added by dropping a new module in this package and wiring it into
`build_target()` in base.py.
"""

from mirtage.targets.base import (
    RateLimiter,
    SampleResult,
    TargetBackend,
    backoff_wait,
    build_target,
)
from mirtage.targets.flux import FluxLocalTarget

__all__ = [
    "RateLimiter",
    "SampleResult",
    "TargetBackend",
    "backoff_wait",
    "build_target",
    "FluxLocalTarget",
]
