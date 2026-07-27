"""Target backends for the T2I model under test.

Each backend implements the `TargetBackend` Protocol (see base.py) so the PAIR
loop can swap implementations without changes. Three are shipped: FLUX.2-klein-4B
via mflux (Apple Silicon), FLUX.2-klein-4B via HuggingFace diffusers (NVIDIA
CUDA), and Qwen-Image 20B via diffusers (NVIDIA CUDA). Cloud backends (DALL-E,
Imagen, SDXL on Vertex) can be added by dropping a new module in this package
and wiring it into `build_target()` in base.py.

Only the mflux backend is re-exported here. The CUDA ones are not, so that
`import ouroboros.targets` never pulls in the optional `[diffusers]` stack.
"""

from ouroboros.targets.base import (
    RateLimiter,
    SampleResult,
    TargetBackend,
    backoff_wait,
    build_target,
)
from ouroboros.targets.flux import FluxLocalTarget

__all__ = [
    "RateLimiter",
    "SampleResult",
    "TargetBackend",
    "backoff_wait",
    "build_target",
    "FluxLocalTarget",
]
