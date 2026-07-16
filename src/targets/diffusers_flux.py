"""FLUX.2-klein-4B via HuggingFace diffusers on NVIDIA CUDA.

Drop-in replacement for FluxLocalTarget on GPU cloud deployments (RunPod,
Lambda, Colab) where Apple Silicon / mflux is unavailable.

Install the ``[diffusers]`` extra first:
    pip install -e ".[diffusers]"
"""
from __future__ import annotations

import asyncio
import io
import logging

from ouroboros.targets.base import SampleResult

logger = logging.getLogger(__name__)

_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# VRAM estimates for FLUX.2-klein-4B (4B weights + VAE + text encoders)
_VRAM_GB: dict[int, float] = {4: 3.5, 8: 6.5}
_VRAM_BF16 = 11.0


class FluxDiffusersTarget:
    """FLUX.2-klein-4B via diffusers + CUDA.

    Parameters
    ----------
    steps:
        Inference steps. 4 is optimal for the klein distilled model.
    width / height:
        Output resolution in pixels.
    quantize_bits:
        4 → 4-bit NF4 (bitsandbytes, ~3.5 GB VRAM).
        8 → 8-bit (bitsandbytes, ~6.5 GB VRAM).
        Anything else → bfloat16 full precision (~11 GB VRAM).
    seed_base:
        Base RNG seed; sample i uses seed_base + i * 1000.
    """

    name = "flux-diffusers"

    def __init__(
        self,
        *,
        steps: int = 4,
        width: int = 512,
        height: int = 512,
        quantize_bits: int = 4,
        seed_base: int = 42,
    ) -> None:
        self._steps = steps
        self._width = width
        self._height = height
        self._quantize_bits = quantize_bits
        self._seed_base = seed_base
        self._pipe = None
        self.estimated_peak_ram_gb: float = _VRAM_GB.get(quantize_bits, _VRAM_BF16)

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import torch
        from diffusers import FluxPipeline

        logger.info(
            "Loading %s (quantize_bits=%d) …", _MODEL_ID, self._quantize_bits
        )

        if self._quantize_bits == 4:
            from diffusers import FluxTransformer2DModel
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                _MODEL_ID,
                subfolder="transformer",
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
            )
            self._pipe = FluxPipeline.from_pretrained(
                _MODEL_ID,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.enable_model_cpu_offload()

        elif self._quantize_bits == 8:
            from diffusers import FluxTransformer2DModel
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            transformer = FluxTransformer2DModel.from_pretrained(
                _MODEL_ID,
                subfolder="transformer",
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
            )
            self._pipe = FluxPipeline.from_pretrained(
                _MODEL_ID,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.enable_model_cpu_offload()

        else:
            # Full bfloat16 (~11 GB VRAM for 4B klein model)
            self._pipe = FluxPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

        logger.info("FLUX.2-klein-4B loaded.")

    # ------------------------------------------------------------------
    # TargetBackend protocol
    # ------------------------------------------------------------------

    async def generate_m(self, prompt: str, m: int) -> list[SampleResult]:
        """Generate *m* images via asyncio thread pool.

        Unlike FluxLocalTarget (MLX), PyTorch CUDA is thread-safe so we can
        safely offload to a thread without breaking the GPU stream.
        """
        return await asyncio.to_thread(self._generate_sync, prompt, m)

    def _generate_sync(self, prompt: str, m: int) -> list[SampleResult]:
        import torch

        self._load()
        results: list[SampleResult] = []
        for i in range(m):
            generator = torch.Generator("cuda").manual_seed(self._seed_base + i * 1000)
            try:
                output = self._pipe(
                    prompt=prompt,
                    num_inference_steps=self._steps,
                    width=self._width,
                    height=self._height,
                    generator=generator,
                    guidance_scale=0.0,   # schnell is guidance-free
                    max_sequence_length=256,
                )
                buf = io.BytesIO()
                output.images[0].save(buf, format="PNG")
                results.append(SampleResult(outcome="image", image_bytes=buf.getvalue()))
            except Exception as exc:
                logger.warning("FluxDiffusers generation error (sample %d): %s", i, exc)
                results.append(SampleResult(outcome="error", error=str(exc)))
        return results

    async def aclose(self) -> None:
        """Release GPU memory."""
        if self._pipe is not None:
            import torch

            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("FluxDiffusersTarget: VRAM released.")
