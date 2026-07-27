"""Qwen-Image via HuggingFace diffusers on NVIDIA CUDA.

Second T2I model family alongside FLUX.2-klein, so a measured skew can be
attributed to a model rather than to the FLUX family. Cloud GPU only (RunPod,
Lambda, Colab) — there is no mflux/Apple Silicon path for this one.

Install the ``[diffusers]`` extra first:
    pip install -e ".[diffusers]"
"""
from __future__ import annotations

import asyncio
import io
import logging

from ouroboros.targets.base import SampleResult

logger = logging.getLogger(__name__)

_MODEL_ID = "Qwen/Qwen-Image"

# Qwen-Image is not guidance-distilled: unlike klein (steps=4, guidance=1.0) it
# needs real classifier-free guidance, which the pipeline exposes as
# ``true_cfg_scale``. 4.0 with an empty negative prompt is the reference config
# from the model card; the negative prompt is deliberately contentless so it
# cannot steer demographics and confound the bias measurement.
_TRUE_CFG_SCALE = 4.0
_NEGATIVE_PROMPT = " "

# VRAM estimates: 20B MMDiT transformer + Qwen2.5-VL-7B text encoder + VAE.
# Both transformer and text encoder are quantized, which is what makes the
# 4-bit configuration fit on a 24 GB card.
_VRAM_GB: dict[int, float] = {4: 18.0, 8: 30.0}
_VRAM_BF16 = 60.0


class QwenImageTarget:
    """Qwen-Image (20B MMDiT) via diffusers + CUDA.

    Parameters
    ----------
    steps:
        Inference steps. 50 is the reference value; Qwen-Image is undistilled,
        so low step counts degrade badly.
    width / height:
        Output resolution in pixels. The model is trained around 1k, so 1024
        is the default rather than klein's 512.
    quantize_bits:
        4 → 4-bit NF4 on transformer + text encoder (~18 GB VRAM).
        8 → 8-bit on both (~30 GB VRAM).
        Anything else → bfloat16 full precision (~60 GB VRAM, A100 80GB class).
    seed_base:
        Base RNG seed; sample i uses seed_base + i * 1000.

    Like FLUX, Qwen-Image has no prompt-level safety filter, so this backend
    never returns ``"refused"`` — only ``"image"`` or ``"error"``. The loop's
    refusal pivot stays a dead branch here.
    """

    name = "qwen-image"

    def __init__(
        self,
        *,
        steps: int = 50,
        width: int = 1024,
        height: int = 1024,
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
        from diffusers import DiffusionPipeline

        logger.info(
            "Loading %s (quantize_bits=%d, steps=%d, %dx%d, true_cfg_scale=%.1f) …",
            _MODEL_ID, self._quantize_bits, self._steps,
            self._width, self._height, _TRUE_CFG_SCALE,
        )

        if self._quantize_bits in (4, 8):
            from diffusers.quantizers import PipelineQuantizationConfig

            if self._quantize_bits == 4:
                quant_kwargs = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                }
            else:
                quant_kwargs = {"load_in_8bit": True}

            # Both components must be quantized: the Qwen2.5-VL-7B text encoder
            # alone is ~15 GB in bfloat16, which would blow the VRAM budget even
            # with a 4-bit transformer.
            quant_config = PipelineQuantizationConfig(
                quant_backend=f"bitsandbytes_{self._quantize_bits}bit",
                quant_kwargs=quant_kwargs,
                components_to_quantize=["transformer", "text_encoder"],
            )
            # No device_map here — it conflicts with enable_model_cpu_offload().
            self._pipe = DiffusionPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
            )
            self._pipe.enable_model_cpu_offload()

        else:
            self._pipe = DiffusionPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

        logger.info("Qwen-Image loaded.")

    # ------------------------------------------------------------------
    # TargetBackend protocol
    # ------------------------------------------------------------------

    async def generate_m(self, prompt: str, m: int) -> list[SampleResult]:
        """Generate *m* images via asyncio thread pool.

        Like FluxDiffusersTarget and unlike FluxLocalTarget (MLX), PyTorch CUDA
        is thread-safe so offloading to a thread does not break the GPU stream.
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
                    negative_prompt=_NEGATIVE_PROMPT,
                    num_inference_steps=self._steps,
                    true_cfg_scale=_TRUE_CFG_SCALE,
                    width=self._width,
                    height=self._height,
                    generator=generator,
                )
                buf = io.BytesIO()
                output.images[0].save(buf, format="PNG")
                results.append(SampleResult(outcome="image", image_bytes=buf.getvalue()))
            except Exception as exc:
                logger.warning("QwenImage generation error (sample %d): %s", i, exc)
                results.append(SampleResult(outcome="error", error=str(exc)))
        return results

    async def aclose(self) -> None:
        """Release GPU memory.

        Note this is expensive for Qwen-Image: the next generate_m re-downloads
        nothing but does re-quantize a 20B transformer, which takes minutes.
        Prefer --no-aggressive-unload when running this backend; the CLI warns
        about it at startup.
        """
        if self._pipe is not None:
            import torch

            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("QwenImageTarget: VRAM released.")
