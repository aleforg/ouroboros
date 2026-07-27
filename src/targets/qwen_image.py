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
import os

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

# Headroom over the weight estimate for activations and fragmentation. Above
# this the pipeline is kept resident on the GPU; below it, weights are offloaded
# to system RAM. CPU offload costs real throughput on a quantized 20B — measured
# ~100 s/image at 4 steps on a 48 GB A6000, ~10x what resident inference needs —
# so it must be the fallback for small cards, never the default on a big one.
_VRAM_HEADROOM_GB = 6.0

# Escape hatch for benchmarking or a card whose free VRAM the auto-policy reads
# wrong: OUROBOROS_QWEN_CPU_OFFLOAD=1 forces offload, =0 forces resident.
_OFFLOAD_ENV = "OUROBOROS_QWEN_CPU_OFFLOAD"

# Opt-in torch.compile on the transformer. Off by default: it trades a large
# one-off compilation for per-step throughput, which only pays back when the
# target stays loaded across many batches (i.e. --no-aggressive-unload). Set
# OUROBOROS_QWEN_COMPILE=1, and optionally OUROBOROS_QWEN_COMPILE_MODE to one of
# torch.compile's modes ("reduce-overhead", "max-autotune").
_COMPILE_ENV = "OUROBOROS_QWEN_COMPILE"
_COMPILE_MODE_ENV = "OUROBOROS_QWEN_COMPILE_MODE"


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

    def _use_cpu_offload(self, total_vram_gb: float) -> bool:
        """Offload weights to system RAM only when they would not fit resident."""
        forced = os.environ.get(_OFFLOAD_ENV)
        if forced is not None:
            use = forced.strip() not in ("0", "false", "False", "")
            logger.info("%s=%s → cpu_offload=%s", _OFFLOAD_ENV, forced, use)
            return use
        needed = self.estimated_peak_ram_gb + _VRAM_HEADROOM_GB
        use = total_vram_gb < needed
        logger.info(
            "GPU has %.1f GB, this config needs ~%.1f GB → %s",
            total_vram_gb, needed,
            "cpu offload (slow, but it fits)" if use else "keeping the pipeline resident",
        )
        return use

    def _maybe_compile(self, cpu_offload: bool) -> None:
        """Opt-in ``torch.compile`` on the transformer.

        Off unless ``OUROBOROS_QWEN_COMPILE`` says otherwise. Compilation is a
        fixed cost paid on the first batch and amortized over every later one,
        so it is worth it for a long run with the target resident and actively
        harmful with aggressive unload, where each batch would recompile.

        Dynamo errors are suppressed rather than raised: this is a speed knob,
        and a graph break in a quantized 20B should degrade to eager execution,
        not turn a multi-day run into a batch of `SampleResult(outcome="error")`.
        """
        flag = os.environ.get(_COMPILE_ENV, "").strip()
        if flag in ("", "0", "false", "False"):
            return
        if cpu_offload:
            # accelerate's offload hooks move weights between devices mid-graph;
            # compiling on top of that produces recompiles, not speedups.
            logger.warning(
                "%s is set but the pipeline is CPU-offloaded — skipping compile.",
                _COMPILE_ENV,
            )
            return

        import torch

        mode = os.environ.get(_COMPILE_MODE_ENV) or None
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
        except Exception as exc:  # very old torch — compile is best-effort
            logger.debug("torch._dynamo unavailable (%s)", exc)

        self._pipe.transformer = torch.compile(self._pipe.transformer, mode=mode)
        logger.info(
            "torch.compile enabled on the transformer (mode=%s). "
            "The first batch pays compilation; later batches are the payoff.",
            mode or "default",
        )

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
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cpu_offload = self._use_cpu_offload(total_vram_gb)

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
            # device_map and enable_model_cpu_offload() are mutually exclusive:
            # the first pins the pipeline on the GPU, the second hands placement
            # to accelerate's hooks. A bitsandbytes-quantized pipeline cannot be
            # moved with .to("cuda") afterwards, so the choice is made here.
            self._pipe = DiffusionPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                **({} if cpu_offload else {"device_map": "cuda"}),
            )
            if cpu_offload:
                self._pipe.enable_model_cpu_offload()

        else:
            self._pipe = DiffusionPipeline.from_pretrained(
                _MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

        self._maybe_compile(cpu_offload)
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

        `del` alone does not free a resident pipeline: diffusers components
        cross-reference each other, and accelerate's device hooks reference the
        modules they wrap, so the whole graph is a reference cycle that only the
        cyclic collector breaks. Under CPU offload this was invisible — the
        weights lived in system RAM and there was nothing on the GPU to leak.
        """
        if self._pipe is None:
            return

        import gc

        import torch

        # Drop accelerate's hooks first; they hold references to the modules
        # (and, on the offload path, to their weight maps).
        try:
            from accelerate.hooks import remove_hook_from_module

            for component in ("transformer", "text_encoder", "vae"):
                module = getattr(self._pipe, component, None)
                if module is not None:
                    remove_hook_from_module(module, recurse=True)
        except Exception as exc:  # accelerate missing or API drift — not fatal
            logger.debug("QwenImageTarget: hook removal skipped (%s)", exc)

        del self._pipe
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("QwenImageTarget: VRAM released.")
