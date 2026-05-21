from __future__ import annotations

import gc
import io
import logging

from ouroboros.targets.base import SampleResult

logger = logging.getLogger(__name__)


class FluxLocalTarget:
    """Local FLUX.2-klein-4B target via mflux (Apple Silicon / MLX).

    Images are generated sequentially (mflux is synchronous, single-context).
    Each call to generate_m returns m images with deterministic seeds: seed_base+i.
    Unlike cloud targets, this backend exposes explicit image seeds → results are
    exactly reproducible given the same model weights and quantization level.

    Safety filters: FLUX.2-klein has no built-in safety filter. All outcomes are
    either 'image' or 'error'. The 'refused' outcome will never appear — the
    refusal-pivot in the loop is effectively a no-op for this backend.
    """

    name = "flux"
    estimated_peak_ram_gb = 5.0  # FLUX.2-klein-4B 4-bit peak on Apple Silicon (4B weights + Qwen3 encoder)

    def __init__(
        self,
        quantize: int = 4,
        steps: int = 4,
        width: int = 512,
        height: int = 512,
        seed_base: int = 42,
    ) -> None:
        self._quantize = quantize
        self._steps = steps
        self._width = width
        self._height = height
        self._seed_base = seed_base
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        logger.info(
            "Loading FLUX.2-klein-4B @ %d-bit (first call — downloads weights if not cached) …",
            self._quantize,
        )
        self._model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), quantize=self._quantize)
        logger.info("FLUX model loaded.")

    def _generate_one_sync(self, prompt: str, seed: int) -> SampleResult:
        try:
            generated = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=self._steps,
                width=self._width,
                height=self._height,
            )
            # mflux returns a GeneratedImage wrapper; .image is the underlying PIL.Image
            pil = getattr(generated, "image", generated)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return SampleResult(outcome="image", image_bytes=buf.getvalue())
        except Exception as exc:
            logger.warning("FLUX generation failed (seed=%d): %s", seed, exc)
            return SampleResult(outcome="error", error=str(exc))

    async def generate_m(self, prompt: str, m: int) -> list[SampleResult]:
        """Generate m images sequentially on the current thread.

        Note: MLX binds a GPU stream to the thread that created the model. To avoid
        'There is no Stream(gpu, 1) in current thread' errors we keep both load and
        generation on the asyncio event loop thread (no asyncio.to_thread). This
        blocks the loop for ~5-15 s per image, which is acceptable because at this
        point in the PAIR loop nothing else runs concurrently (attacker is unloaded,
        judge has not started).
        """
        self._load()
        results: list[SampleResult] = []
        for i in range(m):
            seed = self._seed_base + i
            result = self._generate_one_sync(prompt, seed)
            results.append(result)
            logger.debug("FLUX image %d/%d done — outcome=%s", i + 1, m, result.outcome)
        return results

    async def aclose(self) -> None:
        """Unload the model from MLX/Metal memory to free RAM for the next attacker call."""
        if self._model is None:
            return
        self._model = None
        gc.collect()
        try:
            import mlx.core as mx

            # mx.clear_cache replaces deprecated mx.metal.clear_cache
            clear_cache = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
            if clear_cache:
                clear_cache()
        except Exception:
            pass
        logger.debug("FLUX model unloaded from RAM.")
