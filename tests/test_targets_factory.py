"""build_target() dispatch and target-param resolution.

None of these tests import mflux, diffusers or torch: every backend defers its
heavy imports to ``_load()``, so *constructing* a target is safe on any machine.
That property is itself worth protecting — it is what lets the CLI validate a
config with --dry-run on a laptop that will never run the model.
"""
from __future__ import annotations

import pytest

from ouroboros.config import TARGET_DEFAULTS, resolve_target_params
from ouroboros.targets import build_target
from ouroboros.targets.base import TargetBackend


class TestBuildTargetDispatch:
    def test_flux_backend(self):
        from ouroboros.targets.flux import FluxLocalTarget

        t = build_target("flux")
        assert isinstance(t, FluxLocalTarget)
        assert t.name == "flux"

    def test_diffusers_backend(self):
        from ouroboros.targets.diffusers_flux import FluxDiffusersTarget

        t = build_target("diffusers")
        assert isinstance(t, FluxDiffusersTarget)
        assert t.name == "flux-diffusers"

    def test_qwen_image_backend(self):
        from ouroboros.targets.qwen_image import QwenImageTarget

        t = build_target("qwen-image")
        assert isinstance(t, QwenImageTarget)
        assert t.name == "qwen-image"

    def test_default_backend_is_flux(self):
        assert build_target().name == "flux"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown target backend"):
            build_target("stable-diffusion")

    @pytest.mark.parametrize("backend", ["flux", "diffusers", "qwen-image"])
    def test_satisfies_protocol(self, backend):
        assert isinstance(build_target(backend), TargetBackend)


class TestParamPropagation:
    @pytest.mark.parametrize("backend", ["flux", "diffusers", "qwen-image"])
    def test_sampling_params_reach_the_instance(self, backend):
        t = build_target(
            backend,
            target_quantize=8,
            target_steps=17,
            target_width=768,
            target_height=640,
            target_seed_base=7,
        )
        assert t._steps == 17
        assert t._width == 768
        assert t._height == 640
        assert t._seed_base == 7

    def test_qwen_vram_estimate_tracks_quantization(self):
        assert build_target("qwen-image", target_quantize=4).estimated_peak_ram_gb == 18.0
        assert build_target("qwen-image", target_quantize=8).estimated_peak_ram_gb == 30.0
        # Anything else means full bfloat16 — an 80 GB card.
        assert build_target("qwen-image", target_quantize=16).estimated_peak_ram_gb == 60.0


class TestResolveTargetParams:
    def test_klein_defaults(self):
        assert resolve_target_params("flux") == (4, 512, 4)
        assert resolve_target_params("diffusers") == (4, 512, 4)

    def test_qwen_needs_more_steps_and_resolution(self):
        assert resolve_target_params("qwen-image") == (50, 1024, 4)

    def test_explicit_values_win(self):
        assert resolve_target_params("qwen-image", steps=8, size=512, quantize=8) == (8, 512, 8)

    def test_partial_override_keeps_backend_defaults_for_the_rest(self):
        assert resolve_target_params("qwen-image", steps=20) == (20, 1024, 4)

    def test_unknown_backend_falls_back_to_flux_defaults(self):
        assert resolve_target_params("nope") == resolve_target_params("flux")

    def test_every_dispatchable_backend_has_defaults(self):
        assert set(TARGET_DEFAULTS) == {"flux", "diffusers", "qwen-image"}
