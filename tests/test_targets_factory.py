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


class TestQwenCompileOptIn:
    """torch.compile decision table. No CUDA needed: torch.compile is patched,
    and the point under test is when we call it, not what it produces."""

    @staticmethod
    def _target_with_fake_pipe():
        import types

        from ouroboros.targets.qwen_image import QwenImageTarget

        t = QwenImageTarget()
        t._pipe = types.SimpleNamespace(transformer="<raw>")
        return t

    def _run(self, monkeypatch, env: dict, cpu_offload: bool):
        torch = pytest.importorskip("torch")
        seen = {}

        def fake_compile(mod, mode=None):
            seen["mode"] = mode
            return "<compiled>"

        monkeypatch.setattr(torch, "compile", fake_compile)
        for key in ("OUROBOROS_QWEN_COMPILE", "OUROBOROS_QWEN_COMPILE_MODE"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        target = self._target_with_fake_pipe()
        target._maybe_compile(cpu_offload)
        return target._pipe.transformer, seen.get("mode")

    def test_off_by_default(self, monkeypatch):
        transformer, _ = self._run(monkeypatch, {}, cpu_offload=False)
        assert transformer == "<raw>"

    def test_explicit_zero_is_off(self, monkeypatch):
        transformer, _ = self._run(
            monkeypatch, {"OUROBOROS_QWEN_COMPILE": "0"}, cpu_offload=False
        )
        assert transformer == "<raw>"

    def test_enabled_on_the_resident_path(self, monkeypatch):
        transformer, mode = self._run(
            monkeypatch, {"OUROBOROS_QWEN_COMPILE": "1"}, cpu_offload=False
        )
        assert transformer == "<compiled>"
        assert mode is None  # torch.compile's own default

    def test_mode_is_forwarded(self, monkeypatch):
        _, mode = self._run(
            monkeypatch,
            {"OUROBOROS_QWEN_COMPILE": "1", "OUROBOROS_QWEN_COMPILE_MODE": "max-autotune"},
            cpu_offload=False,
        )
        assert mode == "max-autotune"

    def test_skipped_under_cpu_offload(self, monkeypatch):
        # accelerate moves weights mid-graph; compiling on top of that recompiles
        # instead of speeding up.
        transformer, _ = self._run(
            monkeypatch, {"OUROBOROS_QWEN_COMPILE": "1"}, cpu_offload=True
        )
        assert transformer == "<raw>"
