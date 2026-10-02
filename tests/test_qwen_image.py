"""Qwen-Image-2.1 pipeline contract, exercised without downloads or CUDA."""
from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from ouroboros.config import QWEN_IMAGE_MODEL_ID, RunConfig, resolve_target_params
from ouroboros.targets.qwen_image import QwenImageTarget


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = SimpleNamespace(loads=[], generations=[], offloads=0, devices=[])

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.loads.append((model_id, kwargs))
            return cls()

        def enable_model_cpu_offload(self):
            calls.offloads += 1

        def to(self, device):
            calls.devices.append(device)
            return self

        def __call__(self, **kwargs):
            calls.generations.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    class Generator:
        def __init__(self, device):
            assert device == "cuda"

        def manual_seed(self, seed):
            self.seed = seed
            return self

    torch = ModuleType("torch")
    torch.bfloat16 = "<bf16>"
    torch.Generator = Generator
    torch.cuda = SimpleNamespace(
        get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3)
    )
    diffusers = ModuleType("diffusers")
    diffusers.QwenImage21Pipeline = Pipeline
    quantizers = ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    monkeypatch.delenv("OUROBOROS_QWEN_COMPILE", raising=False)
    monkeypatch.delenv("OUROBOROS_QWEN_CPU_OFFLOAD", raising=False)
    return calls


@pytest.mark.parametrize("bits", [4, 8, 16])
@pytest.mark.parametrize("offload", [False, True])
def test_loads_21_pipeline_with_quantization_and_placement(
    fake_runtime, monkeypatch, bits, offload,
):
    monkeypatch.setenv("OUROBOROS_QWEN_CPU_OFFLOAD", "1" if offload else "0")
    target = QwenImageTarget(quantize_bits=bits)
    target._load()
    target._load()  # loading is once per lifecycle

    assert len(fake_runtime.loads) == 1
    model_id, kwargs = fake_runtime.loads[0]
    assert model_id == QWEN_IMAGE_MODEL_ID
    assert kwargs["torch_dtype"] == "<bf16>"
    assert fake_runtime.offloads == int(offload)
    if bits in (4, 8):
        quant = kwargs["quantization_config"]
        assert quant["components_to_quantize"] == ["transformer", "text_encoder"]
        assert quant["quant_backend"] == f"bitsandbytes_{bits}bit"
        assert quant["quant_kwargs"][f"load_in_{bits}bit"] is True
        if bits == 4:
            assert quant["quant_kwargs"]["bnb_4bit_quant_type"] == "nf4"
        assert kwargs.get("device_map") == (None if offload else "cuda")
        assert fake_runtime.devices == []
    else:
        assert "quantization_config" not in kwargs
        assert fake_runtime.devices == ([] if offload else ["cuda"])


@pytest.mark.asyncio
async def test_generation_uses_21_defaults_without_cfg_and_preserves_seeds(fake_runtime):
    target = QwenImageTarget()
    assert (target._steps, target._width, target._quantize_bits) == resolve_target_params("qwen-image")
    samples = await target.generate_m("Photo portrait of an engineer", 2)
    await target.generate_m("Photo portrait of an engineer", 1)

    assert len(fake_runtime.loads) == 1
    calls = fake_runtime.generations
    for call in calls:
        assert call["num_inference_steps"] == 40
        assert (call["width"], call["height"]) == (1024, 1024)
        assert call["true_cfg_scale"] == 1.0
        assert "negative_prompt" not in call
    assert [call["generator"].seed for call in calls] == [42, 1042, 1000042]
    for sample in samples:
        assert sample.outcome == "image"
        assert Image.open(io.BytesIO(sample.image_bytes)).format == "PNG"


def test_old_diffusers_has_actionable_error(fake_runtime, monkeypatch):
    monkeypatch.delattr(sys.modules["diffusers"], "QwenImage21Pipeline")
    with pytest.raises(ImportError, match="Reinstall the updated extra"):
        QwenImageTarget()._load()


def test_qwen_model_version_is_recorded_in_config():
    cfg = RunConfig(target_backend="qwen-image")
    assert asdict(cfg)["target_model_id"] == QWEN_IMAGE_MODEL_ID
    assert RunConfig().target_model_id is None
    with pytest.raises(ValueError, match="Unsupported Qwen target model"):
        RunConfig(target_backend="qwen-image", target_model_id="Qwen/Qwen-Image")


def test_cli_dry_run_records_21_defaults(tmp_path, monkeypatch):
    from ouroboros.cli import main

    monkeypatch.setattr(sys, "argv", [
        "ouroboros", "run", "--target-backend", "qwen-image", "--dry-run",
        "--no-aggressive-unload", "--output-dir", str(tmp_path),
    ])
    main()

    paths = list(tmp_path.glob("*/meta.json"))
    assert len(paths) == 1
    config = json.loads(paths[0].read_text())["config"]
    assert config["target_model_id"] == QWEN_IMAGE_MODEL_ID
    assert config["target_steps"] == 40
    assert config["target_width"] == config["target_height"] == 1024
    assert config["target_quantize"] == 4
