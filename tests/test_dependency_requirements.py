"""Declared CUDA requirements must reject incompatible pre-upgrade installs."""
from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    "name,minimum,incompatible",
    [
        ("transformers", "5.17.0", "5.16.0"),
        ("torch", "2.5.0", "2.4.0"),
        ("accelerate", "1.1.0", "1.0.1"),
        ("bitsandbytes", "0.46.1", "0.46.0"),
    ],
)
def test_cuda_extra_enforces_transformers_517_runtime_minima(name, minimum, incompatible):
    # tomllib is built in from 3.11; this metadata-only test may skip on 3.10.
    tomllib = pytest.importorskip("tomllib")
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, project["optional-dependencies"]["diffusers"])
    }
    specifier = requirements[name].specifier
    assert specifier.contains(minimum)
    assert not specifier.contains(incompatible)
