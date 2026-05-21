import pytest
from pathlib import Path
from ouroboros.storage import save_image
from ouroboros.seeds import Seed
from ouroboros.config import RunConfig
from unittest.mock import AsyncMock, MagicMock

def test_save_image_int(tmp_path):
    png_data = b"fake-png-bytes"
    rel_path = save_image(
        run_dir=tmp_path,
        seed_id="test-seed",
        iter_idx=3,
        sample_idx=0,
        png_bytes=png_data
    )
    
    assert rel_path == "images/test-seed/iter_03/sample_0.png"
    abs_path = tmp_path / rel_path
    assert abs_path.exists()
    assert abs_path.read_bytes() == png_data

def test_save_image_str(tmp_path):
    png_data = b"fake-png-bytes"
    rel_path = save_image(
        run_dir=tmp_path,
        seed_id="test-seed",
        iter_idx="baseline",
        sample_idx=1,
        png_bytes=png_data
    )
    
    assert rel_path == "images/test-seed/baseline/sample_1.png"
    abs_path = tmp_path / rel_path
    assert abs_path.exists()
    assert abs_path.read_bytes() == png_data
