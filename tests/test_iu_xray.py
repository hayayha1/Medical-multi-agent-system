from pathlib import Path

import pytest

from app.datasets.iu_xray import IUXRayDataset


def test_finds_multiple_case_views(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "CXR100_IM-0001-1001.png").write_bytes(b"png")
    (image_dir / "CXR100_IM-0001-2001.png").write_bytes(b"png")

    dataset = IUXRayDataset(str(tmp_path))
    images = dataset.resolve_request("unused", "CXR100", [])

    assert len(images) == 2
    assert all(Path(item).is_absolute() for item in images)


def test_rejects_path_outside_dataset(tmp_path: Path):
    dataset = IUXRayDataset(str(tmp_path))
    with pytest.raises(ValueError):
        dataset.resolve_relative_paths(["../outside.png"])

