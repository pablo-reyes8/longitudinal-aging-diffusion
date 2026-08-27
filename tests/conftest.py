from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


def write_image(path: Path, color=(128, 64, 32), size=(19, 13), mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        value = color if isinstance(color, int) else color[0]
    elif mode == "RGBA":
        value = tuple(color) if len(color) == 4 else (*color, 255)
    else:
        value = color
    Image.new(mode, size, value).save(path)


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    for person_index in range(8):
        person = tmp_path / f"id_{person_index:04d}"
        for age in (10 + person_index, 20 + person_index, 30 + person_index):
            write_image(person / f"{age}.png", color=(age, 2 * age, 3 * age))
    return tmp_path


@pytest.fixture
def tiny_fgnet_root(tmp_path: Path) -> Path:
    root = tmp_path / "fgnet_flat"
    filenames = (
        "001A02.JPG", "001A05.JPG", "001A43a.JPG", "001A43b.JPG",
        "002A03.JPG", "002A10.JPG", "002A38.JPG",
    )
    for index, filename in enumerate(filenames):
        write_image(root / filename, color=(20 + index, 40 + index, 60 + index))
    return root
