from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data import FaceAgingDataset, build_image_manifest, build_prompts, get_person_word
from conftest import write_image


def _dataset(root: Path, **kwargs) -> FaceAgingDataset:
    manifest, _ = build_image_manifest(root)
    return FaceAgingDataset(root, manifest, **kwargs)


def test_resize_rgb_integrity_and_pixel_range(tmp_path: Path):
    for mode, suffix in (("RGB", "jpg"), ("L", "png"), ("RGBA", "png")):
        write_image(tmp_path / "id_a" / f"10.{suffix}", color=0 if mode == "L" else (0, 0, 0), mode=mode)
        write_image(tmp_path / "id_a" / f"20.{suffix}", color=255 if mode == "L" else (255, 255, 255), mode=mode)
        dataset = _dataset(tmp_path, image_size=32)
        sample = dataset[0]
        assert sample["source_image"].shape == (3, 32, 32)
        assert sample["target_image"].shape == (3, 32, 32)
        assert sample["source_image"].dtype == torch.float32
        assert torch.isfinite(sample["source_image"]).all()
        assert sample["source_image"].min() >= -1
        assert sample["target_image"].max() <= 1
        for path in (tmp_path / "id_a").iterdir():
            path.unlink()


def test_center_crop_then_resize_not_stretch(tmp_path: Path):
    person = tmp_path / "id_a"
    person.mkdir()
    wide = np.zeros((10, 20, 3), dtype=np.uint8)
    wide[:, :5] = (255, 0, 0)
    wide[:, 5:15] = (0, 255, 0)
    wide[:, 15:] = (0, 0, 255)
    Image.fromarray(wide).save(person / "10.png")
    Image.fromarray(wide).save(person / "20.png")
    sample = _dataset(tmp_path, image_size=10)[0]
    # Center crop keeps the green middle square; red/blue outer borders disappear.
    restored = sample["source_image"].add(1).mul(127.5)
    assert restored[1].mean() > 240
    assert restored[0].mean() < 15 and restored[2].mean() < 15


def test_synchronized_horizontal_flip(tmp_path: Path):
    person = tmp_path / "id_a"
    person.mkdir()
    asymmetric = np.zeros((8, 8, 3), dtype=np.uint8)
    asymmetric[:, :2] = 255
    Image.fromarray(asymmetric).save(person / "10.png")
    Image.fromarray(asymmetric).save(person / "20.png")
    unflipped = _dataset(tmp_path, image_size=8, horizontal_flip_prob=0)[0]
    flipped = _dataset(tmp_path, image_size=8, horizontal_flip_prob=1)[0]
    assert torch.equal(flipped["source_image"], torch.flip(unflipped["source_image"], dims=(2,)))
    assert torch.equal(flipped["target_image"], torch.flip(unflipped["target_image"], dims=(2,)))


def test_random_target_epoch_resampling_and_reset(tmp_path: Path):
    for age in (20, 30, 40, 50, 60):
        write_image(tmp_path / "id_a" / f"{age}.png")
    dataset = _dataset(tmp_path, pair_strategy="random_target", seed=17)
    observed = []
    for epoch in range(12):
        dataset.set_epoch(epoch)
        pair = dataset.pair_for_index(0)
        observed.append(pair.target_age)
        assert pair.target_age > pair.source_age
    assert len(set(observed)) > 1
    dataset.set_epoch(4)
    first = dataset.pair_for_index(0)
    dataset.set_epoch(4)
    assert dataset.pair_for_index(0) == first


def test_all_and_random_target_membership(tmp_path: Path):
    for age in (10, 20, 30, 40):
        write_image(tmp_path / "id_a" / f"{age}.png")
    exhaustive = _dataset(tmp_path, pair_strategy="all", min_age_gap=5, max_age_gap=25)
    sampled = _dataset(tmp_path, pair_strategy="random_target", min_age_gap=5, max_age_gap=25)
    universe = set(exhaustive.all_pairs)
    for epoch in range(20):
        sampled.set_epoch(epoch)
        assert all(sampled.pair_for_index(i) in universe for i in range(len(sampled)))


def test_prompts_and_dynamic_boundaries():
    prompts = build_prompts(27, 52, prompt_style="selfage")
    assert prompts["source_prompt"] == "photo of a person as 27-year-old"
    assert prompts["target_prompt"] == "photo of a person as 52-year-old"
    assert build_prompts(27, 52, prompt_style="fading")["target_prompt"] == "photo of a 52 year old person"
    male = {age: get_person_word(age, "male", True) for age in (4, 5, 14, 15, 64, 65)}
    female = {age: get_person_word(age, "female", True) for age in (4, 5, 14, 15, 64, 65)}
    assert list(male.values()) == ["baby", "boy", "boy", "man", "man", "elderly"]
    assert list(female.values()) == ["baby", "girl", "girl", "woman", "woman", "elderly"]
    assert all(get_person_word(age, None, True) == "person" for age in male)


def test_corrupt_image_has_path_specific_error(tmp_path: Path):
    write_image(tmp_path / "id_a" / "10.png")
    corrupt = tmp_path / "id_a" / "20.jpg"
    corrupt.write_bytes(b"not an image")
    dataset = _dataset(tmp_path)
    with pytest.raises(RuntimeError, match=r"20\.jpg"):
        dataset[0]


def test_source_target_are_distinct_tensors(tmp_path: Path):
    write_image(tmp_path / "id_a" / "10.png")
    write_image(tmp_path / "id_a" / "20.png")
    sample = _dataset(tmp_path)[0]
    assert sample["source_path"] != sample["target_path"]
    assert sample["source_image"].data_ptr() != sample["target_image"].data_ptr()
