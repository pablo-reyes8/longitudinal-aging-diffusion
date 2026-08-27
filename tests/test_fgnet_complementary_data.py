from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data import (
    PairRecord,
    build_face_aging_dataloaders,
    build_fgnet_manifest,
    build_pair_index,
    parse_fgnet_filename,
    select_complementary_fgnet_pairs,
)


def _pair(person: str, source_age: int, target_age: int, index: int = 0) -> PairRecord:
    return PairRecord(
        person_id=person,
        source_index=index,
        target_index=index + 1,
        source_age=source_age,
        target_age=target_age,
        delta_age=target_age - source_age,
        source_path=f"{person}_{source_age}_{index}.jpg",
        target_path=f"{person}_{target_age}_{index}.jpg",
    )


def test_fgnet_filename_parser_supports_duplicates_and_rejects_noise():
    assert parse_fgnet_filename("001A02.JPG") == ("fgnet_001", 2, 0, ".jpg")
    assert parse_fgnet_filename("001A43a.JPG") == ("fgnet_001", 43, 1, ".jpg")
    assert parse_fgnet_filename("001A43b.jpeg") == ("fgnet_001", 43, 2, ".jpeg")
    assert parse_fgnet_filename("age_43.jpg") is None
    assert parse_fgnet_filename("001A43.pts") is None


def test_fgnet_sample_manifest_and_all_pairs_are_exact(tiny_fgnet_root: Path):
    root, manifest, audit = build_fgnet_manifest(tiny_fgnet_root)
    assert root.name == "fgnet_flat"
    assert audit["identities"] == 2
    assert audit["images"] == 7
    assert len([row for row in manifest if row.person_id == "fgnet_001"]) == 4
    assert [(row.age, row.same_age_index) for row in manifest if row.age == 43] == [(43, 1), (43, 2)]
    assert len(build_pair_index(manifest)) == 8


def test_scarcity_selector_prefers_missing_child_to_adult_transition():
    primary = [_pair("col", 30, 45, index) for index in range(5)]
    candidates = [
        _pair("fgnet_001", 30, 45),
        _pair("fgnet_002", 2, 43),
    ]
    selected, audit = select_complementary_fgnet_pairs(
        primary, candidates, count=1, seed=9
    )
    assert [(pair.source_age, pair.target_age) for pair in selected] == [(2, 43)]
    assert audit["selected_pairs"] == 1


def test_scarcity_selector_balances_identities_inside_same_transition_cell():
    candidates = [
        _pair(person, 2, 43, index)
        for person in ("fgnet_001", "fgnet_002")
        for index in range(4)
    ]
    selected, _ = select_complementary_fgnet_pairs([], candidates, count=4, seed=11)
    counts = {person: sum(pair.person_id == person for pair in selected) for person in ("fgnet_001", "fgnet_002")}
    assert counts == {"fgnet_001": 2, "fgnet_002": 2}


def test_kaggle_ratio_adds_to_primary_without_cutting_it(tiny_root: Path, tiny_fgnet_root: Path, capsys):
    baseline, _ = build_face_aging_dataloaders(
        tiny_root,
        batch_size=4,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
    )
    primary_count = len(baseline["train"].dataset)
    loaders, metadata = build_face_aging_dataloaders(
        tiny_root,
        batch_size=4,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
        include_kaggle=True,
        kaggle_path=tiny_fgnet_root,
        kaggle_proportion=0.40,
    )
    train = loaders["train"].dataset
    expected_fgnet = round(primary_count * 0.40)
    assert train.primary_observations == primary_count
    assert train.complementary_observations == expected_fgnet
    assert len(train) == primary_count + expected_fgnet
    assert metadata["kaggle"]["selected_pairs"] == expected_fgnet
    assert metadata["kaggle"]["available_pairs"] == 8
    assert len(loaders["val"].dataset) < len(train)
    printed = capsys.readouterr().out
    assert "Pair sources" in printed and "Colombian" in printed and "FG-NET" in printed


def test_kaggle_proportion_one_uses_all_combinations(tiny_root: Path, tiny_fgnet_root: Path):
    loaders, metadata = build_face_aging_dataloaders(
        tiny_root,
        batch_size=64,
        num_workers=0,
        train_drop_last=False,
        include_kaggle=True,
        kaggle_path=tiny_fgnet_root,
        kaggle_proportion=1.0,
    )
    assert metadata["kaggle"]["available_pairs"] == 8
    assert metadata["kaggle"]["selected_pairs"] == 8
    assert loaders["train"].dataset.complementary_observations == 8


def test_fgnet_uses_higher_reverse_probability_and_common_preprocessing(tiny_root: Path, tiny_fgnet_root: Path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root,
        image_size=32,
        batch_size=16,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
        include_bidirectional_pairs=True,
        reverse_pair_prob=0.20,
        include_kaggle=True,
        kaggle_path=tiny_fgnet_root,
        kaggle_proportion=1.0,
        kaggle_reverse_pair_prob=0.50,
    )
    dataset = loaders["train"].dataset
    fgnet = dataset.complementary
    reverse = total = 0
    for epoch in range(200):
        fgnet.set_epoch(epoch)
        for index in range(len(fgnet)):
            reverse += fgnet.pair_for_index(index).delta_age < 0
            total += 1
    assert reverse / total == pytest.approx(0.50, abs=0.02)
    sample = fgnet[0]
    assert sample["source_image"].shape == (3, 32, 32)
    assert sample["target_image"].shape == (3, 32, 32)
    assert sample["source_image"].dtype == torch.float32
    assert -1 <= sample["source_image"].min() <= sample["source_image"].max() <= 1


def test_combined_dataset_tags_source_and_keeps_evaluation_colombian(tiny_root: Path, tiny_fgnet_root: Path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root,
        batch_size=512,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
        include_kaggle=True,
        kaggle_path=tiny_fgnet_root,
        kaggle_proportion=1.0,
    )
    batch = next(iter(loaders["train"]))
    assert set(batch["source_dataset"]) == {"colombian", "fgnet"}
    assert all(not row.person_id.startswith("fgnet_") for row in loaders["val"].dataset.manifest)
    assert all(not row.person_id.startswith("fgnet_") for row in loaders["test"].dataset.manifest)


@pytest.mark.parametrize("proportion", (-0.1, 1.1))
def test_kaggle_proportion_validation(tiny_root: Path, tiny_fgnet_root: Path, proportion: float):
    with pytest.raises(ValueError, match="kaggle_proportion"):
        build_face_aging_dataloaders(
            tiny_root,
            include_kaggle=True,
            kaggle_path=tiny_fgnet_root,
            kaggle_proportion=proportion,
        )


def test_include_kaggle_requires_path(tiny_root: Path):
    with pytest.raises(ValueError, match="requires kaggle_path"):
        build_face_aging_dataloaders(tiny_root, include_kaggle=True)
