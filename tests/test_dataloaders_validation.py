from __future__ import annotations

from pathlib import Path

import torch

from data import build_face_aging_dataloaders, run_data_pipeline_validation


def _metadata_sequence(loader):
    output = []
    for batch in loader:
        output.extend(zip(
            batch["person_id"], batch["source_path"], batch["target_path"],
            batch["source_age"].tolist(), batch["target_age"].tolist(),
        ))
    return output


def test_loader_contract_metadata_alignment_and_split_leakage(tiny_root: Path):
    loaders, metadata = build_face_aging_dataloaders(
        tiny_root, image_size=24, batch_size=3, num_workers=0,
        train_drop_last=False, seed=19,
    )
    batch = next(iter(loaders["train"]))
    assert batch["source_image"].shape[1:] == (3, 24, 24)
    assert batch["target_image"].shape[1:] == (3, 24, 24)
    assert torch.equal(batch["delta_age"], batch["target_age"] - batch["source_age"])
    for index, filename in enumerate(batch["source_filename"]):
        assert int(filename.split(".")[0].split("_")[0]) == batch["source_age"][index]
        assert str(batch["source_age"][index].item()) in batch["source_prompt"][index]
    ids = metadata["split_assignments"]
    split_ids = {s: {pid for pid, split in ids.items() if split == s} for s in ("train", "val", "test")}
    assert split_ids["train"].isdisjoint(split_ids["val"] | split_ids["test"])


def test_evaluation_repeated_epochs_are_identical(tiny_root: Path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, batch_size=2, num_workers=0, train_drop_last=False, seed=3
    )
    assert _metadata_sequence(loaders["val"]) == _metadata_sequence(loaders["val"])
    assert _metadata_sequence(loaders["test"]) == _metadata_sequence(loaders["test"])


def test_multiworker_restart_reproducibility(tiny_root: Path):
    sequences = {}
    for workers in (0, 2, 4):
        kwargs = dict(root_dir=tiny_root, image_size=16, batch_size=2, num_workers=workers,
                      train_drop_last=False, seed=29, persistent_workers=False)
        first, _ = build_face_aging_dataloaders(**kwargs)
        second, _ = build_face_aging_dataloaders(**kwargs)
        sequence_a = _metadata_sequence(first["train"])
        sequence_b = _metadata_sequence(second["train"])
        assert sequence_a == sequence_b
        assert all(target_age > source_age for *_, source_age, target_age in sequence_a)
        sequences[workers] = sequence_a
    assert sequences[0] == sequences[2] == sequences[4]


def test_set_epoch_reaches_persistent_workers(tiny_root: Path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=16, batch_size=2, num_workers=2,
        train_drop_last=False, train_shuffle=False, seed=31, persistent_workers=True,
    )
    dataset = loaders["train"].dataset
    dataset.set_epoch(0)
    epoch_zero = _metadata_sequence(loaders["train"])
    dataset.set_epoch(1)
    epoch_one = _metadata_sequence(loaders["train"])
    dataset.set_epoch(0)
    epoch_zero_again = _metadata_sequence(loaders["train"])
    assert epoch_zero != epoch_one
    assert epoch_zero == epoch_zero_again


def test_validation_report_is_structured_and_passes(tiny_root: Path):
    report = run_data_pipeline_validation(tiny_root, validate_images=True)
    assert report["passed"], report["errors"]
    for key in ("errors", "warnings", "manifest_stats", "split_stats", "pair_stats", "age_stats", "delta_age_stats", "sampling_stats"):
        assert key in report
    assert report["manifest_stats"]["identities"] == 8
    assert report["pair_stats"]["total_valid_pairs"] == 24


def test_loader_injects_zero_delta_only_into_training_split(tiny_root: Path):
    loaders, metadata = build_face_aging_dataloaders(
        tiny_root,
        batch_size=16,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
        include_zero_delta_pairs=True,
        zero_delta_pair_prob=1.0,
    )
    train_batch = next(iter(loaders["train"]))
    assert torch.all(train_batch["delta_age"] == 0)
    assert all(
        source == target
        for source, target in zip(train_batch["source_path"], train_batch["target_path"])
    )
    for split in ("val", "test"):
        assert all(
            pair.delta_age > 0
            for pair in metadata["datasets"][split].all_pairs
        )
        assert metadata["datasets"][split].include_zero_delta_pairs is False
    assert metadata["config"]["include_zero_delta_pairs"] is True
    assert metadata["config"]["zero_delta_pair_prob"] == 1.0
