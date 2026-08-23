"""High-level factory for leakage-free train/validation/test DataLoaders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FaceAgingDataset, collate_face_aging_batch
from .indexing import (
    ImageRecord,
    build_identity_splits,
    build_image_manifest,
    records_for_split,
)
from .validation import summarize_pipeline


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_face_aging_dataloaders(
    root_dir: str | Path,
    *,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    train_pair_strategy: str = "random_target",
    eval_pair_strategy: str = "all",
    min_age_gap: int = 1,
    max_age_gap: int | None = None,
    prompt_style: str = "selfage",
    dynamic_person_word: bool = False,
    gender_by_person: Mapping[str, str | None] | None = None,
    horizontal_flip_prob: float = 0.0,
    manifest_path: str | Path | None = None,
    split_path: str | Path | None = None,
    use_cached_manifest: bool = True,
    use_cached_splits: bool = True,
    min_age: int | None = None,
    max_age: int | None = None,
    normalize_over_100: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    train_shuffle: bool = True,
    train_drop_last: bool = True,
    eval_drop_last: bool = False,
) -> tuple[dict[str, DataLoader], dict]:
    """Build all loaders from a single root path and return rich metadata."""
    root = Path(root_dir).expanduser().resolve()
    manifest, scan_audit = build_image_manifest(
        root,
        manifest_path=manifest_path,
        use_cached_manifest=use_cached_manifest,
        min_age=min_age,
        max_age=max_age,
        normalize_over_100=normalize_over_100,
        gender_by_person=gender_by_person,
    )
    assignments = build_identity_splits(
        manifest,
        split_ratios=split_ratios,
        seed=seed,
        split_path=split_path,
        use_cached_splits=use_cached_splits,
    )
    datasets: dict[str, FaceAgingDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        split_manifest: list[ImageRecord] = records_for_split(manifest, assignments, split)
        strategy = train_pair_strategy if split == "train" else eval_pair_strategy
        flip_prob = horizontal_flip_prob if split == "train" else 0.0
        dataset = FaceAgingDataset(
            root,
            split_manifest,
            image_size=image_size,
            pair_strategy=strategy,
            min_age_gap=min_age_gap,
            max_age_gap=max_age_gap,
            prompt_style=prompt_style,
            dynamic_person_word=dynamic_person_word,
            horizontal_flip_prob=flip_prob,
            seed=seed + split_index,
        )
        datasets[split] = dataset
        generator = torch.Generator().manual_seed(seed + split_index)
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train_shuffle if split == "train" else False,
            drop_last=train_drop_last if split == "train" else eval_drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            worker_init_fn=_seed_worker,
            generator=generator,
            collate_fn=collate_face_aging_batch,
        )
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        loaders[split] = DataLoader(**kwargs)

    metadata = summarize_pipeline(root, manifest, assignments, datasets, scan_audit)
    metadata.update({
        "root_dir": str(root),
        "manifest": manifest,
        "split_assignments": assignments,
        "datasets": datasets,
        "scan_audit": scan_audit,
        "config": {
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "split_ratios": tuple(split_ratios),
            "seed": seed,
            "train_pair_strategy": train_pair_strategy,
            "eval_pair_strategy": eval_pair_strategy,
            "min_age_gap": min_age_gap,
            "max_age_gap": max_age_gap,
            "prompt_style": prompt_style,
        },
    })
    return loaders, metadata
