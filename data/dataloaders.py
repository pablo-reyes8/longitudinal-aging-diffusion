"""High-level factory for leakage-free train/validation/test DataLoaders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CombinedFaceAgingDataset, FaceAgingDataset, collate_face_aging_batch
from .fgnet import (
    build_fgnet_manifest,
    select_complementary_fgnet_pairs,
)
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
    include_zero_delta_pairs: bool = False,
    zero_delta_pair_prob: float = 0.20,
    include_bidirectional_pairs: bool = False,
    reverse_pair_prob: float = 0.20,
    include_kaggle: bool = False,
    kaggle_path: str | Path | None = None,
    kaggle_proportion: float = 0.40,
    kaggle_reverse_pair_prob: float = 0.50,
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
    if not 0.0 <= kaggle_proportion <= 1.0:
        raise ValueError("kaggle_proportion must be in [0, 1]")
    if not 0.0 <= kaggle_reverse_pair_prob <= 1.0:
        raise ValueError("kaggle_reverse_pair_prob must be in [0, 1]")
    if include_kaggle and kaggle_path is None:
        raise ValueError("include_kaggle=True requires kaggle_path")
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
    datasets: dict[str, FaceAgingDataset | CombinedFaceAgingDataset] = {}
    loaders: dict[str, DataLoader] = {}
    kaggle_metadata = {
        "enabled": bool(include_kaggle),
        "root_dir": None,
        "images": 0,
        "identities": 0,
        "available_pairs": 0,
        "selected_pairs": 0,
        "selected_identities": 0,
        "kaggle_proportion": float(kaggle_proportion),
        "selection_mode": "all" if kaggle_proportion == 1.0 else "scarcity_aware_budget",
    }
    for split_index, split in enumerate(("train", "val", "test")):
        split_manifest: list[ImageRecord] = records_for_split(manifest, assignments, split)
        strategy = train_pair_strategy if split == "train" else eval_pair_strategy
        flip_prob = horizontal_flip_prob if split == "train" else 0.0
        dataset: FaceAgingDataset | CombinedFaceAgingDataset = FaceAgingDataset(
            root,
            split_manifest,
            image_size=image_size,
            pair_strategy=strategy,
            min_age_gap=min_age_gap,
            max_age_gap=max_age_gap,
            prompt_style=prompt_style,
            dynamic_person_word=dynamic_person_word,
            horizontal_flip_prob=flip_prob,
            include_zero_delta_pairs=include_zero_delta_pairs if split == "train" else False,
            zero_delta_pair_prob=zero_delta_pair_prob,
            include_bidirectional_pairs=(
                include_bidirectional_pairs if split == "train" else False
            ),
            reverse_pair_prob=reverse_pair_prob,
            seed=seed + split_index,
        )
        if split == "train" and include_kaggle:
            fgnet_root, fgnet_manifest, fgnet_audit = build_fgnet_manifest(
                kaggle_path, min_age=min_age, max_age=max_age
            )
            fgnet_all_pairs = FaceAgingDataset(
                fgnet_root,
                fgnet_manifest,
                pair_strategy="all",
                min_age_gap=min_age_gap,
                max_age_gap=max_age_gap,
                seed=seed + 10_000,
            ).all_pairs
            selected_count = (
                len(fgnet_all_pairs)
                if kaggle_proportion == 1.0
                else min(
                    len(fgnet_all_pairs),
                    max(0, round(len(dataset) * kaggle_proportion)),
                )
            )
            selected_pairs, selection_audit = select_complementary_fgnet_pairs(
                dataset.all_pairs,
                fgnet_all_pairs,
                count=selected_count,
                seed=seed,
            )
            complementary = FaceAgingDataset(
                fgnet_root,
                fgnet_manifest,
                image_size=image_size,
                pair_strategy="all",
                min_age_gap=min_age_gap,
                max_age_gap=max_age_gap,
                prompt_style=prompt_style,
                dynamic_person_word=dynamic_person_word,
                horizontal_flip_prob=horizontal_flip_prob,
                include_zero_delta_pairs=False,
                zero_delta_pair_prob=0.0,
                include_bidirectional_pairs=include_bidirectional_pairs,
                reverse_pair_prob=kaggle_reverse_pair_prob,
                pair_records=selected_pairs,
                seed=seed + 10_000,
            )
            combined = CombinedFaceAgingDataset(dataset, complementary)
            combined.include_kaggle = True
            combined.kaggle_proportion = float(kaggle_proportion)
            combined.kaggle_available_pairs = len(fgnet_all_pairs)
            combined.kaggle_selected_pairs = len(selected_pairs)
            dataset = combined
            kaggle_metadata.update({
                **fgnet_audit,
                **selection_audit,
                "enabled": True,
                "reverse_pair_prob": (
                    float(kaggle_reverse_pair_prob) if include_bidirectional_pairs else 0.0
                ),
            })
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
        "kaggle": kaggle_metadata,
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
            "include_zero_delta_pairs": bool(include_zero_delta_pairs),
            "zero_delta_pair_prob": float(zero_delta_pair_prob),
            "include_bidirectional_pairs": bool(include_bidirectional_pairs),
            "reverse_pair_prob": float(reverse_pair_prob),
            "include_kaggle": bool(include_kaggle),
            "kaggle_path": str(Path(kaggle_path).expanduser().resolve()) if kaggle_path else None,
            "kaggle_proportion": float(kaggle_proportion),
            "kaggle_reverse_pair_prob": float(kaggle_reverse_pair_prob),
        },
    })
    train_dataset = datasets["train"]
    primary_observations = int(getattr(train_dataset, "primary_observations", len(train_dataset)))
    complementary_observations = int(getattr(train_dataset, "complementary_observations", 0))
    primary_canonical_pairs = len(getattr(train_dataset, "all_pairs", ()))
    print(
        " Pair sources | "
        f"Colombian canonical={primary_canonical_pairs:,}, epoch_observations={primary_observations:,} | "
        f"FG-NET available={kaggle_metadata['available_pairs']:,}, selected={complementary_observations:,}, "
        f"reverse_prob={kaggle_metadata.get('reverse_pair_prob', 0.0):.2f} | "
        f"combined_epoch_observations={len(train_dataset):,}"
    )
    return loaders, metadata
