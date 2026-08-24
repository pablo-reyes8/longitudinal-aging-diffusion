"""Audits and reusable validation report for the complete data pipeline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from .indexing import ImageRecord, PairRecord, build_pair_index
from .prompts import build_prompts

if TYPE_CHECKING:
    from .dataset import FaceAgingDataset


AGE_BANDS = ((0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, None))
DELTA_BANDS = ((0, 0), (1, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, None))


def _describe(values: Sequence[int | float]) -> dict:
    if not values:
        return {
            "count": 0, "min": None, "q25": None, "median": None,
            "mean": None, "q75": None, "q95": None, "q99": None, "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _band_counts(values: Sequence[int], bands: Sequence[tuple[int, int | None]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for low, high in bands:
        label = f"{low}+" if high is None else f"{low}-{high}"
        counts[label] = sum(value >= low and (high is None or value <= high) for value in values)
    return counts


def _shares(counts: Sequence[int]) -> dict[str, float]:
    ordered = sorted(counts, reverse=True)
    total = sum(ordered)
    if not total:
        return {"top_1_percent": 0.0, "top_5_percent": 0.0, "top_10_percent": 0.0}
    output = {}
    for percent in (1, 5, 10):
        n = max(1, math.ceil(len(ordered) * percent / 100))
        output[f"top_{percent}_percent"] = sum(ordered[:n]) / total
    return output


def summarize_pipeline(
    root_dir: Path,
    manifest: Sequence[ImageRecord],
    assignments: Mapping[str, str],
    datasets: Mapping[str, "FaceAgingDataset"],
    scan_audit: Mapping,
) -> dict:
    ids = sorted({row.person_id for row in manifest})
    image_counts = Counter(row.person_id for row in manifest)
    raw_directory_files = sum(1 for p in root_dir.glob("*/*") if p.is_file())
    split_stats: dict[str, dict] = {}
    age_stats: dict[str, dict] = {}
    delta_stats: dict[str, dict] = {}
    pair_by_identity: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        split_rows = [row for row in manifest if assignments[row.person_id] == split]
        pairs = datasets[split].all_pairs
        for pair in pairs:
            pair_by_identity[pair.person_id] += 1
        ages = [row.age for row in split_rows]
        deltas = [pair.delta_age for pair in pairs]
        split_stats[split] = {
            "identities": len({row.person_id for row in split_rows}),
            "images": len(split_rows),
            "dataset_observations": len(datasets[split]),
            "valid_forward_pairs": len(pairs),
        }
        age_stats[split] = {**_describe(ages), "by_age": dict(sorted(Counter(ages).items())), "bands": _band_counts(ages, AGE_BANDS)}
        delta_stats[split] = {**_describe(deltas), "bands": _band_counts(deltas, DELTA_BANDS)}
    pair_values = list(pair_by_identity.values())
    repeated = Counter((row.person_id, row.age) for row in manifest)
    repeated_groups = {f"{pid}:{age}": count for (pid, age), count in repeated.items() if count > 1}
    sampling_stats = _sampling_diagnostic(datasets.get("train"), n_samples=10_000)
    longitudinal = []
    for person_id in ids:
        person_ages = [row.age for row in manifest if row.person_id == person_id]
        longitudinal.append({
            "person_id": person_id,
            "n_images": len(person_ages),
            "n_unique_ages": len(set(person_ages)),
            "age_min": min(person_ages),
            "age_max": max(person_ages),
            "age_span": max(person_ages) - min(person_ages),
            "n_valid_pairs": pair_by_identity[person_id],
        })
    return {
        "manifest_stats": {
            "identities": len(ids),
            "files_seen": raw_directory_files,
            "valid_images": len(manifest),
            "skipped_unparseable": len(scan_audit.get("skipped_files", [])),
            "normalized_ages": len(scan_audit.get("normalized_ages", [])),
            "age": _describe([row.age for row in manifest]),
            "images_per_identity": _describe(list(image_counts.values())),
        },
        "split_stats": split_stats,
        "age_stats": age_stats,
        "delta_age_stats": delta_stats,
        "pair_stats": {
            "total_valid_pairs": sum(pair_values),
            "pairs_per_identity": _describe(pair_values),
            "concentration": _shares(pair_values),
        },
        "sampling_stats": sampling_stats,
        "longitudinal_stats": {
            "per_identity": longitudinal,
            "age_span": _describe([row["age_span"] for row in longitudinal]),
            "unique_ages_per_identity": _describe([row["n_unique_ages"] for row in longitudinal]),
        },
        "same_age_duplicates": {
            "groups": repeated_groups,
            "identities_with_duplicates": len({key.split(":", 1)[0] for key in repeated_groups}),
            "largest_multiplicity": max(repeated_groups.values(), default=1),
        },
    }


def _sampling_diagnostic(dataset: "FaceAgingDataset" | None, n_samples: int) -> dict:
    if dataset is None or len(dataset) == 0:
        return {
            "samples": 0, "top_identities": [], "identity_concentration": _shares([]),
            "source_age": _describe([]), "target_age": _describe([]), "delta_age": _describe([]),
        }
    original_epoch = dataset.epoch
    identities: Counter[str] = Counter()
    source_ages: list[int] = []
    target_ages: list[int] = []
    deltas: list[int] = []
    for draw in range(n_samples):
        dataset.set_epoch(draw // len(dataset))
        pair = dataset.pair_for_index(draw % len(dataset))
        identities[pair.person_id] += 1
        source_ages.append(pair.source_age)
        target_ages.append(pair.target_age)
        deltas.append(pair.delta_age)
    dataset.set_epoch(original_epoch)
    return {
        "samples": n_samples,
        "top_identities": identities.most_common(10),
        "identity_concentration": _shares(list(identities.values())),
        "source_age": _describe(source_ages),
        "target_age": _describe(target_ages),
        "delta_age": {**_describe(deltas), "bands": _band_counts(deltas, DELTA_BANDS)},
    }


def _pair_key(pair: PairRecord) -> tuple[str, str, str]:
    return pair.person_id, pair.source_path, pair.target_path


def _validate_corrupt_images(root: Path, manifest: Sequence[ImageRecord]) -> list[str]:
    errors = []
    for row in manifest:
        path = row.path(root)
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:  # this is an audit boundary, and the path is reported
            errors.append(f"Unreadable image {path}: {type(exc).__name__}")
    return errors


def run_data_pipeline_validation(
    root_dir: str | Path,
    *,
    seed: int = 42,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    min_age_gap: int = 1,
    max_age_gap: int | None = None,
    validate_images: bool = True,
    manifest_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> dict:
    """Run structural, leakage, pair, prompt, path, and optional decode checks."""
    from .dataset import FaceAgingDataset
    from .indexing import build_identity_splits, build_image_manifest, records_for_split

    root = Path(root_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    manifest, scan_audit = build_image_manifest(root, manifest_path=manifest_path)
    assignments = build_identity_splits(
        manifest, split_ratios=split_ratios, seed=seed, split_path=split_path
    )
    datasets = {
        split: FaceAgingDataset(
            root,
            records_for_split(manifest, assignments, split),
            pair_strategy="all",
            min_age_gap=min_age_gap,
            max_age_gap=max_age_gap,
            seed=seed,
        )
        for split in ("train", "val", "test")
    }
    summary = summarize_pipeline(root, manifest, assignments, datasets, scan_audit)

    id_sets = {split: {row.person_id for row in datasets[split].manifest} for split in datasets}
    path_sets = {split: {str(row.path(root).resolve()) for row in datasets[split].manifest} for split in datasets}
    pair_sets = {split: {_pair_key(p) for p in datasets[split].all_pairs} for split in datasets}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if id_sets[left] & id_sets[right]:
            errors.append(f"Identity leakage between {left} and {right}")
        if path_sets[left] & path_sets[right]:
            errors.append(f"Resolved image-path leakage between {left} and {right}")
        if pair_sets[left] & pair_sets[right]:
            errors.append(f"Pair leakage between {left} and {right}")

    for split, dataset in datasets.items():
        brute_force = set()
        for i, source in enumerate(dataset.manifest):
            for j, target in enumerate(dataset.manifest):
                gap = target.age - source.age
                if source.person_id == target.person_id and gap >= min_age_gap and (max_age_gap is None or gap <= max_age_gap):
                    brute_force.add((source.person_id, source.relative_path, target.relative_path))
        production = {_pair_key(pair) for pair in dataset.all_pairs}
        if production != brute_force:
            errors.append(f"Pair index differs from brute-force oracle in {split}")
        for pair in dataset.all_pairs:
            source = dataset.manifest[pair.source_index]
            target = dataset.manifest[pair.target_index]
            if source.person_id != target.person_id or source.person_id != pair.person_id:
                errors.append(f"Cross-identity pair in {split}: {_pair_key(pair)}")
            if target.age <= source.age or pair.delta_age != target.age - source.age:
                errors.append(f"Invalid direction/delta in {split}: {_pair_key(pair)}")
            if source.relative_path == target.relative_path:
                errors.append(f"Source-target alias in {split}: {source.relative_path}")
            for style in ("selfage", "fading"):
                prompts = build_prompts(source.age, target.age, prompt_style=style)
                if str(source.age) not in prompts["source_prompt"] or str(target.age) not in prompts["target_prompt"]:
                    errors.append(f"Prompt-age mismatch in {split}: {_pair_key(pair)}")

    duplicate_rows = len(manifest) - len({(r.person_id, r.relative_path, r.age, r.same_age_index) for r in manifest})
    duplicate_paths = len(manifest) - len({r.relative_path for r in manifest})
    if duplicate_rows or duplicate_paths:
        errors.append(f"Duplicate manifest data: rows={duplicate_rows}, paths={duplicate_paths}")
    if validate_images:
        errors.extend(_validate_corrupt_images(root, manifest))
    if scan_audit["skipped_files"]:
        warnings.append(f"Skipped {len(scan_audit['skipped_files'])} unparseable files")
    if scan_audit["normalized_ages"]:
        warnings.append(f"Normalized {len(scan_audit['normalized_ages'])} suspicious age labels")
    for split in ("train", "val", "test"):
        if summary["split_stats"][split]["identities"] == 0:
            warnings.append(f"Split {split} has no identities")
        elif summary["split_stats"][split]["valid_forward_pairs"] == 0:
            warnings.append(f"Split {split} has no valid longitudinal pairs")
    if summary["pair_stats"]["concentration"]["top_10_percent"] > 0.5 and len(manifest) >= 10:
        warnings.append("Top 10% of identities contribute more than half of all pairs")
    info.append("Identity, resolved-path, and pair split sets are disjoint")
    info.append("Production pair index was compared with an independent brute-force oracle")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        **summary,
        "scan_audit": scan_audit,
    }
