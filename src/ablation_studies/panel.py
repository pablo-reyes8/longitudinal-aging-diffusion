"""Fixed longitudinal evaluation panels shared by every ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_REQUIRED = {"sample_id", "identity_id", "source_path", "target_path", "source_age", "target_age"}


def _normalize_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    missing = sorted(_REQUIRED.difference(row))
    if missing:
        raise ValueError(f"Evaluation pair {index} is missing: {', '.join(missing)}")
    if not str(row["identity_id"]).strip():
        raise ValueError("Every longitudinal pair requires a non-empty identity_id for the same person")
    source_age, target_age = int(row["source_age"]), int(row["target_age"])
    return {
        "sample_id": str(row["sample_id"]),
        "identity_id": str(row["identity_id"]),
        "source_path": str(row["source_path"]),
        "target_path": str(row["target_path"]),
        "source_age": source_age,
        "target_age": target_age,
        "delta_age": target_age - source_age,
    }


def build_fixed_evaluation_panel(
    val_loader,
    *,
    size: int | None = 64,
    explicit_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic panel without decoding validation images."""
    if explicit_pairs is not None:
        rows = [_normalize_row(row, index) for index, row in enumerate(explicit_pairs)]
    else:
        if val_loader is None:
            raise ValueError("val_loader is required when explicit_pairs is not provided")
        dataset = val_loader.dataset
        if not hasattr(dataset, "pair_for_index"):
            raise TypeError("Validation dataset must expose pair_for_index()")
        count = len(dataset) if size is None else min(len(dataset), int(size))
        if count < 1:
            raise ValueError("The fixed validation panel cannot be empty")
        root = Path(dataset.root_dir)
        rows = []
        for index in range(count):
            pair = dataset.pair_for_index(index)
            rows.append(_normalize_row({
                "sample_id": f"{pair.person_id}_{pair.source_age}_{pair.target_age}_{index:05d}",
                "identity_id": pair.person_id,
                "source_path": root / pair.source_path,
                "target_path": root / pair.target_path,
                "source_age": pair.source_age,
                "target_age": pair.target_age,
            }, index))
    if not rows:
        raise ValueError("The fixed validation panel cannot be empty")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation panel sample_id values must be unique")
    return rows


def persist_fixed_evaluation_panel(
    panel: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Create the shared panel once, rejecting accidental cross-case drift."""
    destination = Path(path)
    normalized = [_normalize_row(row, index) for index, row in enumerate(panel)]
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                "The fixed validation panel differs from the panel already used by another ablation"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return destination

