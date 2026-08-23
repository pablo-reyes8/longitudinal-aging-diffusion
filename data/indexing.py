"""Deterministic scanning, age parsing, identity splits, and pair indexing."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_FILENAME_RE = re.compile(r"^(\d+)(?:_(\d+))?\.(jpg|jpeg|png|webp)$", re.IGNORECASE)
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    person_id: str
    relative_path: str
    filename: str
    age: int
    same_age_index: int
    extension: str
    raw_age: int
    age_was_normalized: bool = False
    gender: str | None = None

    def path(self, root_dir: str | Path) -> Path:
        return Path(root_dir).expanduser().resolve() / Path(self.relative_path)


@dataclass(frozen=True)
class PairRecord:
    person_id: str
    source_index: int
    target_index: int
    source_age: int
    target_age: int
    delta_age: int
    source_path: str
    target_path: str


def _normalize_suspicious_age(raw_age: int) -> tuple[int, bool]:
    """Apply the project rule for malformed ages such as ``501 -> 51``.

    Only an interior zero is removed, and only if doing so produces a plausible
    age no greater than 100. Values that cannot be corrected unambiguously are
    retained so optional ``min_age``/``max_age`` filters can audit or exclude them.
    """
    if raw_age <= 100:
        return raw_age, False
    digits = str(raw_age)
    for position in range(1, len(digits) - 1):
        if digits[position] == "0":
            candidate = int(digits[:position] + digits[position + 1 :])
            if 0 <= candidate <= 100:
                return candidate, True
    return raw_age, False


def parse_age_filename(
    filename: str,
    *,
    normalize_over_100: bool = True,
) -> tuple[int, int, str, int, bool] | None:
    """Parse ``age[_same_age_index].extension`` without extracting stray digits.

    Returns ``(age, same_age_index, extension, raw_age, was_normalized)`` or
    ``None`` when the whole filename does not match the convention.
    """
    match = _FILENAME_RE.fullmatch(Path(filename).name)
    if match is None:
        return None
    raw_age = int(match.group(1))
    age, normalized = (
        _normalize_suspicious_age(raw_age) if normalize_over_100 else (raw_age, False)
    )
    same_age_index = int(match.group(2) or 0)
    return age, same_age_index, f".{match.group(3).lower()}", raw_age, normalized


def _save_manifest(rows: Sequence[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [
        "person_id", "relative_path", "filename", "age", "same_age_index",
        "extension", "raw_age", "age_was_normalized", "gender",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _load_manifest(path: Path) -> list[ImageRecord]:
    rows: list[ImageRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(ImageRecord(
                person_id=row["person_id"],
                relative_path=Path(row["relative_path"]).as_posix(),
                filename=row["filename"],
                age=int(row["age"]),
                same_age_index=int(row["same_age_index"]),
                extension=row["extension"],
                raw_age=int(row.get("raw_age") or row["age"]),
                age_was_normalized=str(row.get("age_was_normalized", "false")).lower()
                in {"1", "true", "yes"},
                gender=row.get("gender") or None,
            ))
    return sorted(rows, key=_image_sort_key)


def _image_sort_key(row: ImageRecord) -> tuple:
    return (row.person_id, row.age, row.same_age_index, row.filename.lower(), row.relative_path)


def build_image_manifest(
    root_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    use_cached_manifest: bool = True,
    min_age: int | None = None,
    max_age: int | None = None,
    normalize_over_100: bool = True,
    gender_by_person: Mapping[str, str | None] | None = None,
) -> tuple[list[ImageRecord], dict]:
    """Scan one level of identity folders once and return canonical records + audit."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")
    cache = Path(manifest_path).expanduser() if manifest_path else None
    if cache is not None and cache.exists() and use_cached_manifest:
        rows = _load_manifest(cache)
        return rows, {
            "loaded_from_cache": True,
            "manifest_path": str(cache.resolve()),
            "skipped_files": [],
            "normalized_ages": [asdict(r) for r in rows if r.age_was_normalized],
            "filtered_ages": [],
        }

    skipped: list[dict] = []
    normalized: list[dict] = []
    filtered: list[dict] = []
    rows: list[ImageRecord] = []
    genders = gender_by_person or {}
    for identity_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for image_path in sorted((p for p in identity_dir.iterdir() if p.is_file()), key=lambda p: p.name.lower()):
            parsed = parse_age_filename(image_path.name, normalize_over_100=normalize_over_100)
            relative = image_path.relative_to(root).as_posix()
            if parsed is None:
                skipped.append({"relative_path": relative, "reason": "invalid_filename"})
                continue
            age, same_age_index, extension, raw_age, was_normalized = parsed
            if (min_age is not None and age < min_age) or (max_age is not None and age > max_age):
                filtered.append({"relative_path": relative, "raw_age": raw_age, "age": age})
                continue
            record = ImageRecord(
                person_id=identity_dir.name,
                relative_path=relative,
                filename=image_path.name,
                age=age,
                same_age_index=same_age_index,
                extension=extension,
                raw_age=raw_age,
                age_was_normalized=was_normalized,
                gender=genders.get(identity_dir.name),
            )
            rows.append(record)
            if was_normalized:
                normalized.append({"relative_path": relative, "raw_age": raw_age, "age": age})
    rows.sort(key=_image_sort_key)
    if cache is not None:
        _save_manifest(rows, cache)
    return rows, {
        "loaded_from_cache": False,
        "manifest_path": str(cache.resolve()) if cache else None,
        "skipped_files": skipped,
        "normalized_ages": normalized,
        "filtered_ages": filtered,
    }


def _split_counts(n: int, ratios: Sequence[float]) -> list[int]:
    positive = [i for i, ratio in enumerate(ratios) if ratio > 0]
    counts = [0, 0, 0]
    if n >= len(positive):
        for i in positive:
            counts[i] = 1
        remaining = n - len(positive)
        if remaining:
            scaled = [remaining * ratios[i] / sum(ratios) for i in range(3)]
            base = [int(v) for v in scaled]
            counts = [counts[i] + base[i] for i in range(3)]
            for i in sorted(range(3), key=lambda j: (scaled[j] - base[j], ratios[j]), reverse=True)[: remaining - sum(base)]:
                counts[i] += 1
    else:
        for i in sorted(positive, key=lambda j: ratios[j], reverse=True)[:n]:
            counts[i] = 1
    return counts


def build_identity_splits(
    manifest: Sequence[ImageRecord],
    *,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    split_path: str | Path | None = None,
    use_cached_splits: bool = True,
) -> dict[str, str]:
    """Assign whole identities to splits; optionally load/save the assignment CSV."""
    if len(split_ratios) != 3 or any(r < 0 for r in split_ratios) or sum(split_ratios) <= 0:
        raise ValueError("split_ratios must contain three non-negative values with a positive sum")
    split_file = Path(split_path).expanduser() if split_path else None
    known_ids = sorted({row.person_id for row in manifest})
    if split_file is not None and split_file.exists() and use_cached_splits:
        with split_file.open(newline="", encoding="utf-8") as handle:
            assignments = {row["person_id"]: row["split"] for row in csv.DictReader(handle)}
        invalid = set(assignments.values()) - set(_SPLITS)
        missing = set(known_ids) - set(assignments)
        if invalid or missing:
            raise ValueError(f"Invalid split file {split_file}; missing={sorted(missing)}, invalid={sorted(invalid)}")
        return {person_id: assignments[person_id] for person_id in known_ids}

    shuffled = known_ids[:]
    random.Random(seed).shuffle(shuffled)
    counts = _split_counts(len(shuffled), split_ratios)
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(_SPLITS, counts):
        for person_id in shuffled[offset : offset + count]:
            assignments[person_id] = split
        offset += count
    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["person_id", "split"])
            writer.writeheader()
            writer.writerows({"person_id": key, "split": assignments[key]} for key in sorted(assignments))
    return assignments


def build_pair_index(
    manifest: Sequence[ImageRecord],
    *,
    min_age_gap: int = 1,
    max_age_gap: int | None = None,
) -> list[PairRecord]:
    """Build all valid within-identity, image-level forward pairs."""
    if min_age_gap < 1:
        raise ValueError("min_age_gap must be >= 1 for forward-only aging")
    if max_age_gap is not None and max_age_gap < min_age_gap:
        raise ValueError("max_age_gap must be >= min_age_gap")
    pairs: list[PairRecord] = []
    by_identity: dict[str, list[tuple[int, ImageRecord]]] = {}
    for index, row in enumerate(manifest):
        by_identity.setdefault(row.person_id, []).append((index, row))
    for person_id in sorted(by_identity):
        images = sorted(by_identity[person_id], key=lambda item: _image_sort_key(item[1]))
        for source_index, source in images:
            for target_index, target in images:
                gap = target.age - source.age
                if gap < min_age_gap or (max_age_gap is not None and gap > max_age_gap):
                    continue
                pairs.append(PairRecord(
                    person_id=person_id,
                    source_index=source_index,
                    target_index=target_index,
                    source_age=source.age,
                    target_age=target.age,
                    delta_age=gap,
                    source_path=source.relative_path,
                    target_path=target.relative_path,
                ))
    pairs.sort(key=lambda p: (p.person_id, p.source_path, p.target_path))
    return pairs


def records_for_split(
    manifest: Iterable[ImageRecord], assignments: Mapping[str, str], split: str
) -> list[ImageRecord]:
    return [row for row in manifest if assignments[row.person_id] == split]
