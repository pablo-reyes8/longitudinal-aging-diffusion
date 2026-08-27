"""FG-NET parsing and scarcity-aware complementary pair selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import heapq
from pathlib import Path
import re
from typing import Sequence

from .indexing import ImageRecord, PairRecord, build_pair_index


_FGNET_RE = re.compile(
    r"^(?P<person>\d{3})A(?P<age>\d{2})(?P<duplicate>[a-z]?)\.(?P<ext>jpe?g|png|webp)$",
    re.IGNORECASE,
)
_AGE_BANDS = ((0, 4), (5, 9), (10, 17), (18, 29), (30, 44), (45, 59), (60, None))
_GAP_BANDS = ((1, 4), (5, 9), (10, 19), (20, 29), (30, 39), (40, None))


def parse_fgnet_filename(filename: str) -> tuple[str, int, int, str] | None:
    """Parse names such as ``001A43a.JPG`` into identity, age and duplicate."""
    match = _FGNET_RE.fullmatch(Path(filename).name)
    if match is None:
        return None
    duplicate = match.group("duplicate").lower()
    duplicate_index = 0 if not duplicate else ord(duplicate) - ord("a") + 1
    return (
        f"fgnet_{match.group('person')}",
        int(match.group("age")),
        duplicate_index,
        f".{match.group('ext').lower()}",
    )


def _resolve_images_root(root_dir: str | Path) -> Path:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FG-NET path does not exist or is not a directory: {root}")
    if any(parse_fgnet_filename(path.name) for path in root.iterdir() if path.is_file()):
        return root
    for candidate in (root / "images", root / "FGNET" / "images"):
        if candidate.is_dir() and any(
            parse_fgnet_filename(path.name) for path in candidate.iterdir() if path.is_file()
        ):
            return candidate.resolve()
    raise ValueError(
        "FG-NET path contains no images named like 001A02.JPG; pass the flat images folder"
    )


def build_fgnet_manifest(
    root_dir: str | Path,
    *,
    min_age: int | None = None,
    max_age: int | None = None,
) -> tuple[Path, list[ImageRecord], dict]:
    """Scan a flat FG-NET image folder and return project-compatible records."""
    root = _resolve_images_root(root_dir)
    rows: list[ImageRecord] = []
    skipped = []
    filtered = []
    for path in sorted((item for item in root.iterdir() if item.is_file()), key=lambda p: p.name.lower()):
        parsed = parse_fgnet_filename(path.name)
        if parsed is None:
            skipped.append({"filename": path.name, "reason": "invalid_fgnet_filename"})
            continue
        person_id, age, duplicate_index, extension = parsed
        if (min_age is not None and age < min_age) or (max_age is not None and age > max_age):
            filtered.append({"filename": path.name, "age": age})
            continue
        rows.append(ImageRecord(
            person_id=person_id,
            relative_path=path.name,
            filename=path.name,
            age=age,
            same_age_index=duplicate_index,
            extension=extension,
            raw_age=age,
        ))
    rows.sort(key=lambda row: (row.person_id, row.age, row.same_age_index, row.filename.lower()))
    if not rows:
        raise ValueError(f"No valid FG-NET images found in {root}")
    return root, rows, {
        "root_dir": str(root),
        "images": len(rows),
        "identities": len({row.person_id for row in rows}),
        "skipped_files": skipped,
        "filtered_ages": filtered,
    }


def _band(value: int, bands) -> str:
    for low, high in bands:
        if value >= low and (high is None or value <= high):
            return f"{low}+" if high is None else f"{low}-{high}"
    return "outside"


def transition_cell(pair: PairRecord) -> tuple[str, str, str]:
    """Coarse semantic cell used to compare longitudinal coverage."""
    younger, older = sorted((pair.source_age, pair.target_age))
    return _band(younger, _AGE_BANDS), _band(older, _AGE_BANDS), _band(older - younger, _GAP_BANDS)


def _stable_rank(seed: int, *values: object) -> int:
    payload = "|".join(map(str, (seed, *values))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _identity_balanced_group(pairs: Sequence[PairRecord], seed: int) -> list[PairRecord]:
    by_person: dict[str, list[PairRecord]] = defaultdict(list)
    for pair in pairs:
        by_person[pair.person_id].append(pair)
    for person_id, values in by_person.items():
        values.sort(key=lambda pair: _stable_rank(seed, person_id, pair.source_path, pair.target_path))
    people = sorted(by_person, key=lambda person_id: _stable_rank(seed, "person", person_id))
    ordered = []
    while people:
        next_people = []
        for person_id in people:
            ordered.append(by_person[person_id].pop())
            if by_person[person_id]:
                next_people.append(person_id)
        people = next_people
    return ordered


def select_complementary_fgnet_pairs(
    primary_pairs: Sequence[PairRecord],
    fgnet_pairs: Sequence[PairRecord],
    *,
    count: int,
    seed: int = 42,
) -> tuple[list[PairRecord], dict]:
    """Fill scarce transition cells first while balancing FG-NET identities."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(fgnet_pairs):
        selected = list(fgnet_pairs)
    elif count == 0:
        selected = []
    else:
        primary_profile = Counter(transition_cell(pair) for pair in primary_pairs)
        groups: dict[tuple[str, str, str], list[PairRecord]] = defaultdict(list)
        for pair in fgnet_pairs:
            groups[transition_cell(pair)].append(pair)
        ordered_groups = {
            cell: _identity_balanced_group(values, seed)
            for cell, values in groups.items()
        }
        offsets = {cell: 0 for cell in ordered_groups}
        selected_per_cell: Counter = Counter()
        heap = [
            (primary_profile[cell], _stable_rank(seed, "cell", *cell), cell)
            for cell in ordered_groups
        ]
        heapq.heapify(heap)
        selected = []
        while heap and len(selected) < count:
            _, tie_break, cell = heapq.heappop(heap)
            offset = offsets[cell]
            selected.append(ordered_groups[cell][offset])
            offsets[cell] += 1
            selected_per_cell[cell] += 1
            if offsets[cell] < len(ordered_groups[cell]):
                heapq.heappush(
                    heap,
                    (primary_profile[cell] + selected_per_cell[cell], tie_break, cell),
                )
    return selected, {
        "requested_pairs": int(count),
        "available_pairs": len(fgnet_pairs),
        "selected_pairs": len(selected),
        "selected_identities": len({pair.person_id for pair in selected}),
        "selected_transition_cells": {
            "|".join(cell): value
            for cell, value in sorted(Counter(transition_cell(pair) for pair in selected).items())
        },
    }


def summarize_fgnet_manifest(rows: Sequence[ImageRecord]) -> list[dict]:
    """Serializable rows useful in exploratory notebooks."""
    return [asdict(row) for row in rows]
