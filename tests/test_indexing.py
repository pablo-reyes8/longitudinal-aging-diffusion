from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from data import (
    build_identity_splits,
    build_image_manifest,
    build_pair_index,
    parse_age_filename,
)
from conftest import write_image


def test_age_parser_valid_invalid_and_normalization():
    expected = {
        "52.jpg": 52,
        "38_1.jpg": 38,
        "7.PNG": 7,
        "38_999.jpeg": 38,
        "100.webp": 100,
        "501.jpg": 51,
    }
    for filename, age in expected.items():
        assert parse_age_filename(filename)[0] == age
    assert parse_age_filename("501.jpg")[3:] == (501, True)
    for filename in ("age_38.jpg", "38-old.jpg", "foo.jpg", "38_abc.jpg", ".jpg", "-1.jpg"):
        assert parse_age_filename(filename) is None


def test_manifest_audits_invalid_and_preserves_same_age_duplicates(tmp_path: Path):
    write_image(tmp_path / "id_a" / "38.jpg")
    write_image(tmp_path / "id_a" / "38_1.jpg")
    write_image(tmp_path / "id_a" / "501.jpg")
    write_image(tmp_path / "id_a" / "bad.jpg")
    manifest, audit = build_image_manifest(tmp_path)
    assert [(r.age, r.same_age_index) for r in manifest] == [(38, 0), (38, 1), (51, 0)]
    assert len(audit["skipped_files"]) == 1
    assert audit["normalized_ages"][0]["raw_age"] == 501


def test_exhaustive_pair_counts_and_gap_filters(tmp_path: Path):
    for filename in ("10.jpg", "20.jpg", "20_1.jpg", "30.jpg"):
        write_image(tmp_path / "id_a" / filename)
    manifest, _ = build_image_manifest(tmp_path)
    assert len(build_pair_index(manifest)) == 5
    assert len(build_pair_index(manifest, min_age_gap=15)) == 1
    assert len(build_pair_index(manifest, min_age_gap=5, max_age_gap=10)) == 4


def test_pair_index_matches_random_brute_force(tmp_path: Path):
    rng = random.Random(827)
    for identity in range(100):
        for image_index in range(rng.randint(1, 9)):
            age = rng.randint(0, 90)
            write_image(tmp_path / f"id_{identity:03d}" / f"{age}_{image_index}.png")
    manifest, _ = build_image_manifest(tmp_path)
    for min_gap, max_gap in ((1, None), (3, 30), (10, 15), (40, 80)):
        production = {(p.person_id, p.source_path, p.target_path) for p in build_pair_index(manifest, min_age_gap=min_gap, max_age_gap=max_gap)}
        oracle = set()
        for source in manifest:
            for target in manifest:
                gap = target.age - source.age
                if source.person_id == target.person_id and gap >= min_gap and (max_gap is None or gap <= max_gap):
                    oracle.add((source.person_id, source.relative_path, target.relative_path))
        assert production == oracle


def test_identity_split_determinism_leakage_and_changed_seed(tiny_root: Path):
    manifest, _ = build_image_manifest(tiny_root)
    first = build_identity_splits(manifest, seed=42)
    second = build_identity_splits(manifest, seed=42)
    changed = build_identity_splits(manifest, seed=43)
    assert first == second
    assert first != changed
    sets = {split: {pid for pid, assigned in first.items() if assigned == split} for split in ("train", "val", "test")}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_manifest_and_split_reload_and_root_relocation(tiny_root: Path, tmp_path_factory):
    manifest_csv = tiny_root.parent / "manifest.csv"
    split_csv = tiny_root.parent / "splits.csv"
    original, _ = build_image_manifest(tiny_root, manifest_path=manifest_csv)
    splits = build_identity_splits(original, split_path=split_csv)
    relocated = tmp_path_factory.mktemp("relocated") / "dataset"
    shutil.copytree(tiny_root, relocated)
    reloaded, audit = build_image_manifest(relocated, manifest_path=manifest_csv)
    reloaded_splits = build_identity_splits(reloaded, split_path=split_csv)
    assert original == reloaded
    assert splits == reloaded_splits
    assert audit["loaded_from_cache"]
    assert all(row.path(relocated).exists() for row in reloaded)


def test_filesystem_creation_order_does_not_change_manifest(tmp_path: Path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    files = [("id_b", "30.png"), ("id_a", "20_1.png"), ("id_a", "10.png"), ("id_a", "20.png")]
    for identity, filename in files:
        write_image(root_a / identity / filename)
    for identity, filename in reversed(files):
        write_image(root_b / identity / filename)
    rows_a, _ = build_image_manifest(root_a)
    rows_b, _ = build_image_manifest(root_b)
    signature = lambda rows: [(r.person_id, r.filename, r.age, r.same_age_index) for r in rows]
    assert signature(rows_a) == signature(rows_b)
