from __future__ import annotations

from pathlib import Path

import pytest


def _metrics_config(root: Path) -> dict:
    return {
        "adaface_repo_path": str(root / "AdaFace"),
        "adaface_checkpoint_path": str(root / "checkpoints" / "adaface.ckpt"),
        "dex_prototxt_path": str(root / "checkpoints" / "age.prototxt"),
        "dex_checkpoint_path": str(root / "checkpoints" / "dex.caffemodel"),
        "kid_inception_weights_path": str(root / "checkpoints" / "inception.pth"),
        "device": "cpu",
        "local_files_only": True,
        "evaluation": {"batch_size": 8, "kid_seed": 2026},
    }


def test_prepare_metrics_config_downloads_missing_files_and_reuses_them(monkeypatch, tmp_path):
    import src.quantitative_metrics.assets as assets

    sources = tmp_path / "sources"
    sources.mkdir()
    payloads = {
        "adaface_checkpoint_path": b"adaface-checkpoint",
        "dex_prototxt_path": b"name: 'dex-age'",
        "dex_checkpoint_path": b"dex-checkpoint",
        "kid_inception_weights_path": b"inception-weights",
    }
    specs = {}
    for key, payload in payloads.items():
        source = sources / key
        source.write_bytes(payload)
        specs[key] = assets.DownloadSpec(url=source.as_uri(), min_bytes=len(payload))
    monkeypatch.setattr(assets, "DOWNLOAD_SPECS", specs)

    repo = tmp_path / "runtime" / "AdaFace"
    (repo / "face_alignment").mkdir(parents=True)
    (repo / "net.py").write_text("# network", encoding="utf-8")
    (repo / "face_alignment" / "align.py").write_text("# aligner", encoding="utf-8")
    config = _metrics_config(tmp_path / "runtime")

    prepared = assets.prepare_metrics_config(config)

    assert prepared == config
    for key, payload in payloads.items():
        assert Path(prepared[key]).read_bytes() == payload

    monkeypatch.setattr(
        assets,
        "DOWNLOAD_SPECS",
        {
            key: assets.DownloadSpec(url="file:///does-not-exist", min_bytes=len(payload))
            for key, payload in payloads.items()
        },
    )
    assert assets.prepare_metrics_config(config) == config


def test_prepare_metrics_config_repairs_incomplete_files_atomically(monkeypatch, tmp_path):
    import src.quantitative_metrics.assets as assets

    config = _metrics_config(tmp_path / "runtime")
    repo = Path(config["adaface_repo_path"])
    (repo / "face_alignment").mkdir(parents=True)
    (repo / "net.py").write_text("# network", encoding="utf-8")
    (repo / "face_alignment" / "align.py").write_text("# aligner", encoding="utf-8")

    complete = b"complete-payload"
    source = tmp_path / "complete.bin"
    source.write_bytes(complete)
    specs = {}
    for key in (
        "adaface_checkpoint_path",
        "dex_prototxt_path",
        "dex_checkpoint_path",
        "kid_inception_weights_path",
    ):
        destination = Path(config[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad")
        specs[key] = assets.DownloadSpec(
            url=source.as_uri(), min_bytes=1, exact_bytes=len(complete)
        )
    monkeypatch.setattr(assets, "DOWNLOAD_SPECS", specs)

    assets.prepare_metrics_config(config)

    assert all(Path(config[key]).read_bytes() == complete for key in specs)
    assert not list((tmp_path / "runtime").rglob("*.part"))


def test_failed_download_attempts_use_distinct_temporary_files(monkeypatch, tmp_path):
    import src.quantitative_metrics.assets as assets

    destinations = []

    def fail_download(_url, destination):
        destinations.append(destination)
        destination.write_bytes(b"partial")
        raise RuntimeError("connection interrupted")

    monkeypatch.setattr(assets, "_download_http", fail_download)
    target = tmp_path / "asset.bin"
    spec = assets.DownloadSpec(url="https://example.test/asset", exact_bytes=20)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="connection interrupted"):
            assets._ensure_download(target, spec)

    assert destinations[0] != destinations[1]
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".part"]


def test_prepare_metrics_config_clones_missing_adaface_repository(monkeypatch, tmp_path):
    import src.quantitative_metrics.assets as assets

    config = _metrics_config(tmp_path / "runtime")
    for key in assets.DOWNLOAD_SPECS:
        destination = Path(config[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"already-present")
    monkeypatch.setattr(
        assets,
        "DOWNLOAD_SPECS",
        {
            key: assets.DownloadSpec(url=spec.url, min_bytes=1)
            for key, spec in assets.DOWNLOAD_SPECS.items()
        },
    )

    def clone(destination):
        (destination / "face_alignment").mkdir(parents=True)
        (destination / "net.py").write_text("# network", encoding="utf-8")
        (destination / "face_alignment" / "align.py").write_text("# aligner", encoding="utf-8")

    monkeypatch.setattr(assets, "_clone_adaface_repository", clone)

    prepared = assets.prepare_metrics_config(config)

    assert (Path(prepared["adaface_repo_path"]) / "net.py").is_file()
    assert prepared["device"] == "cpu"
    assert prepared["local_files_only"] is True
