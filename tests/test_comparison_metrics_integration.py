from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def test_comparison_loader_adds_local_metrics_bundle(monkeypatch, tmp_path):
    from src.comparisong import load_models

    expected = {"identity_encoder": object()}
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("src.quantitative_metrics.load_quantitative_metrics", fake_loader)
    bundle = load_models.load_aging_models(
        root=tmp_path,
        models=(),
        load_metrics=True,
        metrics_config={
            "adaface_repo_path": "/models/AdaFace",
            "adaface_checkpoint_path": "/models/adaface.ckpt",
            "dex_prototxt_path": "/models/age.prototxt",
            "dex_checkpoint_path": "/models/dex.caffemodel",
        },
    )

    assert bundle["quantitative_metrics"] is expected
    assert captured["adaface_repo_path"] == "/models/AdaFace"
    assert captured["local_files_only"] is True


def test_build_comparison_manifest_maps_same_person_targets_by_age(tmp_path):
    from src.comparisong.generate_images import build_comparison_manifest

    output = tmp_path / "outputs"
    (output / "SAM").mkdir(parents=True)
    generated_40 = output / "SAM" / "age_040.png"
    generated_60 = output / "SAM" / "age_060.png"
    target_40 = tmp_path / "same_person_40.png"
    for path in (generated_40, generated_60, target_40):
        Image.new("RGB", (8, 8), "gray").save(path)

    frame = build_comparison_manifest(
        results={"SAM": {40: Image.new("RGB", (8, 8)), 60: Image.new("RGB", (8, 8))}},
        source_path=output / "source.png",
        source_age=20,
        target_ages=[40, 60],
        models=["SAM"],
        output_dir=output,
        target_images={40: target_40},
    )

    assert frame["sample_id"].tolist() == ["age-040", "age-060"]
    assert frame["target_path"].tolist() == [str(target_40), None]
    assert frame["generated_path"].tolist() == [str(generated_40), str(generated_60)]


def test_age_image_requires_loaded_metric_bundle_before_inference(tmp_path):
    from src.comparisong import age_image

    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "gray").save(source)
    with pytest.raises(ValueError, match="load_metrics=True"):
        age_image(
            bundle={"device": "cpu", "baseline_models": []},
            image=source,
            source_age=20,
            target_ages=[40],
            models=[],
            show=False,
            output_dir=tmp_path / "out",
            metrics=True,
        )


def test_comparison_cli_parses_metrics_and_target_images():
    from scripts.compare_models import build_parser, parse_target_images

    args = build_parser().parse_args(
        [
            "--image", "face.jpg", "--source-age", "20", "--target-ages", "40", "60",
            "--output-dir", "outputs", "--metrics",
            "--target-image", "40=person_40.jpg", "--target-image", "60=https://host/person_60.jpg",
            "--adaface-repo-path", "/models/AdaFace",
            "--adaface-checkpoint-path", "/models/adaface.ckpt",
            "--dex-prototxt-path", "/models/age.prototxt",
            "--dex-checkpoint-path", "/models/dex.caffemodel",
            "--kid-inception-weights-path", "/models/weights-inception-2015-12-05-6726825d.pth",
        ]
    )

    assert args.metrics is True
    assert parse_target_images(args.target_image) == {
        40: "person_40.jpg", 60: "https://host/person_60.jpg"
    }


def test_target_image_parser_rejects_invalid_assignment():
    from scripts.compare_models import parse_target_images

    with pytest.raises(ValueError, match="AGE=PATH"):
        parse_target_images(["person_40.jpg"])
