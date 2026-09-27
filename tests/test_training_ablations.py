from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data import build_face_aging_dataloaders
from src.ablation_studies import (
    AblationEpochEvaluator,
    build_fixed_evaluation_panel,
    canonical_training_ablation_config,
    persist_fixed_evaluation_panel,
    resolve_training_ablation_config,
)


def test_training_ablation_cases_change_only_the_declared_mechanism():
    baseline = canonical_training_ablation_config()
    case1 = resolve_training_ablation_config(1)
    case2 = resolve_training_ablation_config(2)
    case3 = resolve_training_ablation_config(3)
    case4 = resolve_training_ablation_config(4)
    case5 = resolve_training_ablation_config(5)

    assert case1["model"]["use_age_delta_conditioning"] is False
    assert case2["model"]["adapter_type"] == "none"
    assert case3["loss"]["use_relative_age_loss"] is False
    assert case3["loss"]["relative_age_weight"] == 0.0
    assert case4["data"]["include_zero_delta_pairs"] is False
    assert case4["data"]["zero_delta_pair_prob"] == 0.0
    assert case4["loss"]["use_preservation_loss"] is False
    assert case4["loss"]["preservation_weight"] == 0.0
    assert case4["loss"]["use_small_delta_weighting"] is True
    assert case5["loss"]["identity_weight"] == 0.0

    case1["training"]["num_epochs"] = 2
    assert baseline["training"]["num_epochs"] == 14
    assert canonical_training_ablation_config()["training"]["num_epochs"] == 14


def test_training_ablation_resolution_applies_high_level_overrides_and_rejects_inference_cases():
    resolved = resolve_training_ablation_config(
        3,
        {
            "training": {"num_epochs": 2, "lr_lora": 9e-6},
            "monitoring": {"strength": 0.22},
        },
    )
    assert resolved["training"]["num_epochs"] == 2
    assert resolved["training"]["lr_lora"] == 9e-6
    assert resolved["monitoring"]["strength"] == 0.22
    assert resolved["loss"]["use_relative_age_loss"] is False
    with pytest.raises(NotImplementedError, match="inference ablations"):
        resolve_training_ablation_config(101)

    from src.ablation_studies import ablation_studies
    with pytest.raises(NotImplementedError, match="inference ablations"):
        ablation_studies(case=101, dataset_root="unused", metrics_config=None)


def test_fixed_panel_is_deterministic_and_rejects_cross_case_drift(tiny_root, tmp_path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_shuffle=False,
    )
    first = build_fixed_evaluation_panel(loaders["val"], size=2)
    second = build_fixed_evaluation_panel(loaders["val"], size=2)
    assert first == second
    assert len(first) == 2
    assert all(row["identity_id"] and row["source_path"] and row["target_path"] for row in first)

    panel_path = tmp_path / "fixed_validation_panel.json"
    assert persist_fixed_evaluation_panel(first, panel_path) == panel_path
    assert json.loads(panel_path.read_text(encoding="utf-8")) == first
    persist_fixed_evaluation_panel(second, panel_path)
    changed = [dict(row) for row in second]
    changed[0]["target_age"] += 1
    with pytest.raises(ValueError, match="fixed validation panel"):
        persist_fixed_evaluation_panel(changed, panel_path)


def test_explicit_panel_requires_same_person_identity_label():
    pairs = [{
        "sample_id": "p1_20_40",
        "identity_id": "person_1",
        "source_path": "https://example.test/person1_age20.jpg",
        "target_path": "https://example.test/person1_age40.jpg",
        "source_age": 20,
        "target_age": 40,
    }]
    panel = build_fixed_evaluation_panel(None, explicit_pairs=pairs)
    assert panel[0]["delta_age"] == 20

    invalid = [dict(pairs[0], identity_id="")]
    with pytest.raises(ValueError, match="identity_id"):
        build_fixed_evaluation_panel(None, explicit_pairs=invalid)


def test_resume_rejects_resolved_configuration_drift_before_overwriting(tmp_path):
    from src.ablation_studies.runner import _prepare_resolved_config

    path = tmp_path / "resolved_config.json"
    _prepare_resolved_config(path, {"training": {"num_epochs": 14}}, resume=False, overwrite=False)
    with pytest.raises(ValueError, match="configuration differs"):
        _prepare_resolved_config(
            path, {"training": {"num_epochs": 2}}, resume=True, overwrite=False
        )
    assert json.loads(path.read_text())["training"]["num_epochs"] == 14


def test_overwrite_archives_previous_case_instead_of_mixing_artifacts(tmp_path):
    from src.ablation_studies.runner import _archive_existing_case

    case_dir = tmp_path / "case_00_full"
    case_dir.mkdir()
    (case_dir / "epoch_metrics.csv").write_text("old", encoding="utf-8")
    archived = _archive_existing_case(case_dir)
    assert not case_dir.exists()
    assert archived is not None and (archived / "epoch_metrics.csv").read_text() == "old"


class _IdentityMetric:
    def embed_batch(self, images):
        return np.stack([np.asarray(image).mean((0, 1)) for image in images])


class _AgeMetric:
    def predict_batch(self, images):
        return np.asarray([np.asarray(image).mean() / 4 for image in images])


class _KidMetric:
    def compute(self, real, generated, *, subsets, subset_size, seed):
        return 0.1, 0.01


def test_epoch_evaluator_generates_fixed_panel_and_appends_metrics(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (400, 400), (20, 30, 40)).save(source)
    Image.new("RGB", (400, 400), (80, 90, 100)).save(target)
    panel = build_fixed_evaluation_panel(None, explicit_pairs=[{
        "sample_id": "person_1_20_40",
        "identity_id": "person_1",
        "source_path": source,
        "target_path": target,
        "source_age": 20,
        "target_age": 40,
    }])
    calls = []

    def infer_backend(**kwargs):
        calls.append(kwargs)
        return {"image": Image.new("RGB", (24, 24), (70, 80, 90))}

    evaluator = AblationEpochEvaluator(
        panel=panel,
        metrics_bundle={
            "aligner": lambda image: image.resize((16, 16)),
            "identity_encoder": _IdentityMetric(),
            "age_estimator": _AgeMetric(),
            "kid_metric": _KidMetric(),
            "metadata": {},
        },
        output_dir=tmp_path / "case_03",
        model_name="case_03_no_relative_age",
        ablation_case=3,
        inference_config={"strength": 0.25, "num_inference_steps": 4, "seed": 2026},
        evaluation_config={"compute_kid": True},
        infer_backend=infer_backend,
    )
    report = evaluator.on_epoch(bundle={"name": "fake"}, epoch=0)

    assert isinstance(calls[0]["image"], Image.Image)
    assert calls[0]["source_age"] == 20 and calls[0]["target_age"] == 40
    assert calls[0]["strength"] == 0.25
    manifest = pd.read_csv(report["manifest_path"])
    assert manifest.loc[0, "ablation_case"] == 3
    assert manifest.loc[0, "epoch"] == 1
    assert manifest.loc[0, "identity_id"] == "person_1"
    epoch_metrics = pd.read_csv(tmp_path / "case_03" / "epoch_metrics.csv")
    assert epoch_metrics["epoch"].tolist() == [1]
    assert (tmp_path / "case_03" / "epoch_predictions" / "epoch_001" / "generated").is_dir()

    final = evaluator.evaluate_final(bundle={"name": "fake"})
    assert (tmp_path / "case_03" / "final_predictions.csv").is_file()
    assert (tmp_path / "case_03" / "final_metrics.json").is_file()
    assert final["n_samples"] == 1


def test_epoch_evaluator_resolves_remote_source_before_core_inference(tmp_path):
    target = tmp_path / "target.png"
    Image.new("RGB", (24, 24), "gray").save(target)
    seen = []

    evaluator = AblationEpochEvaluator(
        panel=[{
            "sample_id": "remote", "identity_id": "same",
            "source_path": "https://example.test/source.jpg",
            "target_path": str(target), "source_age": 20,
            "target_age": 40, "delta_age": 20,
        }],
        metrics_bundle={}, output_dir=tmp_path, model_name="test",
        ablation_case=0, inference_config={"image_size": 400},
        source_loader=lambda value, **kwargs: Image.new("RGB", (400, 400), "gray"),
        infer_backend=lambda **kwargs: seen.append(kwargs) or Image.new("RGB", (16, 16)),
    )
    evaluator.evaluation_config = {"compute_id_source": False, "compute_id_gt": False,
                                   "compute_age_mae": False, "compute_kid": False}
    evaluator.on_epoch(bundle={}, epoch=0)
    assert isinstance(seen[0]["image"], Image.Image)


def test_high_level_runner_executes_cases_sequentially_and_applies_case_changes(
    monkeypatch, tmp_path
):
    import src.ablation_studies.runner as runner

    events = []
    fake_loader = type("Loader", (), {"dataset": object()})()
    monkeypatch.setattr(
        runner,
        "build_face_aging_dataloaders",
        lambda root, **config: ({"train": fake_loader, "val": fake_loader}, {"root": str(root)}),
    )
    monkeypatch.setattr(
        runner,
        "build_fixed_evaluation_panel",
        lambda *args, **kwargs: [{
            "sample_id": "same_20_40", "identity_id": "same",
            "source_path": "source.jpg", "target_path": "target.jpg",
            "source_age": 20, "target_age": 40, "delta_age": 20,
        }],
    )
    monkeypatch.setattr(runner, "load_quantitative_metrics", lambda **kwargs: {"metadata": {}})
    monkeypatch.setattr(
        runner, "set_seed", lambda seed, deterministic=False: events.append(("seed", seed, deterministic))
    )

    def build_bundle(**config):
        events.append(("bundle", config["adapter_type"], config["use_age_delta_conditioning"]))
        return {
            "scheduler_train": object(), "vae": object(),
            "identity_encoder": object(), "age_estimator": object(),
        }

    monkeypatch.setattr(runner, "build_face_aging_diffusion_bundle", build_bundle)
    monkeypatch.setattr(runner, "FaceAgingDiffusionLoss", lambda **config: {"loss": config})

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.case = kwargs["ablation_case"]

        def on_epoch(self, **kwargs):
            return {"case": self.case}

        def evaluate_final(self, **kwargs):
            events.append(("final", self.case))
            return {"n_samples": 1, "summary": [{"dex_age_mae": 2.0}]}

    monkeypatch.setattr(runner, "AblationEpochEvaluator", FakeEvaluator)

    def train(**kwargs):
        events.append(("train", kwargs["use_age_delta_conditioning"], kwargs["checkpoint_dir"]))
        kwargs["epoch_end_callback"](bundle=kwargs["bundle"], epoch=0)
        return {"best_epoch": 0, "best_metric": 0.5, "history": {}}

    monkeypatch.setattr(runner, "TRAIN_AGGING_MODEL", train)

    reports = runner.ablation_studies(
        case=[1, 2],
        dataset_root=tmp_path / "data",
        kaggle_path=tmp_path / "fgnet",
        monitoring_image=tmp_path / "monitor.jpg",
        monitoring_source_age=26,
        evaluation_source_image="https://example.test/same20.jpg",
        evaluation_target_image="https://example.test/same40.jpg",
        evaluation_source_age=20,
        evaluation_target_age=40,
        evaluation_identity_id="same_person",
        metrics_config={"metrics_bundle": {"metadata": {}}},
        num_epochs=2,
        lr_lora=9e-6,
        output_root=tmp_path / "ablations",
        device="cpu",
        dtype="float32",
    )

    assert list(reports) == [1, 2]
    assert events[0] == ("seed", 42, False)
    assert events[1][:3] == ("bundle", "dora", False)
    assert events[4] == ("seed", 42, False)
    assert events[5][:3] == ("bundle", "none", True)
    for case, slug in ((1, "no_numeric_age"), (2, "no_dora")):
        case_dir = tmp_path / "ablations" / f"case_{case:02d}_{slug}"
        resolved = json.loads((case_dir / "resolved_config.json").read_text())
        assert resolved["training"]["num_epochs"] == 2
        assert resolved["training"]["lr_lora"] == 9e-6
        assert (case_dir / "run_metadata.json").is_file()
