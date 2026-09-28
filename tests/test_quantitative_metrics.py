from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


class MeanIdentity:
    def __init__(self):
        self.calls = 0

    def embed_batch(self, images):
        self.calls += len(images)
        vectors = []
        for image in images:
            rgb = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            vectors.append(rgb / np.linalg.norm(rgb))
        return np.stack(vectors)


class RecordingIdentity(MeanIdentity):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def embed_batch(self, images):
        self.batch_sizes.append(len(images))
        return super().embed_batch(images)


class RedChannelAge:
    def predict_batch(self, images):
        return np.asarray(
            [np.asarray(image, dtype=np.float32)[..., 0].mean() / 255.0 * 100.0 for image in images]
        )


class FixedKid:
    def compute(self, real_images, generated_images, **kwargs):
        assert len(real_images) == len(generated_images) == 2
        return 0.012, 0.003


class CountingKid:
    def __init__(self):
        self.sizes = []

    def compute(self, real_images, generated_images, **kwargs):
        self.sizes.append((len(real_images), len(generated_images)))
        return float(len(real_images)), 0.0


class InspectingKid:
    def __init__(self):
        self.real_pixels = []
        self.seeds = []

    def compute(self, real_images, generated_images, **kwargs):
        self.real_pixels.extend(tuple(np.asarray(image)[0, 0]) for image in real_images)
        self.seeds.append(kwargs.get("seed"))
        return 0.1, 0.01


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (16, 16), color).save(path)
    return path


def _bundle(identity=None, aligner=None, kid=None):
    return {
        "identity_encoder": identity or MeanIdentity(),
        "age_estimator": RedChannelAge(),
        "aligner": aligner or (lambda image: image),
        "kid_metric": kid or FixedKid(),
        "metadata": {"identity_model": "test-adaface", "age_model": "test-dex"},
    }


def test_evaluator_computes_exact_cosines_age_mae_kid_and_files(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (255, 0, 0))
    target_1 = _image(tmp_path / "target_1.png", (0, 255, 0))
    target_2 = _image(tmp_path / "target_2.png", (0, 0, 255))
    generated_1 = _image(tmp_path / "generated_1.png", (255, 255, 0))
    generated_2 = _image(tmp_path / "generated_2.png", (255, 0, 255))
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "p1-40",
                "source_path": source,
                "target_path": target_1,
                "generated_path": generated_1,
                "source_age": 20,
                "target_age": 40,
                "model_name": "MODEL_A",
            },
            {
                "sample_id": "p1-60",
                "source_path": source,
                "target_path": target_2,
                "generated_path": generated_2,
                "source_age": 20,
                "target_age": 60,
                "model_name": "MODEL_A",
            },
        ]
    )

    result = evaluate_aging_outputs(
        predictions,
        metrics_bundle=_bundle(),
        output_dir=tmp_path / "evaluation",
        kid_subsets=7,
        kid_subset_size=2,
    )

    per_sample = result["per_sample"]
    summary = result["summary"].iloc[0]
    assert per_sample["adaface_cosine_source"].tolist() == pytest.approx([2**-0.5, 2**-0.5])
    assert per_sample["adaface_cosine_gt"].tolist() == pytest.approx([2**-0.5, 2**-0.5])
    assert per_sample["dex_predicted_age"].tolist() == pytest.approx([100.0, 100.0])
    assert summary["dex_age_mae"] == pytest.approx(50.0)
    assert summary["kid_mean"] == pytest.approx(0.012)
    assert summary["kid_std"] == pytest.approx(0.003)
    assert summary["n_samples"] == 2
    assert summary["n_valid_identity"] == 2
    assert summary["n_valid_age"] == 2

    for name in (
        "prediction_manifest.csv",
        "per_sample_metrics.csv",
        "metrics_summary.csv",
        "metrics_summary.json",
        "failure_log.csv",
        "evaluation_metadata.json",
    ):
        assert (tmp_path / "evaluation" / name).is_file()
    metadata = json.loads((tmp_path / "evaluation" / "evaluation_metadata.json").read_text())
    assert metadata["models"]["identity_model"] == "test-adaface"
    assert metadata["kid"]["subset_size"] == 2


def test_partial_ground_truth_keeps_rows_and_marks_kid_unavailable(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (50, 60, 70))
    generated_1 = _image(tmp_path / "generated_1.png", (70, 60, 50))
    generated_2 = _image(tmp_path / "generated_2.png", (80, 60, 40))
    target = _image(tmp_path / "target.png", (60, 70, 50))
    predictions = pd.DataFrame(
        [
            dict(sample_id="a", source_path=source, target_path=target, generated_path=generated_1,
                 source_age=20, target_age=30, model_name="M"),
            dict(sample_id="b", source_path=source, target_path=None, generated_path=generated_2,
                 source_age=20, target_age=40, model_name="M"),
        ]
    )

    result = evaluate_aging_outputs(predictions, metrics_bundle=_bundle(), output_dir=tmp_path / "eval")

    assert len(result["per_sample"]) == 2
    assert result["per_sample"]["adaface_cosine_gt"].notna().sum() == 1
    summary = result["summary"].iloc[0]
    assert summary["n_valid_identity_gt"] == 1
    assert np.isnan(summary["kid_mean"])
    assert summary["kid_status"] == "insufficient_samples"


def test_alignment_failure_is_recorded_not_silently_removed(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (20, 20, 20))
    generated = _image(tmp_path / "generated.png", (30, 30, 30))

    def failing_aligner(image):
        if np.asarray(image).mean() > 25:
            return None
        return image

    predictions = pd.DataFrame(
        [dict(sample_id="failed", source_path=source, target_path=None, generated_path=generated,
              source_age=20, target_age=40, model_name="M")]
    )
    result = evaluate_aging_outputs(
        predictions, metrics_bundle=_bundle(aligner=failing_aligner), output_dir=tmp_path / "eval"
    )

    row = result["per_sample"].iloc[0]
    assert not bool(row["identity_face_detected"])
    assert not bool(row["age_face_detected"])
    assert np.isnan(row["adaface_cosine_source"])
    failures = pd.read_csv(tmp_path / "eval" / "failure_log.csv")
    assert failures["sample_id"].tolist() == ["failed"]
    assert failures["stage"].tolist() == ["generated_alignment"]


def test_missing_generated_output_is_recorded_as_baseline_failure(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (20, 20, 20))
    predictions = pd.DataFrame(
        [dict(sample_id="missing", source_path=source, target_path=None, generated_path=None,
              source_age=20, target_age=40, model_name="FAILED_BASELINE")]
    )

    result = evaluate_aging_outputs(
        predictions, metrics_bundle=_bundle(), output_dir=tmp_path / "eval"
    )

    assert len(result["per_sample"]) == 1
    assert result["failures"].iloc[0]["stage"] == "generated_missing"


def test_source_embedding_cache_uses_image_content_hash(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    identity = MeanIdentity()
    source = _image(tmp_path / "source.png", (90, 80, 70))
    generated_1 = _image(tmp_path / "generated_1.png", (70, 80, 90))
    generated_2 = _image(tmp_path / "generated_2.png", (60, 80, 100))
    rows = [
        dict(sample_id="a", source_path=source, target_path=None, generated_path=generated_1,
             source_age=20, target_age=30, model_name="M"),
        dict(sample_id="b", source_path=source, target_path=None, generated_path=generated_2,
             source_age=20, target_age=40, model_name="M"),
    ]

    evaluate_aging_outputs(pd.DataFrame(rows), metrics_bundle=_bundle(identity=identity), output_dir=tmp_path / "e")

    assert identity.calls == 3  # one source plus two generated images


def test_identity_backend_is_evaluated_in_configured_batches(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    identity = RecordingIdentity()
    source = _image(tmp_path / "source.png", (90, 80, 70))
    rows = []
    for index in range(5):
        generated = _image(tmp_path / f"generated_{index}.png", (20 + index, 80, 100))
        rows.append(
            dict(sample_id=str(index), source_path=source, target_path=None,
                 generated_path=generated, source_age=20, target_age=40, model_name="M")
        )

    evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=_bundle(identity=identity),
        output_dir=tmp_path / "eval", batch_size=2
    )

    assert identity.batch_sizes == [2, 2, 2]


def test_manifest_rejects_missing_required_columns(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    with pytest.raises(ValueError, match="generated_path"):
        evaluate_aging_outputs(
            pd.DataFrame([{"sample_id": "x"}]), metrics_bundle=_bundle(), output_dir=tmp_path
        )


def test_dex_reports_opencv_compatibility_error_when_caffe_reader_is_missing(
    monkeypatch, tmp_path
):
    from src.quantitative_metrics.backends import DexAgeEstimator

    prototxt = tmp_path / "age.prototxt"
    checkpoint = tmp_path / "age.caffemodel"
    prototxt.write_text("name: 'age'", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    fake_cv2 = SimpleNamespace(__version__="5.0.0", dnn=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    with pytest.raises(RuntimeError, match="OpenCV 4.*readNetFromCaffe"):
        DexAgeEstimator(prototxt, checkpoint, device="cpu")


def test_disabled_kid_does_not_require_kid_backend(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (30, 40, 50))
    generated = _image(tmp_path / "generated.png", (40, 50, 60))
    bundle = _bundle()
    del bundle["kid_metric"]

    result = evaluate_aging_outputs(
        pd.DataFrame([dict(sample_id="a", source_path=source, target_path=None,
                           generated_path=generated, source_age=20, target_age=30,
                           model_name="M")]),
        metrics_bundle=bundle,
        output_dir=tmp_path / "eval",
        compute_kid=False,
    )

    assert result["summary"].iloc[0]["kid_status"] == "disabled"


def test_kid_uses_original_rgb_images_and_receives_deterministic_seed(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    kid = InspectingKid()
    source = _image(tmp_path / "source.png", (10, 10, 10))
    rows = []
    for index, color in enumerate(((1, 2, 3), (4, 5, 6))):
        target = _image(tmp_path / f"target_{index}.png", color)
        generated = _image(tmp_path / f"generated_{index}.png", (20, 20, 20))
        rows.append(dict(sample_id=str(index), source_path=source, target_path=target,
                         generated_path=generated, source_age=20, target_age=40, model_name="M"))

    def destructive_aligner(_image):
        return Image.new("RGB", (112, 112), (200, 200, 200))

    bundle = _bundle(aligner=destructive_aligner, kid=kid)
    evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=bundle, output_dir=tmp_path / "eval", kid_seed=123
    )

    assert kid.real_pixels == [(1, 2, 3), (4, 5, 6)]
    assert kid.seeds == [123]


def test_kid_rejects_different_target_sample_sets_between_models(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (10, 10, 10))
    rows = []
    for model, sample_ids in (("A", ("one", "two")), ("B", ("one", "three"))):
        for sample_id in sample_ids:
            target = _image(tmp_path / f"target_{sample_id}.png", (20, 30, 40))
            generated = _image(tmp_path / f"generated_{model}_{sample_id}.png", (30, 40, 50))
            rows.append(dict(sample_id=sample_id, source_path=source, target_path=target,
                             generated_path=generated, source_age=20, target_age=40, model_name=model))

    result = evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=_bundle(), output_dir=tmp_path / "eval"
    )

    assert result["summary"]["kid_status"].tolist() == ["incomparable_target_set"] * 2
    assert result["summary"]["kid_mean"].isna().all()


def test_identity_backend_failure_isolated_and_recorded_per_sample(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    class FragileIdentity(MeanIdentity):
        def embed_batch(self, images):
            if any(np.asarray(image)[0, 0, 0] == 99 for image in images):
                raise RuntimeError("synthetic backend failure")
            return super().embed_batch(images)

    source = _image(tmp_path / "source.png", (10, 20, 30))
    good = _image(tmp_path / "good.png", (20, 30, 40))
    bad = _image(tmp_path / "bad.png", (99, 30, 40))
    rows = [
        dict(sample_id="good", source_path=source, target_path=None, generated_path=good,
             source_age=20, target_age=30, model_name="M"),
        dict(sample_id="bad", source_path=source, target_path=None, generated_path=bad,
             source_age=20, target_age=40, model_name="M"),
    ]

    result = evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=_bundle(identity=FragileIdentity()),
        output_dir=tmp_path / "eval", batch_size=8
    )

    assert not np.isnan(
        result["per_sample"].set_index("sample_id").loc["good", "adaface_cosine_source"]
    )
    assert np.isnan(result["per_sample"].set_index("sample_id").loc["bad", "adaface_cosine_source"])
    assert "identity_backend" in result["failures"]["stage"].tolist()


def test_torchmetrics_kid_refuses_missing_local_inception_weights(tmp_path):
    from src.quantitative_metrics.backends import TorchMetricsKid

    metric = TorchMetricsKid(device="cpu", inception_weights_path=tmp_path / "missing.pth")
    images = [Image.new("RGB", (16, 16), "gray") for _ in range(2)]
    with pytest.raises(FileNotFoundError, match="Inception weights"):
        metric.compute(images, images, subsets=1, subset_size=2, seed=1)


def test_dex_expected_age_uses_softmax_probabilities():
    from src.quantitative_metrics.backends import expected_dex_age

    logits = np.full((1, 101), -100.0, dtype=np.float32)
    logits[0, 20] = 0.0
    logits[0, 60] = 0.0
    assert expected_dex_age(logits)[0] == pytest.approx(40.0, abs=1e-5)

    spatial_logits = logits[:, :, None, None]
    assert expected_dex_age(spatial_logits)[0] == pytest.approx(40.0, abs=1e-5)

    probabilities = np.zeros((1, 101), dtype=np.float32)
    probabilities[0, 35] = 0.25
    probabilities[0, 45] = 0.75
    assert expected_dex_age(probabilities)[0] == pytest.approx(42.5)


def test_adaface_preprocessing_is_bgr_and_normalized():
    from src.quantitative_metrics.backends import adaface_input_tensor

    image = Image.new("RGB", (112, 112), (255, 128, 0))
    tensor = adaface_input_tensor(image)
    assert tuple(tensor.shape) == (3, 112, 112)
    assert tensor[:, 0, 0].tolist() == pytest.approx(
        [-1.0, 128 / 127.5 - 1.0, 1.0], abs=1e-6
    )


def test_adaface_lightning_state_dict_keeps_only_backbone_parameters():
    from src.quantitative_metrics.backends import clean_adaface_state_dict

    state = {
        "model.layer.weight": "backbone",
        "head.kernel": "classifier",
    }
    assert clean_adaface_state_dict(state) == {"layer.weight": "backbone"}


def test_auto_device_resolves_to_cpu_without_cuda(monkeypatch):
    import torch

    from src.quantitative_metrics.backends import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_torch_device("auto")) == "cpu"


def test_adaface_aligner_loads_mtcnn_directly_without_training_modules(tmp_path):
    from src.quantitative_metrics.backends import AdaFaceAligner

    face_alignment = tmp_path / "AdaFace" / "face_alignment"
    face_alignment.mkdir(parents=True)
    (face_alignment / "mtcnn.py").write_text(
        """
from PIL import Image

class MTCNN:
    def __init__(self, device, crop_size):
        self.device = device
        self.crop_size = crop_size

    def align_multi(self, image, limit=None):
        return [object()], [image.resize(self.crop_size)]
""",
        encoding="utf-8",
    )

    aligner = AdaFaceAligner(tmp_path / "AdaFace", device="cpu")
    aligned = aligner(Image.new("RGB", (20, 20), "white"))

    assert aligned.size == (112, 112)


def test_kid_is_computed_per_ablation_group_not_pooled(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    kid = CountingKid()
    source = _image(tmp_path / "source.png", (50, 60, 70))
    rows = []
    for case in ("full", "no_identity"):
        for index in range(2):
            target = _image(tmp_path / f"target_{case}_{index}.png", (60 + index, 70, 50))
            generated = _image(tmp_path / f"generated_{case}_{index}.png", (70 + index, 60, 50))
            rows.append(
                dict(sample_id=str(index), source_path=source, target_path=target,
                     generated_path=generated, source_age=20, target_age=40,
                     model_name="OURS", ablation_case=case)
            )

    result = evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=_bundle(kid=kid), output_dir=tmp_path / "eval"
    )

    assert kid.sizes == [(2, 2), (2, 2)]
    assert result["summary"]["kid_mean"].tolist() == [2.0, 2.0]


def test_kid_is_not_computed_on_cherry_picked_subset_after_generation_failure(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs

    source = _image(tmp_path / "source.png", (50, 60, 70))
    rows = []
    for index in range(3):
        target = _image(tmp_path / f"target_{index}.png", (60 + index, 70, 50))
        generated = None
        if index < 2:
            generated = _image(tmp_path / f"generated_{index}.png", (70 + index, 60, 50))
        rows.append(
            dict(sample_id=str(index), source_path=source, target_path=target,
                 generated_path=generated, source_age=20, target_age=40, model_name="M")
        )

    result = evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=_bundle(), output_dir=tmp_path / "eval"
    )

    summary = result["summary"].iloc[0]
    assert np.isnan(summary["kid_mean"])
    assert summary["kid_status"] == "incomplete_generated_set"


def test_inference_wrapper_saves_outputs_builds_manifest_and_evaluates(tmp_path):
    from src.quantitative_metrics import run_inference_and_evaluate

    source = _image(tmp_path / "source.png", (40, 50, 60))
    target_1 = _image(tmp_path / "target_1.png", (80, 50, 60))
    target_2 = _image(tmp_path / "target_2.png", (100, 50, 60))
    samples = pd.DataFrame(
        [
            dict(sample_id="a", source_path=source, target_path=target_1, source_age=20, target_age=40),
            dict(sample_id="b", source_path=source, target_path=target_2, source_age=20, target_age=60),
        ]
    )

    def infer_fn(source_path, source_age, target_age, strength):
        assert source_path == source
        assert source_age == 20
        return Image.new("RGB", (16, 16), (target_age, int(strength * 100), 0))

    result = run_inference_and_evaluate(
        infer_fn,
        samples,
        inference_config={"strength": 0.3},
        evaluation_config={"metrics_bundle": _bundle()},
        output_dir=tmp_path / "run",
        model_name="OURS",
    )

    assert len(result["manifest"]) == 2
    assert all(Path(path).is_file() for path in result["manifest"]["generated_path"])
    assert result["evaluation"]["summary"].iloc[0]["model_name"] == "OURS"


def test_inference_wrapper_keeps_failed_generation_in_manifest(tmp_path):
    from src.quantitative_metrics import run_inference_and_evaluate

    source = _image(tmp_path / "source.png", (40, 50, 60))
    samples = pd.DataFrame([
        dict(sample_id="failed", source_path=source, target_path=None, source_age=20, target_age=40)
    ])

    def infer_fn(**kwargs):
        raise RuntimeError("generation failed")

    result = run_inference_and_evaluate(
        infer_fn, samples, inference_config={},
        evaluation_config={"metrics_bundle": _bundle(), "compute_kid": False},
        output_dir=tmp_path / "run", model_name="OURS"
    )

    assert pd.isna(result["manifest"].iloc[0]["generated_path"])
    assert result["manifest"].iloc[0]["inference_error"] == "generation failed"
