from importlib import import_module

import numpy as np
import pandas as pd
from PIL import Image
import pytest


api = import_module("src.comparisong.evaluate_test_set")


def make_dataset(root):
    for identity, ages in {"id_1": (19, 30, 35, 45, 60, 70), "id_2": (35, 53)}.items():
        folder = root / identity
        folder.mkdir(parents=True)
        for age in ages:
            Image.new("RGB", (16, 16), (age, 50, 80)).save(folder / f"{age}.png")
    return root


def test_selection_limits_each_identity_and_prioritizes_large_gaps(tmp_path):
    root = make_dataset(tmp_path / "data")
    Image.new("RGB", (16, 16)).save(root / "id_1" / "19_1.jpg")
    Image.new("RGB", (16, 16)).save(root / "id_1" / "999.jpg")
    selected = api.select_test_transitions(root)

    assert selected.groupby("identity_id").size().to_dict() == {"id_1": 5, "id_2": 1}
    first = selected.iloc[0]
    assert (first.source_age, first.target_age) == (19, 70)
    assert first.source_path.endswith("19.png")
    assert list(selected[selected.identity_id == "id_1"].delta_age) == [51, 41, 40, 35, 30]
    pd.testing.assert_frame_equal(selected, api.select_test_transitions(root))
    both = api.select_test_transitions(root, direction="both")
    assert (both.delta_age < 0).any() and (both.delta_age > 0).any()


class Identity:
    def embed_batch(self, images):
        return np.ones((len(images), 3))


class Age:
    def predict_batch(self, images):
        return np.full(len(images), 50.0)


class Kid:
    def __init__(self):
        self.calls = []

    def compute(self, real, generated, **kwargs):
        self.calls.append((len(real), len(generated)))
        return 0.01, 0.001


def setup_pipeline(monkeypatch, tmp_path):
    import src.inference as inference

    root = make_dataset(tmp_path / "data")
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    kid = Kid()
    metrics = {"aligner": lambda image: image, "identity_encoder": Identity(), "age_estimator": Age(), "kid_metric": kid}
    events = []

    def load(*args, **kwargs):
        events.append(("load_ours", kwargs))
        return {}

    def generate(**kwargs):
        events.append(("ours", kwargs))
        return {"image": Image.new("RGB", (16, 16), "gray")}

    def literature(bundle, **kwargs):
        events.append(("SAM", kwargs))
        return {"SAM": {kwargs["target_ages"][0]: Image.new("RGB", (16, 16), "white")}}

    monkeypatch.setattr(inference, "load_face_aging_inference_bundle", load)
    monkeypatch.setattr(inference, "infer_face_aging_direct", generate)
    monkeypatch.setattr(api, "age_image", literature)
    return root, checkpoint, metrics, kid, events


def test_pipeline_common_panel_wide_csv_global_kid_and_resume(monkeypatch, tmp_path):
    root, checkpoint, metrics, kid, events = setup_pipeline(monkeypatch, tmp_path)
    kwargs = dict(
        models=["ours", "SAM"], comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        max_transitions_per_id=2, output_dir=tmp_path / "results", device="cpu",
    )
    result = api.evaluate_test_set(root, checkpoint, **kwargs)

    assert [event[0] for event in events[:5]] == ["load_ours", "ours", "ours", "SAM", "SAM"]
    own_events = [options for name, options in events if name == "ours"]
    assert len(own_events) == 3 and all(options["image_size"] == 400 for options in own_events)
    assert kid.calls == [(2, 2), (2, 2), (3, 3), (3, 3)]
    assert len(result["per_transition"]) == 3
    assert "SAM__dex_age_abs_error" in result["per_transition"]
    assert "ours__adaface_cosine_gt" in result["per_transition"]
    assert result["paths"]["paper"].is_file()
    assert set(result["summary"].n_samples) == {3}

    event_count = len(events)
    api.evaluate_test_set(root, checkpoint, **kwargs)
    assert len(events) == event_count  # cached generations, metrics recomputed
    with pytest.raises(ValueError, match="another protocol"):
        api.evaluate_test_set(root, checkpoint, **{**kwargs, "seed": 9})


def test_preflight_failure_stops_before_bulk_and_keeps_completed_image(monkeypatch, tmp_path):
    root, checkpoint, metrics, kid, events = setup_pipeline(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("SAM dependency unavailable")

    monkeypatch.setattr(api, "age_image", fail)
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        api.evaluate_test_set(
            root, checkpoint, tmp_path / "results", models=["ours", "SAM"],
            comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        )
    assert len([name for name, _ in events if name == "ours"]) == 2
    assert kid.calls == [(2, 2)]
    assert len(list((tmp_path / "results" / "generated").rglob("*.png"))) == 2


def test_own_strength_map_is_exact_and_loaded_only_once(monkeypatch, tmp_path):
    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)
    api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["ours"], metrics_bundle=metrics,
        max_transitions_per_id=1, target_age_strength_map={70: 0.29, 53: 0.27},
    )
    assert len([name for name, _ in events if name == "load_ours"]) == 1
    assert [options["strength"] for name, options in events if name == "ours"] == [0.29, 0.27]


def test_picking_reuses_metrics_and_preserves_prompt_options(monkeypatch, tmp_path):
    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)
    captured = []
    generated = tmp_path / "picked.png"
    Image.new("RGB", (16, 16), "gray").save(generated)

    def pick(*args, **kwargs):
        captured.append(kwargs)
        return {"generated_paths": {70: generated}}

    monkeypatch.setattr(api, "evaluate_aging_inference_picking", pick)
    api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["ours"], metrics_bundle=metrics,
        max_transitions_per_id=1, our_mode="picking",
        diagnostic_config={"generate_assisted_prompt_variant": True, "negative_prompt": "blur"},
    )
    assert len(captured) == 2
    assert all(call["metrics_bundle"] is metrics for call in captured)
    assert all(call["diagnostic_backend"] == "metrics" for call in captured)
    assert all(call["diagnostic_config"]["negative_prompt"] == "blur" for call in captured)
    assert all(call["diagnostic_config"]["image_size"] == 400 for call in captured)


def test_setup_reuses_existing_loader_and_prepares_metrics_on_cpu(monkeypatch, tmp_path):
    root, checkpoint, metrics, _, _ = setup_pipeline(monkeypatch, tmp_path)
    observed = {}

    def prepare(*args, **kwargs):
        observed["prepare"] = kwargs
        return {"ready": True}

    def load(**kwargs):
        observed["loader"] = kwargs
        return {"SAM": {}}

    monkeypatch.setattr(api, "prepare_quantitative_metrics", prepare)
    monkeypatch.setattr(api, "load_quantitative_metrics", lambda **kwargs: metrics)
    monkeypatch.setattr(api, "load_aging_models", load)
    api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["SAM"],
        max_transitions_per_id=1, model_loader_config={"root": "existing-models"},
    )

    assert observed["prepare"]["device"] == "cpu"
    assert observed["loader"] == {"root": "existing-models", "models": ["SAM"], "load_metrics": False}
