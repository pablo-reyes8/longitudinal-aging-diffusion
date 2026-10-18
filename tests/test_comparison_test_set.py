from importlib import import_module

import numpy as np
import pandas as pd
from PIL import Image
import pytest


api = import_module("src.comparisong.evaluate_test_set")


def make_dataset(root, identities=None):
    for identity, ages in (identities or {"id_1": (19, 30, 35, 45, 60, 70), "id_2": (35, 53)}).items():
        folder = root / identity
        folder.mkdir(parents=True)
        for age in ages:
            Image.new("RGB", (16, 16), (age, 50, 80)).save(folder / f"{age}.png")
    return root


def test_selection_keeps_all_aging_pairs_and_prioritizes_large_gaps(tmp_path):
    root = make_dataset(tmp_path / "data")
    Image.new("RGB", (16, 16)).save(root / "id_1" / "19_1.jpg")
    Image.new("RGB", (16, 16)).save(root / "id_1" / "999.jpg")
    selected = api.select_test_transitions(root)

    assert selected.groupby("identity_id").size().to_dict() == {"id_1": 15, "id_2": 1}
    first = selected.iloc[0]
    assert (first.source_age, first.target_age) == (19, 70)
    assert first.source_path.endswith("19.png")
    assert list(selected[selected.identity_id == "id_1"].delta_age)[:5] == [51, 41, 40, 35, 30]
    pd.testing.assert_frame_equal(selected, api.select_test_transitions(root))
    pd.testing.assert_frame_equal(selected, api.select_test_transitions(root, max_transitions_per_id=5))
    both = api.select_test_transitions(root, direction="both")
    assert (both.delta_age < 0).any() and (both.delta_age > 0).any()


def test_both_keeps_all_aging_and_adds_thirty_percent_reverse_pairs(tmp_path):
    root = tmp_path / "data"
    for identity in ("id_1", "id_2"):
        folder = root / identity
        folder.mkdir(parents=True)
        for age in (19, 30, 35, 45, 60, 70):
            Image.new("RGB", (8, 8)).save(folder / f"{age}.png")

    selected = api.select_test_transitions(root, direction="both")

    assert len(selected) == 39
    assert (selected.delta_age > 0).sum() == 30
    assert (selected.delta_age < 0).sum() == 9
    forward = selected[selected.delta_age > 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(forward, api.select_test_transitions(root, direction="aging"))
    assert sorted(selected[selected.delta_age < 0].delta_age.abs(), reverse=True) == [51, 51, 41, 41, 40, 40, 35, 35, 30]
    pd.testing.assert_frame_equal(selected, api.select_test_transitions(root, direction="both"))


def test_both_keeps_available_directions_with_only_two_photos(tmp_path):
    root = tmp_path / "data"
    folder = root / "id_1"
    folder.mkdir(parents=True)
    for age in (20, 60):
        Image.new("RGB", (8, 8)).save(folder / f"{age}.png")

    selected = api.select_test_transitions(root, direction="both")

    assert sorted(selected.delta_age) == [-40, 40]


def test_internal_gender_labels_and_explicit_overrides_reach_model(monkeypatch, tmp_path):
    from src.comparisong.identity_metadata import TEST_ID_GENDERS

    assert len(TEST_ID_GENDERS) == 100
    assert TEST_ID_GENDERS["id_1"] == "male"
    assert TEST_ID_GENDERS["id_2"] == "female"
    assert TEST_ID_GENDERS["id_1062"] == "male"
    assert TEST_ID_GENDERS["id_1135"] == "female"
    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)
    result = api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["SAM"],
        comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
    )
    labels = result["transitions"].groupby("identity_id").gender.first().to_dict()
    assert labels == {"id_1": "male", "id_2": "female"}
    assert all(options["gender"] == ("female" if "id_2" in options["image"] else "male")
               for name, options in events if name == "SAM")
    overridden = api.evaluate_test_set(
        root, checkpoint, tmp_path / "overridden", models=["SAM"],
        comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        gender_by_id={"id_1": "M", "id_2": "H"},
    )
    assert overridden["transitions"].groupby("identity_id").gender.first().to_dict() == {
        "id_1": "female", "id_2": "male",
    }


def test_unknown_identity_needs_explicit_gender_for_fading(tmp_path):
    root = make_dataset(tmp_path / "data", {"id_9999": (20, 60)})
    with pytest.raises(ValueError, match="Missing gender for id_9999"):
        api.evaluate_test_set(root, tmp_path / "unused.pt", tmp_path / "out", models=["FADING"])


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

    root = make_dataset(tmp_path / "data", {"id_1": (19, 35, 70), "id_2": (35, 53)})
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
        our_mode="direct",
        max_transitions_per_id=2, output_dir=tmp_path / "results", device="cpu",
    )
    result = api.evaluate_test_set(root, checkpoint, **kwargs)

    assert [event[0] for event in events[:5]] == ["load_ours", "ours", "ours", "SAM", "SAM"]
    own_events = [options for name, options in events if name == "ours"]
    assert len(own_events) == 4 and all(options["image_size"] == 400 for options in own_events)
    assert kid.calls == [(2, 2), (2, 2), (2, 2), (4, 4), (4, 4)]
    assert len(result["per_transition"]) == 4
    assert "SAM__dex_age_abs_error" in result["per_transition"]
    assert "ours_strength_0.35__adaface_cosine_gt" in result["per_transition"]
    assert result["paths"]["paper"].is_file()
    assert set(result["summary"].n_samples) == {4}

    event_count = len(events)
    api.evaluate_test_set(root, checkpoint, **kwargs)
    assert len(events) == event_count  # cached generations, metrics recomputed
    with pytest.raises(ValueError, match="another protocol"):
        api.evaluate_test_set(root, checkpoint, **{**kwargs, "seed": 9})


def test_model_preflight_failure_is_recorded_and_other_models_complete(monkeypatch, tmp_path):
    root, checkpoint, metrics, kid, events = setup_pipeline(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("SAM dependency unavailable")

    monkeypatch.setattr(api, "age_image", fail)
    result = api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["ours", "SAM"],
        comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        our_mode="direct",
    )
    assert len([name for name, _ in events if name == "ours"]) == len(result["transitions"])
    summary = result["summary"].set_index("model_name")
    assert summary.loc["SAM", "n_failed_generation"] == len(result["transitions"])
    assert summary.loc["ours_strength_0.35", "n_failed_generation"] == 0
    assert result["paths"]["failures"].is_file()


@pytest.mark.parametrize("failed_model", ["SAM", "ours"])
def test_late_image_failure_continues_and_resume_only_retries_when_requested(monkeypatch, tmp_path, failed_model):
    import src.inference as inference

    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)
    attempts = []
    original = api.age_image if failed_model == "SAM" else inference.infer_face_aging_direct

    def fail_for_last_identity(*args, **kwargs):
        image = kwargs["image"]
        attempts.append(str(image))
        if "id_2" in str(image):
            raise RuntimeError("Could not find face in image. Try another!")
        return original(*args, **kwargs)

    owner = api if failed_model == "SAM" else inference
    method = "age_image" if failed_model == "SAM" else "infer_face_aging_direct"
    monkeypatch.setattr(owner, method, fail_for_last_identity)
    kwargs = dict(
        models=["SAM", "ours"], comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        our_mode="direct", max_transitions_per_id=2,
    )
    result = api.evaluate_test_set(root, checkpoint, tmp_path / "results", **kwargs)
    report_name = "SAM" if failed_model == "SAM" else "ours_strength_0.35"
    summary = result["summary"].set_index("model_name")
    assert summary.loc[report_name, "n_failed_generation"] == 1
    assert summary.loc[report_name, "n_generated"] == 3
    assert summary.loc[report_name, "kid_status"] == "incomplete_generated_set"
    assert result["per_transition"][f"{report_name}__inference_status"].tolist().count("failed") == 1
    assert len(pd.read_csv(result["paths"]["failures"])) == 1

    attempt_count = len(attempts)
    api.evaluate_test_set(root, checkpoint, tmp_path / "results", **kwargs)
    assert len(attempts) == attempt_count

    retried = []

    def fixed(*args, **kwargs):
        retried.append(kwargs["image"])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, fixed)
    recovered = api.evaluate_test_set(root, checkpoint, tmp_path / "results", retry_failed=True, **kwargs)
    assert len(retried) == 1
    assert recovered["summary"].n_failed_generation.sum() == 0
    assert pd.read_csv(recovered["paths"]["failures"]).empty


def test_own_strength_map_is_exact_and_loaded_only_once(monkeypatch, tmp_path):
    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)
    api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["ours"], metrics_bundle=metrics,
        max_transitions_per_id=1, target_age_strength_map={35: 0.22, 70: 0.29, 53: 0.27},
        our_mode="direct",
    )
    assert len([name for name, _ in events if name == "load_ours"]) == 1
    assert [options["strength"] for name, options in events if name == "ours"] == [0.29, 0.29, 0.22, 0.27]


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
    assert len(captured) == 4
    assert all(call["metrics_bundle"] is metrics for call in captured)
    assert all(call["diagnostic_backend"] == "metrics" for call in captured)
    assert all(call["diagnostic_config"]["negative_prompt"] == "blur" for call in captured)
    assert all(call["diagnostic_config"]["image_size"] == 400 for call in captured)
    assert all(call["diagnostic_config"]["max_trials_per_target"] == 8 for call in captured)
    assert all(call["search_method"] == "smart" for call in captured)


def test_default_reports_direct_and_smart_picking_on_the_same_panel(monkeypatch, tmp_path):
    root, checkpoint, metrics, kid, events = setup_pipeline(monkeypatch, tmp_path)
    captured = []
    generated = tmp_path / "picked.png"
    Image.new("RGB", (16, 16), "gray").save(generated)

    def pick(*args, **kwargs):
        captured.append(kwargs)
        return {"generated_paths": {70: generated}}

    monkeypatch.setattr(api, "evaluate_aging_inference_picking", pick)
    result = api.evaluate_test_set(
        root, checkpoint, tmp_path / "results", models=["ours"], metrics_bundle=metrics,
        max_transitions_per_id=1, our_inference_config={"strength": 0.22},
    )
    assert set(result["summary"].model_name) == {"ours_strength_0.22", "ours_picking"}
    assert "ours_picking__dex_age_abs_error" in result["per_transition"]
    assert len(captured) == 4
    assert all(call["diagnostic_config"]["max_trials_per_target"] == 8 for call in captured)
    assert all(list(call["target_age_strength_map"].values()) == [0.22] for call in captured)
    assert [options["strength"] for name, options in events if name == "ours"] == [0.22] * 4
    assert len([name for name, _ in events if name == "load_ours"]) == 1
    assert kid.calls == [(2, 2), (2, 2), (2, 2), (4, 4), (4, 4)]


@pytest.mark.parametrize("backend", ["age_estimator", "identity_encoder", "kid_metric"])
def test_metrics_smoke_failure_prevents_any_generation(monkeypatch, tmp_path, backend):
    root, checkpoint, metrics, _, events = setup_pipeline(monkeypatch, tmp_path)

    class Broken:
        def fail(self, *args, **kwargs):
            raise RuntimeError("metric backend unavailable")

        predict_batch = embed_batch = compute = fail

    metrics[backend] = Broken()
    with pytest.raises(RuntimeError, match="Metrics smoke failed before any generation"):
        api.evaluate_test_set(
            root, checkpoint, tmp_path / "results", models=["SAM", "ours"],
            comparison_bundle={"SAM": {}}, metrics_bundle=metrics,
        )
    assert events == []
    assert not (tmp_path / "results" / "generated").exists()


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
