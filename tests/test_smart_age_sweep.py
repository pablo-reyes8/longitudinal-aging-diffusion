from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.inference import (
    adaptive_mivolo_confidence_margin,
    diagnose_checkpoint_smart_age_sweep,
)


def _bundle():
    return {"identity_encoder": object(), "age_estimator": object()}


def _fake_inference(**kwargs):
    strength = float(kwargs["strength"])
    predicted_age = 20.0 + 100.0 * strength
    return {
        "image": Image.new("RGB", (24, 24), (int(strength * 255), 80, 60)),
        "diagnostics": {
            "predicted_source_age": 30.0,
            "predicted_generated_age": predicted_age,
            "identity_cosine_source_generated": 1.0 - strength,
        },
    }


def test_adaptive_mivolo_confidence_is_wider_for_small_deltas():
    assert adaptive_mivolo_confidence_margin(0) == 4.0
    assert adaptive_mivolo_confidence_margin(12.5) == 3.0
    assert adaptive_mivolo_confidence_margin(25) == 2.0
    assert adaptive_mivolo_confidence_margin(100) == 2.0
    with pytest.raises(ValueError):
        adaptive_mivolo_confidence_margin(10, small_delta_margin_years=2, large_delta_margin_years=4)


def test_smart_sweep_searches_strength_uses_bias_band_and_zero_shortcut(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "src.inference.smart_age_sweep.load_face_aging_adapter_for_inference",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr("src.inference.smart_age_sweep.infer_face_aging", _fake_inference)
    frame = diagnose_checkpoint_smart_age_sweep(
        checkpoint_path=tmp_path / "adapter.pt",
        bundle=_bundle(),
        source_image=Image.new("RGB", (24, 24), "gray"),
        source_age=26,
        target_ages=[26, 65],
        output_dir=tmp_path / "results",
        target_age_strength_map={26: 0.04, 65: 0.25},
        image_size=24,
    )

    zero, old = frame.iloc[0], frame.iloc[1]
    assert zero["trials_run"] == 1 and zero["strength"] == 0.04
    assert old["trials_run"] == 5
    assert old["strength"] == pytest.approx(0.37)
    assert old["expected_mivolo_delta"] == pytest.approx(-3.19 + 0.841 * 39)
    assert old["confidence_margin_years"] == 2.0
    trials = frame.attrs["trials"]
    assert trials.groupby("target_age").size().to_dict() == {26.0: 1, 65.0: 5}
    assert trials.groupby("target_age")["selected_best"].sum().to_dict() == {
        26.0: 1,
        65.0: 1,
    }
    assert Path(frame.attrs["grid_path"]).exists()
    assert Path(frame.attrs["trials_csv_path"]).exists()
    assert Path(frame.attrs["summary_csv_path"]).exists()
    assert frame.attrs["all_trials_grid_path"] is None
    assert not list((tmp_path / "results").rglob("trial_*.png"))


def test_smart_sweep_nearest_prior_unbiased_mode_and_micro_search_are_reproducible(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "src.inference.smart_age_sweep.load_face_aging_adapter_for_inference",
        lambda *args, **kwargs: {},
    )

    def always_too_young(**kwargs):
        return {
            "image": Image.new("RGB", (20, 20), "gray"),
            "diagnostics": {
                "predicted_source_age": 30.0,
                "predicted_generated_age": 0.0,
                "identity_cosine_source_generated": 0.8,
            },
        }

    monkeypatch.setattr(
        "src.inference.smart_age_sweep.infer_face_aging", always_too_young
    )
    kwargs = dict(
        checkpoint_path=tmp_path / "adapter.pt",
        bundle=_bundle(),
        source_image=Image.new("RGB", (20, 20), "gray"),
        source_age=26,
        target_ages=[65],
        target_age_strength_map={12: 0.10, 55: 0.25},
        use_bias_corrected_mivolo_target=False,
        enable_guidance_micro_search=True,
        save_all_trials=True,
        image_size=20,
    )
    first = diagnose_checkpoint_smart_age_sweep(
        **kwargs, output_dir=tmp_path / "first"
    )
    second = diagnose_checkpoint_smart_age_sweep(
        **kwargs, output_dir=tmp_path / "second"
    )
    first_trials = first.attrs["trials"].drop(columns="image_path")
    second_trials = second.attrs["trials"].drop(columns="image_path")
    pd.testing.assert_frame_equal(first_trials, second_trials)
    assert first_trials["strength"].tolist() == pytest.approx(
        [0.25, 0.30, 0.33, 0.33, 0.33]
    )
    # Trial four changes only age guidance; trial five resets it and changes text guidance.
    assert first_trials.loc[3, "age_guidance_scale"] == 3.25
    assert first_trials.loc[3, "text_guidance_scale"] == 7.0
    assert first_trials.loc[4, "age_guidance_scale"] == 3.0
    assert first_trials.loc[4, "text_guidance_scale"] == 7.25
    assert Path(first.attrs["all_trials_grid_path"]).exists()
    assert len(list((tmp_path / "first" / "smart_trials").glob("*.png"))) == 5


def test_assisted_smart_sweep_reuses_base_winners_without_repeating_search(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "src.inference.smart_age_sweep.load_face_aging_adapter_for_inference",
        lambda *args, **kwargs: {},
    )
    calls = []

    def recording_inference(**kwargs):
        calls.append(kwargs)
        return _fake_inference(**kwargs)

    monkeypatch.setattr(
        "src.inference.smart_age_sweep.infer_face_aging", recording_inference
    )
    frame = diagnose_checkpoint_smart_age_sweep(
        checkpoint_path=tmp_path / "adapter.pt",
        bundle=_bundle(),
        source_image=Image.new("RGB", (24, 24), "gray"),
        source_age=26,
        target_ages=[26, 65],
        output_dir=tmp_path / "assisted",
        target_age_strength_map={26: 0.04, 65: 0.25},
        generate_assisted_prompt_variant=True,
        prompt_assistance_scale=0.35,
        negative_prompt_assistance_scale=0.60,
        source_mouth_state="closed",
        image_size=24,
    )

    assisted_calls = [call for call in calls if call.get("target_prompt")]
    base_calls = [call for call in calls if not call.get("target_prompt")]
    assert len(base_calls) == 6  # one zero-delta pass + five search trials
    assert len(assisted_calls) == 2  # exactly one pass for each selected winner
    assert [call["strength"] for call in assisted_calls] == pytest.approx(
        frame["strength"].tolist()
    )
    assert all("keep lips closed" in call["target_prompt"] for call in assisted_calls)
    assert all("open mouth" in call["negative_prompt"] for call in assisted_calls)
    assert all(call["prompt_assistance_scale"] == 0.35 for call in assisted_calls)
    assert all(
        call["negative_prompt_assistance_scale"] == 0.60
        for call in assisted_calls
    )
    assert frame.attrs["assisted"]["trials_run"].tolist() == [1, 1]
    assert frame.attrs["assisted"]["prompt_assistance_scale"].tolist() == [0.35, 0.35]
    for attr in (
        "grid_path",
        "assisted_grid_path",
        "comparison_grid_path",
        "trials_csv_path",
        "assisted_trials_csv_path",
        "summary_csv_path",
        "assisted_summary_csv_path",
    ):
        assert Path(frame.attrs[attr]).exists()
