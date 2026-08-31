from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.inference import (
    compute_face_aging_diagnostics,
    diagnose_conditioning_sources,
    diagnose_checkpoint_age_sweep,
    diagnose_checkpoint_strength_sweep,
    infer_face_aging_direct,
)
from src.loss import AgeEstimatorAdapter, IdentityEncoderAdapter
from src.training import (
    REAL_IDENTITY_MEAN,
    REAL_IDENTITY_MEDIAN,
    TARGET_INTERCEPT,
    TARGET_SLOPE,
    atomic_torch_save,
    build_inference_payload,
    compute_directional_age_metrics,
    fit_age_response_calibration,
    fit_directional_age_calibrations,
)
from src.training.sampling_monitor import run_face_aging_monitor
from src.inference.checkpoint_diagnostics import _save_annotated_grid
from training_fakes import make_training_bundle


class ChannelIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        return images.mean(dim=(2, 3)) * self.scale


class MeanAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(100.0))

    def forward(self, images):
        return images.mean(dim=(1, 2, 3)) * self.scale


def attach_diagnostics(bundle):
    bundle["identity_encoder"] = IdentityEncoderAdapter(ChannelIdentity())
    bundle["age_estimator"] = AgeEstimatorAdapter(MeanAge())
    return bundle


def test_diagnostic_returns_scalar_age_and_controlled_identity_ordering():
    bundle = attach_diagnostics(make_training_bundle())
    red = torch.zeros(1, 3, 20, 20)
    red[:, 0] = 1
    green = torch.zeros_like(red)
    green[:, 1] = 1
    same = compute_face_aging_diagnostics(bundle, red, red, 50, image_size=16)
    different = compute_face_aging_diagnostics(bundle, red, green, 50, image_size=16)
    assert isinstance(same["predicted_generated_age"], float)
    assert same["target_age"] == 50.0
    assert same["identity_cosine_source_generated"] > different["identity_cosine_source_generated"]


def test_inference_diagnostics_do_not_change_generated_output():
    bundle = attach_diagnostics(make_training_bundle(seed=880))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    kwargs = dict(
        bundle=bundle, image=image, target_age=65, source_age=30,
        num_inference_steps=4, strength=0.5, image_size=32, seed=91,
    )
    plain = infer_face_aging_direct(**kwargs, compute_diagnostics=False)
    diagnosed = infer_face_aging_direct(**kwargs, compute_diagnostics=True)
    assert torch.equal(plain["image_tensor"], diagnosed["image_tensor"])
    assert plain["diagnostics"] is None
    assert set(diagnosed["diagnostics"]) == {
        "target_age", "predicted_source_age", "predicted_generated_age",
        "target_delta_age", "predicted_delta_age", "delta_age_error",
        "identity_cosine_source_generated",
    }


def test_training_sweep_writes_epoch_rows_appends_history_and_calibration(tmp_path, capsys):
    bundle = attach_diagnostics(make_training_bundle(seed=881))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    for epoch in (0, 1):
        report = run_face_aging_monitor(
            bundle=bundle, image=image, epoch=epoch, output_dir=tmp_path,
            target_age=[30, 50, 65], source_age=25,
            num_inference_steps=3, strength=0.5, seed=2026, image_size=32,
        )
        epoch_csv = tmp_path / f"epoch_{epoch + 1:03d}" / f"sampling_diagnostics_epoch_{epoch + 1:03d}.csv"
        frame = pd.read_csv(epoch_csv)
        assert len(frame) == 3
        assert frame["target_age"].tolist() == [30.0, 50.0, 65.0]
        assert frame["age_calibration_intercept"].nunique() == 1
        assert frame["age_calibration_slope"].nunique() == 1
        assert frame["age_calibration_r2"].nunique() == 1
        assert frame["age_calibration_score"].nunique() == 1
        assert report["age_calibration"] is not None
        assert report["age_direction"]["forward_mae"] is not None
        assert "forward_calibration_slope" in report["directional_calibration"]
        assert report["diagnostics_csv"] == str(epoch_csv)
        strength_report = report["strength_sweep"]
        assert strength_report["strengths"] == [0.20, 0.27, 0.35, 0.40]
        assert Path(strength_report["grid_path"]).name == "strength_age_sweeps.png"
        assert Path(strength_report["grid_path"]).exists()
        assert Path(strength_report["summary_csv"]).exists()
        epoch_dir = tmp_path / f"epoch_{epoch + 1:03d}"
        assert len(list(epoch_dir.glob("strength*.png"))) == 1
        assert not [path for path in epoch_dir.iterdir() if path.is_dir()]
    history = pd.read_csv(tmp_path / "sampling_diagnostics_history.csv")
    assert len(history) == 6
    assert history.groupby("epoch").size().to_dict() == {1: 3, 2: 3}
    printed = capsys.readouterr().out
    assert printed.count("Age calibration | all: a=") == 2
    assert printed.count("Age direction   | forward_MAE=") == 2


def test_age_response_calibration_exact_linear_oracle_and_edge_cases():
    rows = [
        {"target_delta_age": x, "predicted_delta_age": 3.0 + 1.25 * x}
        for x in (-20, 0, 10, 40)
    ]
    calibration = fit_age_response_calibration(rows)
    assert calibration == pytest.approx({
        "age_calibration_intercept": 3.0,
        "age_calibration_slope": 1.25,
        "age_calibration_r2": 1.0,
        "age_calibration_score": 10.29,
    })
    assert (TARGET_INTERCEPT, TARGET_SLOPE) == (-3.19, 0.84)
    assert (REAL_IDENTITY_MEAN, REAL_IDENTITY_MEDIAN) == (0.532, 0.589)
    target_fit = fit_age_response_calibration([
        {"target_delta_age": x, "predicted_delta_age": TARGET_INTERCEPT + TARGET_SLOPE * x}
        for x in (-20, 0, 20)
    ])
    assert target_fit["age_calibration_score"] == pytest.approx(0.0, abs=1e-12)
    assert fit_age_response_calibration(rows[:1]) is None
    assert fit_age_response_calibration([
        {"target_delta_age": 5, "predicted_delta_age": 1},
        {"target_delta_age": 5, "predicted_delta_age": 2},
    ]) is None


def test_directional_age_metrics_exact_oracle_ignores_zero_delta():
    metrics = compute_directional_age_metrics([
        {"target_delta_age": 10, "predicted_delta_age": 8},
        {"target_delta_age": 20, "predicted_delta_age": 24},
        {"target_delta_age": -10, "predicted_delta_age": -7},
        {"target_delta_age": -20, "predicted_delta_age": -25},
        {"target_delta_age": 0, "predicted_delta_age": 99},
    ])
    assert metrics == pytest.approx({
        "forward_mae": 3.0,
        "forward_bias": 1.0,
        "reverse_mae": 4.0,
        "reverse_bias": -1.0,
    })


def test_directional_calibration_exact_and_insufficient_direction_is_nan():
    rows = [
        {"target_delta_age": delta, "predicted_delta_age": 2.0 + 0.5 * delta}
        for delta in (-30, -10, 0, 10, 30)
    ]
    metrics = fit_directional_age_calibrations(rows)
    assert metrics == pytest.approx({
        "forward_calibration_intercept": 2.0,
        "forward_calibration_slope": 0.5,
        "forward_calibration_r2": 1.0,
        "reverse_calibration_intercept": 2.0,
        "reverse_calibration_slope": 0.5,
        "reverse_calibration_r2": 1.0,
    })
    forward_only = fit_directional_age_calibrations(rows[-2:])
    assert torch.isnan(torch.tensor(forward_only["reverse_calibration_slope"]))


def test_monitoring_history_migrates_old_csv_schema(tmp_path):
    history_path = tmp_path / "sampling_diagnostics_history.csv"
    history_path.write_text(
        "epoch,source_age,target_age,target_delta_age,predicted_delta_age\n"
        "1,25,30,5,4\n",
        encoding="utf-8",
    )
    bundle = attach_diagnostics(make_training_bundle(seed=889))
    report = run_face_aging_monitor(
        bundle=bundle,
        image=Image.new("RGB", (38, 32), (110, 75, 55)),
        epoch=1,
        output_dir=tmp_path,
        target_age=[30, 50, 65],
        source_age=25,
        num_inference_steps=2,
        strength=0.5,
        image_size=32,
    )
    history = pd.read_csv(history_path)
    assert set([
        "age_calibration_intercept", "age_calibration_slope",
        "age_calibration_r2", "age_calibration_score",
        "forward_calibration_intercept", "forward_calibration_slope",
        "forward_calibration_r2", "reverse_calibration_intercept",
        "reverse_calibration_slope", "reverse_calibration_r2",
        "forward_mae", "forward_bias", "reverse_mae", "reverse_bias",
    ]).issubset(history.columns)
    assert len(history) == 4
    assert report["age_calibration"] is not None


def test_checkpoint_diagnostic_age_error_and_generation_match_normal_inference(tmp_path):
    original = make_training_bundle(seed=882)
    checkpoint = atomic_torch_save(
        build_inference_payload(original, {"image_size": 32}),
        tmp_path / "epoch_005" / "adapter_inference.pt",
    )
    rebuilt = attach_diagnostics(make_training_bundle(seed=882))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    output_dir = tmp_path / "diagnostic"
    frame = diagnose_checkpoint_age_sweep(
        checkpoint_path=checkpoint,
        bundle=rebuilt,
        source_image=image,
        source_age=26,
        target_ages=[35, 65],
        output_dir=output_dir,
        mode="direct",
        num_inference_steps=3,
        strength=0.5,
        seed=77,
        image_size=32,
    )
    assert frame.columns.tolist() == [
        "checkpoint", "source_age", "target_age", "target_delta_age",
        "predicted_source_age", "predicted_generated_age", "predicted_delta_age",
        "age_error", "delta_age_error", "identity_cosine", "mode", "strength",
        "num_inference_steps", "text_guidance_scale", "text_reference_mode",
        "age_guidance_scale", "image_guidance_scale", "seed",
    ]
    assert torch.equal(
        torch.tensor(frame["age_error"].to_numpy()),
        torch.tensor((frame["predicted_generated_age"] - frame["target_age"]).to_numpy()),
    )
    normal = infer_face_aging_direct(
        bundle=rebuilt, image=image, target_age=35, source_age=26,
        num_inference_steps=3, strength=0.5, seed=77, image_size=32,
        compute_diagnostics=True,
    )
    with Image.open(output_dir / "age_035.png") as saved:
        assert list(saved.getdata()) == list(normal["image"].getdata())
    assert (output_dir / "age_sweep.png").exists()
    assert (output_dir / "sampling_diagnostics.csv").exists()


def test_checkpoint_strength_sweep_saves_only_combined_grid_and_summary(tmp_path):
    original = make_training_bundle(seed=883)
    checkpoint = atomic_torch_save(
        build_inference_payload(original, {"image_size": 32}),
        tmp_path / "epoch_004" / "adapter_inference.pt",
    )
    rebuilt = attach_diagnostics(make_training_bundle(seed=883))
    output_dir = tmp_path / "strength_diagnostic"
    frame = diagnose_checkpoint_strength_sweep(
        checkpoint_path=checkpoint,
        bundle=rebuilt,
        source_image=Image.new("RGB", (38, 32), (110, 75, 55)),
        source_age=26,
        target_ages=[16, 18, 35, 65],
        strengths=[0.20, 0.40],
        output_dir=output_dir,
        num_inference_steps=2,
        image_size=32,
    )
    assert frame["strength"].tolist() == [0.20, 0.40]
    assert frame["forward_calibration_slope"].notna().all()
    assert frame["reverse_calibration_slope"].notna().all()
    assert (output_dir / "strength_age_sweeps.png").exists()
    assert (output_dir / "strength_sweep_summary.csv").exists()
    assert not list(output_dir.glob("age_*.png"))


def test_checkpoint_grid_places_rejuvenation_before_source_and_aging_after(tmp_path):
    colors = {
        65: (255, 0, 0),
        16: (0, 255, 0),
        35: (0, 0, 255),
    }
    results = [
        {
            "image": Image.new("RGB", (16, 16), colors[age]),
            "diagnostics": {
                "target_age": float(age),
                "predicted_generated_age": float(age),
                "predicted_delta_age": float(age - 30),
                "identity_cosine_source_generated": 1.0,
            },
        }
        for age in (65, 16, 35)  # deliberately not chronological
    ]
    path = _save_annotated_grid(
        source_image=Image.new("RGB", (16, 16), (0, 0, 0)),
        source_age=30,
        results=results,
        image_size=16,
        output_path=tmp_path / "ordered.png",
    )
    with Image.open(path) as grid:
        assert grid.size == (64, 80)
        assert [grid.getpixel((8 + 16 * index, 8)) for index in range(4)] == [
            colors[16],
            (0, 0, 0),
            colors[35],
            colors[65],
        ]


def test_conditioning_isolation_runs_nine_matched_cases_without_writing(tmp_path, capsys):
    bundle = attach_diagnostics(make_training_bundle(seed=883))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    output = diagnose_conditioning_sources(
        bundle,
        image,
        source_age=26,
        target_ages=[30, 40, 65],
        num_inference_steps=2,
        strength=0.35,
        seed=2026,
        image_size=32,
        display_result=False,
    )
    frame = output["dataframe"]
    assert len(frame) == 9
    assert frame["condition"].tolist() == ["full"] * 3 + ["delta_only"] * 3 + ["text_only"] * 3
    assert frame.groupby("condition")["seed"].unique().map(list).to_dict() == {
        "delta_only": [2026], "full": [2026], "text_only": [2026],
    }
    assert frame.loc[frame.condition == "full", "effective_delta_age"].tolist() == [4, 14, 39]
    assert frame.loc[frame.condition == "delta_only", "effective_delta_age"].tolist() == [4, 14, 39]
    assert frame.loc[frame.condition == "text_only", "effective_delta_age"].tolist() == [0, 0, 0]
    assert frame.loc[frame.condition == "delta_only", "target_prompt"].unique().tolist() == ["photo of a person"]
    assert output["grid"].size == (90 + 3 * 32, 30 + 3 * (32 + 54))
    assert not list(tmp_path.iterdir())
    printed = capsys.readouterr().out
    assert "CONDITIONING ISOLATION DIAGNOSTIC" in printed
    assert "delta_only" in printed and "text_only" in printed


def test_checkpoint_sweeps_compare_all_referenced_cfg_modes_and_write_metadata(tmp_path):
    original = make_training_bundle(seed=884)
    checkpoint = atomic_torch_save(
        build_inference_payload(original, {"image_size": 32}),
        tmp_path / "adapter.pt",
    )
    rebuilt = attach_diagnostics(make_training_bundle(seed=884))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    predictions = {}
    for reference_mode in ("source_age", "generic", "null"):
        frame = diagnose_checkpoint_age_sweep(
            checkpoint_path=checkpoint,
            bundle=rebuilt,
            source_image=image,
            source_age=26,
            target_ages=[30, 40, 65],
            output_dir=tmp_path / reference_mode,
            mode="direct",
            text_reference_mode=reference_mode,
            age_guidance_scale=3.0 if reference_mode != "null" else 7.0,
            num_inference_steps=2,
            strength=0.35,
            seed=2026,
            image_size=32,
        )
        assert frame["text_reference_mode"].unique().tolist() == [reference_mode]
        assert (tmp_path / reference_mode / "sampling_diagnostics.csv").exists()
        predictions[reference_mode] = frame["predicted_generated_age"].to_numpy()
    assert not torch.equal(
        torch.tensor(predictions["source_age"]), torch.tensor(predictions["generic"])
    )
    assert not torch.equal(
        torch.tensor(predictions["source_age"]), torch.tensor(predictions["null"])
    )
