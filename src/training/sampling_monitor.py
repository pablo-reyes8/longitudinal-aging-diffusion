"""Callback-only deterministic monitoring; no fake inference implementation."""

from __future__ import annotations

from collections.abc import Sequence
import csv
import inspect
from pathlib import Path

import torch

from .age_calibration import compute_directional_age_metrics, fit_age_response_calibration


DIAGNOSTIC_CSV_FIELDS = [
    "epoch", "source_age", "target_age", "target_delta_age", "predicted_source_age",
    "predicted_generated_age", "predicted_delta_age", "age_error", "delta_age_error",
    "identity_cosine", "mode", "text_reference_mode", "age_guidance_scale", "seed",
    "age_calibration_intercept", "age_calibration_slope", "age_calibration_r2",
    "age_calibration_score",
]


def _diagnostic_row(*, epoch: int, result) -> dict | None:
    diagnostics = result.get("diagnostics")
    if diagnostics is None:
        return None
    return {
        "epoch": int(epoch + 1),
        "source_age": result.get("source_age"),
        "target_age": diagnostics["target_age"],
        "target_delta_age": diagnostics["target_delta_age"],
        "predicted_source_age": diagnostics["predicted_source_age"],
        "predicted_generated_age": diagnostics["predicted_generated_age"],
        "predicted_delta_age": diagnostics["predicted_delta_age"],
        "age_error": diagnostics["predicted_generated_age"] - diagnostics["target_age"],
        "delta_age_error": diagnostics["delta_age_error"],
        "identity_cosine": diagnostics["identity_cosine_source_generated"],
        "mode": result["mode"],
        "text_reference_mode": result["text_reference_mode"],
        "age_guidance_scale": result["age_guidance_scale"],
        "seed": result["seed"],
    }


def _write_diagnostic_csvs(rows: list[dict], *, epoch: int, epoch_dir: Path, history_dir: Path):
    if not rows:
        return None, None
    epoch_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = epoch_dir / f"sampling_diagnostics_epoch_{epoch + 1:03d}.csv"
    with epoch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    history_path = history_dir / "sampling_diagnostics_history.csv"
    if history_path.exists() and history_path.stat().st_size > 0:
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != DIAGNOSTIC_CSV_FIELDS:
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {field: row.get(field, "") for field in DIAGNOSTIC_CSV_FIELDS}
                    for row in existing_rows
                )
    write_header = not history_path.exists() or history_path.stat().st_size == 0
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return epoch_path, history_path


def normalize_monitoring_ages(target_age) -> tuple[list[int], bool]:
    """Return validated ages and whether the caller requested a sweep."""
    is_sweep = isinstance(target_age, Sequence) and not isinstance(target_age, (str, bytes))
    raw_ages = list(target_age) if is_sweep else [target_age]
    if not raw_ages or any(isinstance(age, bool) or not isinstance(age, int) for age in raw_ages):
        raise ValueError("monitoring_target_age must be an int or a non-empty sequence of ints")
    ages = [int(age) for age in raw_ages]
    if any(age < 0 or age > 120 for age in ages):
        raise ValueError("monitoring target ages must be in [0, 120]")
    if len(set(ages)) != len(ages):
        raise ValueError("monitoring target ages must be unique to avoid overwritten files")
    return ages, is_sweep


def run_face_aging_monitor(
    *, bundle, image, epoch: int, output_dir, loss_fn=None, target_prompt=None,
    target_age=None, source_prompt=None, source_age=None,
    mode="direct", use_inverse_diffusion=None, num_inference_steps=30,
    strength=0.35, text_guidance_scale=7.0, image_guidance_scale=1.5,
    text_reference_mode="source_age", age_guidance_scale=3.0,
    seed=2026, image_size=256, compute_diagnostics: bool = True,
):
    """Generate one edit or an ordered age sweep from the same fixed image."""
    from src.inference import generate_age_sweep, infer_face_aging, save_inference_image

    ages = None
    is_sweep = False
    if target_age is not None:
        ages, is_sweep = normalize_monitoring_ages(target_age)
    identity_encoder = bundle.get("identity_encoder") or getattr(loss_fn, "identity_encoder", None)
    age_estimator = bundle.get("age_estimator") or getattr(loss_fn, "age_estimator", None)
    diagnostics_enabled = bool(
        compute_diagnostics and identity_encoder is not None and age_estimator is not None
    )
    if is_sweep:
        if target_prompt is not None:
            raise ValueError("target_prompt cannot be combined with a monitoring age sequence")
        epoch_dir = Path(output_dir) / f"epoch_{epoch + 1:03d}"
        sweep = generate_age_sweep(
            bundle=bundle, image=image, ages=ages,
            output_path=epoch_dir / "age_sweep.png",
            annotate_diagnostics=diagnostics_enabled,
            include_source=True,
            source_prompt=source_prompt, source_age=source_age,
            mode=mode, use_inverse_diffusion=use_inverse_diffusion,
            num_inference_steps=num_inference_steps, strength=strength,
            text_guidance_scale=text_guidance_scale,
            text_reference_mode=text_reference_mode,
            age_guidance_scale=age_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=seed, image_size=image_size,
            compute_diagnostics=diagnostics_enabled,
            identity_encoder=identity_encoder,
            age_estimator=age_estimator,
        )
        samples = []
        diagnostic_rows = []
        for age, result in zip(ages, sweep["results"]):
            path = save_inference_image(result, epoch_dir / f"age_{age:03d}.png")
            samples.append({
                "target_age": age,
                "output_path": str(path),
                "target_prompt": result["target_prompt"],
                "start_timestep": result["metadata"]["start_timestep"],
                "diagnostics": result.get("diagnostics"),
            })
            row = _diagnostic_row(epoch=epoch, result=result)
            if row is not None:
                diagnostic_rows.append(row)
        calibration = fit_age_response_calibration(diagnostic_rows)
        directional = compute_directional_age_metrics(diagnostic_rows)
        if calibration is not None:
            for row in diagnostic_rows:
                row.update(calibration)
            print(
                " Age calibration | "
                f"intercept={calibration['age_calibration_intercept']:.4f} | "
                f"slope={calibration['age_calibration_slope']:.4f} | "
                f"R2={calibration['age_calibration_r2']:.4f} | "
                f"score={calibration['age_calibration_score']:.4f}"
            )
        elif compute_diagnostics:
            print(" Age calibration | unavailable (need MiVOLO diagnostics at >=2 distinct ages)")
        if diagnostic_rows:
            formatted = {
                key: "n/a" if value is None else f"{value:.4f}"
                for key, value in directional.items()
            }
            print(
                " Age direction   | "
                f"forward_MAE={formatted['forward_mae']} | "
                f"forward_bias={formatted['forward_bias']} | "
                f"reverse_MAE={formatted['reverse_mae']} | "
                f"reverse_bias={formatted['reverse_bias']}"
            )
        epoch_csv, history_csv = _write_diagnostic_csvs(
            diagnostic_rows, epoch=epoch, epoch_dir=epoch_dir, history_dir=Path(output_dir)
        )
        return {
            "output_dir": str(epoch_dir),
            "grid_path": str(sweep["output_path"]),
            "target_ages": ages,
            "samples": samples,
            "mode": sweep["results"][0]["mode"],
            "seed": int(seed),
            "diagnostics_csv": str(epoch_csv) if epoch_csv else None,
            "diagnostics_history_csv": str(history_csv) if history_csv else None,
            "age_calibration": calibration,
            "age_direction": directional,
        }

    result = infer_face_aging(
        bundle=bundle, image=image,
        target_prompt=target_prompt, target_age=ages[0] if ages else None,
        source_prompt=source_prompt, source_age=source_age,
        mode=mode, use_inverse_diffusion=use_inverse_diffusion,
        num_inference_steps=num_inference_steps, strength=strength,
        text_guidance_scale=text_guidance_scale,
        text_reference_mode=text_reference_mode,
        age_guidance_scale=age_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        seed=seed, image_size=image_size,
        compute_diagnostics=diagnostics_enabled,
        identity_encoder=identity_encoder,
        age_estimator=age_estimator,
    )
    path = save_inference_image(result, Path(output_dir) / f"epoch_{epoch + 1:03d}.png")
    row = _diagnostic_row(epoch=epoch, result=result)
    epoch_csv, history_csv = _write_diagnostic_csvs(
        [row] if row is not None else [],
        epoch=epoch,
        epoch_dir=Path(output_dir),
        history_dir=Path(output_dir),
    )
    return {
        "output_path": str(path), "mode": result["mode"],
        "target_prompt": result["target_prompt"], "target_age": result["target_age"],
        "seed": result["seed"], "start_timestep": result["metadata"]["start_timestep"],
        "diagnostics": result.get("diagnostics"),
        "diagnostics_csv": str(epoch_csv) if epoch_csv else None,
        "diagnostics_history_csv": str(history_csv) if history_csv else None,
    }


def sample_monitoring_images(sample_fn, **kwargs):
    if sample_fn is None:
        return {"status": "NOT RUN", "reason": "sample_fn was not supplied"}
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(sample_fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = kwargs if accepts_var_kwargs else {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    with torch.inference_mode():
        result = sample_fn(**supported)
    return {"status": "PASSED", "result": result}
