"""Thin model-agnostic inference-to-evaluation bridge."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image

from .evaluator import evaluate_aging_outputs


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def _as_image(result) -> Image.Image:
    if isinstance(result, dict):
        result = result.get("image")
    if isinstance(result, Image.Image):
        return result.convert("RGB")
    if isinstance(result, (str, Path)):
        return Image.open(result).convert("RGB")
    raise TypeError("infer_fn must return a PIL image, image path, or {'image': ...}")


def run_inference_and_evaluate(
    infer_fn,
    samples,
    inference_config,
    evaluation_config,
    output_dir,
    *,
    model_name="model",
):
    """Generate a fixed panel, write its manifest, then run the independent evaluator."""
    frame = samples.copy() if isinstance(samples, pd.DataFrame) else pd.DataFrame(samples)
    required = {"sample_id", "source_path", "target_path", "source_age", "target_age"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"samples is missing required columns: {', '.join(missing)}")

    output = Path(output_dir)
    generated_dir = output / "generated" / _safe_name(model_name)
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for sample in frame.to_dict(orient="records"):
        generated_path = None
        inference_error = None
        try:
            generated = _as_image(
                infer_fn(
                    source_path=sample["source_path"],
                    source_age=sample["source_age"],
                    target_age=sample["target_age"],
                    **dict(inference_config or {}),
                )
            )
            generated_path = generated_dir / f"{_safe_name(sample['sample_id'])}.png"
            generated.save(generated_path)
        except Exception as error:
            inference_error = str(error)
        manifest_rows.append(
            {
                **sample,
                "generated_path": None if generated_path is None else str(generated_path),
                "model_name": sample.get("model_name", model_name),
                "inference_error": inference_error,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "prediction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    options = dict(evaluation_config or {})
    metrics_bundle = options.pop("metrics_bundle", None)
    if metrics_bundle is None:
        raise ValueError("evaluation_config must include metrics_bundle")
    options.setdefault("output_dir", output / "evaluation")
    evaluation = evaluate_aging_outputs(manifest, metrics_bundle=metrics_bundle, **options)
    return {"manifest": manifest, "manifest_path": manifest_path, "evaluation": evaluation}
