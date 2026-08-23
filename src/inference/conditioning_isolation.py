"""In-memory diagnostic that isolates text and numerical age conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from PIL import Image, ImageDraw
import torch

from .checkpoint_loading import load_face_aging_inference_bundle
from .infer_face_aging import infer_face_aging


CONDITIONING_DIAGNOSTIC_COLUMNS = [
    "condition", "source_age", "target_age", "true_delta_age",
    "effective_delta_age", "predicted_source_age", "predicted_generated_age",
    "predicted_delta_age", "delta_age_error", "identity_cosine",
    "target_prompt", "seed", "strength", "num_inference_steps",
    "text_guidance_scale", "image_guidance_scale",
]


_CONDITIONS = (
    ("full", "Full"),
    ("delta_only", "Delta only"),
    ("text_only", "Text only"),
)


def _build_grid(results: Mapping[str, list[dict[str, Any]]], target_ages: list[int]) -> Image.Image:
    first_image = results["full"][0]["image"]
    if not isinstance(first_image, Image.Image):
        raise TypeError("Conditioning diagnostic requires PIL inference output")
    panel_width, image_height = first_image.size
    row_label_width, header_height, annotation_height = 90, 30, 54
    panel_height = image_height + annotation_height
    grid = Image.new(
        "RGB",
        (row_label_width + panel_width * len(target_ages), header_height + panel_height * len(_CONDITIONS)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for column, age in enumerate(target_ages):
        draw.text(
            (row_label_width + column * panel_width + 6, 8),
            f"Target {age}", fill="black",
        )
    for row, (condition, label) in enumerate(_CONDITIONS):
        row_y = header_height + row * panel_height
        draw.text((6, row_y + image_height // 2), label, fill="black")
        for column, result in enumerate(results[condition]):
            image = result["image"]
            diagnostics = result["diagnostics"]
            x = row_label_width + column * panel_width
            grid.paste(image, (x, row_y))
            annotation = (
                f"Pred: {diagnostics['predicted_generated_age']:.1f}\n"
                f"Delta pred: {diagnostics['predicted_delta_age']:.1f}\n"
                f"ID: {diagnostics['identity_cosine_source_generated']:.3f}"
            )
            draw.multiline_text((x + 5, row_y + image_height + 3), annotation, fill="black", spacing=1)
    return grid


def _print_diagnostic(frame: pd.DataFrame, *, source_age: int, target_ages: list[int]) -> None:
    print("\n" + "=" * 112)
    print(" CONDITIONING ISOLATION DIAGNOSTIC")
    print("=" * 112)
    print(
        f" Source age: {source_age}  |  Targets: {target_ages}  |  "
        "Conditions: full / delta only / text only"
    )
    print("-" * 112)
    view = frame[[
        "condition", "target_age", "true_delta_age", "effective_delta_age",
        "predicted_source_age", "predicted_generated_age", "predicted_delta_age",
        "delta_age_error", "identity_cosine",
    ]].copy()
    print(view.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    print("-" * 112)
    print(" Prompt policy:")
    print("   full       = numeric CLIP prompt + real delta")
    print("   delta_only = generic CLIP prompt + real delta")
    print("   text_only  = numeric CLIP prompt + forced delta 0")
    print("=" * 112 + "\n")


def diagnose_conditioning_sources(
    bundle,
    source_image,
    source_age: int = 26,
    target_ages: Iterable[int] = (30, 40, 65),
    *,
    use_inverse_diffusion: bool = False,
    num_inference_steps: int = 30,
    strength: float = 0.35,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    seed: int = 2026,
    image_size: int = 256,
    display_result: bool = True,
) -> dict[str, Any]:
    """Run the fixed 3x3 diagnostic, print it, and keep every artifact in memory."""
    ages = [int(age) for age in target_ages]
    if not ages:
        raise ValueError("target_ages must not be empty")
    if bundle.get("age_delta_conditioner") is None:
        raise ValueError("The bundle has no age-delta conditioner to isolate")
    if bundle.get("identity_encoder") is None or bundle.get("age_estimator") is None:
        raise ValueError("Identity and age auxiliary models are required for this diagnostic")

    results: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _CONDITIONS}
    rows = []
    for condition, _ in _CONDITIONS:
        for target_age in ages:
            numeric_prompt = f"photo of a person as {target_age}-year-old"
            target_prompt = "photo of a person" if condition == "delta_only" else numeric_prompt
            override_delta = 0.0 if condition == "text_only" else None
            result = infer_face_aging(
                bundle=bundle,
                image=source_image,
                source_age=int(source_age),
                target_age=target_age,
                target_prompt=target_prompt,
                use_inverse_diffusion=use_inverse_diffusion,
                num_inference_steps=num_inference_steps,
                strength=strength,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                seed=seed,
                image_size=image_size,
                output_type="pil",
                compute_diagnostics=True,
                override_delta_age=override_delta,
            )
            diagnostics = result["diagnostics"]
            if diagnostics is None:
                raise RuntimeError("Auxiliary diagnostics unexpectedly returned None")
            effective_delta = float(result["metadata"]["delta_age"])
            results[condition].append(result)
            rows.append({
                "condition": condition,
                "source_age": int(source_age),
                "target_age": target_age,
                "true_delta_age": float(target_age - source_age),
                "effective_delta_age": effective_delta,
                "predicted_source_age": diagnostics["predicted_source_age"],
                "predicted_generated_age": diagnostics["predicted_generated_age"],
                "predicted_delta_age": diagnostics["predicted_delta_age"],
                "delta_age_error": diagnostics["delta_age_error"],
                "identity_cosine": diagnostics["identity_cosine_source_generated"],
                "target_prompt": result["target_prompt"],
                "seed": int(seed),
                "strength": float(strength),
                "num_inference_steps": int(num_inference_steps),
                "text_guidance_scale": float(text_guidance_scale),
                "image_guidance_scale": float(image_guidance_scale),
            })
    frame = pd.DataFrame(rows, columns=CONDITIONING_DIAGNOSTIC_COLUMNS)
    grid = _build_grid(results, ages)
    _print_diagnostic(frame, source_age=int(source_age), target_ages=ages)
    if display_result:
        try:
            from IPython.display import display
        except ImportError:
            print("IPython is unavailable; inspect result['grid'] to view the image grid.")
        else:
            display(grid)
    return {"dataframe": frame, "grid": grid, "results": results}


def diagnose_checkpoint_conditioning_sources(
    checkpoint_path: str | Path,
    source_image,
    *,
    source_age: int = 26,
    target_ages: Iterable[int] = (30, 40, 65),
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    auxiliary_trust_remote_code: bool = True,
    **diagnostic_kwargs,
) -> dict[str, Any]:
    """Load a checkpoint and run the in-memory diagnostic from only its path and an image."""
    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if dtype is None:
        dtype = (
            torch.bfloat16 if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if resolved_device.type == "cuda"
            else torch.float32
        )
    bundle = load_face_aging_inference_bundle(
        checkpoint_path,
        device=resolved_device,
        dtype=dtype,
        local_files_only=local_files_only,
        token=token,
        load_auxiliary_models=True,
        auxiliary_dtype=torch.float32,
        auxiliary_trust_remote_code=auxiliary_trust_remote_code,
    )
    output = diagnose_conditioning_sources(
        bundle,
        source_image,
        source_age=source_age,
        target_ages=target_ages,
        **diagnostic_kwargs,
    )
    output["bundle"] = bundle
    return output
