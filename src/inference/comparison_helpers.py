"""Thin diagnostic helpers built exclusively on the public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from .infer_face_aging import infer_face_aging
from .inference_utils import prepare_inference_image, tensor_to_pil


def _labeled_grid(
    images: list[Image.Image], labels: list[str], *, labels_below: bool = False
) -> Image.Image:
    width, height = images[0].size
    label_height = max(28, max(label.count("\n") + 1 for label in labels) * 14 + 8)
    grid = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        image_y = 0 if labels_below else label_height
        label_y = height + 4 if labels_below else 5
        grid.paste(image, (index * width, image_y))
        draw.multiline_text((index * width + 5, label_y), label, fill="black", spacing=1)
    return grid


def _age_diagnostic_label(result) -> str:
    diagnostics = result.get("diagnostics")
    if diagnostics is None:
        return f"Target: {result['target_age']}"
    delta_label = (
        f"Delta: {diagnostics['predicted_delta_age']:.1f}\n"
        if diagnostics.get("predicted_delta_age") is not None else ""
    )
    strength_label = (
        f"Strength: {result['strength']:.3f}\n"
        if result.get("strength") is not None else ""
    )
    return (
        f"Target: {diagnostics['target_age']:.0f}\n"
        f"Pred: {diagnostics['predicted_generated_age']:.1f}\n"
        f"{delta_label}"
        f"{strength_label}"
        f"ID: {diagnostics['identity_cosine_source_generated']:.2f}"
    )


def compare_inference_modes(*, bundle, image, output_path: str | Path | None = None, **kwargs):
    direct = infer_face_aging(bundle=bundle, image=image, mode="direct", **kwargs)
    inverse = infer_face_aging(bundle=bundle, image=image, mode="inverse", **kwargs)
    source = tensor_to_pil(prepare_inference_image(image, image_size=kwargs.get("image_size", 256)).div(2).add(0.5))
    grid = _labeled_grid([source, direct["image"], inverse["image"]], ["source", "direct", "inverse"])
    saved = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        grid.save(saved)
    return {"grid": grid, "source": source, "direct": direct, "inverse": inverse, "output_path": saved}


def generate_age_sweep(
    *, bundle, image, ages: Iterable[int], output_path: str | Path | None = None,
    annotate_diagnostics: bool = False, include_source: bool = False, **kwargs,
):
    ordered_ages = [int(age) for age in ages]
    if not ordered_ages:
        raise ValueError("ages must not be empty")
    if annotate_diagnostics:
        kwargs["compute_diagnostics"] = True
    results = [infer_face_aging(bundle=bundle, image=image, target_age=age, **kwargs) for age in ordered_ages]
    generated_labels = (
        [_age_diagnostic_label(result) for result in results]
        if annotate_diagnostics else [str(age) for age in ordered_ages]
    )
    images = [result["image"] for result in results]
    labels = generated_labels
    if include_source:
        source = tensor_to_pil(
            prepare_inference_image(
                image, image_size=kwargs.get("image_size", 256)
            ).div(2).add(0.5)
        )
        source_age = kwargs.get("source_age")
        source_label = "Original" + (f"\nAge: {source_age}" if source_age is not None else "")
        if source_age is None:
            images = [source, *images]
            labels = [source_label, *labels]
        else:
            chronological = sorted(
                zip(ordered_ages, images, labels), key=lambda item: item[0]
            )
            source_position = sum(age < float(source_age) for age, _, _ in chronological)
            ordered_images = [image for _, image, _ in chronological]
            ordered_labels = [label for _, _, label in chronological]
            images = [
                *ordered_images[:source_position], source,
                *ordered_images[source_position:],
            ]
            labels = [
                *ordered_labels[:source_position], source_label,
                *ordered_labels[source_position:],
            ]
    grid = _labeled_grid(images, labels, labels_below=annotate_diagnostics)
    saved = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        grid.save(saved)
    return {"ages": ordered_ages, "results": results, "grid": grid, "output_path": saved}


def generate_strength_age_sweep(
    *,
    bundle,
    image,
    ages: Iterable[int],
    strengths: Sequence[float],
    output_path: str | Path | None = None,
    annotate_diagnostics: bool = True,
    include_source: bool = True,
    precomputed_sweeps: dict[float, dict] | None = None,
    **kwargs,
):
    """Render several fixed-strength age sweeps into one lossless comparison grid."""
    values = [float(value) for value in strengths]
    ordered_ages = [int(age) for age in ages]
    if not ordered_ages:
        raise ValueError("ages must not be empty")
    if not values or any(not 0 < value <= 1 for value in values):
        raise ValueError("strengths must be a non-empty sequence with values in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("strengths must be unique")
    kwargs.pop("strength", None)
    kwargs["use_delta_dependent_strength"] = False
    cached = precomputed_sweeps or {}
    sweeps = []
    for value in values:
        sweep = cached.get(value)
        if sweep is None:
            sweep = generate_age_sweep(
                bundle=bundle,
                image=image,
                ages=ordered_ages,
                output_path=None,
                annotate_diagnostics=annotate_diagnostics,
                include_source=include_source,
                strength=value,
                **kwargs,
            )
        sweeps.append(sweep)
    label_width = 128
    row_gap = 8
    width = label_width + max(sweep["grid"].width for sweep in sweeps)
    height = sum(sweep["grid"].height for sweep in sweeps) + row_gap * (len(sweeps) - 1)
    combined = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(combined)
    y = 0
    for value, sweep in zip(values, sweeps):
        grid = sweep["grid"]
        draw.text((8, y + 12), f"Strength\n{value:.3f}", fill="black", spacing=4)
        combined.paste(grid, (label_width, y))
        y += grid.height + row_gap
    saved = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        # PNG preserves native 256px cells for detailed notebook zooming.
        combined.save(saved, format="PNG", optimize=False)
    return {
        "strengths": values,
        "sweeps": sweeps,
        "grid": combined,
        "output_path": saved,
    }
