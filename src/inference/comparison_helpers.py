"""Thin diagnostic helpers built exclusively on the public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
    return (
        f"Target: {diagnostics['target_age']:.0f}\n"
        f"Pred: {diagnostics['predicted_generated_age']:.1f}\n"
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
        images = [source, *images]
        labels = [source_label, *labels]
    grid = _labeled_grid(images, labels, labels_below=annotate_diagnostics)
    saved = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        grid.save(saved)
    return {"ages": ordered_ages, "results": results, "grid": grid, "output_path": saved}
