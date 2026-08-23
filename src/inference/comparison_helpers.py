"""Thin diagnostic helpers built exclusively on the public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .infer_face_aging import infer_face_aging
from .inference_utils import prepare_inference_image, tensor_to_pil


def _labeled_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    header = 28
    grid = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        grid.paste(image, (index * width, header))
        draw.text((index * width + 6, 6), label, fill="black")
    return grid


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


def generate_age_sweep(*, bundle, image, ages: Iterable[int], output_path: str | Path | None = None, **kwargs):
    ordered_ages = [int(age) for age in ages]
    if not ordered_ages:
        raise ValueError("ages must not be empty")
    results = [infer_face_aging(bundle=bundle, image=image, target_age=age, **kwargs) for age in ordered_ages]
    grid = _labeled_grid([result["image"] for result in results], [str(age) for age in ordered_ages])
    saved = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        grid.save(saved)
    return {"ages": ordered_ages, "results": results, "grid": grid, "output_path": saved}
