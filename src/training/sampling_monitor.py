"""Callback-only deterministic monitoring; no fake inference implementation."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
from pathlib import Path

import torch


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
    *, bundle, image, epoch: int, output_dir, target_prompt=None,
    target_age=None, source_prompt=None, source_age=None,
    mode="direct", use_inverse_diffusion=None, num_inference_steps=30,
    strength=0.45, text_guidance_scale=7.0, image_guidance_scale=1.5,
    seed=2026, image_size=256,
):
    """Generate one edit or an ordered age sweep from the same fixed image."""
    from src.inference import generate_age_sweep, infer_face_aging, save_inference_image

    ages = None
    is_sweep = False
    if target_age is not None:
        ages, is_sweep = normalize_monitoring_ages(target_age)
    if is_sweep:
        if target_prompt is not None:
            raise ValueError("target_prompt cannot be combined with a monitoring age sequence")
        epoch_dir = Path(output_dir) / f"epoch_{epoch + 1:03d}"
        sweep = generate_age_sweep(
            bundle=bundle, image=image, ages=ages,
            output_path=epoch_dir / "age_sweep.png",
            source_prompt=source_prompt, source_age=source_age,
            mode=mode, use_inverse_diffusion=use_inverse_diffusion,
            num_inference_steps=num_inference_steps, strength=strength,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=seed, image_size=image_size,
        )
        samples = []
        for age, result in zip(ages, sweep["results"]):
            path = save_inference_image(result, epoch_dir / f"age_{age:03d}.png")
            samples.append({
                "target_age": age,
                "output_path": str(path),
                "target_prompt": result["target_prompt"],
                "start_timestep": result["metadata"]["start_timestep"],
            })
        return {
            "output_dir": str(epoch_dir),
            "grid_path": str(sweep["output_path"]),
            "target_ages": ages,
            "samples": samples,
            "mode": sweep["results"][0]["mode"],
            "seed": int(seed),
        }

    result = infer_face_aging(
        bundle=bundle, image=image,
        target_prompt=target_prompt, target_age=ages[0] if ages else None,
        source_prompt=source_prompt, source_age=source_age,
        mode=mode, use_inverse_diffusion=use_inverse_diffusion,
        num_inference_steps=num_inference_steps, strength=strength,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        seed=seed, image_size=image_size,
    )
    path = save_inference_image(result, Path(output_dir) / f"epoch_{epoch + 1:03d}.png")
    return {
        "output_path": str(path), "mode": result["mode"],
        "target_prompt": result["target_prompt"], "target_age": result["target_age"],
        "seed": result["seed"], "start_timestep": result["metadata"]["start_timestep"],
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
