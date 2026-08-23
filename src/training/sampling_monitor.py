"""Callback-only deterministic monitoring; no fake inference implementation."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch


def run_face_aging_monitor(
    *, bundle, image, epoch: int, output_dir, target_prompt=None,
    target_age=None, source_prompt=None, source_age=None,
    mode="direct", use_inverse_diffusion=None, num_inference_steps=30,
    strength=0.45, text_guidance_scale=7.0, image_guidance_scale=1.5,
    seed=2026, image_size=256,
):
    """Generate the same fixed diagnostic edit at every requested epoch."""
    from src.inference import infer_face_aging, save_inference_image

    result = infer_face_aging(
        bundle=bundle, image=image,
        target_prompt=target_prompt, target_age=target_age,
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
