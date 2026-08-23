"""Structured numerical and optional end-to-end inference preflight."""

from __future__ import annotations

from typing import Any

import torch

from .cfg_guidance import combine_three_way_cfg
from .prompt_building import build_inference_prompt_pack


def run_inference_pipeline_validation(*, smoke_fn=None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    full = torch.tensor([1.0, 2.0, 3.0])
    image = torch.tensor([0.5, 1.0, 1.5])
    uncond = torch.tensor([-1.0, 0.0, 1.0])
    observed = combine_three_way_cfg(
        full, image, uncond, text_guidance_scale=7, image_guidance_scale=1.5
    )
    expected = uncond + 1.5 * (image - uncond) + 7 * (full - image)
    cfg_error = float((observed - expected).abs().max())
    if cfg_error != 0:
        errors.append(f"CFG analytical oracle failed: max_error={cfg_error}")
    prompt_selfage = build_inference_prompt_pack(target_age=65, prompt_style="selfage")
    prompt_fading = build_inference_prompt_pack(target_age=65, prompt_style="fading")
    prompt_passed = (
        prompt_selfage["target_prompt"] == "photo of a person as 65-year-old"
        and prompt_fading["target_prompt"] == "photo of a 65 year old person"
    )
    if not prompt_passed:
        errors.append("Prompt construction differs from the training contract")
    smoke = {"status": "NOT RUN", "passed": None}
    if smoke_fn is not None:
        try:
            result = smoke_fn()
            smoke = {"status": "PASSED", "passed": True, "result": result}
        except Exception as exc:
            smoke = {"status": "FAILED", "passed": False, "error": repr(exc)}
            errors.append(f"Inference smoke failed: {exc}")
    else:
        warnings.append("Real SD1.5/GPU inference is NOT RUN until a server smoke_fn is supplied")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "cfg_guidance": {"passed": cfg_error == 0, "max_absolute_error": cfg_error},
        "prompt_building": {"passed": prompt_passed, "selfage": prompt_selfage, "fading": prompt_fading},
        "preprocessing": {"status": "covered by strict pytest suite"},
        "vae_scaling": {"status": "covered by strict pytest suite"},
        "direct": {"status": "covered by strict pytest suite"},
        "inverse": {"status": "covered by strict pytest suite"},
        "checkpoint_loading": {"status": "covered by strict pytest suite"},
        "real_model_smoke": smoke,
    }
