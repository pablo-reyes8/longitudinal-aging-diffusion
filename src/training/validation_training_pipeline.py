"""Structured preflight report for the training pipeline."""

from __future__ import annotations

from typing import Any

import torch

from .conditioning_dropout import sample_conditioning_dropout
from .timestep_sampling import sample_diffusion_timesteps, scheduler_num_train_timesteps


def run_training_pipeline_validation(
    scheduler,
    *,
    sample_count: int = 100_000,
    conditioning_dropout_prob: float = 0.05,
    smoke_fn=None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    generator = torch.Generator().manual_seed(8801)
    timesteps = sample_diffusion_timesteps(sample_count, scheduler, "cpu", generator)
    total = scheduler_num_train_timesteps(scheduler)
    quartiles = torch.clamp(timesteps * 4 // total, max=3)
    fractions = [float((quartiles == index).float().mean()) for index in range(4)]
    timestep_passed = all(abs(fraction - 0.25) < 0.01 for fraction in fractions)
    if not timestep_passed:
        errors.append(f"Uniform timestep quartiles failed: {fractions}")
    masks = sample_conditioning_dropout(
        sample_count, conditioning_dropout_prob,
        generator=torch.Generator().manual_seed(8802),
    )
    dropout_observed = {
        name: float(masks[name].float().mean())
        for name in ("text_only", "both", "image_only", "none")
    }
    expected = {
        "text_only": conditioning_dropout_prob,
        "both": conditioning_dropout_prob,
        "image_only": conditioning_dropout_prob,
        "none": 1 - 3 * conditioning_dropout_prob,
    }
    dropout_passed = all(abs(dropout_observed[name] - expected[name]) < 0.01 for name in expected)
    if not dropout_passed:
        errors.append(f"Conditioning dropout distribution failed: {dropout_observed}")
    smoke = {"status": "NOT RUN", "passed": None}
    if smoke_fn is not None:
        try:
            smoke_result = smoke_fn()
            smoke = {"status": "PASSED", "passed": True, "result": smoke_result}
        except Exception as exc:
            smoke = {"status": "FAILED", "passed": False, "error": repr(exc)}
            errors.append(f"Training smoke failed: {exc}")
    else:
        warnings.append("Real-model/GPU smoke is NOT RUN until a smoke_fn is supplied on the training server")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "step_mechanics": {"status": "covered by strict pytest suite"},
        "timestep_sampling": {"passed": timestep_passed, "quartile_fractions": fractions, "samples": sample_count},
        "conditioning_dropout": {"passed": dropout_passed, "observed": dropout_observed, "expected": expected},
        "gradient_accumulation": {"status": "covered by strict pytest suite"},
        "optimizer": {"status": "covered by strict pytest suite"},
        "lr_scheduler": {"status": "covered by strict pytest suite"},
        "mixed_precision": {"status": "covered by strict pytest suite"},
        "checkpointing": {"status": "covered by strict pytest suite"},
        "resume": {"status": "covered by strict pytest suite"},
        "validation": {"status": "covered by strict pytest suite"},
        "memory": {"status": "NOT RUN" if not torch.cuda.is_available() else "available during real smoke"},
        "real_model_smoke": smoke,
    }
