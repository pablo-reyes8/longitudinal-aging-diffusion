"""Deterministic per-sample prompt regularization for age conditioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import math
import torch


def validate_prompt_policy(
    target_prompt_policy: str,
    generic_prompt_prob: float,
    numeric_prompt_prob: float,
) -> None:
    if target_prompt_policy not in {"numeric", "generic", "mixed"}:
        raise ValueError("target_prompt_policy must be 'numeric', 'generic', or 'mixed'")
    if not 0 <= generic_prompt_prob <= 1 or not 0 <= numeric_prompt_prob <= 1:
        raise ValueError("Prompt probabilities must be in [0,1]")
    if not math.isclose(generic_prompt_prob + numeric_prompt_prob, 1.0, abs_tol=1e-8):
        raise ValueError("generic_prompt_prob and numeric_prompt_prob must sum to 1")


def select_training_prompts(
    batch: Mapping[str, Any],
    *,
    target_prompt_policy: str = "mixed",
    generic_prompt_prob: float = 0.30,
    numeric_prompt_prob: float = 0.70,
    generator: torch.Generator | None = None,
    random_values: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Select numeric or generic text independently for every batch sample."""
    validate_prompt_policy(target_prompt_policy, generic_prompt_prob, numeric_prompt_prob)
    numeric_prompts = list(batch["target_prompt"])
    generic_prompts = list(batch["generic_prompt"])
    if len(numeric_prompts) != len(generic_prompts):
        raise ValueError("target_prompt and generic_prompt batch sizes must match")
    batch_size = len(numeric_prompts)
    if target_prompt_policy == "numeric":
        generic_mask = torch.zeros(batch_size, dtype=torch.bool)
    elif target_prompt_policy == "generic":
        generic_mask = torch.ones(batch_size, dtype=torch.bool)
    else:
        if random_values is None:
            generator_device = torch.device(getattr(generator, "device", "cpu")) if generator is not None else torch.device("cpu")
            random_values = torch.rand(batch_size, generator=generator, device=generator_device)
        values = torch.as_tensor(random_values).flatten().cpu()
        if values.shape != (batch_size,):
            raise ValueError(f"prompt random_values must have shape [{batch_size}]")
        if not torch.isfinite(values).all() or bool(((values < 0) | (values >= 1)).any()):
            raise ValueError("prompt random_values must be finite values in [0,1)")
        generic_mask = values < float(generic_prompt_prob)
    selected = [
        generic_prompts[index] if bool(generic_mask[index]) else numeric_prompts[index]
        for index in range(batch_size)
    ]
    generic_count = int(generic_mask.sum())
    return {
        "prompts": selected,
        "generic_mask": generic_mask,
        "numeric_mask": ~generic_mask,
        "generic_count": generic_count,
        "numeric_count": batch_size - generic_count,
        "generic_fraction": generic_count / batch_size if batch_size else 0.0,
        "numeric_fraction": (batch_size - generic_count) / batch_size if batch_size else 0.0,
        "policy": target_prompt_policy,
    }
