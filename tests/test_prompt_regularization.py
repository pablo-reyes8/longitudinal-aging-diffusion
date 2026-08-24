from __future__ import annotations

import torch

from src.training import select_training_prompts


def make_prompt_batch(size: int):
    return {
        "target_prompt": [f"numeric-{index}" for index in range(size)],
        "generic_prompt": ["generic"] * size,
    }


def test_prompt_policy_probability_boundaries_select_expected_text():
    batch = make_prompt_batch(6)
    numeric = select_training_prompts(
        batch, target_prompt_policy="mixed", generic_prompt_prob=0.0,
        numeric_prompt_prob=1.0, random_values=torch.linspace(0, 0.9, 6),
    )
    generic = select_training_prompts(
        batch, target_prompt_policy="mixed", generic_prompt_prob=1.0,
        numeric_prompt_prob=0.0, random_values=torch.linspace(0, 0.9, 6),
    )
    assert numeric["prompts"] == batch["target_prompt"] and numeric["numeric_count"] == 6
    assert generic["prompts"] == batch["generic_prompt"] and generic["generic_count"] == 6


def test_mixed_prompt_frequency_and_fixed_seed_determinism():
    batch = make_prompt_batch(20_000)
    first = select_training_prompts(
        batch, generator=torch.Generator().manual_seed(44),
        target_prompt_policy="mixed", generic_prompt_prob=0.30, numeric_prompt_prob=0.70,
    )
    second = select_training_prompts(
        batch, generator=torch.Generator().manual_seed(44),
        target_prompt_policy="mixed", generic_prompt_prob=0.30, numeric_prompt_prob=0.70,
    )
    assert first["prompts"] == second["prompts"]
    assert torch.equal(first["generic_mask"], second["generic_mask"])
    assert abs(first["generic_fraction"] - 0.30) < 0.01


def test_explicit_numeric_and_generic_policies_ignore_randomness():
    batch = make_prompt_batch(4)
    numeric = select_training_prompts(batch, target_prompt_policy="numeric")
    generic = select_training_prompts(batch, target_prompt_policy="generic")
    assert numeric["prompts"] == batch["target_prompt"]
    assert generic["prompts"] == batch["generic_prompt"]
