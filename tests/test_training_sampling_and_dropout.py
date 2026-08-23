from __future__ import annotations

import torch

from model_fakes import FakeScheduler
from src.training import (
    apply_conditioning_dropout,
    deterministic_validation_timesteps,
    run_training_pipeline_validation,
    sample_conditioning_dropout,
    sample_diffusion_timesteps,
)


def test_uniform_timestep_sampler_100k_histogram_and_exact_range():
    scheduler = FakeScheduler(num_train_timesteps=1000)
    samples = sample_diffusion_timesteps(
        100_000, scheduler, "cpu", torch.Generator().manual_seed(11)
    )
    histogram = torch.bincount(samples // 100, minlength=10).float() / len(samples)
    assert samples.dtype == torch.long and samples.shape == (100_000,)
    assert samples.min() == 0 and samples.max() == 999
    assert torch.all((histogram - 0.1).abs() < 0.004)


def test_restricted_timestep_range_inclusive_and_boundaries_reachable():
    scheduler = FakeScheduler(num_train_timesteps=1000)
    samples = sample_diffusion_timesteps(
        100_000, scheduler, "cpu", torch.Generator().manual_seed(12),
        min_timestep=100, max_timestep=700,
    )
    assert samples.min() == 100 and samples.max() == 700
    assert torch.all((samples >= 100) & (samples <= 700))


def test_deterministic_validation_spans_every_quartile():
    scheduler = FakeScheduler(num_train_timesteps=1000)
    first = deterministic_validation_timesteps(12, scheduler, "cpu", batch_index=0)
    second = deterministic_validation_timesteps(12, scheduler, "cpu", batch_index=0)
    quartiles = first * 4 // 1000
    assert torch.equal(first, second)
    assert set(quartiles.tolist()) == {0, 1, 2, 3}


def test_conditioning_dropout_100k_probabilities_and_zero_case():
    masks = sample_conditioning_dropout(
        100_000, 0.05, generator=torch.Generator().manual_seed(13)
    )
    expected = {"text_only": 0.05, "both": 0.05, "image_only": 0.05, "none": 0.85}
    for name, probability in expected.items():
        assert abs(float(masks[name].float().mean()) - probability) < 0.003
    zero = sample_conditioning_dropout(1000, 0, generator=torch.Generator().manual_seed(14))
    assert zero["none"].all() and not zero["text_dropped"].any() and not zero["image_dropped"].any()


def test_conditioning_dropout_four_states_have_exact_semantics():
    text = torch.arange(24.0).reshape(4, 3, 2)
    source = torch.arange(16.0).reshape(4, 1, 2, 2) + 10
    null = torch.full((1, 3, 2), -7.0)
    # text-only, both, image-only, none for p=.2
    masks = sample_conditioning_dropout(4, 0.2, random_values=torch.tensor([0.1, 0.3, 0.5, 0.9]))
    conditioned_text, conditioned_source = apply_conditioning_dropout(text, source, null, masks)
    assert torch.equal(conditioned_text[0], null[0]) and torch.equal(conditioned_source[0], source[0])
    assert torch.equal(conditioned_text[1], null[0]) and torch.count_nonzero(conditioned_source[1]) == 0
    assert torch.equal(conditioned_text[2], text[2]) and torch.count_nonzero(conditioned_source[2]) == 0
    assert torch.equal(conditioned_text[3], text[3]) and torch.equal(conditioned_source[3], source[3])


def test_structured_training_preflight_is_honest_about_real_smoke():
    report = run_training_pipeline_validation(FakeScheduler())
    assert report["passed"]
    assert report["real_model_smoke"]["status"] == "NOT RUN"
    assert report["real_model_smoke"]["passed"] is None
