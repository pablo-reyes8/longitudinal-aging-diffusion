from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from src.inference import (
    combine_referenced_text_cfg,
    combine_referenced_three_way_cfg,
    combine_three_way_cfg,
    create_inference_scheduler,
    ddim_forward_step,
    ddim_invert_source_image,
    edit_from_inverted_latent,
    model_output_to_x0_epsilon,
    predict_three_way_cfg,
)
from src.model import build_conditioned_unet_input, encode_prompts
from training_fakes import make_training_bundle


def test_guidance_formula_and_three_critical_scale_sanities():
    full = torch.tensor([[[[4.0]]]])
    image = torch.tensor([[[[2.0]]]])
    uncond = torch.tensor([[[[-1.0]]]])
    observed = combine_three_way_cfg(full, image, uncond, text_guidance_scale=7, image_guidance_scale=1.5)
    oracle = uncond + 1.5 * (image - uncond) + 7 * (full - image)
    assert torch.equal(observed, oracle)
    assert torch.equal(combine_three_way_cfg(full, image, uncond, text_guidance_scale=0, image_guidance_scale=0), uncond)
    assert torch.equal(combine_three_way_cfg(full, image, uncond, text_guidance_scale=0, image_guidance_scale=1), image)
    assert torch.equal(combine_three_way_cfg(full, image, uncond, text_guidance_scale=1, image_guidance_scale=1), full)


def test_referenced_cfg_critical_invariants_and_legacy_equivalence():
    target = torch.tensor([[[[5.0]]]])
    reference = torch.tensor([[[[2.0]]]])
    reference_no_image = torch.tensor([[[[-1.0]]]])
    assert torch.equal(
        combine_referenced_text_cfg(target, reference, age_guidance_scale=0), reference
    )
    assert torch.equal(
        combine_referenced_text_cfg(reference, reference, age_guidance_scale=9), reference
    )
    observed = combine_referenced_three_way_cfg(
        target, reference, reference_no_image,
        age_guidance_scale=3, image_guidance_scale=1.5,
    )
    oracle = reference + 3 * (target - reference) + 0.5 * (reference - reference_no_image)
    assert torch.equal(observed, oracle)
    assert torch.equal(
        combine_referenced_three_way_cfg(
            target, reference, reference_no_image,
            age_guidance_scale=7, image_guidance_scale=1.5,
        ),
        combine_three_way_cfg(
            target, reference, reference_no_image,
            text_guidance_scale=7, image_guidance_scale=1.5,
        ),
    )


def test_batched_cfg_branches_match_three_slow_independent_forwards():
    bundle = make_training_bundle()
    bundle["unet"].eval()
    torch.manual_seed(19)
    target, source = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    full_embeddings = encode_prompts(bundle, ["age 40", "age 70"])
    null = encode_prompts(bundle, [""]).expand(2, -1, -1)
    guided, branches = predict_three_way_cfg(
        bundle=bundle, target_latents=target, timestep=torch.tensor(30),
        source_latents=source, full_text_embeddings=full_embeddings,
        null_text_embeddings=null, return_branches=True,
    )
    slow_full = bundle["unet"](build_conditioned_unet_input(target, source), torch.tensor(30), encoder_hidden_states=full_embeddings, return_dict=True).sample
    slow_image = bundle["unet"](build_conditioned_unet_input(target, source), torch.tensor(30), encoder_hidden_states=null, return_dict=True).sample
    slow_uncond = bundle["unet"](build_conditioned_unet_input(target, torch.zeros_like(source)), torch.tensor(30), encoder_hidden_states=null, return_dict=True).sample
    assert torch.allclose(branches["full"], slow_full, atol=2e-7)
    assert torch.allclose(branches["image"], slow_image, atol=2e-7)
    assert torch.allclose(branches["uncond"], slow_uncond, atol=2e-7)
    oracle = combine_three_way_cfg(slow_full, slow_image, slow_uncond)
    assert torch.allclose(guided, oracle, atol=5e-7)


def test_referenced_cfg_uses_supplied_reference_embedding_and_changes_prediction():
    bundle = make_training_bundle(seed=512)
    bundle["unet"].eval()
    target, source = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    target_embeddings = encode_prompts(bundle, ["age 40", "age 70"])
    source_reference = encode_prompts(bundle, ["age 26", "age 26"])
    generic_reference = encode_prompts(bundle, ["photo of a person", "photo of a person"])
    null = encode_prompts(bundle, [""])
    source_guided, branches = predict_three_way_cfg(
        bundle=bundle, target_latents=target, timestep=torch.tensor(30),
        source_latents=source, full_text_embeddings=target_embeddings,
        reference_text_embeddings=source_reference, null_text_embeddings=null,
        age_guidance_scale=3.0, return_branches=True,
    )
    generic_guided = predict_three_way_cfg(
        bundle=bundle, target_latents=target, timestep=torch.tensor(30),
        source_latents=source, full_text_embeddings=target_embeddings,
        reference_text_embeddings=generic_reference, null_text_embeddings=null,
        age_guidance_scale=3.0,
    )
    slow_reference = bundle["unet"](
        build_conditioned_unet_input(target, source), torch.tensor(30),
        encoder_hidden_states=source_reference, return_dict=True,
    ).sample
    assert torch.allclose(branches["reference"], slow_reference, atol=2e-7)
    assert source_guided.shape == target.shape and torch.isfinite(source_guided).all()
    assert not torch.equal(source_guided, generic_guided)


def test_cfg_uses_exactly_one_unet_forward_with_three_times_the_batch():
    bundle = make_training_bundle()
    original_forward = bundle["unet"].forward
    observed_batches = []

    def counting_forward(self, sample, *args, **kwargs):
        observed_batches.append(sample.shape[0])
        return original_forward(sample, *args, **kwargs)

    bundle["unet"].forward = MethodType(counting_forward, bundle["unet"])
    target = torch.randn(2, 4, 4, 4)
    source = torch.randn_like(target)
    full = encode_prompts(bundle, ["age 40", "age 70"])
    null = encode_prompts(bundle, [""])
    prediction = predict_three_way_cfg(
        bundle=bundle,
        target_latents=target,
        timestep=torch.tensor(30),
        source_latents=source,
        full_text_embeddings=full,
        null_text_embeddings=null,
    )
    assert prediction.shape == target.shape
    assert observed_batches == [6]


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_ddim_forward_formula_recovers_known_next_state(prediction_type):
    bundle = make_training_bundle()
    scheduler = bundle["scheduler_infer"]
    scheduler.config.prediction_type = prediction_type
    torch.manual_seed(20)
    x0, epsilon = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    current, following = 10, 70
    alpha_current = scheduler.alphas_cumprod[current]
    alpha_following = scheduler.alphas_cumprod[following]
    sample = alpha_current.sqrt() * x0 + (1 - alpha_current).sqrt() * epsilon
    output = epsilon if prediction_type == "epsilon" else alpha_current.sqrt() * epsilon - (1 - alpha_current).sqrt() * x0
    recovered_x0, recovered_epsilon = model_output_to_x0_epsilon(output, sample, current, scheduler)
    next_sample = ddim_forward_step(sample, output, current, following, scheduler)
    expected = alpha_following.sqrt() * x0 + (1 - alpha_following).sqrt() * epsilon
    assert torch.allclose(recovered_x0, x0, atol=2e-6)
    assert torch.allclose(recovered_epsilon, epsilon, atol=2e-6)
    assert torch.allclose(next_sample, expected, atol=2e-6)


def test_zero_prediction_inversion_then_reverse_reconstructs_latent():
    bundle = make_training_bundle()
    def zero_forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
        return SimpleNamespace(sample=torch.zeros(sample.shape[0], 4, *sample.shape[2:], device=sample.device, dtype=sample.dtype))
    bundle["unet"].forward = MethodType(zero_forward, bundle["unet"])
    scheduler = create_inference_scheduler(bundle)
    source = torch.randn(1, 4, 4, 4) * 0.2
    inverted = ddim_invert_source_image(
        bundle=bundle, source_latents=source, source_prompt="photo of a person",
        scheduler=scheduler, num_inference_steps=10, use_cfg=False,
    )
    reconstructed = edit_from_inverted_latent(
        bundle=bundle, inverted_latents=inverted["inverted_latents"],
        source_latents=source, target_prompt="photo of a person", scheduler=scheduler,
        denoising_timesteps=inverted["denoising_timesteps"], use_cfg=False,
    )["latents"]
    assert torch.allclose(reconstructed, source, atol=2e-4, rtol=2e-4)
