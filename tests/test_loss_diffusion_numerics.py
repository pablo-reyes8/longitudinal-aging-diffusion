from __future__ import annotations

import math
import random

import pytest
import torch

from loss_fakes import NumericalScheduler
from loss_reference import (
    reference_coefficients,
    reference_min_snr_weights,
    reference_per_sample_mse,
    reference_velocity,
    reference_x0,
)
from src.loss import (
    compute_diffusion_loss,
    compute_min_snr_weights,
    extract_scheduler_coefficients,
    get_diffusion_target,
    predict_x0_from_model_output,
)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_perfect_prediction_zero_and_exact_x0_full_timestep_range(prediction_type):
    scheduler = NumericalScheduler(prediction_type)
    torch.manual_seed(101)
    x0, noise = torch.randn(8, 3, 4, 5, dtype=torch.float64), torch.randn(8, 3, 4, 5, dtype=torch.float64)
    timesteps = torch.tensor([0, 1, 10, 100, 500, 998, 999, 377])
    alpha, sigma = reference_coefficients(scheduler, timesteps, x0.ndim, x0.dtype)
    noisy = alpha * x0 + sigma * noise
    exact = noise if prediction_type == "epsilon" else reference_velocity(x0, noise, alpha, sigma)
    production_target = get_diffusion_target(scheduler, x0, noise, timesteps)
    loss, per_sample, _ = compute_diffusion_loss(exact, production_target, scheduler=scheduler, timesteps=timesteps)
    recovered = predict_x0_from_model_output(exact, noisy, timesteps, scheduler)
    oracle_recovered = reference_x0(exact, noisy, alpha, sigma, prediction_type)
    assert loss.item() == 0.0
    assert torch.count_nonzero(per_sample) == 0
    assert (recovered - x0).abs().max() < 2e-12
    assert (recovered - x0).norm() / x0.norm() < 2e-13
    assert torch.equal(recovered, oracle_recovered)


def test_per_sample_broadcast_matches_python_loop():
    scheduler = NumericalScheduler("epsilon")
    torch.manual_seed(4)
    x0, noise = torch.randn(4, 2, 3, 5, dtype=torch.float64), torch.randn(4, 2, 3, 5, dtype=torch.float64)
    timesteps = torch.tensor([0, 37, 401, 999])
    alpha, sigma = reference_coefficients(scheduler, timesteps, 4, torch.float64)
    noisy = alpha * x0 + sigma * noise
    vectorized = predict_x0_from_model_output(noise, noisy, timesteps, scheduler)
    independent = []
    for index, timestep in enumerate(timesteps.tolist()):
        alpha_i = scheduler.alphas_cumprod[timestep].double()
        independent.append((noisy[index] - (1 - alpha_i).sqrt() * noise[index]) / alpha_i.sqrt())
    assert torch.allclose(vectorized, torch.stack(independent), atol=1e-12, rtol=1e-12)


def test_coefficient_extraction_shape_device_dtype_range_and_values():
    scheduler = NumericalScheduler()
    reference = torch.randn(4, 2, 3, 3, dtype=torch.float64)
    timesteps = torch.tensor([0, 13, 587, 999])
    values = extract_scheduler_coefficients(scheduler, timesteps, reference)
    expected = scheduler.alphas_cumprod[timesteps]
    assert values["alpha_bar"].shape == (4,)
    assert values["alpha_bar_broadcast"].shape == (4, 1, 1, 1)
    assert values["alpha_bar"].dtype == torch.float64
    assert values["alpha_bar"].device == reference.device
    assert torch.equal(values["alpha_bar"], expected)
    assert torch.isfinite(values["alpha_bar"]).all()
    assert ((values["alpha_bar"] >= 0) & (values["alpha_bar"] <= 1)).all()
    assert torch.all(values["alpha_bar"][:-1] >= values["alpha_bar"][1:])


def test_scheduler_add_noise_against_independent_manual_formula():
    scheduler = NumericalScheduler()
    x0, noise = torch.randn(5, 2, 3, 4, dtype=torch.float64), torch.randn(5, 2, 3, 4, dtype=torch.float64)
    timesteps = torch.tensor([0, 1, 99, 501, 999])
    alpha, sigma = reference_coefficients(scheduler, timesteps, 4, torch.float64)
    assert torch.allclose(scheduler.add_noise(x0, noise, timesteps), alpha * x0 + sigma * noise, atol=1e-14)


def test_exact_mse_reduction_and_batch_aggregation():
    scheduler = NumericalScheduler()
    pred = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 1, 2, 2)
    target = torch.zeros_like(pred)
    loss, per_sample, _ = compute_diffusion_loss(pred, target, scheduler=scheduler, timesteps=torch.tensor([2]))
    assert loss.item() == 3.5 and per_sample.tolist() == [3.5]
    pred = torch.cat([pred, 2 * pred])
    target = torch.zeros_like(pred)
    loss, per_sample, _ = compute_diffusion_loss(pred, target, scheduler=scheduler, timesteps=torch.tensor([2, 7]))
    assert per_sample.tolist() == [3.5, 14.0]
    assert loss.item() == 8.75


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_min_snr_disabled_exactly_matches_plain_mse(prediction_type):
    scheduler = NumericalScheduler(prediction_type)
    pred, target = torch.randn(4, 2, 3, 3), torch.randn(4, 2, 3, 3)
    timesteps = torch.tensor([1, 50, 500, 999])
    loss, per_sample, weights = compute_diffusion_loss(pred, target, scheduler=scheduler, timesteps=timesteps, min_snr_gamma=None)
    oracle = reference_per_sample_mse(pred, target).float()
    assert weights is None
    assert torch.equal(per_sample, oracle)
    assert torch.equal(loss, oracle.mean())


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_min_snr_weights_against_official_independent_formula(prediction_type):
    scheduler = NumericalScheduler(prediction_type)
    timesteps = torch.tensor([0, 20, 400, 999])
    reference = torch.randn(4, 1, 2, 2)
    gamma = 5.0
    production = compute_min_snr_weights(scheduler, timesteps, reference, gamma)
    oracle = reference_min_snr_weights(scheduler, timesteps, gamma, prediction_type).float()
    assert torch.allclose(production, oracle, atol=1e-7, rtol=1e-6)


def test_wrong_formula_and_wrong_v_target_failure_injections_are_detected():
    scheduler = NumericalScheduler("v_prediction")
    x0, noise = torch.randn(3, 1, 2, 2, dtype=torch.float64), torch.randn(3, 1, 2, 2, dtype=torch.float64)
    timesteps = torch.tensor([1, 400, 999])
    alpha, sigma = reference_coefficients(scheduler, timesteps, 4, torch.float64)
    noisy = alpha * x0 + sigma * noise
    velocity = alpha * noise - sigma * x0
    correct = predict_x0_from_model_output(velocity, noisy, timesteps, scheduler)
    wrong_x0 = noisy - velocity
    assert (correct - x0).abs().max() < 1e-12
    assert (wrong_x0 - x0).abs().max() > 0.1
    correct_loss = (velocity - velocity).square().mean()
    wrong_target_loss = (velocity - noise).square().mean()
    assert correct_loss == 0 and wrong_target_loss > 0.01


def test_randomized_diffusion_properties_100_cases():
    rng = random.Random(5519)
    torch.manual_seed(5519)
    for case in range(100):
        prediction_type = rng.choice(["epsilon", "v_prediction"])
        scheduler = NumericalScheduler(prediction_type, steps=rng.randint(20, 200))
        shape = (rng.randint(1, 5), rng.randint(1, 4), rng.randint(1, 6), rng.randint(1, 6))
        x0, noise = torch.randn(*shape, dtype=torch.float64), torch.randn(*shape, dtype=torch.float64)
        timesteps = torch.tensor([rng.randrange(len(scheduler.alphas_cumprod)) for _ in range(shape[0])])
        alpha, sigma = reference_coefficients(scheduler, timesteps, 4, torch.float64)
        noisy = alpha * x0 + sigma * noise
        target = noise if prediction_type == "epsilon" else alpha * noise - sigma * x0
        recovered = predict_x0_from_model_output(target, noisy, timesteps, scheduler)
        assert torch.allclose(recovered, x0, atol=2e-11, rtol=2e-11), (case, prediction_type, shape, timesteps)
        perturbed = target + 0.1
        loss, per_sample, _ = compute_diffusion_loss(perturbed, target, scheduler=scheduler, timesteps=timesteps)
        oracle = reference_per_sample_mse(perturbed, target)
        assert torch.allclose(per_sample, oracle, atol=1e-14)
        assert torch.allclose(loss, oracle.mean(), atol=1e-14)


def test_float32_and_low_precision_consistency_diagnostics():
    scheduler = NumericalScheduler("epsilon")
    torch.manual_seed(77)
    x64, noise64 = torch.randn(4, 2, 3, 3, dtype=torch.float64), torch.randn(4, 2, 3, 3, dtype=torch.float64)
    timesteps = torch.tensor([0, 10, 500, 999])
    alpha, sigma = reference_coefficients(scheduler, timesteps, 4, torch.float64)
    noisy64 = alpha * x64 + sigma * noise64
    recovered64 = predict_x0_from_model_output(noise64, noisy64, timesteps, scheduler)
    recovered32 = predict_x0_from_model_output(noise64.float(), noisy64.float(), timesteps, scheduler)
    assert (recovered64 - x64).abs().max() < 2e-12
    assert (recovered32.double() - x64).abs().max() < 2e-3
    for dtype, tolerance in ((torch.bfloat16, 3.0), (torch.float16, 3.0)):
        recovered = predict_x0_from_model_output(noise64.to(dtype), noisy64.to(dtype), timesteps, scheduler)
        assert torch.isfinite(recovered).all()
        assert (recovered.double() - x64).abs().max() < tolerance


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_not_silently_valid_losses(bad):
    scheduler = NumericalScheduler()
    pred = torch.zeros(1, 1, 2, 2); pred[0, 0, 0, 0] = bad
    loss, _, _ = compute_diffusion_loss(pred, torch.zeros_like(pred), scheduler=scheduler, timesteps=torch.tensor([0]))
    assert not torch.isfinite(loss)


def test_shape_timestep_prediction_and_gamma_failures_are_clear():
    scheduler = NumericalScheduler()
    x = torch.randn(2, 1, 2, 2)
    with pytest.raises(ValueError, match="shapes differ"):
        predict_x0_from_model_output(x, torch.randn(1, 1, 2, 2), torch.tensor([0, 1]), scheduler)
    with pytest.raises(ValueError, match="timesteps must have shape"):
        predict_x0_from_model_output(x, x, torch.tensor([0]), scheduler)
    scheduler.config.prediction_type = "sample"
    with pytest.raises(ValueError, match="Unsupported"):
        get_diffusion_target(scheduler, x, x, torch.tensor([0, 1]))
    scheduler.config.prediction_type = "epsilon"
    with pytest.raises(ValueError, match="positive"):
        compute_min_snr_weights(scheduler, torch.tensor([0, 1]), x, 0)
