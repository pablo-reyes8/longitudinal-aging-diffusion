from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from loss_fakes import TinyAgeModel, TinyIdentityModel, TinyVAE
from src.loss import (
    AgeEstimatorAdapter,
    IdentityEncoderAdapter,
    expected_age_from_logits,
    identity_cosine_loss,
    sd_image_to_01,
)


def central_difference(function, tensor, flat_index, step=1e-6):
    plus, minus = tensor.detach().clone(), tensor.detach().clone()
    plus.view(-1)[flat_index] += step
    minus.view(-1)[flat_index] -= step
    return (function(plus) - function(minus)) / (2 * step)


@pytest.mark.parametrize(
    "predicted,reference,expected",
    [
        ([1.0, 0.0], [1.0, 0.0], 0.0),
        ([0.5, math.sqrt(0.75)], [1.0, 0.0], 0.5),
        ([0.0, 1.0], [1.0, 0.0], 1.0),
        ([-0.5, math.sqrt(0.75)], [1.0, 0.0], 1.5),
        ([-1.0, 0.0], [1.0, 0.0], 2.0),
    ],
)
def test_identity_cosine_geometry_oracle(predicted, reference, expected):
    loss = identity_cosine_loss(torch.tensor([predicted]), torch.tensor([reference]))
    assert loss.item() == pytest.approx(expected, abs=2e-7)


def test_identity_scale_invariance_batch_reduction_and_monotonicity():
    reference = torch.tensor([[1.0, 0.0]])
    vector = torch.tensor([[0.3, 0.7]])
    assert torch.allclose(identity_cosine_loss(vector, reference), identity_cosine_loss(100 * vector, reference))
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    references = reference.repeat(3, 1)
    per_sample = identity_cosine_loss(predicted, references, reduction="none")
    assert torch.allclose(per_sample, torch.tensor([0.0, 1.0, 2.0]))
    assert per_sample.mean() == 1.0
    cosines = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    vectors = torch.stack([cosines, (1 - cosines.square()).sqrt()], dim=1)
    losses = identity_cosine_loss(vectors, reference.repeat(5, 1), reduction="none")
    assert torch.all(losses[:-1] > losses[1:])


def test_randomized_identity_property_200_cases():
    torch.manual_seed(88)
    rng = random.Random(88)
    for _ in range(200):
        shape = (rng.randint(1, 8), rng.randint(2, 20))
        first, second = torch.randn(*shape), torch.randn(*shape)
        production = identity_cosine_loss(first, second, reduction="none")
        oracle = 1 - F.cosine_similarity(first, second, dim=-1)
        assert torch.allclose(production, oracle, atol=2e-7, rtol=2e-6)


def test_identity_finite_difference_and_frozen_encoder_jacobian_flow():
    torch.manual_seed(9)
    model = TinyIdentityModel().double()
    adapter = IdentityEncoderAdapter(model)
    image = torch.rand(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64)
    function = lambda value: identity_cosine_loss(adapter(value), reference)
    analytical = torch.autograd.grad(function(image), image)[0]
    for index in range(image.numel()):
        numerical = central_difference(function, image, index)
        assert analytical.view(-1)[index].item() == pytest.approx(numerical.item(), rel=2e-5, abs=2e-7)
    assert analytical.norm() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("loss_type", ["l1", "mse"])
def test_age_loss_exact_values_symmetry_and_monotonicity(loss_type):
    target = torch.tensor([30.0])
    errors = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
    positive = (target + errors - target).abs() if loss_type == "l1" else (target + errors - target).square()
    negative = (target - errors - target).abs() if loss_type == "l1" else (target - errors - target).square()
    assert torch.equal(positive, negative)
    assert torch.all(positive[:-1] < positive[1:])
    assert positive[-2].item() == (5.0 if loss_type == "l1" else 25.0)


def test_expected_age_classifier_exact_oracle_and_values():
    probabilities = torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float64)
    logits = probabilities.clamp_min(1e-30).log()
    assert expected_age_from_logits(logits).item() == pytest.approx(2.5, abs=1e-12)
    custom = expected_age_from_logits(torch.tensor([[0.0, 0.0]], dtype=torch.float64), [10, 20])
    assert custom.item() == 15.0


def test_expected_age_logits_finite_difference():
    logits = torch.tensor([[0.2, -0.1, 0.7, 1.2]], dtype=torch.float64, requires_grad=True)
    objective = lambda value: (expected_age_from_logits(value) - 1.3).square().mean()
    analytical = torch.autograd.grad(objective(logits), logits)[0]
    for index in range(logits.numel()):
        numerical = central_difference(objective, logits, index)
        assert analytical.view(-1)[index].item() == pytest.approx(numerical.item(), rel=2e-6, abs=2e-8)


def test_age_estimator_frozen_but_image_gradient_connected_and_nonzero():
    model = TinyAgeModel().double()
    adapter = AgeEstimatorAdapter(model, output_type="scalar")
    image = torch.rand(2, 1, 3, 3, dtype=torch.float64, requires_grad=True)
    loss = (adapter(image) - torch.tensor([22.0, 25.0], dtype=torch.float64)).abs().mean()
    gradient = torch.autograd.grad(loss, image)[0]
    assert gradient is not None and gradient.norm() > 0 and torch.isfinite(gradient).all()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_vae_decoder_frozen_jacobian_and_finite_difference():
    vae = TinyVAE().double().eval().requires_grad_(False)
    latent = torch.rand(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    function = lambda value: vae.decode(value / vae.config.scaling_factor).sample.square().mean()
    analytical = torch.autograd.grad(function(latent), latent)[0]
    for index in range(latent.numel()):
        numerical = central_difference(function, latent, index)
        assert analytical.view(-1)[index].item() == pytest.approx(numerical.item(), rel=2e-6, abs=2e-8)
    assert analytical.norm() > 0
    assert all(parameter.grad is None for parameter in vae.parameters())


def test_vae_scaling_roundtrip_and_wrong_scaling_failure_injection():
    vae = TinyVAE(scaling_factor=0.25).double()
    raw = torch.randn(2, 1, 3, 3, dtype=torch.float64)
    scaled = raw * vae.config.scaling_factor
    correct = vae.decode(scaled / vae.config.scaling_factor).sample
    direct = vae.decode(raw).sample
    wrong = vae.decode(scaled).sample
    assert torch.equal(correct, direct)
    assert (wrong - correct).abs().max() > 0.1


def test_sd_image_conversion_is_canonical_and_clamped():
    images = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert torch.equal(sd_image_to_01(images), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert torch.equal(sd_image_to_01(images, clamp=False), torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5]))


def test_auxiliary_adapters_stay_eval_after_train_calls():
    identity = IdentityEncoderAdapter(nn.Sequential(nn.Dropout(0.9), nn.Flatten(), nn.Linear(4, 2)))
    age = AgeEstimatorAdapter(nn.Sequential(nn.Dropout(0.9), nn.Flatten(), nn.Linear(4, 1)))
    identity.train(True); age.train(True)
    assert not identity.training and not identity.model.training
    assert not age.training and not age.model.training


def test_adapter_shape_contracts_are_explicit():
    with pytest.raises(ValueError, match="Age logits"):
        expected_age_from_logits(torch.randn(4))
    with pytest.raises(ValueError, match="age_values"):
        expected_age_from_logits(torch.randn(2, 4), [1, 2])
    bad_identity = IdentityEncoderAdapter(nn.Identity())
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        bad_identity(torch.randn(2))


@pytest.mark.parametrize("loss_type", ["l1", "mse"])
def test_randomized_age_loss_property_200_cases(loss_type):
    torch.manual_seed(991)
    for batch_size in range(1, 9):
        for _ in range(25):
            predicted = torch.randn(batch_size, dtype=torch.float64) * 20 + 40
            target = torch.randn(batch_size, dtype=torch.float64) * 20 + 40
            production = (predicted - target).abs() if loss_type == "l1" else (predicted - target).square()
            oracle = torch.stack([
                abs(predicted[i] - target[i]) if loss_type == "l1" else (predicted[i] - target[i]) ** 2
                for i in range(batch_size)
            ])
            assert torch.equal(production, oracle)
