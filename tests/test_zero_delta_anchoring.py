from __future__ import annotations

import pytest
import torch

from loss_fakes import NumericalScheduler, TinyVAE
from src.loss import FaceAgingDiffusionLoss


def _preservation_loss(**kwargs) -> FaceAgingDiffusionLoss:
    return FaceAgingDiffusionLoss(
        scheduler=NumericalScheduler(steps=100, dtype=torch.float64),
        vae=TinyVAE().double(),
        identity_weight=0,
        age_weight=0,
        use_preservation_loss=True,
        preservation_weight=0.10,
        preservation_loss_type="l1",
        preservation_max_delta=2,
        use_small_delta_weighting=False,
        clamp_pred_x0=False,
        **kwargs,
    )


def _preservation_inputs(
    loss_fn: FaceAgingDiffusionLoss,
    deltas: list[float],
    *,
    prediction_offset: float = 0.0,
) -> dict:
    batch = len(deltas)
    source_images = torch.linspace(-0.6, 0.6, batch * 4, dtype=torch.float64).reshape(batch, 1, 2, 2)
    latent_mean = loss_fn.vae.encode(source_images).latent_dist.mean
    target_latents = latent_mean * loss_fn.vae.config.scaling_factor
    noise = torch.linspace(-0.2, 0.2, batch * 4, dtype=torch.float64).reshape_as(target_latents)
    timesteps = torch.full((batch,), 10, dtype=torch.long)
    noisy = loss_fn.scheduler.add_noise(target_latents, noise, timesteps)
    model_pred = (noise + prediction_offset).detach().requires_grad_(True)
    delta_tensor = torch.tensor(deltas, dtype=torch.float64)
    return {
        "model_pred": model_pred,
        "noise": noise,
        "noisy_target_latents": noisy,
        "target_latents": target_latents,
        "timesteps": timesteps,
        "source_images": source_images,
        "target_images": source_images.clone(),
        "source_ages": torch.full((batch,), 26.0),
        "target_ages": torch.full((batch,), 26.0) + delta_tensor,
        "delta_ages": delta_tensor,
        "return_per_sample": True,
    }


def test_preservation_is_zero_and_skips_decode_when_no_delta_is_eligible():
    class DecodeCountingVAE(TinyVAE):
        def __init__(self):
            super().__init__()
            self.decode_calls = 0

        def decode(self, latents):
            self.decode_calls += 1
            return super().decode(latents)

    vae = DecodeCountingVAE().double()
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=NumericalScheduler(steps=100, dtype=torch.float64),
        vae=vae,
        identity_weight=0,
        age_weight=0,
        use_preservation_loss=True,
        preservation_weight=0.1,
        preservation_max_delta=2,
        use_small_delta_weighting=False,
    )
    output = loss_fn(**_preservation_inputs(loss_fn, [3, 10], prediction_offset=0.1))
    assert output["loss_preservation"].item() == 0
    assert output["metrics"]["preservation_count"] == 0
    assert output["metrics"]["preservation_active_fraction"] == 0
    assert vae.decode_calls == 0


def test_exact_source_reconstruction_has_zero_preservation_loss():
    loss_fn = _preservation_loss()
    output = loss_fn(**_preservation_inputs(loss_fn, [0, -2, -3]))
    assert output["preservation_indices"].tolist() == [0, 1]
    assert output["loss_preservation"].item() == pytest.approx(0.0, abs=1e-12)
    assert output["metrics"]["preservation_active_fraction"] == pytest.approx(2 / 3)


def test_preservation_is_positive_finite_monotonic_and_has_gradients():
    loss_fn = _preservation_loss()
    small_inputs = _preservation_inputs(loss_fn, [0, 1, 8], prediction_offset=0.03)
    large_inputs = _preservation_inputs(loss_fn, [0, 1, 8], prediction_offset=0.15)
    small = loss_fn(**small_inputs)
    large = loss_fn(**large_inputs)
    assert torch.isfinite(small["loss_preservation"])
    assert 0 < small["loss_preservation"] < large["loss_preservation"]
    large["weighted_preservation"].backward()
    gradient = large_inputs["model_pred"].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[:2].abs().sum() > 0
    assert gradient[2].abs().sum() == 0


def _diffusion_inputs(loss_fn: FaceAgingDiffusionLoss) -> dict:
    model_pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).reshape(3, 1, 1, 1).requires_grad_()
    noise = torch.zeros_like(model_pred)
    target_latents = torch.zeros_like(model_pred)
    timesteps = torch.tensor([1, 2, 3])
    noisy = loss_fn.scheduler.add_noise(target_latents, noise, timesteps)
    return {
        "model_pred": model_pred,
        "noise": noise,
        "noisy_target_latents": noisy,
        "target_latents": target_latents,
        "timesteps": timesteps,
        "source_ages": torch.tensor([20, 20, 20]),
        "target_ages": torch.tensor([20, 15, 26]),
        "delta_ages": torch.tensor([0, -5, 6]),
        "return_per_sample": True,
    }


def test_small_delta_weighting_matches_manual_per_sample_oracle():
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=NumericalScheduler(steps=10, dtype=torch.float64),
        vae=TinyVAE().double(),
        identity_weight=0,
        age_weight=0,
        use_preservation_loss=False,
        use_small_delta_weighting=True,
        small_delta_threshold=5,
        small_delta_weight=2.0,
    )
    output = loss_fn(**_diffusion_inputs(loss_fn))
    raw = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    weights = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64)
    assert torch.equal(output["small_delta_sample_weights"], weights)
    assert torch.allclose(output["loss_diff_per_sample_unweighted"], raw)
    assert output["loss_diff"].item() == pytest.approx(float((raw * weights).mean()))
    assert output["metrics"]["small_delta_count"] == 2
    assert output["metrics"]["small_delta_fraction"] == pytest.approx(2 / 3)


def test_disabling_small_delta_weighting_exactly_restores_unweighted_loss():
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=NumericalScheduler(steps=10, dtype=torch.float64),
        vae=TinyVAE().double(),
        identity_weight=0,
        age_weight=0,
        use_preservation_loss=False,
        use_small_delta_weighting=False,
    )
    output = loss_fn(**_diffusion_inputs(loss_fn))
    assert torch.equal(
        output["loss_diff_per_sample"],
        output["loss_diff_per_sample_unweighted"],
    )
    assert output["loss_diff"].item() == pytest.approx((1 + 4 + 9) / 3)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"preservation_weight": -0.1}, "weights"),
        ({"preservation_loss_type": "cosine"}, "preservation_loss_type"),
        ({"preservation_max_delta": -1}, "preservation_max_delta"),
        ({"small_delta_threshold": -1}, "small_delta_threshold"),
        ({"small_delta_weight": 0.5}, "small_delta_weight"),
    ],
)
def test_zero_delta_loss_configuration_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaceAgingDiffusionLoss(
            scheduler=NumericalScheduler(),
            vae=TinyVAE(),
            identity_weight=0,
            age_weight=0,
            **kwargs,
        )
