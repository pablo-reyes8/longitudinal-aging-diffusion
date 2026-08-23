"""Cross-check against the real Diffusers scheduler when installed."""

from __future__ import annotations

import pytest
import torch

from src.loss import get_diffusion_target, predict_x0_from_model_output

diffusers = pytest.importorskip("diffusers", reason="optional real Diffusers validation")


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_real_ddpm_scheduler_target_and_x0_round_trip(prediction_type):
    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type=prediction_type,
    )
    torch.manual_seed(9102)
    clean = torch.randn(6, 4, 5, 5, dtype=torch.float64)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 1, 17, 500, 998, 999])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    target = get_diffusion_target(scheduler, clean, noise, timesteps)
    if prediction_type == "epsilon":
        assert torch.equal(target, noise)
    else:
        assert torch.allclose(target, scheduler.get_velocity(clean, noise, timesteps), atol=1e-13)
    recovered = predict_x0_from_model_output(target, noisy, timesteps, scheduler)
    assert torch.allclose(recovered, clean, atol=3e-11, rtol=3e-11)
