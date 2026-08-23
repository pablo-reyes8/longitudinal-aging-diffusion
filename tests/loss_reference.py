"""Independent float64 oracles for diffusion-loss tests only.

This module deliberately does not import ``src.loss`` so a shared implementation
mistake cannot make production code and its numerical oracle agree accidentally.
"""

from __future__ import annotations

import torch


def reference_coefficients(scheduler, timesteps, ndim, dtype=torch.float64, device="cpu"):
    alpha_bar = torch.stack(
        [scheduler.alphas_cumprod[int(timestep)].double() for timestep in timesteps]
    ).to(device=device, dtype=dtype)
    shape = (len(timesteps),) + (1,) * (ndim - 1)
    alpha_bar = alpha_bar.reshape(shape)
    return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()


def reference_velocity(clean, noise, alpha, sigma):
    return alpha * noise - sigma * clean


def reference_x0(model_output, noisy, alpha, sigma, prediction_type):
    if prediction_type == "epsilon":
        return (noisy - sigma * model_output) / alpha
    if prediction_type == "v_prediction":
        return alpha * noisy - sigma * model_output
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def reference_per_sample_mse(prediction, target):
    # Preserve the input dtype: this oracle also verifies the production
    # reduction order and native-precision semantics exactly.
    return (prediction - target).square().flatten(1).mean(1)


def reference_min_snr_weights(scheduler, timesteps, gamma, prediction_type):
    alpha_bar = torch.stack(
        [scheduler.alphas_cumprod[int(timestep)].double() for timestep in timesteps]
    )
    snr = alpha_bar / (1.0 - alpha_bar)
    capped = torch.minimum(snr, torch.full_like(snr, float(gamma)))
    denominator = snr if prediction_type == "epsilon" else snr + 1.0
    return capped / denominator
