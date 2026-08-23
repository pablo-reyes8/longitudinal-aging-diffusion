"""Numerically stable diffusion targets, coefficients, and x0 reconstruction."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


SUPPORTED_PREDICTION_TYPES = ("epsilon", "v_prediction")


def _config_value(config: Any, key: str) -> Any:
    return config[key] if isinstance(config, Mapping) else getattr(config, key)


def get_prediction_type(scheduler: Any) -> str:
    prediction_type = str(_config_value(scheduler.config, "prediction_type"))
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(
            f"Unsupported scheduler prediction_type={prediction_type!r}; "
            f"expected one of {SUPPORTED_PREDICTION_TYPES}"
        )
    return prediction_type


def _calculation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _validate_timesteps(timesteps: torch.Tensor, batch_size: int, total_steps: int) -> None:
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
        raise ValueError(f"timesteps must have shape [{batch_size}], got {tuple(timesteps.shape)}")
    if timesteps.dtype not in (torch.int32, torch.int64):
        raise TypeError("timesteps must use an integer dtype")
    if timesteps.numel() and (int(timesteps.min()) < 0 or int(timesteps.max()) >= total_steps):
        raise ValueError(f"timesteps must be in [0, {total_steps - 1}]")


def extract_scheduler_coefficients(
    scheduler: Any,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather per-sample alpha coefficients on the reference device/dtype.

    Half/bfloat inputs use FP32 coefficient arithmetic; float64 remains float64
    for strict numerical validation.
    """
    if reference.ndim < 1:
        raise ValueError("reference must include a batch dimension")
    if not hasattr(scheduler, "alphas_cumprod"):
        raise AttributeError("scheduler must expose alphas_cumprod")
    alphas = scheduler.alphas_cumprod
    if not isinstance(alphas, torch.Tensor) or alphas.ndim != 1:
        raise ValueError("scheduler.alphas_cumprod must be a one-dimensional tensor")
    _validate_timesteps(timesteps, reference.shape[0], len(alphas))
    if timesteps.device != reference.device:
        raise ValueError("timesteps and reference must be on the same device")
    dtype = _calculation_dtype(reference.dtype)
    indices = timesteps.to(device=alphas.device, dtype=torch.long)
    alpha_bar = alphas.index_select(0, indices).to(device=reference.device, dtype=dtype)
    if not torch.isfinite(alpha_bar).all() or (alpha_bar < 0).any() or (alpha_bar > 1).any():
        raise ValueError("scheduler alphas_cumprod contains invalid values")
    shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    alpha_bar_broadcast = alpha_bar.reshape(shape)
    return {
        "alpha_bar": alpha_bar,
        "alpha_bar_broadcast": alpha_bar_broadcast,
        "sqrt_alpha_bar": alpha_bar_broadcast.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar_broadcast).clamp_min(0).sqrt(),
    }


def get_diffusion_target(
    scheduler: Any,
    target_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if target_latents.shape != noise.shape:
        raise ValueError(f"target_latents and noise shapes differ: {target_latents.shape} vs {noise.shape}")
    prediction_type = get_prediction_type(scheduler)
    if prediction_type == "epsilon":
        return noise
    if not hasattr(scheduler, "get_velocity"):
        raise AttributeError("v_prediction requires scheduler.get_velocity")
    return scheduler.get_velocity(target_latents, noise, timesteps)


def compute_min_snr_weights(
    scheduler: Any,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("min_snr_gamma must be positive")
    coefficients = extract_scheduler_coefficients(scheduler, timesteps, reference)
    alpha_bar = coefficients["alpha_bar"]
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(torch.finfo(alpha_bar.dtype).tiny)
    clipped = torch.minimum(snr, torch.full_like(snr, float(gamma)))
    prediction_type = get_prediction_type(scheduler)
    denominator = snr if prediction_type == "epsilon" else snr + 1.0
    return clipped / denominator.clamp_min(torch.finfo(snr.dtype).tiny)


def compute_diffusion_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scheduler: Any,
    timesteps: torch.Tensor,
    loss_type: str = "mse",
    min_snr_gamma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if model_pred.shape != target.shape:
        raise ValueError(f"model_pred and diffusion target shapes differ: {model_pred.shape} vs {target.shape}")
    if model_pred.ndim < 2:
        raise ValueError("model_pred must have a batch dimension and at least one feature dimension")
    if loss_type != "mse":
        raise ValueError("V1 only supports diffusion_loss_type='mse'")
    calc_dtype = _calculation_dtype(model_pred.dtype)
    per_element = F.mse_loss(model_pred.to(calc_dtype), target.to(calc_dtype), reduction="none")
    per_sample = per_element.flatten(start_dim=1).mean(dim=1)
    weights = None
    if min_snr_gamma is not None:
        weights = compute_min_snr_weights(scheduler, timesteps, model_pred, min_snr_gamma)
        per_sample = per_sample * weights
    return per_sample.mean(), per_sample, weights


def predict_x0_from_model_output(
    model_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    if model_pred.shape != noisy_latents.shape:
        raise ValueError(f"model_pred and noisy_latents shapes differ: {model_pred.shape} vs {noisy_latents.shape}")
    if model_pred.device != noisy_latents.device:
        raise ValueError("model_pred and noisy_latents must be on the same device")
    coefficients = extract_scheduler_coefficients(scheduler, timesteps, noisy_latents)
    calc_dtype = coefficients["sqrt_alpha_bar"].dtype
    prediction = model_pred.to(calc_dtype)
    noisy = noisy_latents.to(calc_dtype)
    alpha = coefficients["sqrt_alpha_bar"]
    sigma = coefficients["sqrt_one_minus_alpha_bar"]
    prediction_type = get_prediction_type(scheduler)
    if prediction_type == "epsilon":
        tiny = torch.finfo(calc_dtype).tiny
        return (noisy - sigma * prediction) / alpha.clamp_min(tiny)
    return alpha * noisy - sigma * prediction


def sd_image_to_01(images: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
    converted = images.to(_calculation_dtype(images.dtype)).mul(0.5).add(0.5)
    return converted.clamp(0, 1) if clamp else converted
