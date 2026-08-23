"""Deterministic DDIM inversion and reverse editing loops."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch

from src.model import encode_prompts

from .cfg_guidance import predict_three_way_cfg
from .inference_utils import scheduler_prediction_type, scheduler_set_timesteps


def _alpha_cumprod(scheduler, timestep: int, reference: torch.Tensor) -> torch.Tensor:
    if timestep < 0:
        value = getattr(scheduler, "final_alpha_cumprod", torch.tensor(1.0))
    else:
        value = scheduler.alphas_cumprod[int(timestep)]
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def model_output_to_x0_epsilon(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    timestep: int,
    scheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = _alpha_cumprod(scheduler, timestep, sample)
    sqrt_alpha = alpha.sqrt()
    sqrt_beta = (1 - alpha).sqrt()
    prediction_type = scheduler_prediction_type(scheduler)
    if prediction_type == "epsilon":
        epsilon = model_output
        x0 = (sample - sqrt_beta * epsilon) / sqrt_alpha
    elif prediction_type == "v_prediction":
        x0 = sqrt_alpha * sample - sqrt_beta * model_output
        epsilon = sqrt_alpha * model_output + sqrt_beta * sample
    elif prediction_type == "sample":
        x0 = model_output
        epsilon = (sample - sqrt_alpha * x0) / sqrt_beta.clamp_min(torch.finfo(sample.dtype).eps)
    else:
        raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}")
    return x0, epsilon


def ddim_forward_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    current_timestep: int,
    next_timestep: int,
    scheduler,
) -> torch.Tensor:
    """Deterministic clean-to-noisy DDIM step used for inversion."""
    x0, epsilon = model_output_to_x0_epsilon(model_output, sample, current_timestep, scheduler)
    next_alpha = _alpha_cumprod(scheduler, next_timestep, sample)
    result = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
    if not torch.isfinite(result).all():
        raise FloatingPointError("DDIM inversion produced NaN/Inf")
    return result


def _scale_model_input(scheduler, latents: torch.Tensor, timestep):
    method = getattr(scheduler, "scale_model_input", None)
    return method(latents, timestep) if method is not None else latents


def scheduler_reverse_step(scheduler, model_output, timestep, sample, generator=None):
    signature = inspect.signature(scheduler.step)
    kwargs: dict[str, Any] = {}
    if "eta" in signature.parameters:
        kwargs["eta"] = 0.0
    if "generator" in signature.parameters:
        kwargs["generator"] = generator
    result = scheduler.step(model_output, timestep, sample, **kwargs)
    return result.prev_sample if hasattr(result, "prev_sample") else result[0]


@torch.no_grad()
def ddim_invert_source_image(
    *,
    bundle,
    source_latents: torch.Tensor,
    source_prompt: str,
    scheduler,
    num_inference_steps: int = 50,
    inversion_strength: float = 1.0,
    text_guidance_scale: float = 1.0,
    image_guidance_scale: float = 1.0,
    negative_prompt: str = "",
    use_cfg: bool = True,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    if not 0 < inversion_strength <= 1:
        raise ValueError("inversion_strength must be in (0,1]")
    if num_inference_steps < 2:
        raise ValueError("DDIM inversion requires at least two inference steps")
    device = source_latents.device
    timesteps = scheduler_set_timesteps(scheduler, num_inference_steps, device)
    ascending = timesteps.flip(0)
    transitions = max(1, min(len(ascending) - 1, int(round((len(ascending) - 1) * inversion_strength))))
    selected = ascending[: transitions + 1]
    source_embeddings = encode_prompts(bundle, [source_prompt] * source_latents.shape[0], device=device)
    null_embeddings = encode_prompts(bundle, [negative_prompt], device=device)
    latents = source_latents.clone()
    trajectory = [latents.detach().cpu().clone()] if return_intermediates else None
    for index in range(len(selected) - 1):
        current_t = selected[index]
        next_t = selected[index + 1]
        model_latents = _scale_model_input(scheduler, latents, current_t)
        prediction = predict_three_way_cfg(
            bundle=bundle, target_latents=model_latents, timestep=current_t,
            source_latents=source_latents,
            full_text_embeddings=source_embeddings,
            null_text_embeddings=null_embeddings,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            use_cfg=use_cfg,
        )
        latents = ddim_forward_step(
            latents, prediction.to(latents.dtype), int(current_t), int(next_t), scheduler
        )
        if trajectory is not None:
            trajectory.append(latents.detach().cpu().clone())
    return {
        "inverted_latents": latents,
        "start_timestep": int(selected[-1]),
        "denoising_timesteps": selected.flip(0),
        "inversion_timesteps": selected,
        "trajectory": trajectory,
    }


@torch.no_grad()
def edit_from_inverted_latent(
    *,
    bundle,
    inverted_latents: torch.Tensor,
    source_latents: torch.Tensor,
    target_prompt: str,
    scheduler,
    denoising_timesteps: torch.Tensor,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    negative_prompt: str = "",
    use_cfg: bool = True,
    generator=None,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    device = inverted_latents.device
    embeddings = encode_prompts(bundle, [target_prompt] * inverted_latents.shape[0], device=device)
    null_embeddings = encode_prompts(bundle, [negative_prompt], device=device)
    latents = inverted_latents.clone()
    trajectory = [latents.detach().cpu().clone()] if return_intermediates else None
    guided_norms = []
    for timestep in denoising_timesteps:
        model_latents = _scale_model_input(scheduler, latents, timestep)
        prediction = predict_three_way_cfg(
            bundle=bundle, target_latents=model_latents, timestep=timestep,
            source_latents=source_latents,
            full_text_embeddings=embeddings,
            null_text_embeddings=null_embeddings,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            use_cfg=use_cfg,
        ).to(latents.dtype)
        guided_norms.append(float(prediction.float().norm()))
        latents = scheduler_reverse_step(scheduler, prediction, timestep, latents, generator)
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"Reverse DDIM step produced NaN/Inf at timestep {int(timestep)}")
        if trajectory is not None:
            trajectory.append(latents.detach().cpu().clone())
    return {"latents": latents, "trajectory": trajectory, "guided_prediction_norms": guided_norms}
