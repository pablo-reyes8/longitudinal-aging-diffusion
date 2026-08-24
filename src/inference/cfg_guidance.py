"""Batched three-way image/text classifier-free guidance."""

from __future__ import annotations

import torch

from src.model import build_conditioned_unet_input

from .inference_utils import inference_autocast


def combine_three_way_cfg(
    eps_full: torch.Tensor,
    eps_image: torch.Tensor,
    eps_uncond: torch.Tensor,
    *,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
) -> torch.Tensor:
    if eps_full.shape != eps_image.shape or eps_full.shape != eps_uncond.shape:
        raise ValueError("All CFG prediction branches must have identical shapes")
    return (
        eps_uncond
        + float(image_guidance_scale) * (eps_image - eps_uncond)
        + float(text_guidance_scale) * (eps_full - eps_image)
    )


def combine_referenced_text_cfg(
    eps_target: torch.Tensor,
    eps_reference: torch.Tensor,
    *,
    age_guidance_scale: float = 3.0,
) -> torch.Tensor:
    """Guide the target-age prediction relative to a meaningful text reference."""
    if eps_target.shape != eps_reference.shape:
        raise ValueError("Target and reference CFG predictions must have identical shapes")
    return eps_reference + float(age_guidance_scale) * (eps_target - eps_reference)


def combine_referenced_three_way_cfg(
    eps_target: torch.Tensor,
    eps_reference: torch.Tensor,
    eps_uncond: torch.Tensor,
    *,
    age_guidance_scale: float = 3.0,
    image_guidance_scale: float = 1.5,
) -> torch.Tensor:
    """Apply referenced text guidance while retaining source-image guidance."""
    text_guided = combine_referenced_text_cfg(
        eps_target, eps_reference, age_guidance_scale=age_guidance_scale
    )
    return text_guided + (float(image_guidance_scale) - 1.0) * (
        eps_reference - eps_uncond
    )


def predict_three_way_cfg(
    *,
    bundle,
    target_latents: torch.Tensor,
    timestep: torch.Tensor | int,
    source_latents: torch.Tensor,
    full_text_embeddings: torch.Tensor,
    null_text_embeddings: torch.Tensor,
    reference_text_embeddings: torch.Tensor | None = None,
    text_guidance_scale: float = 7.0,
    age_guidance_scale: float | None = None,
    image_guidance_scale: float = 1.5,
    use_cfg: bool = True,
    age_conditioning: torch.Tensor | None = None,
    return_branches: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = target_latents.shape[0]
    if source_latents.shape != target_latents.shape:
        raise ValueError("Source and target latent shapes must match")
    if full_text_embeddings.shape[0] != batch:
        raise ValueError("Full text embedding batch mismatch")
    if null_text_embeddings.shape[0] == 1:
        null_text_embeddings = null_text_embeddings.expand(batch, -1, -1)
    if null_text_embeddings.shape != full_text_embeddings.shape:
        raise ValueError("Null and full embedding shapes must match")
    reference_embeddings = (
        null_text_embeddings if reference_text_embeddings is None else reference_text_embeddings
    )
    if reference_embeddings.shape[0] == 1:
        reference_embeddings = reference_embeddings.expand(batch, -1, -1)
    if reference_embeddings.shape != full_text_embeddings.shape:
        raise ValueError("Reference and full embedding shapes must match")
    if not use_cfg:
        model_input = build_conditioned_unet_input(target_latents, source_latents)
        with inference_autocast(bundle, target_latents.device):
            kwargs = {"encoder_hidden_states": full_text_embeddings, "return_dict": True}
            if age_conditioning is not None:
                kwargs["timestep_cond"] = age_conditioning
            prediction = bundle["unet"](model_input, timestep, **kwargs).sample
        return (prediction, {"full": prediction}) if return_branches else prediction
    targets = torch.cat((target_latents, target_latents, target_latents), dim=0)
    sources = torch.cat((source_latents, source_latents, torch.zeros_like(source_latents)), dim=0)
    embeddings = torch.cat((full_text_embeddings, reference_embeddings, null_text_embeddings), dim=0)
    if torch.is_tensor(timestep) and timestep.ndim > 0:
        if timestep.numel() == 1:
            expanded_timestep = timestep.expand(3 * batch)
        elif timestep.shape == (batch,):
            expanded_timestep = timestep.repeat(3)
        else:
            raise ValueError("timestep must be scalar, [1], or [B]")
    else:
        expanded_timestep = timestep
    model_input = build_conditioned_unet_input(targets, sources)
    unet_kwargs = {"encoder_hidden_states": embeddings, "return_dict": True}
    if age_conditioning is not None:
        if age_conditioning.shape[0] != batch:
            raise ValueError("Age-conditioning batch mismatch")
        unet_kwargs["timestep_cond"] = age_conditioning.repeat(3, 1)
    with inference_autocast(bundle, target_latents.device):
        prediction = bundle["unet"](
            model_input, expanded_timestep,
            **unet_kwargs,
        ).sample
    eps_target, eps_reference, eps_uncond = prediction.chunk(3, dim=0)
    guided = combine_referenced_three_way_cfg(
        eps_target, eps_reference, eps_uncond,
        age_guidance_scale=(
            float(text_guidance_scale) if age_guidance_scale is None else float(age_guidance_scale)
        ),
        image_guidance_scale=image_guidance_scale,
    )
    if not torch.isfinite(guided).all():
        raise FloatingPointError("Three-way guided prediction contains NaN/Inf")
    branches = {
        "full": eps_target,
        "target": eps_target,
        "image": eps_reference,
        "reference": eps_reference,
        "uncond": eps_uncond,
    }
    return (guided, branches) if return_branches else guided
