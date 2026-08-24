"""One direct-corruption forward/loss step for paired face-aging training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.loss import get_diffusion_target
from src.model import (
    build_conditioned_unet_input,
    compute_age_delta_embedding,
    encode_images_to_latents,
    encode_prompts,
)

from .conditioning_dropout import (
    apply_conditioning_dropout,
    conditioning_dropout_statistics,
    sample_conditioning_dropout,
)
from .mixed_precision import autocast_ctx
from .prompt_regularization import select_training_prompts
from .timestep_sampling import sample_diffusion_timesteps, timestep_statistics


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _random_normal_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    generator_device = torch.device(getattr(generator, "device", reference.device)) if generator is not None else reference.device
    value = torch.randn(reference.shape, generator=generator, device=generator_device, dtype=reference.dtype)
    return value.to(reference.device)


def prepare_training_batch(
    *,
    bundle: Mapping[str, Any],
    batch: Mapping[str, Any],
    device: torch.device,
    conditioning_dropout_prob: float = 0.05,
    timestep_sampling: str = "uniform",
    min_train_timestep: int = 0,
    max_train_timestep: int | None = None,
    sample_source_posterior: bool = False,
    sample_target_posterior: bool = True,
    noise_offset: float = 0.0,
    generator: torch.Generator | None = None,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    dropout_random_values: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Encode both images, then corrupt the target once at one sampled t/sample."""
    source_images = batch["source_image"].to(device)
    target_images = batch["target_image"].to(device)
    with torch.no_grad():
        source_latents = encode_images_to_latents(
            bundle, source_images, sample_posterior=sample_source_posterior, generator=generator
        )
        target_latents = encode_images_to_latents(
            bundle, target_images, sample_posterior=sample_target_posterior, generator=generator
        )
    batch_size = target_latents.shape[0]
    scheduler = bundle["scheduler_train"]
    if timesteps is None:
        timesteps = sample_diffusion_timesteps(
            batch_size, scheduler, target_latents.device, generator,
            strategy=timestep_sampling, min_timestep=min_train_timestep,
            max_timestep=max_train_timestep,
        )
    else:
        timesteps = timesteps.to(device=target_latents.device, dtype=torch.long)
    if noise is None:
        noise = _random_normal_like(target_latents, generator)
        if noise_offset:
            offset_shape = (batch_size, target_latents.shape[1], 1, 1)
            offset = torch.randn(offset_shape, generator=generator, device=torch.device(getattr(generator, "device", target_latents.device)) if generator is not None else target_latents.device, dtype=target_latents.dtype).to(target_latents.device)
            noise = noise + float(noise_offset) * offset
    else:
        noise = noise.to(target_latents)
    # Critical: one closed-form add_noise call, never a loop over scheduler steps.
    noisy_target_latents = scheduler.add_noise(target_latents, noise, timesteps)
    masks = sample_conditioning_dropout(
        batch_size,
        conditioning_dropout_prob,
        device=target_latents.device,
        generator=generator,
        random_values=dropout_random_values,
    )
    return {
        "source_images": source_images,
        "target_images": target_images,
        "source_latents": source_latents,
        "target_latents": target_latents,
        "noise": noise,
        "timesteps": timesteps,
        "noisy_target_latents": noisy_target_latents,
        "dropout_masks": masks,
    }


def run_training_step(
    *,
    bundle: Mapping[str, Any],
    loss_fn,
    batch: Mapping[str, Any],
    device: torch.device,
    prompts: Sequence[str] | None = None,
    prompt_key: str = "target_prompt",
    target_prompt_policy: str = "numeric",
    generic_prompt_prob: float = 0.0,
    numeric_prompt_prob: float = 1.0,
    prompt_policy_random_values: torch.Tensor | None = None,
    prepared: dict[str, Any] | None = None,
    amp_enabled: bool = True,
    amp_dtype: str | torch.dtype = "auto",
    conditioning_dropout_prob: float = 0.05,
    timestep_sampling: str = "uniform",
    min_train_timestep: int = 0,
    max_train_timestep: int | None = None,
    sample_source_posterior: bool = False,
    sample_target_posterior: bool = True,
    noise_offset: float = 0.0,
    identity_loss_on_image_dropped_samples: bool = False,
    generator: torch.Generator | None = None,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    dropout_random_values: torch.Tensor | None = None,
    global_step: int = 0,
    return_debug_tensors: bool = False,
) -> dict[str, Any]:
    if prepared is None:
        prepared = prepare_training_batch(
            bundle=bundle, batch=batch, device=device,
            conditioning_dropout_prob=conditioning_dropout_prob,
            timestep_sampling=timestep_sampling,
            min_train_timestep=min_train_timestep,
            max_train_timestep=max_train_timestep,
            sample_source_posterior=sample_source_posterior,
            sample_target_posterior=sample_target_posterior,
            noise_offset=noise_offset, generator=generator,
            timesteps=timesteps, noise=noise,
            dropout_random_values=dropout_random_values,
        )
    if prompts is None and prompt_key == "target_prompt":
        prompt_selection = select_training_prompts(
            batch,
            target_prompt_policy=target_prompt_policy,
            generic_prompt_prob=generic_prompt_prob,
            numeric_prompt_prob=numeric_prompt_prob,
            generator=generator,
            random_values=prompt_policy_random_values,
        )
        selected_prompts = prompt_selection["prompts"]
    else:
        selected_prompts = list(prompts if prompts is not None else batch[prompt_key])
        prompt_selection = {
            "prompts": selected_prompts,
            "generic_mask": None,
            "numeric_mask": None,
            "generic_count": 0,
            "numeric_count": 0,
            "generic_fraction": 0.0,
            "numeric_fraction": 0.0,
            "policy": "explicit",
        }
    if len(selected_prompts) != prepared["target_latents"].shape[0]:
        raise ValueError("Prompt count does not equal batch size")
    with torch.no_grad():
        text_embeddings = encode_prompts(bundle, selected_prompts, device=device)
        null_embeddings = encode_prompts(bundle, [""], device=device)
    conditioned_text, conditioned_source = apply_conditioning_dropout(
        text_embeddings, prepared["source_latents"], null_embeddings, prepared["dropout_masks"]
    )
    model_input = build_conditioned_unet_input(
        prepared["noisy_target_latents"], conditioned_source,
        source_conditioning=bundle.get("source_conditioning", "concat"),
    )
    identity_mask = None
    if not identity_loss_on_image_dropped_samples:
        identity_mask = ~prepared["dropout_masks"]["image_dropped"]
    age_conditioning = compute_age_delta_embedding(
        bundle,
        batch.get("delta_age"),
        batch_size=prepared["target_latents"].shape[0],
        source_age=batch.get("source_age"),
        target_age=batch.get("target_age"),
    )
    unet_kwargs = {
        "encoder_hidden_states": conditioned_text,
        "return_dict": True,
    }
    if age_conditioning is not None:
        unet_kwargs["timestep_cond"] = age_conditioning
    with autocast_ctx(device, enabled=amp_enabled, amp_dtype=amp_dtype):
        model_pred = bundle["unet"](
            model_input,
            prepared["timesteps"],
            **unet_kwargs,
        ).sample
        # Autocast may emit BF16/FP16 while VAE latents are stored in FP32.
        # Compute the objective in latent storage precision; the cast remains
        # differentiable and prevents low-precision auxiliary reconstruction.
        model_pred_for_loss = model_pred.to(prepared["target_latents"].dtype)
        loss_out = loss_fn(
            model_pred=model_pred_for_loss,
            noise=prepared["noise"],
            noisy_target_latents=prepared["noisy_target_latents"],
            target_latents=prepared["target_latents"],
            timesteps=prepared["timesteps"],
            source_images=prepared["source_images"],
            target_images=prepared["target_images"],
            source_ages=batch.get("source_age"),
            target_ages=batch["target_age"],
            delta_ages=batch.get("delta_age"),
            identity_sample_mask=identity_mask,
            global_step=global_step,
            return_per_sample=True,
        )
    raw_target = get_diffusion_target(
        bundle["scheduler_train"], prepared["target_latents"], prepared["noise"], prepared["timesteps"]
    )
    raw_diffusion = (model_pred_for_loss.float() - raw_target.float()).square().flatten(1).mean(1).mean()
    diagnostics = {
        **timestep_statistics(prepared["timesteps"], bundle["scheduler_train"]),
        **conditioning_dropout_statistics(prepared["dropout_masks"]),
        "raw_diffusion_mse": float(raw_diffusion.detach()),
        "source_age_mean": float(batch["source_age"].float().mean()),
        "target_age_mean": float(batch["target_age"].float().mean()),
        "delta_age_mean": float(batch["delta_age"].float().mean()),
    }
    age_conditioner = bundle.get("age_conditioner")
    age_scale = getattr(age_conditioner, "age_scale", None)
    if age_scale is not None:
        diagnostics["age_conditioner_scale"] = float(age_scale.detach())
    result = {
        "loss_out": loss_out,
        "prepared": prepared,
        "diagnostics": diagnostics,
        "prompt_selection": prompt_selection,
    }
    if return_debug_tensors:
        result["debug"] = {
            "model_input": model_input,
            "model_pred": model_pred_for_loss,
            "model_pred_compute_dtype": model_pred.dtype,
            "conditioned_text": conditioned_text,
            "conditioned_source_latents": conditioned_source,
            "age_conditioning": age_conditioning,
        }
    return result
