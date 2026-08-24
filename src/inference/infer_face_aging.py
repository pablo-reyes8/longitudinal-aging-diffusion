"""Public direct and inverse-diffusion face-aging inference APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.model import compute_age_delta_embedding, encode_prompts

from .cfg_guidance import predict_three_way_cfg
from .ddim_inversion import (
    ddim_invert_source_image,
    edit_from_inverted_latent,
    scheduler_reverse_step,
)
from .diagnostics import compute_face_aging_diagnostics
from .inference_utils import (
    create_inference_scheduler,
    decode_latents_to_tensor,
    encode_image_to_latent,
    make_generator,
    module_device_dtype,
    prepare_inference_image,
    randn_like,
    scheduler_set_timesteps,
    tensor_to_pil,
)
from .prompt_building import build_inference_prompt_pack


def _scale_model_input(scheduler, latents, timestep):
    method = getattr(scheduler, "scale_model_input", None)
    return method(latents, timestep) if method is not None else latents


@torch.no_grad()
def _direct_latent_edit(
    *,
    bundle,
    source_latents: torch.Tensor,
    target_prompt: str,
    reference_prompt: str,
    negative_prompt: str,
    scheduler,
    num_inference_steps: int,
    strength: float,
    text_guidance_scale: float,
    age_guidance_scale: float,
    image_guidance_scale: float,
    use_cfg: bool,
    generator,
    age_conditioning: torch.Tensor | None,
    return_intermediates: bool,
) -> dict[str, Any]:
    if not 0 < strength <= 1:
        raise ValueError("strength must be in (0,1]")
    timesteps = scheduler_set_timesteps(scheduler, num_inference_steps, source_latents.device)
    init_steps = max(1, min(num_inference_steps, int(num_inference_steps * strength)))
    start_index = max(0, num_inference_steps - init_steps)
    denoising_timesteps = timesteps[start_index:]
    start_timestep = denoising_timesteps[0]
    noise = randn_like(source_latents, generator)
    repeated_timestep = start_timestep.reshape(1).expand(source_latents.shape[0])
    latents = scheduler.add_noise(source_latents, noise, repeated_timestep)
    initial_latents = latents.detach().cpu().clone()
    embeddings = encode_prompts(bundle, [target_prompt] * source_latents.shape[0], device=source_latents.device)
    reference_embeddings = encode_prompts(
        bundle, [reference_prompt] * source_latents.shape[0], device=source_latents.device
    )
    null_embeddings = encode_prompts(bundle, [negative_prompt], device=source_latents.device)
    trajectory = [initial_latents] if return_intermediates else None
    guided_norms = []
    for timestep in denoising_timesteps:
        model_latents = _scale_model_input(scheduler, latents, timestep)
        prediction = predict_three_way_cfg(
            bundle=bundle,
            target_latents=model_latents,
            timestep=timestep,
            source_latents=source_latents,
            full_text_embeddings=embeddings,
            null_text_embeddings=null_embeddings,
            reference_text_embeddings=reference_embeddings,
            text_guidance_scale=text_guidance_scale,
            age_guidance_scale=age_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            use_cfg=use_cfg,
            age_conditioning=age_conditioning,
        ).to(latents.dtype)
        guided_norms.append(float(prediction.float().norm()))
        latents = scheduler_reverse_step(scheduler, prediction, timestep, latents, generator)
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"Direct inference produced NaN/Inf at timestep {int(timestep)}")
        if trajectory is not None:
            trajectory.append(latents.detach().cpu().clone())
    return {
        "latents": latents,
        "initial_noisy_latents": initial_latents,
        "noise": noise.detach().cpu(),
        "start_timestep": int(start_timestep),
        "denoising_timesteps": denoising_timesteps.detach().cpu(),
        "trajectory": trajectory,
        "guided_prediction_norms": guided_norms,
    }


def _format_images(image_tensor: torch.Tensor, output_type: str):
    if output_type == "tensor":
        return image_tensor
    if output_type == "latent":
        return None
    if output_type != "pil":
        raise ValueError("output_type must be 'pil', 'tensor', or 'latent'")
    images = [tensor_to_pil(image_tensor[index:index + 1]) for index in range(image_tensor.shape[0])]
    return images[0] if len(images) == 1 else images


@torch.inference_mode()
def infer_face_aging(
    *,
    bundle,
    image,
    target_prompt: str | None = None,
    target_age: int | None = None,
    source_prompt: str | None = None,
    source_age: int | None = None,
    mode: str = "direct",
    use_inverse_diffusion: bool | None = None,
    num_inference_steps: int = 50,
    strength: float = 0.35,
    inversion_strength: float = 1.0,
    text_guidance_scale: float = 7.0,
    text_reference_mode: str = "source_age",
    age_guidance_scale: float = 3.0,
    image_guidance_scale: float = 1.5,
    negative_prompt: str = "",
    prompt_style: str = "selfage",
    use_cfg: bool = True,
    seed: int = 42,
    generator=None,
    image_size: int = 256,
    output_type: str = "pil",
    return_dict: bool = True,
    return_latents: bool = False,
    return_intermediates: bool = False,
    compute_diagnostics: bool = True,
    use_age_delta_conditioning: bool | None = None,
    override_delta_age: float | None = None,
    identity_encoder=None,
    age_estimator=None,
    device: str | torch.device | None = None,
) -> dict[str, Any] | Image.Image | torch.Tensor:
    """Age one source image through direct img2img or deterministic DDIM inversion."""
    if use_inverse_diffusion is not None:
        mode = "inverse" if use_inverse_diffusion else "direct"
    if mode not in {"direct", "inverse"}:
        raise ValueError("mode must be 'direct' or 'inverse'")
    if text_reference_mode not in {"null", "generic", "source_age"}:
        raise ValueError("text_reference_mode must be 'null', 'generic', or 'source_age'")
    if age_guidance_scale < 0:
        raise ValueError("age_guidance_scale must be non-negative")
    prompt_pack = build_inference_prompt_pack(
        target_prompt=target_prompt, target_age=target_age,
        source_prompt=source_prompt, source_age=source_age,
        prompt_style=prompt_style, negative_prompt=negative_prompt,
    )
    # Milestone 09 deliberately changes only direct editing. Inverse diffusion
    # remains on the legacy null-referenced guidance path.
    resolved_text_reference_mode = text_reference_mode if mode == "direct" else "null"
    if resolved_text_reference_mode == "source_age":
        reference_prompt = prompt_pack["source_prompt"]
    elif resolved_text_reference_mode == "generic":
        reference_prompt = prompt_pack["generic_prompt"]
    else:
        reference_prompt = negative_prompt
    resolved_age_guidance_scale = (
        float(age_guidance_scale) if mode == "direct" else float(text_guidance_scale)
    )
    model_device, _ = module_device_dtype(bundle["unet"])
    resolved_device = torch.device(device) if device is not None else model_device
    if resolved_device != model_device:
        bundle["unet"].to(resolved_device)
        bundle["vae"].to(resolved_device)
        bundle["text_encoder"].to(resolved_device)
        if bundle.get("age_delta_conditioner") is not None:
            bundle["age_delta_conditioner"].to(resolved_device)
    vae_dtype = module_device_dtype(bundle["vae"])[1]
    source_images = prepare_inference_image(
        image, image_size=image_size, device=resolved_device, dtype=vae_dtype
    )
    source_latents = encode_image_to_latent(bundle, source_images, sample_posterior=False)
    bundle_age_conditioning = bool(bundle.get("use_age_delta_conditioning", False))
    resolved_age_conditioning = (
        bundle_age_conditioning
        if use_age_delta_conditioning is None
        else bool(use_age_delta_conditioning)
    )
    if resolved_age_conditioning and not bundle_age_conditioning:
        raise ValueError("The loaded bundle has no age-delta conditioner")
    if resolved_age_conditioning:
        if prompt_pack["source_age"] is None or prompt_pack["target_age"] is None:
            raise ValueError(
                "source_age and target_age are required when age-delta conditioning is enabled"
            )
        true_delta_value = float(prompt_pack["target_age"] - prompt_pack["source_age"])
        delta_value = (
            true_delta_value
            if override_delta_age is None
            else float(override_delta_age)
        )
        effective_target_age_value = float(prompt_pack["source_age"]) + delta_value
        edit_age_conditioning = compute_age_delta_embedding(
            bundle,
            torch.full((source_latents.shape[0],), delta_value, device=resolved_device),
            batch_size=source_latents.shape[0],
            source_age=torch.full(
                (source_latents.shape[0],), float(prompt_pack["source_age"]), device=resolved_device
            ),
            target_age=torch.full(
                (source_latents.shape[0],), effective_target_age_value, device=resolved_device
            ),
        )
        source_age_conditioning = compute_age_delta_embedding(
            bundle,
            torch.zeros(source_latents.shape[0], device=resolved_device),
            batch_size=source_latents.shape[0],
            source_age=torch.full(
                (source_latents.shape[0],), float(prompt_pack["source_age"]), device=resolved_device
            ),
            target_age=torch.full(
                (source_latents.shape[0],), float(prompt_pack["source_age"]), device=resolved_device
            ),
        )
    else:
        true_delta_value = (
            float(prompt_pack["target_age"] - prompt_pack["source_age"])
            if prompt_pack["source_age"] is not None and prompt_pack["target_age"] is not None
            else None
        )
        delta_value = None
        effective_target_age_value = None
        edit_age_conditioning = source_age_conditioning = None
    scheduler = create_inference_scheduler(bundle)
    actual_generator = make_generator(resolved_device, seed, generator)
    previous_mode = bundle["unet"].training
    previous_age_mode = (
        bundle["age_delta_conditioner"].training
        if bundle.get("age_delta_conditioner") is not None else None
    )
    bundle["unet"].eval(); bundle["vae"].eval(); bundle["text_encoder"].eval()
    if bundle.get("age_delta_conditioner") is not None:
        bundle["age_delta_conditioner"].eval()
    try:
        if mode == "direct":
            edit = _direct_latent_edit(
                bundle=bundle, source_latents=source_latents,
                target_prompt=prompt_pack["target_prompt"],
                reference_prompt=reference_prompt,
                negative_prompt=negative_prompt,
                scheduler=scheduler, num_inference_steps=num_inference_steps,
                strength=strength,
                text_guidance_scale=text_guidance_scale,
                age_guidance_scale=resolved_age_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                use_cfg=use_cfg, generator=actual_generator,
                age_conditioning=edit_age_conditioning,
                return_intermediates=return_intermediates,
            )
            inversion = None
        else:
            inversion = ddim_invert_source_image(
                bundle=bundle, source_latents=source_latents,
                source_prompt=prompt_pack["source_prompt"], scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                inversion_strength=inversion_strength,
                # Unit scales reconstruct the full source condition during inversion.
                text_guidance_scale=1.0, image_guidance_scale=1.0,
                negative_prompt=negative_prompt, use_cfg=use_cfg,
                age_conditioning=source_age_conditioning,
                return_intermediates=return_intermediates,
            )
            edit = edit_from_inverted_latent(
                bundle=bundle, inverted_latents=inversion["inverted_latents"],
                source_latents=source_latents,
                target_prompt=prompt_pack["target_prompt"], scheduler=scheduler,
                denoising_timesteps=inversion["denoising_timesteps"],
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt, use_cfg=use_cfg,
                age_conditioning=edit_age_conditioning,
                generator=actual_generator,
                return_intermediates=return_intermediates,
            )
        image_tensor = decode_latents_to_tensor(bundle, edit["latents"])
    finally:
        bundle["unet"].train(previous_mode)
        if bundle.get("age_delta_conditioner") is not None:
            bundle["age_delta_conditioner"].train(previous_age_mode)
        bundle["vae"].eval(); bundle["text_encoder"].eval()
    formatted = _format_images(image_tensor, output_type)
    diagnostics = None
    if compute_diagnostics and prompt_pack["target_age"] is not None:
        diagnostics = compute_face_aging_diagnostics(
            bundle=bundle,
            source_image=source_images.float().div(2).add(0.5),
            generated_image=image_tensor,
            target_age=prompt_pack["target_age"],
            source_age=prompt_pack["source_age"],
            image_size=image_size,
            identity_encoder=identity_encoder,
            age_estimator=age_estimator,
        )
    result: dict[str, Any] = {
        "image": formatted,
        "image_tensor": image_tensor.detach().cpu(),
        "mode": mode,
        "target_prompt": prompt_pack["target_prompt"],
        "source_prompt": prompt_pack["source_prompt"],
        "source_age": prompt_pack["source_age"],
        "target_age": prompt_pack["target_age"],
        "num_inference_steps": num_inference_steps,
        "strength": strength if mode == "direct" else None,
        "inversion_strength": inversion_strength if mode == "inverse" else None,
        "text_guidance_scale": text_guidance_scale,
        "text_reference_mode": resolved_text_reference_mode,
        "requested_text_reference_mode": text_reference_mode,
        "reference_prompt": reference_prompt,
        "age_guidance_scale": resolved_age_guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "seed": int(seed),
        "diagnostics": diagnostics,
        "metadata": {
            "image_size": image_size,
            "use_cfg": use_cfg,
            "negative_prompt": negative_prompt,
            "prompt_style": prompt_style,
            "prompt_warnings": prompt_pack["warnings"],
            "use_age_delta_conditioning": resolved_age_conditioning,
            "delta_age": delta_value,
            "true_delta_age": true_delta_value,
            "effective_target_age": effective_target_age_value,
            "override_delta_age": override_delta_age,
            "start_timestep": edit.get("start_timestep", inversion.get("start_timestep") if inversion else None),
        },
    }
    if return_latents:
        result["latents"] = edit["latents"].detach().cpu()
    if inversion is not None:
        result["inverted_latents"] = inversion["inverted_latents"].detach().cpu()
        result["start_timestep"] = inversion["start_timestep"]
    if return_intermediates:
        result["intermediates"] = {
            "edit_trajectory": edit["trajectory"],
            "inversion_trajectory": inversion["trajectory"] if inversion else None,
            "guided_prediction_norms": edit["guided_prediction_norms"],
        }
    return result if return_dict else formatted


def infer_face_aging_direct(**kwargs):
    kwargs.pop("mode", None)
    kwargs.pop("use_inverse_diffusion", None)
    return infer_face_aging(mode="direct", **kwargs)


def infer_face_aging_inverse(**kwargs):
    kwargs.pop("mode", None)
    kwargs.pop("use_inverse_diffusion", None)
    return infer_face_aging(mode="inverse", **kwargs)


def save_inference_image(result: dict[str, Any], output_path: str | Path) -> Path:
    image = result.get("image")
    if not isinstance(image, Image.Image):
        image = tensor_to_pil(result["image_tensor"])
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
