"""Public direct and inverse-diffusion face-aging inference APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.model import encode_prompts

from .cfg_guidance import predict_three_way_cfg
from .ddim_inversion import (
    ddim_invert_source_image,
    edit_from_inverted_latent,
    scheduler_reverse_step,
)
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
    negative_prompt: str,
    scheduler,
    num_inference_steps: int,
    strength: float,
    text_guidance_scale: float,
    image_guidance_scale: float,
    use_cfg: bool,
    generator,
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
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            use_cfg=use_cfg,
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
    strength: float = 0.45,
    inversion_strength: float = 1.0,
    text_guidance_scale: float = 7.0,
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
    compute_diagnostics: bool = False,
    identity_encoder=None,
    age_estimator=None,
    device: str | torch.device | None = None,
) -> dict[str, Any] | Image.Image | torch.Tensor:
    """Age one source image through direct img2img or deterministic DDIM inversion."""
    if use_inverse_diffusion is not None:
        mode = "inverse" if use_inverse_diffusion else "direct"
    if mode not in {"direct", "inverse"}:
        raise ValueError("mode must be 'direct' or 'inverse'")
    prompt_pack = build_inference_prompt_pack(
        target_prompt=target_prompt, target_age=target_age,
        source_prompt=source_prompt, source_age=source_age,
        prompt_style=prompt_style, negative_prompt=negative_prompt,
    )
    model_device, _ = module_device_dtype(bundle["unet"])
    resolved_device = torch.device(device) if device is not None else model_device
    if resolved_device != model_device:
        bundle["unet"].to(resolved_device)
        bundle["vae"].to(resolved_device)
        bundle["text_encoder"].to(resolved_device)
    vae_dtype = module_device_dtype(bundle["vae"])[1]
    source_images = prepare_inference_image(
        image, image_size=image_size, device=resolved_device, dtype=vae_dtype
    )
    source_latents = encode_image_to_latent(bundle, source_images, sample_posterior=False)
    scheduler = create_inference_scheduler(bundle)
    actual_generator = make_generator(resolved_device, seed, generator)
    previous_mode = bundle["unet"].training
    bundle["unet"].eval(); bundle["vae"].eval(); bundle["text_encoder"].eval()
    try:
        if mode == "direct":
            edit = _direct_latent_edit(
                bundle=bundle, source_latents=source_latents,
                target_prompt=prompt_pack["target_prompt"], negative_prompt=negative_prompt,
                scheduler=scheduler, num_inference_steps=num_inference_steps,
                strength=strength,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                use_cfg=use_cfg, generator=actual_generator,
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
                generator=actual_generator,
                return_intermediates=return_intermediates,
            )
        image_tensor = decode_latents_to_tensor(bundle, edit["latents"])
    finally:
        bundle["unet"].train(previous_mode)
        bundle["vae"].eval(); bundle["text_encoder"].eval()
    formatted = _format_images(image_tensor, output_type)
    diagnostics = {}
    if compute_diagnostics:
        identity_encoder = identity_encoder or bundle.get("identity_encoder")
        age_estimator = age_estimator or bundle.get("age_estimator")
        source_01 = (source_images.float() / 2 + 0.5).clamp(0, 1)
        if identity_encoder is not None:
            source_embedding = identity_encoder(source_01)
            generated_embedding = identity_encoder(image_tensor.to(source_01.device))
            diagnostics["identity_cosine_source_generated"] = float((source_embedding * generated_embedding).sum(-1).mean())
        if age_estimator is not None:
            diagnostics["predicted_generated_age"] = float(age_estimator(image_tensor.to(source_01.device)).mean())
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
        "image_guidance_scale": image_guidance_scale,
        "seed": int(seed),
        "diagnostics": diagnostics,
        "metadata": {
            "image_size": image_size,
            "use_cfg": use_cfg,
            "negative_prompt": negative_prompt,
            "prompt_style": prompt_style,
            "prompt_warnings": prompt_pack["warnings"],
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
