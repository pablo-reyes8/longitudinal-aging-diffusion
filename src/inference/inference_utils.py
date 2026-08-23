"""Training-consistent image, latent, scheduler, and dtype helpers."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


def module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def inference_autocast(bundle: Mapping[str, Any], device: torch.device):
    dtype = bundle.get("weight_dtype", module_device_dtype(bundle["unet"])[1])
    if device.type in {"cpu", "cuda"} and dtype in {torch.float16, torch.bfloat16}:
        if device.type == "cpu" and dtype == torch.float16:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def prepare_inference_image(
    image: str | Path | Image.Image | torch.Tensor,
    *,
    image_size: int = 256,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """EXIF transpose, RGB, center-square crop, resize, and normalize to [-1,1]."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if isinstance(image, (str, Path)):
        with Image.open(Path(image).expanduser()) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB").copy()
    elif isinstance(image, Image.Image):
        pil = ImageOps.exif_transpose(image).convert("RGB")
    elif torch.is_tensor(image):
        tensor = image.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Tensor image must have shape [3,H,W] or [B,3,H,W]")
        if not torch.isfinite(tensor).all():
            raise ValueError("Tensor image contains NaN/Inf")
        if float(tensor.min()) < -1.0001 or float(tensor.max()) > 1.0001:
            raise ValueError("Tensor image must already be normalized to [-1,1]")
        height, width = tensor.shape[-2:]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        tensor = tensor[..., top:top + side, left:left + side]
        tensor = F.interpolate(tensor.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
        return tensor.to(device=device, dtype=dtype)
    else:
        raise TypeError("image must be a path, PIL.Image, or normalized tensor")
    width, height = pil.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    pil = pil.crop((left, top, left + side, top + side)).resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(pil, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def encode_image_to_latent(bundle, images: torch.Tensor, *, sample_posterior: bool = False, generator=None) -> torch.Tensor:
    vae = bundle["vae"]
    device, dtype = module_device_dtype(vae)
    posterior = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
    if sample_posterior:
        raw = posterior.sample(generator=generator)
    else:
        raw = getattr(posterior, "mean", None)
        if raw is None:
            raw = posterior.mode()
    latents = raw * float(getattr(vae.config, "scaling_factor"))
    if not torch.isfinite(latents).all():
        raise FloatingPointError("VAE encoding produced NaN/Inf")
    return latents


@torch.no_grad()
def decode_latents_to_tensor(bundle, latents: torch.Tensor) -> torch.Tensor:
    vae = bundle["vae"]
    device, dtype = module_device_dtype(vae)
    scaling = float(getattr(vae.config, "scaling_factor"))
    decoded = vae.decode(latents.to(device=device, dtype=dtype) / scaling).sample
    images = (decoded.float() / 2 + 0.5).clamp(0, 1)
    if not torch.isfinite(images).all():
        raise FloatingPointError("VAE decoding produced NaN/Inf")
    return images


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("tensor_to_pil expects one image")
        image = image[0]
    array = image.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def create_inference_scheduler(bundle):
    scheduler = bundle.get("scheduler_infer")
    if scheduler is not None and hasattr(scheduler, "set_timesteps") and hasattr(scheduler, "step"):
        return copy.deepcopy(scheduler)
    try:
        from diffusers import DDIMScheduler
    except ImportError as exc:
        raise ImportError(
            "DDIM inference requires diffusers on the server, or a bundle scheduler_infer "
            "implementing set_timesteps/step."
        ) from exc
    source = bundle["scheduler_train"]
    return DDIMScheduler.from_config(source.config)


def scheduler_set_timesteps(scheduler, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)
        scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler.timesteps


def make_generator(device: torch.device, seed: int, generator=None):
    if generator is not None:
        return generator
    generator_device = device.type if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def randn_like(reference: torch.Tensor, generator) -> torch.Tensor:
    generator_device = torch.device(getattr(generator, "device", reference.device))
    return torch.randn(reference.shape, generator=generator, device=generator_device, dtype=reference.dtype).to(reference.device)


def scheduler_prediction_type(scheduler) -> str:
    config = scheduler.config
    return str(config.get("prediction_type", "epsilon") if isinstance(config, Mapping) else getattr(config, "prediction_type", "epsilon"))
