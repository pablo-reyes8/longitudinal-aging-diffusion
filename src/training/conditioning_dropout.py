"""InstructPix2Pix-style joint text/image conditioning dropout."""

from __future__ import annotations

import torch


def sample_conditioning_dropout(
    batch_size: int,
    probability: float = 0.05,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if probability < 0 or probability > 1 / 3:
        raise ValueError("conditioning dropout probability must be in [0, 1/3]")
    if random_values is None:
        generator_device = torch.device(getattr(generator, "device", "cpu")) if generator is not None else torch.device(device)
        random_values = torch.rand(batch_size, generator=generator, device=generator_device).to(device)
    else:
        random_values = random_values.to(device)
        if random_values.shape != (batch_size,):
            raise ValueError(f"random_values must have shape [{batch_size}]")
    p = float(probability)
    text_only = random_values < p
    both = (random_values >= p) & (random_values < 2 * p)
    image_only = (random_values >= 2 * p) & (random_values < 3 * p)
    text_dropped = text_only | both
    image_dropped = image_only | both
    return {
        "random_values": random_values,
        "text_only": text_only,
        "both": both,
        "image_only": image_only,
        "none": ~(text_only | both | image_only),
        "text_dropped": text_dropped,
        "image_dropped": image_dropped,
    }


def apply_conditioning_dropout(
    text_embeddings: torch.Tensor,
    source_latents: torch.Tensor,
    null_text_embeddings: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = text_embeddings.shape[0]
    if source_latents.shape[0] != batch:
        raise ValueError("Text and source latent batch sizes differ")
    if null_text_embeddings.shape[0] not in {1, batch} or null_text_embeddings.shape[1:] != text_embeddings.shape[1:]:
        raise ValueError("Null text embeddings must have shape [1,L,D] or [B,L,D]")
    nulls = null_text_embeddings.expand_as(text_embeddings)
    text_mask = masks["text_dropped"].view(batch, *([1] * (text_embeddings.ndim - 1)))
    image_mask = masks["image_dropped"].view(batch, *([1] * (source_latents.ndim - 1)))
    conditioned_text = torch.where(text_mask, nulls, text_embeddings)
    conditioned_source = torch.where(image_mask, torch.zeros_like(source_latents), source_latents)
    return conditioned_text, conditioned_source


def conditioning_dropout_statistics(masks: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "text_dropout_fraction": float(masks["text_only"].float().mean()),
        "image_dropout_fraction": float(masks["image_only"].float().mean()),
        "both_dropout_fraction": float(masks["both"].float().mean()),
        "any_text_dropout_fraction": float(masks["text_dropped"].float().mean()),
        "any_image_dropout_fraction": float(masks["image_dropped"].float().mean()),
        "conditioning_kept_fraction": float(masks["none"].float().mean()),
    }
