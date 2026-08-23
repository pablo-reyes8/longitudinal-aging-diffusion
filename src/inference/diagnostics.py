"""Age calibration and identity preservation diagnostics for generated faces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import torch
import torch.nn.functional as F

from .inference_utils import module_device_dtype, prepare_inference_image


def _image_to_01(
    image: str | Path | Image.Image | torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Prepare one image as FP32 BCHW in [0,1] without model-specific transforms."""
    if torch.is_tensor(image):
        tensor = image.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Diagnostic tensor must have shape [3,H,W] or [B,3,H,W]")
        if not torch.isfinite(tensor).all():
            raise ValueError("Diagnostic image contains NaN/Inf")
        minimum, maximum = float(tensor.min()), float(tensor.max())
        if minimum >= -1.0001 and maximum <= 1.0001:
            normalized = tensor.float() if minimum < -1e-6 else tensor.float().mul(2).sub(1)
        else:
            raise ValueError("Diagnostic tensor must be in [0,1] or [-1,1]")
        return prepare_inference_image(
            normalized, image_size=image_size, dtype=torch.float32
        ).div(2).add(0.5).clamp(0, 1)
    return prepare_inference_image(
        image, image_size=image_size, dtype=torch.float32
    ).div(2).add(0.5).clamp(0, 1)


def _module_device(module: torch.nn.Module) -> torch.device:
    return module_device_dtype(module)[0]


@torch.inference_mode()
def compute_face_aging_diagnostics(
    bundle: Mapping[str, Any],
    source_image: str | Path | Image.Image | torch.Tensor,
    generated_image: str | Path | Image.Image | torch.Tensor,
    target_age: int | float,
    *,
    image_size: int = 256,
    identity_encoder=None,
    age_estimator=None,
) -> dict[str, float] | None:
    """Measure generated age and source/generated identity cosine.

    The already configured loss adapters are called directly, so ArcFace and
    MiVOLO use exactly the same differentiable preprocessing as training.
    Returns ``None`` when either auxiliary adapter is unavailable.
    """
    identity_encoder = identity_encoder or bundle.get("identity_encoder")
    age_estimator = age_estimator or bundle.get("age_estimator")
    if identity_encoder is None or age_estimator is None:
        return None
    if isinstance(target_age, bool) or not isinstance(target_age, (int, float)):
        raise TypeError("target_age must be numeric for diagnostics")

    source_01 = _image_to_01(source_image, image_size=image_size)
    generated_01 = _image_to_01(generated_image, image_size=image_size)
    if source_01.shape[0] != generated_01.shape[0]:
        raise ValueError("Source and generated diagnostic batches must match")

    identity_device = _module_device(identity_encoder)
    source_embeddings = identity_encoder(source_01.to(identity_device))
    generated_embeddings = identity_encoder(generated_01.to(identity_device))
    cosine = F.cosine_similarity(source_embeddings.float(), generated_embeddings.float(), dim=-1)

    age_device = _module_device(age_estimator)
    predicted_age = age_estimator(generated_01.to(age_device)).float()
    if predicted_age.numel() != generated_01.shape[0]:
        raise ValueError("Age estimator must produce one scalar per generated image")

    return {
        "target_age": float(target_age),
        "predicted_generated_age": float(predicted_age.mean().cpu()),
        "identity_cosine_source_generated": float(cosine.mean().cpu()),
    }
