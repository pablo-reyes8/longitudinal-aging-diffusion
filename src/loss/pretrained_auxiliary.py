"""Differentiable loaders for the maintained ArcFace and MiVOLO auxiliaries."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .auxiliary_adapters import AgeEstimatorAdapter, IdentityEncoderAdapter


IDENTITY_MODEL_ID = "py-feat/arcface_r50"
AGE_MODEL_ID = "iitolstykh/mivolo_v2"
DEFAULT_IDENTITY_MODEL_ID = IDENTITY_MODEL_ID
DEFAULT_AGE_MODEL_ID = AGE_MODEL_ID


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


class ArcFaceR50InputAdapter(nn.Module):
    """Resize differentiably and call py-feat's ArcFace `[0,1]` wrapper.

    py-feat's ArcFace wrapper performs part of its input normalization in
    float32.  Keeping its BatchNorm-heavy IResNet in float16 can therefore mix
    float activations with half weights under AMP.  ArcFace is intentionally
    kept in float32 and excluded from autocast; gradients still flow through
    the input image while its frozen weights receive no gradients.
    """

    def __init__(self, model: nn.Module, input_size: int = 112) -> None:
        super().__init__()
        self.model = model.float()
        self.input_size = int(input_size)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images_01.device.type, enabled=False):
            faces = F.interpolate(
                images_01.float(),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            return self.model(faces)


class MiVOLOFaceOnlyAgeModel(nn.Module):
    """Differentiable face-only bridge for MiVOLO's face+body architecture.

    The official preprocessing represents a missing crop as a black image before
    ImageNet normalization. We use that exact convention for the unavailable body
    crop instead of pretending that the face crop is also a body crop.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: int = 384,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.model = model
        self.input_size = int(input_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            images_01,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        face_input = (resized - self.mean.to(resized)) / self.std.to(resized)
        body_input = (torch.zeros_like(resized) - self.mean.to(resized)) / self.std.to(resized)
        model_dtype = _module_dtype(self.model)
        output = self.model(
            faces_input=face_input.to(dtype=model_dtype),
            body_input=body_input.to(dtype=model_dtype),
            return_dict=True,
        )
        age = getattr(output, "age_output", None)
        if age is None:
            raise TypeError("MiVOLO output does not expose age_output")
        return age


def _load_arcface(
    model_id: str,
    *,
    revision: str | None,
    token: str | bool | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> nn.Module:
    try:
        from feat.identity_detectors.arcface.arcface_model import ArcFace
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "ArcFace loading requires the optional auxiliary dependencies. "
            "Install this project with: pip install -e '.[auxiliary]'"
        ) from exc
    model = ArcFace(backbone="r50")
    weights = hf_hub_download(
        repo_id=model_id,
        filename="arcface_r50.safetensors",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    missing, unexpected = model.net.load_state_dict(load_file(weights), strict=False)
    real_missing = [name for name in missing if "num_batches_tracked" not in name]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ArcFace checkpoint mismatch: missing={real_missing}, unexpected={list(unexpected)}"
        )
    return model


def _load_mivolo(
    model_id: str,
    *,
    dtype: torch.dtype,
    revision: str | None,
    token: str | bool | None,
    cache_dir: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> nn.Module:
    if not trust_remote_code:
        raise ValueError(
            "MiVOLO uses repository-defined Transformers code. Set trust_remote_code=True "
            "only after reviewing/pinning the repository revision."
        )
    try:
        from transformers import AutoModelForImageClassification
    except ImportError as exc:
        raise ImportError("MiVOLO loading requires transformers and the optional mivolo package") from exc
    return AutoModelForImageClassification.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=dtype,
    )


def load_pretrained_auxiliary_models(
    *,
    identity_model_id: str = DEFAULT_IDENTITY_MODEL_ID,
    age_model_id: str = DEFAULT_AGE_MODEL_ID,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    identity_revision: str | None = None,
    age_revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    activation_checkpointing: bool = True,
) -> dict[str, Any]:
    """Load frozen, differentiable auxiliary adapters resident on one device."""
    resolved_device = torch.device(device)
    resolved_dtype = dtype or (torch.float16 if resolved_device.type == "cuda" else torch.float32)
    if resolved_device.type == "cpu" and resolved_dtype == torch.float16:
        resolved_dtype = torch.float32
    # ArcFace remains FP32 even when the diffusion backbone and MiVOLO use
    # FP16/BF16. See ArcFaceR50InputAdapter for the py-feat dtype constraint.
    arcface = _load_arcface(
        identity_model_id,
        revision=identity_revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device=resolved_device, dtype=torch.float32)
    mivolo = _load_mivolo(
        age_model_id,
        dtype=resolved_dtype,
        revision=age_revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    ).to(resolved_device)
    identity_encoder = IdentityEncoderAdapter(
        ArcFaceR50InputAdapter(arcface),
        activation_checkpointing=activation_checkpointing,
    )
    age_estimator = AgeEstimatorAdapter(
        MiVOLOFaceOnlyAgeModel(mivolo),
        output_type="scalar",
        activation_checkpointing=activation_checkpointing,
    )
    for adapter in (identity_encoder, age_estimator):
        adapter.requires_grad_(False)
        adapter.eval()
    return {
        "identity_encoder": identity_encoder,
        "age_estimator": age_estimator,
        "identity_model_id": identity_model_id,
        "age_model_id": age_model_id,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "identity_dtype": torch.float32,
        "age_dtype": resolved_dtype,
        "activation_checkpointing": bool(activation_checkpointing),
        "mivolo_body_input": "normalized_black_missing_crop",
    }
