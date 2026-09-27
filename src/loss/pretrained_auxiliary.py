"""Differentiable loaders for the maintained ArcFace and MiVOLO auxiliaries."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .auxiliary_adapters import AgeEstimatorAdapter, IdentityEncoderAdapter


IDENTITY_MODEL_ID = "py-feat/arcface_r50"
AGE_MODEL_ID = "iitolstykh/mivolo_v2"
# Immutable Hugging Face snapshot.  The model repository uses remote code;
# keeping this revision fixed prevents a later code upload from silently
# changing the executable model implementation.
AGE_MODEL_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"
DEFAULT_IDENTITY_MODEL_ID = IDENTITY_MODEL_ID
DEFAULT_AGE_MODEL_ID = AGE_MODEL_ID


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
        # The maintained MiVOLO implementation contains normalization paths
        # that promote activations to float32. Half-precision weights then fail
        # in its BatchNorm layers, so the frozen estimator is kept in FP32.
        self.model = model.float()
        self.input_size = int(input_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images_01.device.type, enabled=False):
            resized = F.interpolate(
                images_01.float(),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            face_input = (resized - self.mean.to(resized)) / self.std.to(resized)
            body_input = (torch.zeros_like(resized) - self.mean.to(resized)) / self.std.to(resized)
            output = self.model(
                faces_input=face_input,
                body_input=body_input,
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
            "ArcFace loading requires py-feat from the auxiliary dependencies. "
            "Install with: pip install -e '.[auxiliary]'. "
            "MiVOLO must be installed separately with --no-deps because its "
            "timm pin conflicts with py-feat."
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
    _patch_mivolo_timm_compatibility()
    resolved_revision = revision or AGE_MODEL_REVISION
    return AutoModelForImageClassification.from_pretrained(
        model_id,
        revision=resolved_revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=dtype,
    )


def _patch_mivolo_timm_compatibility() -> None:
    """Bridge MiVOLO's legacy private timm import to current timm releases.

    MiVOLO imports private helpers from timm locations that changed over time:
    ``remap_checkpoint(model, state_dict)`` became
    ``remap_state_dict(state_dict, model)``, and ``split_model_name_tag`` moved
    from ``_pretrained`` to ``_registry``.  The adapters preserve MiVOLO's old
    imports while delegating to the maintained implementations.
    """
    import inspect

    import timm.models._helpers as helpers
    import timm.models._pretrained as pretrained
    import timm.models._registry as registry
    import timm.models.volo as volo

    if not hasattr(helpers, "remap_checkpoint"):
        remap_state_dict = getattr(helpers, "remap_state_dict", None)
        if remap_state_dict is None:
            raise ImportError(
                "Installed timm is incompatible with MiVOLO: neither remap_checkpoint "
                "nor remap_state_dict is available"
            )

        def remap_checkpoint(model, state_dict):
            return remap_state_dict(state_dict, model)

        helpers.remap_checkpoint = remap_checkpoint

    # MiVOLO imports this registry helper from the old _pretrained module.
    # Current timm keeps it in _registry instead.
    if not hasattr(pretrained, "split_model_name_tag"):
        split_model_name_tag = getattr(registry, "split_model_name_tag", None)
        if split_model_name_tag is None:
            raise ImportError(
                "Installed timm is incompatible with MiVOLO: split_model_name_tag "
                "is unavailable"
            )
        pretrained.split_model_name_tag = split_model_name_tag

    # timm added ``pos_drop_rate`` in the middle of VOLO's constructor after
    # MiVOLO's subclass was published.  MiVOLO calls this constructor with the
    # old positional layout, so translate that one legacy call shape.
    if getattr(volo.VOLO, "_face_aging_mivolo_compat", False):
        return
    parameters = inspect.signature(volo.VOLO.__init__).parameters
    if "pos_drop_rate" not in parameters:
        return
    original_init = volo.VOLO.__init__

    def compat_init(self, *args, **kwargs):
        is_legacy_mivolo_call = (
            len(args) == 21
            and callable(args[16])
            and isinstance(args[17], tuple)
            and isinstance(args[18], bool)
            and isinstance(args[19], bool)
        )
        if is_legacy_mivolo_call and "pos_drop_rate" not in kwargs:
            return original_init(
                self,
                layers=args[0],
                img_size=args[1],
                in_chans=args[2],
                num_classes=args[3],
                global_pool=args[4],
                patch_size=args[5],
                stem_hidden_dim=args[6],
                embed_dims=args[7],
                num_heads=args[8],
                downsamples=args[9],
                outlook_attention=args[10],
                mlp_ratio=args[11],
                qkv_bias=args[12],
                drop_rate=args[13],
                pos_drop_rate=0.0,
                attn_drop_rate=args[14],
                drop_path_rate=args[15],
                norm_layer=args[16],
                post_layers=args[17],
                use_aux_head=args[18],
                use_mix_token=args[19],
                pooling_scale=args[20],
                **kwargs,
            )
        return original_init(self, *args, **kwargs)

    volo.VOLO.__init__ = compat_init
    volo.VOLO._face_aging_mivolo_compat = True


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
        dtype=torch.float32,
        revision=age_revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    ).to(device=resolved_device, dtype=torch.float32)
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
        "dtype": torch.float32,
        "requested_dtype": resolved_dtype,
        "identity_dtype": torch.float32,
        "age_dtype": torch.float32,
        "activation_checkpointing": bool(activation_checkpointing),
        "mivolo_body_input": "normalized_black_missing_crop",
    }
