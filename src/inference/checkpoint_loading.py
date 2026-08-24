"""Load inference or training-resume weights without optimizer dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.model import build_face_aging_diffusion_bundle, get_bundle_trainable_named_parameters


PROTECTED_KEYS = (
    "model_id", "adapter_type", "rank", "target_modules",
    "source_conditioning", "unet_in_channels", "unet_cross_attention_dim",
    "use_age_delta_conditioning", "age_conditioning_mode", "age_delta_scale",
    "age_condition_hidden_dim", "age_condition_output_dim",
    "use_age_conditioner_v2", "age_conditioning_version", "num_fourier_frequencies",
    "age_condition_use_raw_scalars", "age_condition_use_gate",
)
V2_ONLY_KEYS = {
    "use_age_conditioner_v2", "age_conditioning_version", "num_fourier_frequencies",
    "age_condition_use_raw_scalars", "age_condition_use_gate",
}


def _checkpoint_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload.get("config") or payload.get("bundle_config") or {})


def _checkpoint_weights(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("adapter_state_dict") or payload.get("trainable_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint contains neither adapter_state_dict nor trainable_state_dict")
    return state


def load_face_aging_adapter_for_inference(
    bundle,
    checkpoint_path: str | Path,
    *,
    strict_config: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
    saved_config = _checkpoint_config(payload)
    current_config = dict(bundle.get("config", {}))
    mismatches = {
        key: {"checkpoint": saved_config.get(key), "bundle": current_config.get(key)}
        for key in PROTECTED_KEYS
        if (key not in V2_ONLY_KEYS or key in saved_config)
        and saved_config.get(key) != current_config.get(key)
    }
    if strict_config and mismatches:
        raise ValueError(f"Inference checkpoint is incompatible with bundle: {mismatches}")
    state = _checkpoint_weights(payload)
    parameters = dict(get_bundle_trainable_named_parameters(bundle))
    expected = set(bundle.get("trainable_param_names", [name for name, parameter in parameters.items() if parameter.requires_grad]))
    if set(state) != expected:
        raise ValueError(f"Checkpoint trainable keys mismatch: missing={sorted(expected-set(state))}, unexpected={sorted(set(state)-expected)}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
            parameter.copy_(value.to(parameter))
    bundle["unet"].eval()
    bundle["vae"].eval()
    bundle["text_encoder"].eval()
    return {
        "checkpoint_kind": payload.get("kind", "legacy_adapter"),
        "loaded_tensors": len(state),
        "config": saved_config,
        "mismatches": mismatches,
        "training_config": payload.get("training_config"),
    }


def load_face_aging_inference_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    strict_config: bool = True,
    **model_loading_kwargs,
):
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
    config = _checkpoint_config(payload)
    adapter = payload.get("adapter_config", {})
    model_id = config.get("model_id") or payload.get("model_id")
    if not model_id:
        raise ValueError("Checkpoint does not record model_id; construct the bundle explicitly")
    rank = config.get("rank", adapter.get("rank", 16))
    alpha = config.get("alpha", adapter.get("alpha", rank))
    dropout = config.get("dropout", adapter.get("dropout", 0.0))
    target_modules = config.get("target_modules", adapter.get("target_modules"))
    kwargs = dict(
        model_id=model_id,
        vae_id=config.get("vae_id") or payload.get("vae_id"),
        adapter_type=config.get("adapter_type", "lora"),
        rank=rank, alpha=alpha, dropout=dropout,
        source_conditioning=config.get("source_conditioning", "concat"),
        use_age_delta_conditioning=config.get("use_age_delta_conditioning", False),
        age_conditioning_mode=config.get("age_conditioning_mode", "delta_mlp"),
        use_age_conditioner_v2=config.get(
            "use_age_conditioner_v2",
            config.get("age_conditioning_version") == "v2_fourier",
        ),
        age_conditioning_version=config.get("age_conditioning_version", "v1_delta"),
        age_delta_scale=config.get("age_delta_scale", 80.0),
        age_condition_hidden_dim=config.get("age_condition_hidden_dim", 128),
        age_condition_output_dim=config.get("age_condition_output_dim"),
        num_fourier_frequencies=config.get("num_fourier_frequencies", 8),
        age_condition_use_raw_scalars=config.get("age_condition_use_raw_scalars", True),
        age_condition_use_gate=config.get("age_condition_use_gate", True),
        device=device, dtype=dtype,
        local_files_only=local_files_only, token=token,
        **model_loading_kwargs,
    )
    if target_modules is not None:
        kwargs["target_modules"] = target_modules
    bundle = build_face_aging_diffusion_bundle(**kwargs)
    report = load_face_aging_adapter_for_inference(bundle, checkpoint_path, strict_config=strict_config)
    bundle["inference_checkpoint_report"] = report
    return bundle
