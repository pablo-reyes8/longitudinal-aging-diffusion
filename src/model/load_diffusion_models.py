"""Single-backbone SD1.x model construction for longitudinal face aging.

Imports of Diffusers and Transformers are deliberately lazy: structural tests,
adapter checkpoint inspection, and helper tests work without either dependency
and never trigger a model download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .DoRa import inject_manual_dora_unet
from .LoRa import DEFAULT_ATTENTION_TARGETS, inject_manual_lora_unet
from .age_conditioning import (
    AgeConditionerV2,
    AgeDeltaConditioner,
    infer_unet_time_embedding_dim,
    install_age_conditioned_time_embedding,
)


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LEGACY_RUNWAY_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_EXTERNAL_VAE_ID = "stabilityai/sd-vae-ft-mse"


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except StopIteration as exc:
        raise ValueError(f"Module {type(module).__name__} has no parameters") from exc
    return parameter.device, parameter.dtype


def resolve_device_dtype(
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = dtype if dtype is not None else (torch.float16 if resolved_device.type == "cuda" else torch.float32)
    if resolved_device.type == "cpu" and resolved_dtype == torch.float16:
        raise ValueError("float16 SD execution on CPU is unsupported; use torch.float32 or bfloat16")
    return resolved_device, resolved_dtype


def is_no_vae_checkpoint(model_id: str | Path | None) -> bool:
    normalized = str(model_id or "").lower()
    return any(marker in normalized for marker in ("novae", "no_vae", "no-vae"))


def resolve_vae_loading_policy(
    model_id: str | Path,
    vae_id: str | Path | None = None,
    force_external_vae: bool | None = None,
) -> bool:
    use_external = bool(force_external_vae) if force_external_vae is not None else (vae_id is not None or is_no_vae_checkpoint(model_id))
    if use_external and vae_id is None:
        raise ValueError(
            f"Checkpoint {model_id!s} requires an external VAE; provide vae_id "
            f"(for example {DEFAULT_EXTERNAL_VAE_ID!r})."
        )
    return use_external


def freeze_module(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "total_tensors": len(parameters),
        "trainable_tensors": sum(parameter.requires_grad for parameter in parameters),
    }


def load_diffusion_components(
    model_id: str | Path = DEFAULT_MODEL_ID,
    *,
    vae_id: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    force_external_vae: bool | None = None,
    revision: str | None = None,
    variant: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load one tokenizer, text encoder, U-Net, VAE and two schedulers."""
    try:
        from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise ImportError(
            "Model loading requires diffusers and transformers. Install them on the server; "
            "the offline structural tests do not require them."
        ) from exc

    resolved_device, resolved_dtype = resolve_device_dtype(device, dtype)
    model_id = str(model_id)
    external_vae = resolve_vae_loading_policy(model_id, vae_id, force_external_vae)
    common = {"revision": revision, "token": token, "local_files_only": local_files_only}
    common = {key: value for key, value in common.items() if value is not None}
    weight_args = {**common, "torch_dtype": resolved_dtype}
    if variant is not None:
        weight_args["variant"] = variant

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **common)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **weight_args).to(resolved_device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **weight_args).to(resolved_device)
    scheduler_train = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **common)
    if external_vae:
        # A model revision/variant belongs to the backbone and may not exist in
        # the independent VAE repository.
        external_vae_args = {
            "torch_dtype": resolved_dtype,
            "local_files_only": local_files_only,
        }
        if token is not None:
            external_vae_args["token"] = token
        vae = AutoencoderKL.from_pretrained(str(vae_id), **external_vae_args).to(resolved_device)
    else:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **weight_args).to(resolved_device)
    scheduler_infer = DDIMScheduler.from_config(scheduler_train.config)
    freeze_module(vae)
    freeze_module(text_encoder)
    freeze_module(unet)
    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "scheduler_train": scheduler_train,
        "scheduler_infer": scheduler_infer,
        "device": resolved_device,
        "weight_dtype": resolved_dtype,
        "uses_external_vae": external_vae,
    }


def _update_unet_in_channels_config(unet: nn.Module, in_channels: int) -> None:
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=in_channels)
    elif isinstance(unet.config, dict):
        unet.config["in_channels"] = in_channels
    else:
        setattr(unet.config, "in_channels", in_channels)


def expand_unet_conv_in_for_source_conditioning(
    unet: nn.Module,
    *,
    source_channels: int = 4,
) -> dict[str, Any]:
    """Expand SD's input convolution from 4 to 8 channels, preserving function."""
    if not hasattr(unet, "conv_in") or not isinstance(unet.conv_in, nn.Conv2d):
        raise TypeError("U-Net must expose conv_in as nn.Conv2d")
    old_conv = unet.conv_in
    original_channels = old_conv.in_channels
    if original_channels != 4:
        raise ValueError(f"Expected an SD1.x 4-channel conv_in, found {original_channels}")
    if old_conv.groups != 1:
        raise ValueError("Grouped conv_in is not supported for source-channel expansion")
    new_channels = original_channels + source_channels
    new_conv = nn.Conv2d(
        new_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :original_channels].copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    _update_unet_in_channels_config(unet, new_channels)
    return {
        "original_in_channels": original_channels,
        "source_channels": source_channels,
        "expanded_in_channels": new_channels,
        "copied_weight_max_error": float((new_conv.weight[:, :original_channels] - old_conv.weight).detach().abs().max()),
        "source_weight_max_abs": float(new_conv.weight[:, original_channels:].detach().abs().max()),
        "bias_copied": old_conv.bias is None or torch.equal(new_conv.bias, old_conv.bias),
    }


def cast_trainable_parameters_to_fp32(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()


def configure_face_aging_trainable_parameters(
    vae: nn.Module,
    text_encoder: nn.Module,
    unet: nn.Module,
    *,
    trainable_dtype: torch.dtype = torch.float32,
) -> tuple[list[nn.Parameter], list[str]]:
    """Freeze everything except adapter tensors and the expanded ``conv_in``."""
    freeze_module(vae)
    freeze_module(text_encoder)
    unet.requires_grad_(False)
    allowed_adapter_markers = (".lora_down.", ".lora_up.", ".magnitude")
    for name, parameter in unet.named_parameters():
        if name.startswith("conv_in.") or any(marker in name for marker in allowed_adapter_markers):
            parameter.requires_grad_(True)
    unet.train()
    if trainable_dtype == torch.float32:
        cast_trainable_parameters_to_fp32(unet)
    elif any(parameter.dtype != trainable_dtype for parameter in unet.parameters() if parameter.requires_grad):
        raise ValueError("Only explicit FP32 promotion is supported for trainable parameters")
    names, parameters = [], []
    for name, parameter in unet.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            parameters.append(parameter)
    if not names or not any(name.startswith("conv_in.") for name in names):
        raise RuntimeError("Trainable policy failed to include conv_in")
    if not any("lora_" in name or name.endswith("magnitude") for name in names):
        raise RuntimeError("Trainable policy found no adapter parameters")
    return parameters, names


def assemble_face_aging_diffusion_bundle(
    components: Mapping[str, Any],
    *,
    model_id: str | Path,
    vae_id: str | Path | None = None,
    adapter_type: str = "lora",
    rank: int = 16,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_ATTENTION_TARGETS,
    source_conditioning: str = "concat",
    use_age_delta_conditioning: bool = True,
    age_conditioning_mode: str = "delta_mlp",
    use_age_conditioner_v2: bool = True,
    age_conditioning_version: str | None = None,
    age_delta_scale: float = 80.0,
    age_condition_hidden_dim: int = 256,
    age_condition_output_dim: int | None = None,
    num_fourier_frequencies: int = 8,
    age_condition_use_raw_scalars: bool = True,
    age_condition_use_gate: bool = True,
    trainable_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> dict[str, Any]:
    """Adapt already loaded components; used by the online builder and offline tests."""
    required = {"vae", "tokenizer", "text_encoder", "unet", "scheduler_train", "scheduler_infer"}
    missing = required - set(components)
    if missing:
        raise KeyError(f"Missing diffusion components: {sorted(missing)}")
    if source_conditioning != "concat":
        raise ValueError("V1 only supports source_conditioning='concat'")
    adapter_type = adapter_type.lower().strip()
    if adapter_type not in {"lora", "dora"}:
        raise ValueError("adapter_type must be 'lora' or 'dora'")
    if age_conditioning_mode != "delta_mlp":
        raise ValueError("V1 supports age_conditioning_mode='delta_mlp'")
    resolved_age_version = age_conditioning_version or (
        "v2_fourier" if use_age_conditioner_v2 else "v1_delta"
    )
    expected_age_version = "v2_fourier" if use_age_conditioner_v2 else "v1_delta"
    if resolved_age_version != expected_age_version:
        raise ValueError(
            f"age_conditioning_version={resolved_age_version!r} conflicts with "
            f"use_age_conditioner_v2={use_age_conditioner_v2}"
        )
    unet = components["unet"]
    conv_report = expand_unet_conv_in_for_source_conditioning(unet)
    injection = inject_manual_lora_unet if adapter_type == "lora" else inject_manual_dora_unet
    injection(
        unet,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_suffixes=tuple(target_modules),
        require_all_targets=True,
        verbose=verbose,
    )
    trainable_params, trainable_names = configure_face_aging_trainable_parameters(
        components["vae"], components["text_encoder"], unet, trainable_dtype=trainable_dtype
    )
    age_conditioner = None
    age_conditioning_config = None
    if use_age_delta_conditioning:
        inferred_output_dim = infer_unet_time_embedding_dim(unet)
        output_dim = inferred_output_dim if age_condition_output_dim is None else int(age_condition_output_dim)
        if output_dim != inferred_output_dim:
            raise ValueError(
                f"age_condition_output_dim={output_dim} must match U-Net time embedding {inferred_output_dim}"
            )
        install_age_conditioned_time_embedding(unet, output_dim)
        unet_device = next(unet.parameters()).device
        if use_age_conditioner_v2:
            age_conditioner = AgeConditionerV2(
                num_fourier_frequencies=num_fourier_frequencies,
                hidden_dim=age_condition_hidden_dim,
                output_dim=output_dim,
                delta_age_scale=age_delta_scale,
                use_raw_scalars=age_condition_use_raw_scalars,
                use_gate=age_condition_use_gate,
            )
            conditioner_prefix = "age_conditioner"
        else:
            age_conditioner = AgeDeltaConditioner(
                hidden_dim=age_condition_hidden_dim,
                output_dim=output_dim,
                age_delta_scale=age_delta_scale,
            )
            conditioner_prefix = "age_delta_conditioner"
        age_conditioner = age_conditioner.to(device=unet_device, dtype=trainable_dtype)
        age_conditioning_config = age_conditioner.get_config()
        conditioner_named = [
            (f"{conditioner_prefix}.{name}", parameter)
            for name, parameter in age_conditioner.named_parameters()
        ]
        trainable_names.extend(name for name, _ in conditioner_named)
        trainable_params.extend(parameter for _, parameter in conditioner_named)
    cross_attention_dim = _config_value(unet.config, "cross_attention_dim")
    config = {
        "model_id": str(model_id),
        "vae_id": str(vae_id) if vae_id is not None else None,
        "adapter_type": adapter_type,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(target_modules),
        "source_conditioning": source_conditioning,
        "unet_in_channels": int(_config_value(unet.config, "in_channels")),
        "unet_cross_attention_dim": cross_attention_dim,
        "trainable_dtype": str(trainable_dtype),
        "use_age_delta_conditioning": bool(use_age_delta_conditioning),
        "age_conditioning_mode": age_conditioning_mode,
        "use_age_conditioner_v2": bool(use_age_conditioner_v2),
        "age_conditioning_version": resolved_age_version,
        "age_delta_scale": float(age_delta_scale),
        "age_condition_hidden_dim": int(age_condition_hidden_dim),
        "age_condition_output_dim": age_conditioning_config["output_dim"] if age_conditioning_config else None,
        "num_fourier_frequencies": int(num_fourier_frequencies),
        "age_condition_use_raw_scalars": bool(age_condition_use_raw_scalars),
        "age_condition_use_gate": bool(age_condition_use_gate),
    }
    bundle = {
        **dict(components),
        "name": f"face-aging-{adapter_type}-{str(model_id).split('/')[-1]}",
        "model_id": str(model_id),
        "vae_id": str(vae_id) if vae_id is not None else None,
        "adapter_type": adapter_type,
        "adapter_config": {
            "rank": int(rank), "alpha": float(alpha), "dropout": float(dropout),
            "target_modules": list(target_modules),
        },
        "adapter_report": unet._face_aging_adapter_report,
        "source_conditioning": source_conditioning,
        "use_age_delta_conditioning": bool(use_age_delta_conditioning),
        "age_conditioning_mode": age_conditioning_mode,
        "use_age_conditioner_v2": bool(use_age_conditioner_v2),
        "age_conditioning_version": resolved_age_version,
        "age_delta_scale": float(age_delta_scale),
        "age_conditioner": age_conditioner,
        # Compatibility alias for code written against milestone 07.
        "age_delta_conditioner": age_conditioner,
        "age_conditioning_config": age_conditioning_config,
        "conv_in_report": conv_report,
        "trainable_params": trainable_params,
        "trainable_param_names": trainable_names,
        "param_stats": {
            "unet": count_parameters(unet),
            "vae": count_parameters(components["vae"]),
            "text_encoder": count_parameters(components["text_encoder"]),
            "age_conditioner": count_parameters(age_conditioner) if age_conditioner is not None else None,
            "age_delta_conditioner": count_parameters(age_conditioner) if age_conditioner is not None else None,
        },
        "config": config,
    }
    if verbose:
        print_parameter_report(bundle)
    return bundle


def build_face_aging_diffusion_bundle(
    model_id: str | Path = DEFAULT_MODEL_ID,
    vae_id: str | Path | None = None,
    *,
    adapter_type: str = "lora",
    rank: int = 16,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_ATTENTION_TARGETS,
    source_conditioning: str = "concat",
    use_age_delta_conditioning: bool = True,
    age_conditioning_mode: str = "delta_mlp",
    use_age_conditioner_v2: bool = True,
    age_conditioning_version: str | None = None,
    age_delta_scale: float = 80.0,
    age_condition_hidden_dim: int = 256,
    age_condition_output_dim: int | None = None,
    num_fourier_frequencies: int = 8,
    age_condition_use_raw_scalars: bool = True,
    age_condition_use_gate: bool = True,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trainable_dtype: torch.dtype = torch.float32,
    force_external_vae: bool | None = None,
    revision: str | None = None,
    variant: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    load_auxiliary_models: bool = False,
    identity_model_id: str = "py-feat/arcface_r50",
    age_model_id: str = "iitolstykh/mivolo_v2",
    auxiliary_dtype: torch.dtype | None = None,
    identity_revision: str | None = None,
    age_revision: str | None = None,
    auxiliary_cache_dir: str | None = None,
    auxiliary_trust_remote_code: bool = False,
    auxiliary_activation_checkpointing: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load one SD1.x backbone and optionally its frozen auxiliary losses."""
    components = load_diffusion_components(
        model_id,
        vae_id=vae_id,
        device=device,
        dtype=dtype,
        force_external_vae=force_external_vae,
        revision=revision,
        variant=variant,
        token=token,
        local_files_only=local_files_only,
    )
    bundle = assemble_face_aging_diffusion_bundle(
        components,
        model_id=model_id,
        vae_id=vae_id,
        adapter_type=adapter_type,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        source_conditioning=source_conditioning,
        use_age_delta_conditioning=use_age_delta_conditioning,
        age_conditioning_mode=age_conditioning_mode,
        use_age_conditioner_v2=use_age_conditioner_v2,
        age_conditioning_version=age_conditioning_version,
        age_delta_scale=age_delta_scale,
        age_condition_hidden_dim=age_condition_hidden_dim,
        age_condition_output_dim=age_condition_output_dim,
        num_fourier_frequencies=num_fourier_frequencies,
        age_condition_use_raw_scalars=age_condition_use_raw_scalars,
        age_condition_use_gate=age_condition_use_gate,
        trainable_dtype=trainable_dtype,
        verbose=verbose,
    )
    if load_auxiliary_models:
        from src.loss import load_pretrained_auxiliary_models

        auxiliary = load_pretrained_auxiliary_models(
            identity_model_id=identity_model_id,
            age_model_id=age_model_id,
            device=bundle["device"],
            dtype=auxiliary_dtype,
            identity_revision=identity_revision,
            age_revision=age_revision,
            token=token,
            cache_dir=auxiliary_cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=auxiliary_trust_remote_code,
            activation_checkpointing=auxiliary_activation_checkpointing,
        )
        bundle.update(auxiliary)
        bundle["config"].update({
            "identity_model_id": identity_model_id,
            "age_model_id": age_model_id,
            "auxiliary_activation_checkpointing": bool(auxiliary_activation_checkpointing),
        })
        if verbose:
            print(
                "Auxiliary models:", identity_model_id, "+", age_model_id,
                f"| device={auxiliary['device']}",
                f"ArcFace dtype={auxiliary['identity_dtype']}",
                f"MiVOLO dtype={auxiliary['age_dtype']}",
                f"| activation checkpointing={auxiliary['activation_checkpointing']}",
            )
    return bundle


def print_parameter_report(bundle: Mapping[str, Any], max_names: int = 30) -> None:
    stats = bundle["param_stats"]
    print(f"\n[{bundle['name']}] model={bundle['model_id']}")
    print("UNet in_channels:", bundle["config"]["unet_in_channels"])
    print("Adapter coverage:", bundle["adapter_report"]["counts_by_target"])
    print(
        "UNet parameters:", f"{stats['unet']['trainable_params']:,} / {stats['unet']['total_params']:,}",
        f"({stats['unet']['trainable_pct']:.4f}% trainable)",
    )
    print("Trainable tensors:", len(bundle["trainable_param_names"]))
    if bundle.get("age_conditioner") is not None:
        stats_age = stats["age_conditioner"]
        print(
            f"Age conditioner ({bundle['age_conditioning_version']}):",
            f"{stats_age['trainable_params']:,} trainable parameters",
            f"| scale={bundle['age_delta_scale']}",
        )
    for name in bundle["trainable_param_names"][:max_names]:
        print(" -", name)
    if len(bundle["trainable_param_names"]) > max_names:
        print(f" - ... {len(bundle['trainable_param_names']) - max_names} more")


def get_bundle_trainable_named_parameters(bundle: Mapping[str, Any]) -> list[tuple[str, nn.Parameter]]:
    named = [
        (name, parameter)
        for name, parameter in bundle["unet"].named_parameters()
        if parameter.requires_grad
    ]
    conditioner = bundle.get("age_conditioner")
    if conditioner is None:
        conditioner = bundle.get("age_delta_conditioner")
    if conditioner is not None:
        prefix = (
            "age_conditioner"
            if bundle.get("age_conditioning_version") == "v2_fourier"
            else "age_delta_conditioner"
        )
        named.extend(
            (f"{prefix}.{name}", parameter)
            for name, parameter in conditioner.named_parameters()
            if parameter.requires_grad
        )
    return named


def build_face_aging_optimizer(
    bundle: Mapping[str, Any],
    *,
    lr_lora: float = 3e-5,
    lr_conv_in: float = 5e-6,
    lr_age_conditioner: float = 1e-4,
    weight_decay: float = 1e-2,
    conv_in_weight_decay: float | None = None,
    age_conditioner_weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
) -> torch.optim.Optimizer:
    named = get_bundle_trainable_named_parameters(bundle)
    conv = [(name, parameter) for name, parameter in named if name.startswith("conv_in.")]
    age_conditioner = [
        (name, parameter) for name, parameter in named
        if name.startswith("age_delta_conditioner.") or name.startswith("age_conditioner.")
    ]
    adapters = [
        (name, parameter) for name, parameter in named
        if not name.startswith("conv_in.")
        and not name.startswith("age_delta_conditioner.")
        and not name.startswith("age_conditioner.")
    ]
    if not conv or not adapters:
        raise RuntimeError(f"Expected both adapter and conv_in parameter groups; conv={len(conv)}, adapters={len(adapters)}")
    if bundle.get("use_age_delta_conditioning", False) and not age_conditioner:
        raise RuntimeError("Age-delta conditioning is enabled but has no optimizer parameters")
    all_ids = [id(parameter) for _, parameter in conv + adapters + age_conditioner]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("A trainable parameter appears in more than one optimizer group")
    conv_decay = weight_decay if conv_in_weight_decay is None else conv_in_weight_decay
    groups = [
        {
            "params": [parameter for _, parameter in adapters],
            "lr": lr_lora,
            "weight_decay": weight_decay,
            "group_name": "adapter",
        },
        {
            "params": [parameter for _, parameter in conv],
            "lr": lr_conv_in,
            "weight_decay": conv_decay,
            "group_name": "conv_in",
        },
    ]
    if age_conditioner:
        groups.append({
            "params": [parameter for _, parameter in age_conditioner],
            "lr": lr_age_conditioner,
            "weight_decay": age_conditioner_weight_decay,
            "group_name": "age_conditioner",
        })
    optimizer = optimizer_cls(
        groups,
        weight_decay=0.0,
        betas=betas,
        eps=eps,
    )
    expected = {id(parameter) for _, parameter in named}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("Optimizer membership does not exactly match the trainable parameter set")
    return optimizer


@torch.no_grad()
def encode_prompts(
    bundle: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tokenizer, text_encoder = bundle["tokenizer"], bundle["text_encoder"]
    model_device, _ = _module_device_dtype(text_encoder)
    resolved_device = torch.device(device) if device is not None else model_device
    tokens = tokenizer(
        list(prompts), padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    kwargs = {"input_ids": tokens.input_ids.to(resolved_device), "return_dict": True}
    if hasattr(tokens, "attention_mask") and tokens.attention_mask is not None:
        kwargs["attention_mask"] = tokens.attention_mask.to(resolved_device)
    hidden = text_encoder(**kwargs).last_hidden_state
    return hidden.to(dtype=dtype) if dtype is not None else hidden


@torch.no_grad()
def encode_images_to_latents(
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    *,
    sample_posterior: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    vae = bundle["vae"]
    device, dtype = _module_device_dtype(vae)
    posterior = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
    raw = posterior.sample(generator=generator) if sample_posterior else posterior.mean
    latents = raw * float(_config_value(vae.config, "scaling_factor"))
    if not torch.isfinite(latents).all():
        raise FloatingPointError("VAE produced non-finite latents")
    return latents


def prepare_source_target_latents(
    bundle: Mapping[str, Any],
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    *,
    sample_posterior: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "source_latents": encode_images_to_latents(bundle, source_images, sample_posterior=sample_posterior, generator=generator),
        "target_latents": encode_images_to_latents(bundle, target_images, sample_posterior=sample_posterior, generator=generator),
    }


def build_conditioned_unet_input(
    noisy_target_latents: torch.Tensor,
    source_latents: torch.Tensor,
    *,
    source_conditioning: str = "concat",
) -> torch.Tensor:
    if source_conditioning != "concat":
        raise ValueError("V1 only supports source_conditioning='concat'")
    if noisy_target_latents.ndim != 4 or source_latents.ndim != 4:
        raise ValueError("Both latent tensors must have shape [B, C, H, W]")
    if noisy_target_latents.shape[0] != source_latents.shape[0] or noisy_target_latents.shape[2:] != source_latents.shape[2:]:
        raise ValueError(f"Source/target latent shapes are incompatible: {tuple(noisy_target_latents.shape)} vs {tuple(source_latents.shape)}")
    if noisy_target_latents.device != source_latents.device:
        raise ValueError("Source and noisy target latents must be on the same device")
    if noisy_target_latents.dtype != source_latents.dtype:
        raise ValueError("Source and noisy target latents must have the same dtype")
    return torch.cat((noisy_target_latents, source_latents), dim=1)


def tokenizer_audit(bundle: Mapping[str, Any], prompts: Sequence[str]) -> list[dict[str, Any]]:
    tokenizer = bundle["tokenizer"]
    audit = []
    for prompt in prompts:
        encoded = tokenizer(prompt, padding=False, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        token_ids = encoded.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids) if hasattr(tokenizer, "convert_ids_to_tokens") else None
        audit.append({"prompt": prompt, "token_ids": token_ids, "tokens": tokens, "non_padding_tokens": len(token_ids)})
    return audit


def save_face_aging_adapter(bundle: Mapping[str, Any], checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in get_bundle_trainable_named_parameters(bundle)
    }
    payload = {"format_version": 1, "config": dict(bundle["config"]), "adapter_state_dict": state}
    torch.save(payload, path)
    return path


def load_face_aging_adapter(
    bundle: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    strict_backbone: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location=map_location, weights_only=True)
    saved = payload.get("config", {})
    current = bundle["config"]
    protected = (
        "model_id", "adapter_type", "rank", "target_modules", "source_conditioning",
        "unet_in_channels", "unet_cross_attention_dim", "use_age_delta_conditioning",
        "age_conditioning_mode", "age_delta_scale", "age_condition_hidden_dim",
        "age_condition_output_dim", "use_age_conditioner_v2", "age_conditioning_version",
        "num_fourier_frequencies", "age_condition_use_raw_scalars", "age_condition_use_gate",
    )
    v2_only = {
        "use_age_conditioner_v2", "age_conditioning_version", "num_fourier_frequencies",
        "age_condition_use_raw_scalars", "age_condition_use_gate",
    }
    mismatches = {
        key: {"checkpoint": saved.get(key), "bundle": current.get(key)}
        for key in protected
        if (key not in v2_only or key in saved) and saved.get(key) != current.get(key)
    }
    if strict_backbone and mismatches:
        raise ValueError(f"Adapter checkpoint is incompatible with this bundle: {mismatches}")
    current_parameters = dict(get_bundle_trainable_named_parameters(bundle))
    expected_names = set(bundle["trainable_param_names"])
    saved_state = payload.get("adapter_state_dict", {})
    if set(saved_state) != expected_names:
        raise ValueError(
            f"Adapter state keys do not match bundle trainables; missing={sorted(expected_names-set(saved_state))}, "
            f"unexpected={sorted(set(saved_state)-expected_names)}"
        )
    with torch.no_grad():
        for name, tensor in saved_state.items():
            parameter = current_parameters[name]
            if parameter.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}: checkpoint={tuple(tensor.shape)}, bundle={tuple(parameter.shape)}")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return {"config": saved, "mismatches": mismatches, "loaded_tensors": len(saved_state)}
