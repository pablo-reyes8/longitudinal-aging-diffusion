"""High-level single-model longitudinal face-aging training orchestration."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from src.model import build_face_aging_optimizer, get_bundle_trainable_named_parameters

from .checkpoints import (
    TrainingCheckpointManager,
    atomic_json_save,
    atomic_torch_save,
    build_inference_payload,
    build_training_payload,
    load_training_checkpoint,
)
from .mixed_precision import ensure_trainable_parameters_fp32, setup_device_and_precision
from .prompt_regularization import validate_prompt_policy
from .sampling_monitor import normalize_monitoring_ages, run_face_aging_monitor, sample_monitoring_images
from .scheduler_warmup import WarmupCosineLR, compute_warmup_steps, estimate_optimizer_steps
from .seed import set_seed
from .train_one_epoch import train_one_epoch
from .validate_one_epoch import validate_one_epoch


def _enable_memory_features(bundle, *, gradient_checkpointing: bool, enable_xformers: bool) -> dict[str, Any]:
    report = {"gradient_checkpointing": "disabled", "xformers": "disabled"}
    unet = bundle["unet"]
    if gradient_checkpointing:
        method = getattr(unet, "enable_gradient_checkpointing", None)
        if method is None:
            report["gradient_checkpointing"] = "unavailable"
        else:
            method()
            report["gradient_checkpointing"] = "enabled"
    if enable_xformers:
        method = getattr(unet, "enable_xformers_memory_efficient_attention", None)
        if method is None:
            report["xformers"] = "unavailable"
        else:
            try:
                method()
                report["xformers"] = "enabled"
            except Exception as exc:
                report["xformers"] = f"unavailable ({type(exc).__name__})"
    return report


def _move_training_objects(bundle, loss_fn, device: torch.device) -> None:
    bundle["unet"].to(device)
    if bundle.get("age_delta_conditioner") is not None:
        bundle["age_delta_conditioner"].to(device)
    bundle["vae"].to(device)
    bundle["text_encoder"].to(device)
    loss_fn.to(device)
    bundle["vae"].eval()
    bundle["text_encoder"].eval()
    if loss_fn.identity_encoder is not None:
        loss_fn.identity_encoder.eval()
    if loss_fn.age_estimator is not None:
        loss_fn.age_estimator.eval()


def _validate_config(**config) -> None:
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive")
    if config["max_train_steps"] is not None and config["max_train_steps"] <= 0:
        raise ValueError("max_train_steps must be positive")
    if config["grad_accum_steps"] <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if config["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be positive")
    if not 0 <= config["conditioning_dropout_prob"] <= 1 / 3:
        raise ValueError("conditioning_dropout_prob must be in [0,1/3]")
    if config["use_ema"]:
        raise ValueError("EMA is intentionally unsupported in V1")


def _loader_batch_size(loader) -> int | None:
    return getattr(loader, "batch_size", None)


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1


def _dataset_identity_count(loader) -> int | None:
    manifest = getattr(getattr(loader, "dataset", None), "manifest", None)
    if manifest is None:
        return None
    return len({record.person_id for record in manifest})


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def train_model(
    *,
    bundle: Mapping[str, Any],
    loss_fn,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    max_train_steps: int | None = None,
    optimizer=None,
    lr_scheduler=None,
    lr_lora: float = 3e-5,
    lr_conv_in: float = 5e-6,
    lr_age_conditioner: float = 1e-4,
    weight_decay: float = 1e-2,
    conv_in_weight_decay: float = 1e-2,
    age_conditioner_weight_decay: float = 1e-2,
    use_age_delta_conditioning: bool | None = None,
    age_conditioning_mode: str = "delta_mlp",
    age_delta_scale: float = 80.0,
    use_age_conditioner_v2: bool | None = None,
    age_conditioning_version: str | None = None,
    use_relative_age_loss: bool | None = None,
    relative_age_weight: float | None = None,
    relative_age_loss_type: str | None = None,
    use_preservation_loss: bool | None = None,
    preservation_weight: float | None = None,
    preservation_loss_type: str | None = None,
    preservation_max_delta: float | None = None,
    use_small_delta_weighting: bool | None = None,
    small_delta_threshold: float | None = None,
    small_delta_weight: float | None = None,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.10,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    amp_enabled: bool = True,
    amp_dtype: str | torch.dtype = "auto",
    scaler=None,
    gradient_checkpointing: bool = True,
    enable_xformers: bool = True,
    conditioning_dropout_prob: float = 0.05,
    target_prompt_policy: str = "mixed",
    generic_prompt_prob: float = 0.30,
    numeric_prompt_prob: float = 0.70,
    identity_loss_on_image_dropped_samples: bool = False,
    timestep_sampling: str = "uniform",
    min_train_timestep: int = 0,
    max_train_timestep: int | None = None,
    double_prompt_prob: float = 0.0,
    age_prompt_weight: float = 0.5,
    generic_prompt_weight: float = 0.5,
    sample_source_posterior: bool = False,
    sample_target_posterior: bool = True,
    noise_offset: float = 0.0,
    min_snr_gamma: float | None = 5.0,
    auxiliary_max_timestep: int | None = None,
    checkpoint_dir: str | Path | None = None,
    save_every_epochs: int = 1,
    save_epoch_checkpoints: bool = True,
    max_epoch_checkpoints: int = 5,
    resume_from: str | Path | None = None,
    strict_resume_config: bool = True,
    monitor: str = "val/loss_total",
    monitor_mode: str = "min",
    validate_every_epochs: int = 1,
    max_val_batches: int | None = None,
    deterministic_validation: bool = True,
    validation_seed: int = 2026,
    sample_every_epochs: int = 1,
    sample_fn=None,
    monitoring_dir: str | Path | None = None,
    monitoring_image=None,
    monitoring_target_prompt: str | None = None,
    monitoring_target_age: int | Sequence[int] | None = None,
    monitoring_source_prompt: str | None = None,
    monitoring_source_age: int | None = None,
    monitoring_mode: str = "direct",
    monitoring_use_inverse_diffusion: bool | None = None,
    monitoring_num_inference_steps: int = 30,
    monitoring_strength: float = 0.35,
    monitoring_text_guidance_scale: float = 7.0,
    monitoring_text_reference_mode: str = "source_age",
    monitoring_age_guidance_scale: float = 3.0,
    monitoring_image_guidance_scale: float = 1.5,
    monitoring_seed: int = 2026,
    monitoring_compute_diagnostics: bool = True,
    log_every: int = 25,
    seed: int = 42,
    deterministic: bool = False,
    device: str | torch.device = "auto",
    max_train_batches: int | None = None,
    skip_nonfinite_loss: bool = True,
    abort_after_nonfinite_steps: int = 5,
    safe_optimizer_step: bool = True,
    use_ema: bool = False,
    image_size: int | None = 256,
    prompt_configuration: dict | None = None,
) -> dict[str, Any]:
    """Train the single SD1.5 editing bundle; `max_train_steps` overrides epochs."""
    _validate_config(
        num_epochs=num_epochs, max_train_steps=max_train_steps,
        grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm,
        conditioning_dropout_prob=conditioning_dropout_prob, use_ema=use_ema,
    )
    if validate_every_epochs <= 0 or save_every_epochs <= 0:
        raise ValueError("validation/save epoch intervals must be positive")
    validate_prompt_policy(target_prompt_policy, generic_prompt_prob, numeric_prompt_prob)
    if monitoring_image is not None and checkpoint_dir is None and monitoring_dir is None:
        raise ValueError("Built-in monitoring requires monitoring_dir or checkpoint_dir")
    if monitoring_image is not None and monitoring_target_prompt is None and monitoring_target_age is None:
        raise ValueError("Built-in monitoring requires monitoring_target_prompt or monitoring_target_age")
    monitoring_ages = None
    if monitoring_target_age is not None:
        monitoring_ages, monitoring_is_sweep = normalize_monitoring_ages(monitoring_target_age)
        if monitoring_is_sweep and monitoring_target_prompt is not None:
            raise ValueError("monitoring_target_prompt cannot be combined with multiple target ages")
    resolved_monitoring_mode = (
        ("inverse" if monitoring_use_inverse_diffusion else "direct")
        if monitoring_use_inverse_diffusion is not None else monitoring_mode
    )
    set_seed(seed, deterministic=deterministic)
    precision = setup_device_and_precision(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler)
    resolved_device = precision["device"]
    _move_training_objects(bundle, loss_fn, resolved_device)
    bundle_age_enabled = bool(bundle.get("use_age_delta_conditioning", False))
    requested_age_enabled = bundle_age_enabled if use_age_delta_conditioning is None else bool(use_age_delta_conditioning)
    if requested_age_enabled != bundle_age_enabled:
        raise ValueError(
            "use_age_delta_conditioning must match the already constructed bundle; rebuild the bundle"
        )
    if requested_age_enabled:
        if age_conditioning_mode != bundle.get("age_conditioning_mode"):
            raise ValueError("age_conditioning_mode does not match the bundle")
        if not math.isclose(float(age_delta_scale), float(bundle.get("age_delta_scale"))):
            raise ValueError("age_delta_scale does not match the bundle")
        requested_v2 = (
            bool(bundle.get("use_age_conditioner_v2", False))
            if use_age_conditioner_v2 is None else bool(use_age_conditioner_v2)
        )
        if requested_v2 != bool(bundle.get("use_age_conditioner_v2", False)):
            raise ValueError("use_age_conditioner_v2 must match the constructed bundle")
        if age_conditioning_version is not None and age_conditioning_version != bundle.get("age_conditioning_version"):
            raise ValueError("age_conditioning_version does not match the constructed bundle")
        if monitoring_image is not None and monitoring_source_age is None:
            raise ValueError(
                "monitoring_source_age is required when the bundle uses age-delta conditioning"
            )
    if use_relative_age_loss is not None:
        loss_fn.use_relative_age_loss = bool(use_relative_age_loss)
    if relative_age_weight is not None:
        if relative_age_weight < 0:
            raise ValueError("relative_age_weight must be non-negative")
        loss_fn.relative_age_weight = float(relative_age_weight)
    if relative_age_loss_type is not None:
        if relative_age_loss_type not in {"l1", "mse"}:
            raise ValueError("relative_age_loss_type must be 'l1' or 'mse'")
        loss_fn.relative_age_loss_type = relative_age_loss_type
    if loss_fn.use_relative_age_loss and loss_fn.relative_age_weight > 0 and loss_fn.age_estimator is None:
        raise ValueError("Enabled relative age loss requires an age estimator")
    if use_preservation_loss is not None:
        loss_fn.use_preservation_loss = bool(use_preservation_loss)
    if preservation_weight is not None:
        if preservation_weight < 0:
            raise ValueError("preservation_weight must be non-negative")
        loss_fn.preservation_weight = float(preservation_weight)
    if preservation_loss_type is not None:
        if preservation_loss_type not in {"l1", "mse"}:
            raise ValueError("preservation_loss_type must be 'l1' or 'mse'")
        loss_fn.preservation_loss_type = preservation_loss_type
    if preservation_max_delta is not None:
        if preservation_max_delta < 0:
            raise ValueError("preservation_max_delta must be non-negative")
        loss_fn.preservation_max_delta = float(preservation_max_delta)
    if use_small_delta_weighting is not None:
        loss_fn.use_small_delta_weighting = bool(use_small_delta_weighting)
    if small_delta_threshold is not None:
        if small_delta_threshold < 0:
            raise ValueError("small_delta_threshold must be non-negative")
        loss_fn.small_delta_threshold = float(small_delta_threshold)
    if small_delta_weight is not None:
        if small_delta_weight < 1:
            raise ValueError("small_delta_weight must be >= 1")
        loss_fn.small_delta_weight = float(small_delta_weight)
    trainables = [parameter for _, parameter in get_bundle_trainable_named_parameters(bundle)]
    ensure_trainable_parameters_fp32(trainables)
    memory_features = _enable_memory_features(
        bundle, gradient_checkpointing=gradient_checkpointing, enable_xformers=enable_xformers
    )
    if optimizer is None:
        optimizer = build_face_aging_optimizer(
            bundle, lr_lora=lr_lora, lr_conv_in=lr_conv_in,
            lr_age_conditioner=lr_age_conditioner,
            weight_decay=weight_decay, conv_in_weight_decay=conv_in_weight_decay,
            age_conditioner_weight_decay=age_conditioner_weight_decay,
        )
    expected_parameter_ids = {id(parameter) for parameter in trainables}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_parameter_ids = expected_parameter_ids - optimizer_parameter_ids
    unexpected_parameter_ids = optimizer_parameter_ids - expected_parameter_ids
    if missing_parameter_ids or unexpected_parameter_ids:
        raise ValueError(
            "Optimizer parameters must exactly match all bundle trainables; "
            f"missing={len(missing_parameter_ids)}, unexpected={len(unexpected_parameter_ids)}"
        )
    loss_fn.min_snr_gamma = float(min_snr_gamma) if min_snr_gamma is not None else None
    loss_fn.auxiliary_max_timestep = auxiliary_max_timestep
    batches_per_epoch = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    steps_per_epoch = estimate_optimizer_steps(batches_per_epoch, grad_accum_steps)
    total_planned_steps = max_train_steps if max_train_steps is not None else steps_per_epoch * num_epochs
    epochs_to_run = math.ceil(total_planned_steps / steps_per_epoch) if max_train_steps is not None else num_epochs
    warmup_steps = compute_warmup_steps(total_planned_steps, warmup_ratio)
    if lr_scheduler is None:
        lr_scheduler = WarmupCosineLR(optimizer, total_planned_steps, warmup_steps, min_lr_ratio)
    loader_batch_size = _loader_batch_size(train_loader)
    effective_batch_size = loader_batch_size * grad_accum_steps * _world_size() if loader_batch_size else None
    training_identity_count = _dataset_identity_count(train_loader)
    training_config = {
        "num_epochs": num_epochs, "epochs_to_run": epochs_to_run,
        "max_train_steps": max_train_steps, "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_epoch": steps_per_epoch, "total_planned_optimizer_steps": total_planned_steps,
        "batch_size": loader_batch_size, "grad_accum_steps": grad_accum_steps,
        "training_samples": len(train_loader.dataset), "training_identities": training_identity_count,
        "effective_batch_size": effective_batch_size,
        "lr_lora": lr_lora, "lr_conv_in": lr_conv_in,
        "lr_age_conditioner": lr_age_conditioner,
        "weight_decay": weight_decay, "conv_in_weight_decay": conv_in_weight_decay,
        "age_conditioner_weight_decay": age_conditioner_weight_decay,
        "use_age_delta_conditioning": requested_age_enabled,
        "age_conditioning_mode": bundle.get("age_conditioning_mode"),
        "use_age_conditioner_v2": bundle.get("use_age_conditioner_v2"),
        "age_conditioning_version": bundle.get("age_conditioning_version"),
        "age_delta_scale": bundle.get("age_delta_scale"),
        "use_relative_age_loss": loss_fn.use_relative_age_loss,
        "relative_age_weight": loss_fn.relative_age_weight,
        "relative_age_loss_type": loss_fn.relative_age_loss_type,
        "use_preservation_loss": loss_fn.use_preservation_loss,
        "preservation_weight": loss_fn.preservation_weight,
        "preservation_loss_type": loss_fn.preservation_loss_type,
        "preservation_max_delta": loss_fn.preservation_max_delta,
        "use_small_delta_weighting": loss_fn.use_small_delta_weighting,
        "small_delta_threshold": loss_fn.small_delta_threshold,
        "small_delta_weight": loss_fn.small_delta_weight,
        "warmup_ratio": warmup_ratio, "warmup_steps": warmup_steps, "min_lr_ratio": min_lr_ratio,
        "max_grad_norm": max_grad_norm,
        "conditioning_dropout_prob": conditioning_dropout_prob,
        "target_prompt_policy": target_prompt_policy,
        "generic_prompt_prob": generic_prompt_prob,
        "numeric_prompt_prob": numeric_prompt_prob,
        "identity_loss_on_image_dropped_samples": identity_loss_on_image_dropped_samples,
        "timestep_sampling": timestep_sampling,
        "min_train_timestep": min_train_timestep,
        "max_train_timestep": max_train_timestep,
        "double_prompt_prob": double_prompt_prob,
        "age_prompt_weight": age_prompt_weight, "generic_prompt_weight": generic_prompt_weight,
        "sample_source_posterior": sample_source_posterior,
        "sample_target_posterior": sample_target_posterior,
        "noise_offset": noise_offset, "min_snr_gamma": min_snr_gamma,
        "auxiliary_max_timestep": auxiliary_max_timestep,
        "amp_dtype": precision["amp_dtype_name"], "gradient_checkpointing": gradient_checkpointing,
        "enable_xformers": enable_xformers, "seed": seed, "deterministic": deterministic,
        "image_size": image_size, "prompt_configuration": prompt_configuration,
        "monitoring_mode": resolved_monitoring_mode,
        "monitoring_num_inference_steps": monitoring_num_inference_steps,
        "monitoring_strength": monitoring_strength,
        "monitoring_text_reference_mode": monitoring_text_reference_mode,
        "monitoring_age_guidance_scale": monitoring_age_guidance_scale,
        "monitoring_seed": monitoring_seed,
        "monitoring_target_ages": monitoring_ages,
        "monitoring_compute_diagnostics": bool(monitoring_compute_diagnostics),
        "include_zero_delta_pairs": bool(
            getattr(train_loader.dataset, "include_zero_delta_pairs", False)
        ),
        "zero_delta_pair_prob": float(
            getattr(train_loader.dataset, "zero_delta_pair_prob", 0.0)
        ),
    }
    root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    trainable_parameters = sum(parameter.numel() for parameter in trainables)
    timestep_end = (
        max_train_timestep
        if max_train_timestep is not None
        else len(bundle["scheduler_train"].alphas_cumprod) - 1
    )
    identity_count_text = training_identity_count if training_identity_count is not None else "unknown"
    checkpoint_text = str(root) if root is not None else "disabled"
    monitoring_text = (
        f"every {sample_every_epochs} epoch(s), mode={resolved_monitoring_mode}, "
        f"strength={monitoring_strength}, ages={monitoring_ages}, seed={monitoring_seed}, "
        f"text_ref={monitoring_text_reference_mode}, age_cfg={monitoring_age_guidance_scale}"
        if monitoring_image is not None else "disabled"
    )
    print("\n" + "=" * 104)
    print(" FACE AGING TRAINING")
    print("=" * 104)
    print(
        f" Runtime       device={resolved_device} | precision={precision['amp_dtype_name']} | "
        f"seed={seed} | deterministic={deterministic}"
    )
    print(
        f" Dataset       samples={len(train_loader.dataset):,} | identities={identity_count_text} | "
        f"batches/epoch={batches_per_epoch} | effective_batch={effective_batch_size}"
    )
    adapter_config = bundle.get("adapter_config", {})
    print(
        f" Model         backbone={bundle.get('model_id', 'unknown')} | "
        f"adapter={bundle.get('adapter_type', 'unknown').upper()} r={adapter_config.get('rank', 'n/a')} "
        f"alpha={adapter_config.get('alpha', 'n/a')} | image_size={image_size}"
    )
    print(
        f" Trainable     {trainable_parameters:,} params in {len(trainables)} tensors | "
        f"source_conditioning={bundle.get('source_conditioning')} | "
        f"age_conditioning={bundle.get('age_conditioning_version')} (scale={bundle.get('age_delta_scale')})"
    )
    print(
        f" Optimization  epochs={epochs_to_run} | steps/epoch={steps_per_epoch} | total_steps={total_planned_steps} | "
        f"accumulation={grad_accum_steps} | clip_norm={max_grad_norm}"
    )
    print(
        f" Learning rate LoRA={lr_lora:.2e} | conv_in={lr_conv_in:.2e} | age_mlp={lr_age_conditioner:.2e} | "
        f"warmup={warmup_steps} steps | min_lr_ratio={min_lr_ratio:.2f}"
    )
    print(
        f" Objective     diffusion={loss_fn.diffusion_weight:g} | identity={loss_fn.identity_weight:g} | "
        f"age_abs={loss_fn.age_weight:g} | age_relative={loss_fn.relative_age_weight:g} "
        f"({'on' if loss_fn.use_relative_age_loss else 'off'}) | Min-SNR={min_snr_gamma}"
    )
    print(
        f" Preservation enabled={loss_fn.use_preservation_loss} | weight={loss_fn.preservation_weight:g} | "
        f"type={loss_fn.preservation_loss_type} | max_delta={loss_fn.preservation_max_delta:g}"
    )
    print(
        f" Small-delta  self_pair_prob={getattr(train_loader.dataset, 'zero_delta_pair_prob', 0.0):.2f} "
        f"(enabled={getattr(train_loader.dataset, 'include_zero_delta_pairs', False)}) | "
        f"weighting={loss_fn.use_small_delta_weighting} | threshold={loss_fn.small_delta_threshold:g} | "
        f"weight={loss_fn.small_delta_weight:g}"
    )
    print(
        f" Sampling      timesteps={min_train_timestep}-{timestep_end} ({timestep_sampling}) | "
        f"cond_dropout={conditioning_dropout_prob:.3f} | source_posterior={'sample' if sample_source_posterior else 'mean'} | "
        f"target_posterior={'sample' if sample_target_posterior else 'mean'}"
    )
    print(
        f" Prompt policy {target_prompt_policy} | numeric={numeric_prompt_prob:.2f} | "
        f"generic={generic_prompt_prob:.2f} | conditioning_dropout={conditioning_dropout_prob:.3f}"
    )
    print(
        f" Aux losses    every={loss_fn.auxiliary_every_n_steps} microbatch(es) | "
        f"sample_fraction={loss_fn.auxiliary_sample_fraction:.2f} | max_timestep={auxiliary_max_timestep}"
    )
    print(
        f" Memory        gradient_checkpointing={memory_features['gradient_checkpointing']} | "
        f"xFormers={memory_features['xformers']}"
    )
    print(
        f" Validation    every={validate_every_epochs} epoch(s) | deterministic={deterministic_validation} | "
        f"monitor={monitor} ({monitor_mode})"
    )
    print(f" Monitoring    {monitoring_text}")
    print(f" Checkpoints   path={checkpoint_text}")
    print(
        f"               save_every={save_every_epochs} epoch(s) | keep_last={max_epoch_checkpoints} | "
        f"resume={resume_from or 'none'}"
    )
    print("=" * 104 + "\n")
    manager = TrainingCheckpointManager(
        root, monitor=monitor, mode=monitor_mode,
        save_epoch_checkpoints=save_epoch_checkpoints,
        max_epoch_checkpoints=max_epoch_checkpoints,
    ) if root is not None else None
    history = {"train": [], "val": [], "epochs": [], "config": training_config}
    start_epoch = global_step = optimizer_step = 0
    generator_device = resolved_device.type if resolved_device.type == "cuda" else "cpu"
    train_generator = torch.Generator(device=generator_device).manual_seed(seed + 10_000)
    if resume_from is not None:
        payload = load_training_checkpoint(
            resume_from, bundle=bundle, loss_fn=loss_fn, optimizer=optimizer,
            lr_scheduler=lr_scheduler, scaler=precision["scaler"],
            strict_config=strict_resume_config, restore_rng=True,
            current_training_config=training_config,
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        optimizer_step = int(payload["optimizer_step"])
        history = payload.get("history", history)
        if payload.get("training_generator_state") is not None:
            train_generator.set_state(payload["training_generator_state"])
        if payload.get("dataloader_generator_state") is not None and getattr(train_loader, "generator", None) is not None:
            train_loader.generator.set_state(payload["dataloader_generator_state"])
        if manager is not None:
            manager.best_metric = payload.get("best_metric")
            manager.best_epoch = payload.get("best_epoch")
        print(f"[resume] epoch={start_epoch + 1}  global_step={global_step}  optimizer_step={optimizer_step}")
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs_to_run):
            if optimizer_step >= total_planned_steps:
                break
            last_epoch = epoch
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            epoch_started = time.perf_counter()
            train_result = train_one_epoch(
                bundle=bundle, loss_fn=loss_fn, train_loader=train_loader,
                optimizer=optimizer, lr_scheduler=lr_scheduler, device=resolved_device,
                epoch=epoch, global_step=global_step, optimizer_step=optimizer_step,
                scaler=precision["scaler"], amp_enabled=precision["amp_enabled"], amp_dtype=amp_dtype,
                grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm,
                conditioning_dropout_prob=conditioning_dropout_prob,
                target_prompt_policy=target_prompt_policy,
                generic_prompt_prob=generic_prompt_prob,
                numeric_prompt_prob=numeric_prompt_prob,
                timestep_sampling=timestep_sampling,
                min_train_timestep=min_train_timestep, max_train_timestep=max_train_timestep,
                double_prompt_prob=double_prompt_prob,
                age_prompt_weight=age_prompt_weight, generic_prompt_weight=generic_prompt_weight,
                identity_loss_on_image_dropped_samples=identity_loss_on_image_dropped_samples,
                sample_source_posterior=sample_source_posterior,
                sample_target_posterior=sample_target_posterior,
                noise_offset=noise_offset, max_batches=max_train_batches,
                max_optimizer_steps=total_planned_steps, log_every=log_every,
                skip_nonfinite_loss=skip_nonfinite_loss,
                abort_after_nonfinite_steps=abort_after_nonfinite_steps,
                safe_optimizer_step=safe_optimizer_step, generator=train_generator,
            )
            global_step, optimizer_step = train_result["global_step"], train_result["optimizer_step"]
            history["train"].append(train_result["metrics"])
            val_result = None
            if (epoch + 1) % validate_every_epochs == 0 or optimizer_step >= total_planned_steps:
                val_result = validate_one_epoch(
                    bundle=bundle, loss_fn=loss_fn, val_loader=val_loader,
                    device=resolved_device, epoch=epoch,
                    amp_enabled=precision["amp_enabled"], amp_dtype=amp_dtype,
                    max_batches=max_val_batches,
                    deterministic_validation=deterministic_validation,
                    validation_seed=validation_seed,
                    min_train_timestep=min_train_timestep,
                    max_train_timestep=max_train_timestep,
                )
                history["val"].append(val_result["metrics"])
            epoch_record = {
                "epoch": epoch, "global_step": global_step, "optimizer_step": optimizer_step,
                "train": train_result["metrics"],
                "val": val_result["metrics"] if val_result else None,
                "duration_seconds": time.perf_counter() - epoch_started,
            }
            history["epochs"].append(epoch_record)
            checkpoint_report = None
            if manager is not None and val_result is not None and ((epoch + 1) % save_every_epochs == 0 or optimizer_step >= total_planned_steps):
                if monitor not in val_result["metrics"]:
                    raise KeyError(f"Checkpoint monitor {monitor!r} is absent from validation metrics")
                monitored = float(val_result["metrics"][monitor])
                improved = manager.is_improved(monitored)
                payload = build_training_payload(
                    bundle=bundle, loss_fn=loss_fn, optimizer=optimizer,
                    lr_scheduler=lr_scheduler, scaler=precision["scaler"],
                    epoch=epoch, batch_position=0, global_step=global_step,
                    optimizer_step=optimizer_step,
                    best_metric=monitored if improved else manager.best_metric,
                    best_epoch=epoch if improved else manager.best_epoch,
                    history=history, training_config=training_config,
                    training_generator_state=train_generator.get_state(),
                    dataloader_generator_state=train_loader.generator.get_state() if getattr(train_loader, "generator", None) is not None else None,
                )
                checkpoint_report = manager.save(
                    training_payload=payload,
                    inference_payload=build_inference_payload(bundle, training_config),
                    epoch=epoch, metric=monitored,
                )
                atomic_json_save(history, manager.root_dir / "history.json")
            sampling_report = None
            monitoring_enabled = sample_fn is not None or monitoring_image is not None
            if monitoring_enabled and sample_every_epochs > 0 and ((epoch + 1) % sample_every_epochs == 0 or optimizer_step >= total_planned_steps):
                selected_sample_fn = sample_fn
                if selected_sample_fn is None:
                    selected_sample_fn = run_face_aging_monitor
                sampling_report = sample_monitoring_images(
                    selected_sample_fn, bundle=bundle, loss_fn=loss_fn, val_loader=val_loader,
                    epoch=epoch, device=resolved_device,
                    output_dir=Path(monitoring_dir) if monitoring_dir else (root / "monitoring" if root else None),
                    seed=monitoring_seed,
                    image=monitoring_image,
                    target_prompt=monitoring_target_prompt,
                    target_age=monitoring_target_age,
                    source_prompt=monitoring_source_prompt,
                    source_age=monitoring_source_age,
                    mode=monitoring_mode,
                    use_inverse_diffusion=monitoring_use_inverse_diffusion,
                    num_inference_steps=monitoring_num_inference_steps,
                    strength=monitoring_strength,
                    text_guidance_scale=monitoring_text_guidance_scale,
                    text_reference_mode=monitoring_text_reference_mode,
                    age_guidance_scale=monitoring_age_guidance_scale,
                    image_guidance_scale=monitoring_image_guidance_scale,
                    image_size=image_size,
                    compute_diagnostics=monitoring_compute_diagnostics,
                )
            epoch_record["checkpoint"] = checkpoint_report
            epoch_record["sampling"] = sampling_report
            validation_value = val_result["metrics"].get("val/loss_total") if val_result else None
            validation_text = f"{validation_value:.4f}" if validation_value is not None else "not_run"
            epoch_duration = float(epoch_record["duration_seconds"])
            checkpoint_status = "not scheduled"
            if checkpoint_report is not None:
                checkpoint_status = "saved"
                if checkpoint_report.get("improved"):
                    checkpoint_status += " (new best)"
            sampling_status = "saved" if sampling_report is not None else "not scheduled"
            print("-" * 104)
            print(
                f" Epoch {epoch + 1:02d}/{epochs_to_run:02d} complete  |  "
                f"train_loss={train_result['metrics'].get('train/loss_total'):.4f}  |  "
                f"val_loss={validation_text}  |  optimizer_step={optimizer_step}  |  "
                f"duration={_format_duration(epoch_duration)}"
            )
            best_value = manager.best_metric if manager is not None else None
            best_text = f"{best_value:.6f}" if best_value is not None else "n/a"
            print(
                f" Checkpoint: {checkpoint_status}  |  Monitoring: {sampling_status}  |  "
                f"Best {monitor}: {best_text}"
            )
            print("-" * 104 + "\n")
    except KeyboardInterrupt:
        if root is not None:
            emergency = build_training_payload(
                bundle=bundle, loss_fn=loss_fn, optimizer=optimizer,
                lr_scheduler=lr_scheduler, scaler=precision["scaler"],
                epoch=max(last_epoch, 0), batch_position=0,
                global_step=global_step, optimizer_step=optimizer_step,
                best_metric=manager.best_metric, best_epoch=manager.best_epoch,
                history=history, training_config=training_config,
                training_generator_state=train_generator.get_state(),
                dataloader_generator_state=train_loader.generator.get_state() if getattr(train_loader, "generator", None) is not None else None,
            )
            atomic_torch_save(emergency, root / "interrupted_training_resume.pt")
        raise
    except Exception as exc:
        cuda_memory = None
        if resolved_device.type == "cuda":
            cuda_memory = {
                "allocated": torch.cuda.memory_allocated(resolved_device),
                "reserved": torch.cuda.memory_reserved(resolved_device),
                "peak": torch.cuda.max_memory_allocated(resolved_device),
            }
        print(
            "TRAINING ERROR | "
            f"epoch={last_epoch} global_step={global_step} optimizer_step={optimizer_step} "
            f"cuda_memory={cuda_memory} exception={exc!r}"
        )
        raise
    history.update({
        "global_step": global_step, "optimizer_step": optimizer_step,
        "best_metric": manager.best_metric if manager else None,
        "best_epoch": manager.best_epoch if manager else None,
    })
    if manager is not None:
        atomic_json_save(history, manager.root_dir / "history.json")
    return {
        "history": history,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "best_metric": manager.best_metric if manager else None,
        "best_epoch": manager.best_epoch if manager else None,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "scaler": precision["scaler"],
        "checkpoint_manager": manager,
        "precision": precision,
        "memory_features": memory_features,
        "config": training_config,
    }


def TRAIN_AGGING_MODEL(**kwargs):
    """Notebook-friendly wrapper retained with the user's requested spelling."""
    return train_model(**kwargs)


TRAIN_AGING_MODEL = TRAIN_AGGING_MODEL
