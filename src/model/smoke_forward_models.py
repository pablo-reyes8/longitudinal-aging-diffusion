"""Offline-friendly structural validation and one-batch model preparation."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .load_diffusion_models import (
    build_conditioned_unet_input,
    build_face_aging_optimizer,
    encode_prompts,
    get_bundle_trainable_named_parameters,
    prepare_source_target_latents,
    tokenizer_audit,
)
from .age_conditioning import compute_age_delta_embedding


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def prepare_face_aging_forward(
    bundle: Mapping[str, Any],
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    target_prompts: Sequence[str],
    *,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    delta_ages: torch.Tensor | None = None,
    source_ages: torch.Tensor | None = None,
    target_ages: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare and execute one denoising prediction, without defining a loss."""
    latents = prepare_source_target_latents(bundle, source_images, target_images)
    source_latents, target_latents = latents["source_latents"], latents["target_latents"]
    scheduler = bundle["scheduler_train"]
    if noise is None:
        noise = torch.randn(target_latents.shape, generator=generator, device=target_latents.device, dtype=target_latents.dtype)
    if timesteps is None:
        total_value = (
            scheduler.config.get("num_train_timesteps")
            if isinstance(scheduler.config, Mapping)
            else getattr(scheduler.config, "num_train_timesteps")
        )
        total = int(total_value)
        timesteps = torch.randint(0, total, (target_latents.shape[0],), generator=generator, device=target_latents.device).long()
    noisy_target = scheduler.add_noise(target_latents, noise, timesteps)
    conditioned_input = build_conditioned_unet_input(
        noisy_target, source_latents, source_conditioning=bundle["source_conditioning"]
    )
    hidden_states = encode_prompts(bundle, target_prompts)
    unet_device = next(bundle["unet"].parameters()).device
    compute_dtype = bundle.get("weight_dtype", next(bundle["unet"].parameters()).dtype)
    conditioned_input = conditioned_input.to(unet_device)
    timesteps = timesteps.to(unet_device)
    hidden_states = hidden_states.to(unet_device)
    age_conditioning = compute_age_delta_embedding(
        bundle, delta_ages, batch_size=conditioned_input.shape[0],
        source_age=source_ages, target_age=target_ages,
    ) if delta_ages is not None else None
    unet_kwargs = {"encoder_hidden_states": hidden_states, "return_dict": True}
    if age_conditioning is not None:
        unet_kwargs["timestep_cond"] = age_conditioning
    with _autocast_context(unet_device, compute_dtype):
        noise_pred = bundle["unet"](
            conditioned_input,
            timesteps,
            **unet_kwargs,
        ).sample
    return {
        **latents,
        "noise": noise,
        "timesteps": timesteps,
        "noisy_target_latents": noisy_target,
        "conditioned_input": conditioned_input,
        "encoder_hidden_states": hidden_states,
        "age_conditioning": age_conditioning,
        "noise_pred": noise_pred,
    }


def inspect_model_batch(batch: Mapping[str, Any], prepared: Mapping[str, torch.Tensor]) -> None:
    for index, person_id in enumerate(batch["person_id"]):
        print(
            person_id,
            f"{int(batch['source_age'][index])} -> {int(batch['target_age'][index])}",
            batch["target_prompt"][index],
        )
    for key in ("source_latents", "target_latents", "conditioned_input", "encoder_hidden_states", "noise_pred"):
        print(f"{key}:", tuple(prepared[key].shape), prepared[key].dtype)


def _gradient_summary(named_parameters) -> dict[str, Any]:
    norms = []
    with_grad = 0
    for _, parameter in named_parameters:
        if parameter.grad is not None:
            with_grad += 1
            norms.append(float(parameter.grad.detach().float().norm()))
    return {
        "parameters": len(named_parameters),
        "with_grad": with_grad,
        "grad_norm_min": min(norms, default=None),
        "grad_norm_mean": sum(norms) / len(norms) if norms else None,
        "grad_norm_max": max(norms, default=None),
    }


def _cuda_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
    }


def run_face_aging_model_validation(
    bundle: Mapping[str, Any],
    *,
    batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a localized structural/numerical validation report.

    When ``batch`` is supplied, it also performs the real data-layer integration,
    scheduler, prompt, U-Net forward, and gradient-routing checks.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    unet, vae, text_encoder = bundle["unet"], bundle["vae"], bundle["text_encoder"]
    config, adapter = bundle["config"], bundle["adapter_report"]
    if config["unet_in_channels"] != 8 or unet.conv_in.in_channels != 8:
        errors.append("Expanded U-Net input channel count is not 8")
    if bundle["conv_in_report"]["source_weight_max_abs"] != 0:
        errors.append("Source-channel conv_in weights were not zero initialized")
    if any(count == 0 for count in adapter["counts_by_target"].values()):
        errors.append(f"Incomplete adapter target coverage: {adapter['counts_by_target']}")
    adapter_actual = sum(
        parameter.numel() for name, parameter in unet.named_parameters()
        if ".lora_down." in name or ".lora_up." in name or name.endswith(".magnitude")
    )
    if adapter_actual != adapter["expected_adapter_parameters"]:
        errors.append(f"Adapter parameter formula mismatch: expected={adapter['expected_adapter_parameters']}, actual={adapter_actual}")
    unet_trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    trainable = get_bundle_trainable_named_parameters(bundle)
    invalid_trainable = [name for name, _ in unet_trainable if not (name.startswith("conv_in.") or ".lora_" in name or name.endswith(".magnitude"))]
    if invalid_trainable:
        errors.append(f"Unexpected trainable U-Net parameters: {invalid_trainable}")
    non_fp32 = [name for name, parameter in trainable if parameter.dtype != torch.float32]
    if non_fp32:
        errors.append(f"Trainable parameters are not FP32: {non_fp32}")
    if any(parameter.requires_grad for parameter in vae.parameters()):
        errors.append("VAE has trainable parameters")
    if any(parameter.requires_grad for parameter in text_encoder.parameters()):
        errors.append("Text encoder has trainable parameters")
    if vae.training or text_encoder.training or not unet.training:
        errors.append(f"Wrong module modes: vae.training={vae.training}, text.training={text_encoder.training}, unet.training={unet.training}")
    unet_refs = [value for value in bundle.values() if value is unet]
    if len(unet_refs) != 1:
        errors.append(f"Bundle contains {len(unet_refs)} direct references to its U-Net; expected exactly one")
    optimizer = build_face_aging_optimizer(bundle)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != {id(p) for _, p in trainable}:
        errors.append("Optimizer membership is not an exact unique match for trainables")

    prompt_report: dict[str, Any] = {}
    latent_report: dict[str, Any] = {}
    gradient_report: dict[str, Any] = {}
    scheduler_report: dict[str, Any] = {}
    if batch is not None:
        prompts = list(batch["target_prompt"])
        first_embeddings = encode_prompts(bundle, prompts)
        second_embeddings = encode_prompts(bundle, prompts)
        prompt_report = {
            "shape": list(first_embeddings.shape),
            "deterministic_max_error": float((first_embeddings - second_embeddings).abs().max()),
            "finite": bool(torch.isfinite(first_embeddings).all()),
            "tokenization": tokenizer_audit(bundle, prompts[: min(2, len(prompts))]),
        }
        if prompt_report["deterministic_max_error"] != 0 or not prompt_report["finite"]:
            errors.append("Frozen prompt encoder is non-deterministic or non-finite")
        prepared = prepare_face_aging_forward(
            bundle, batch["source_image"], batch["target_image"], prompts,
            delta_ages=batch.get("delta_age"),
            source_ages=batch.get("source_age"),
            target_ages=batch.get("target_age"),
        )
        source, target = prepared["source_latents"], prepared["target_latents"]
        latent_report = {
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "conditioned_shape": list(prepared["conditioned_input"].shape),
            "noise_prediction_shape": list(prepared["noise_pred"].shape),
            "finite": all(bool(torch.isfinite(prepared[key]).all()) for key in ("source_latents", "target_latents", "conditioned_input", "noise_pred")),
        }
        if source.shape != target.shape or source.shape[1] != 4:
            errors.append(f"Source/target latent incompatibility: {source.shape}, {target.shape}")
        if prepared["conditioned_input"].shape[1] != 8:
            errors.append("Conditioned U-Net input does not have 8 channels")
        if prepared["noise_pred"].shape != prepared["noise"].shape:
            errors.append("U-Net prediction shape differs from target noise shape")
        if not latent_report["finite"]:
            errors.append("Non-finite value in full preparation path")
        total_value = (
            bundle["scheduler_train"].config.get("num_train_timesteps")
            if isinstance(bundle["scheduler_train"].config, Mapping)
            else getattr(bundle["scheduler_train"].config, "num_train_timesteps")
        )
        boundary_steps = torch.tensor(
            [0, int(total_value) // 2, int(total_value) - 1], device=target.device
        )
        repeated_latent = target[:1].repeat(3, 1, 1, 1)
        repeated_noise = prepared["noise"][:1].repeat(3, 1, 1, 1)
        noisy_once = bundle["scheduler_train"].add_noise(repeated_latent, repeated_noise, boundary_steps)
        noisy_twice = bundle["scheduler_train"].add_noise(repeated_latent, repeated_noise, boundary_steps)
        scheduler_report = {
            "timesteps": boundary_steps.tolist(),
            "deterministic": bool(torch.equal(noisy_once, noisy_twice)),
            "finite": bool(torch.isfinite(noisy_once).all()),
            "norms": [float(item.float().norm()) for item in noisy_once],
        }
        if not scheduler_report["deterministic"] or not scheduler_report["finite"]:
            errors.append("Training scheduler is non-deterministic or non-finite for fixed inputs")
        unet.zero_grad(set_to_none=True)
        if bundle.get("age_delta_conditioner") is not None:
            bundle["age_delta_conditioner"].zero_grad(set_to_none=True)
        prepared["noise_pred"].float().square().mean().backward()
        adapter_named = [(name, p) for name, p in unet_trainable if not name.startswith("conv_in.")]
        conv_named = [(name, p) for name, p in unet_trainable if name.startswith("conv_in.")]
        age_named = [
            (name, p) for name, p in trainable
            if name.startswith("age_delta_conditioner.") or name.startswith("age_conditioner.")
        ]
        frozen_named = [(name, p) for name, p in unet.named_parameters() if not p.requires_grad]
        gradient_report = {
            "adapter": _gradient_summary(adapter_named),
            "conv_in": _gradient_summary(conv_named),
            "age_delta_conditioner": _gradient_summary(age_named),
            "frozen_unet": _gradient_summary(frozen_named),
            "vae": _gradient_summary(list(vae.named_parameters())),
            "text_encoder": _gradient_summary(list(text_encoder.named_parameters())),
        }
        if gradient_report["adapter"]["with_grad"] != gradient_report["adapter"]["parameters"]:
            errors.append("At least one trainable adapter tensor has grad=None")
        if gradient_report["conv_in"]["with_grad"] != gradient_report["conv_in"]["parameters"]:
            errors.append("At least one trainable conv_in tensor has grad=None")
        if age_named and gradient_report["age_delta_conditioner"]["with_grad"] != gradient_report["age_delta_conditioner"]["parameters"]:
            errors.append("At least one age-delta conditioner tensor has grad=None")
        if any(gradient_report[group]["with_grad"] for group in ("frozen_unet", "vae", "text_encoder")):
            errors.append("A frozen parameter received a gradient")
        unet.zero_grad(set_to_none=True)
        if bundle.get("age_delta_conditioner") is not None:
            bundle["age_delta_conditioner"].zero_grad(set_to_none=True)
    else:
        warnings.append("No batch supplied; latent, prompt, forward, and gradient integration tests were not run")

    info.append("Only one U-Net was assembled; adapter injection was in-place")
    info.append("LoRA target suffixes match Diffusers attention projection names")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "model": {"model_id": bundle["model_id"], "vae_id": bundle["vae_id"], "config": dict(config)},
        "adapter": dict(adapter),
        "trainable_parameters": {**bundle["param_stats"]["unet"], "names": list(bundle["trainable_param_names"])},
        "conv_in": dict(bundle["conv_in_report"]),
        "latent_tests": latent_report,
        "prompt_tests": prompt_report,
        "scheduler_tests": scheduler_report,
        "gradient_tests": gradient_report,
        "checkpoint_tests": {"run_separately": True},
        "memory": _cuda_memory(),
    }
