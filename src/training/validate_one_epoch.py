"""Independent, deterministic validation for the face-aging objective."""

from __future__ import annotations

import time
from typing import Any, Mapping

import torch

from .metrics import AGE_BANDS, AGE_GAP_BINS, MetricsTracker, bin_name
from .mixed_precision import move_batch_to_device
from .timestep_sampling import deterministic_validation_timesteps
from .training_step import run_training_step


def _validation_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _stratified_metrics(tracker: MetricsTracker, batch, output: dict, loss_fn) -> None:
    batch_size = output["loss_diff_per_sample"].shape[0]
    total = loss_fn.diffusion_weight * output["loss_diff_per_sample"].detach().float()
    identity = torch.full((batch_size,), float("nan"), device=total.device)
    age = torch.full((batch_size,), float("nan"), device=total.device)
    if output["loss_id_per_sample"].numel():
        indices = output["identity_indices"]
        values = output["loss_id_per_sample"].detach().float()
        identity[indices] = 1.0 - values
        total[indices] += loss_fn.identity_weight * values
    if output["loss_age_per_sample"].numel():
        indices = output["auxiliary_indices"]
        values = output["loss_age_per_sample"].detach().float()
        age[indices] = values if loss_fn.age_loss_type == "l1" else values.sqrt()
        total[indices] += loss_fn.age_weight * values
    if output["loss_relative_age_per_sample"].numel():
        indices = output["auxiliary_indices"]
        values = output["loss_relative_age_per_sample"].detach().float()
        total[indices] += loss_fn.relative_age_weight * values
    if output["loss_preservation_per_sample"].numel():
        indices = output["preservation_indices"]
        values = output["loss_preservation_per_sample"].detach().float()
        total[indices] += loss_fn.preservation_weight * values
    for index in range(batch_size):
        gap = bin_name(float(batch["delta_age"][index]), AGE_GAP_BINS)
        source_band = bin_name(float(batch["source_age"][index]), AGE_BANDS)
        target_band = bin_name(float(batch["target_age"][index]), AGE_BANDS)
        tracker.update({f"gap_{gap}/loss_total": float(total[index])}, weight=1)
        if torch.isfinite(identity[index]):
            tracker.update({f"gap_{gap}/identity_cosine": float(identity[index])}, weight=1)
            tracker.update({f"source_band_{source_band}/identity_cosine": float(identity[index])}, weight=1)
            tracker.update({f"target_band_{target_band}/identity_cosine": float(identity[index])}, weight=1)
        if torch.isfinite(age[index]):
            tracker.update({f"gap_{gap}/age_mae": float(age[index])}, weight=1)
            tracker.update({f"source_band_{source_band}/age_mae": float(age[index])}, weight=1)
            tracker.update({f"target_band_{target_band}/age_mae": float(age[index])}, weight=1)


def validate_one_epoch(
    *,
    bundle: Mapping[str, Any],
    loss_fn,
    val_loader,
    device: torch.device,
    epoch: int = 0,
    amp_enabled: bool = True,
    amp_dtype: str | torch.dtype = "auto",
    max_batches: int | None = None,
    deterministic_validation: bool = True,
    validation_seed: int = 2026,
    min_train_timestep: int = 0,
    max_train_timestep: int | None = None,
) -> dict[str, Any]:
    previous_unet_mode = bundle["unet"].training
    bundle["unet"].eval()
    if bundle.get("age_delta_conditioner") is not None:
        bundle["age_delta_conditioner"].eval()
    bundle["vae"].eval()
    bundle["text_encoder"].eval()
    loss_fn.eval()
    tracker = MetricsTracker()
    timesteps_seen = []
    processed_batches = processed_samples = 0
    limit = len(val_loader) if max_batches is None else min(len(val_loader), int(max_batches))
    generator = _validation_generator(validation_seed) if deterministic_validation else None
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(val_loader):
                if batch_index >= limit:
                    break
                batch = move_batch_to_device(raw_batch, device)
                batch_size = int(batch["source_image"].shape[0])
                timesteps = None
                if deterministic_validation:
                    timesteps = deterministic_validation_timesteps(
                        batch_size, bundle["scheduler_train"], device,
                        batch_index=batch_index,
                        min_timestep=min_train_timestep,
                        max_timestep=max_train_timestep,
                    )
                result = run_training_step(
                    bundle=bundle, loss_fn=loss_fn, batch=batch, device=device,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    conditioning_dropout_prob=0.0,
                    min_train_timestep=min_train_timestep,
                    max_train_timestep=max_train_timestep,
                    sample_source_posterior=False,
                    sample_target_posterior=False,
                    identity_loss_on_image_dropped_samples=True,
                    generator=generator, timesteps=timesteps,
                    # Validation must not alternate auxiliaries with the train
                    # cadence; step zero is always an auxiliary-active step.
                    global_step=0,
                )
                output = result["loss_out"]
                values = {
                    "loss_total": float(output["loss"]),
                    "loss_diff": float(output["loss_diff"]),
                    "loss_id": float(output["loss_id"]),
                    "loss_age": float(output["loss_age"]),
                    "loss_relative_age": float(output["loss_relative_age"]),
                    "loss_preservation": float(output["loss_preservation"]),
                    "weighted_diff": float(output["weighted_diff"]),
                    "weighted_id": float(output["weighted_id"]),
                    "weighted_age": float(output["weighted_age"]),
                    "weighted_relative_age": float(output["weighted_relative_age"]),
                    "weighted_preservation": float(output["weighted_preservation"]),
                    "preservation_active_fraction": output["metrics"]["preservation_active_fraction"],
                    "small_delta_fraction": output["metrics"]["small_delta_fraction"],
                    "small_delta_mean_weight": output["metrics"]["small_delta_mean_weight"],
                    "identity_cosine": output["metrics"]["identity_cosine_mean"],
                    "age_mae": float(output["loss_age"]) if loss_fn.age_loss_type == "l1" else float(output["loss_age"].sqrt()),
                    **result["diagnostics"],
                }
                tracker.update(values, weight=batch_size)
                _stratified_metrics(tracker, batch, output, loss_fn)
                timesteps_seen.append(result["prepared"]["timesteps"].detach().cpu())
                processed_batches += 1
                processed_samples += batch_size
    finally:
        if previous_unet_mode:
            bundle["unet"].train()
            if bundle.get("age_delta_conditioner") is not None:
                bundle["age_delta_conditioner"].train()
        bundle["vae"].eval()
        bundle["text_encoder"].eval()
        loss_fn.train(previous_unet_mode)
    elapsed = time.perf_counter() - start
    all_timesteps = torch.cat(timesteps_seen) if timesteps_seen else torch.empty(0, dtype=torch.long)
    metrics = tracker.averages(prefix="val")
    if all_timesteps.numel():
        values = all_timesteps.double()
        total_timesteps = len(bundle["scheduler_train"].alphas_cumprod)
        quartiles = torch.clamp(all_timesteps * 4 // total_timesteps, max=3)
        metrics.update({
            "val/timestep_mean": float(values.mean()),
            "val/timestep_std": float(values.std(unbiased=False)),
            "val/timestep_min": float(values.min()),
            "val/timestep_max": float(values.max()),
            **{f"val/timestep_q{index + 1}_fraction": float((quartiles == index).float().mean()) for index in range(4)},
        })
    return {
        "metrics": metrics,
        "num_batches": processed_batches,
        "num_samples": processed_samples,
        "timesteps": all_timesteps,
        "deterministic": deterministic_validation,
        "validation_seed": validation_seed,
        "duration_seconds": elapsed,
    }
