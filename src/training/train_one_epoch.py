"""One-epoch mechanics for the single face-aging diffusion model."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import torch

from src.model import get_bundle_trainable_named_parameters

from .metrics import AGE_GAP_BINS, MetricsTracker, bin_name, optimizer_group_lrs
from .mixed_precision import move_batch_to_device, safe_optimizer_step as perform_safe_optimizer_step
from .training_step import run_training_step


def _trainable_parameters(bundle: Mapping[str, Any]) -> list[torch.nn.Parameter]:
    parameters = [parameter for _, parameter in get_bundle_trainable_named_parameters(bundle)]
    if not parameters:
        raise RuntimeError("The U-Net has no trainable parameters")
    return parameters


def set_training_modes(bundle: Mapping[str, Any], loss_fn) -> None:
    bundle["unet"].train()
    if bundle.get("age_delta_conditioner") is not None:
        bundle["age_delta_conditioner"].train()
    bundle["vae"].eval()
    bundle["text_encoder"].eval()
    loss_fn.train()


def _backward(loss: torch.Tensor, scaler) -> None:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def _draw_double_prompt(probability: float, generator: torch.Generator | None) -> bool:
    if probability <= 0:
        return False
    generator_device = torch.device(getattr(generator, "device", "cpu")) if generator is not None else torch.device("cpu")
    return bool(torch.rand((), generator=generator, device=generator_device) < probability)


def _format_eta(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _component_values(first: dict, second: dict | None, first_weight: float, second_weight: float) -> dict[str, float]:
    names = (
        "loss", "loss_diff", "loss_id", "loss_age", "loss_relative_age",
        "weighted_diff", "weighted_id", "weighted_age", "weighted_relative_age",
    )
    result = {}
    for name in names:
        first_value = first.get(name)
        value = first_weight * (float(first_value.detach()) if first_value is not None else 0.0)
        if second is not None:
            second_value = second.get(name)
            value += second_weight * (float(second_value.detach()) if second_value is not None else 0.0)
        result["loss_total" if name == "loss" else name] = value
    first_metrics = first["metrics"]
    second_metrics = second["metrics"] if second is not None else None
    identity_values = []
    if first_metrics["identity_cosine_mean"] is not None:
        identity_values.append((first_weight, first_metrics["identity_cosine_mean"]))
    if second_metrics is not None and second_metrics["identity_cosine_mean"] is not None:
        identity_values.append((second_weight, second_metrics["identity_cosine_mean"]))
    result["identity_cosine"] = sum(weight * value for weight, value in identity_values) if identity_values else None
    result["age_mae"] = first_weight * float(first["loss_age"].detach())
    if second is not None:
        result["age_mae"] += second_weight * float(second["loss_age"].detach())
    first_relative = first.get("loss_relative_age")
    result["relative_age_mae"] = first_weight * (
        float(first_relative.detach()) if first_relative is not None else 0.0
    )
    if second is not None:
        second_relative = second.get("loss_relative_age")
        result["relative_age_mae"] += second_weight * (
            float(second_relative.detach()) if second_relative is not None else 0.0
        )
    result["identity_sample_fraction"] = first_metrics["identity_count"] / max(1, first["loss_diff_per_sample"].shape[0]) if "identity_count" in first_metrics else 0.0
    result["age_sample_fraction"] = first_metrics["age_count"] / max(1, first["loss_diff_per_sample"].shape[0]) if "age_count" in first_metrics else 0.0
    return result


def _update_gap_metrics(tracker: MetricsTracker, batch, loss_out: dict, loss_fn) -> None:
    per_sample = loss_fn.diffusion_weight * loss_out["loss_diff_per_sample"].detach().float()
    identity = torch.full_like(per_sample, float("nan"))
    age = torch.full_like(per_sample, float("nan"))
    if loss_out["loss_id_per_sample"].numel():
        indices = loss_out["identity_indices"]
        values = loss_out["loss_id_per_sample"].detach().float()
        per_sample[indices] += loss_fn.identity_weight * values
        identity[indices] = 1.0 - values
    if loss_out["loss_age_per_sample"].numel():
        indices = loss_out["auxiliary_indices"]
        values = loss_out["loss_age_per_sample"].detach().float()
        per_sample[indices] += loss_fn.age_weight * values
        age[indices] = values if loss_fn.age_loss_type == "l1" else values.sqrt()
    relative_per_sample = loss_out.get("loss_relative_age_per_sample")
    if relative_per_sample is not None and relative_per_sample.numel():
        indices = loss_out["auxiliary_indices"]
        values = relative_per_sample.detach().float()
        per_sample[indices] += loss_fn.relative_age_weight * values
    deltas = batch["delta_age"].detach()
    for index in range(per_sample.shape[0]):
        gap = bin_name(float(deltas[index]), AGE_GAP_BINS)
        tracker.update({f"gap_{gap}/loss_total": float(per_sample[index])}, weight=1)
        if torch.isfinite(identity[index]):
            tracker.update({f"gap_{gap}/identity_cosine": float(identity[index])}, weight=1)
        if torch.isfinite(age[index]):
            tracker.update({f"gap_{gap}/age_mae": float(age[index])}, weight=1)


def train_one_epoch(
    *,
    bundle: Mapping[str, Any],
    loss_fn,
    train_loader,
    optimizer,
    lr_scheduler,
    device: torch.device,
    epoch: int,
    global_step: int = 0,
    optimizer_step: int = 0,
    scaler=None,
    amp_enabled: bool = True,
    amp_dtype: str | torch.dtype = "auto",
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    conditioning_dropout_prob: float = 0.05,
    timestep_sampling: str = "uniform",
    min_train_timestep: int = 0,
    max_train_timestep: int | None = None,
    double_prompt_prob: float = 0.0,
    age_prompt_weight: float = 0.5,
    generic_prompt_weight: float = 0.5,
    identity_loss_on_image_dropped_samples: bool = False,
    sample_source_posterior: bool = False,
    sample_target_posterior: bool = True,
    noise_offset: float = 0.0,
    max_batches: int | None = None,
    max_optimizer_steps: int | None = None,
    log_every: int = 25,
    skip_nonfinite_loss: bool = True,
    abort_after_nonfinite_steps: int = 5,
    safe_optimizer_step: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if not 0 <= double_prompt_prob <= 1:
        raise ValueError("double_prompt_prob must be in [0,1]")
    if age_prompt_weight < 0 or generic_prompt_weight < 0 or not math.isclose(age_prompt_weight + generic_prompt_weight, 1.0):
        raise ValueError("double-prompt weights must be non-negative and sum to 1")
    set_training_modes(bundle, loss_fn)
    parameters = _trainable_parameters(bundle)
    initial_optimizer_step = optimizer_step
    optimizer.zero_grad(set_to_none=True)
    tracker = MetricsTracker()
    timestep_count = timestep_sum = timestep_square_sum = 0.0
    timestep_min, timestep_max = float("inf"), float("-inf")
    timestep_quartile_counts = torch.zeros(4, dtype=torch.long)
    total_batches = len(train_loader) if max_batches is None else min(len(train_loader), int(max_batches))
    iterator = iter(train_loader)
    processed_batches = processed_samples = skipped_nonfinite = skipped_updates = double_prompt_batches = 0
    consecutive_nonfinite = 0
    start = time.perf_counter()
    if log_every:
        print(f" Epoch {epoch + 1:02d} - training progress")
        print(
            "   batch       done    opt.step   total     diffusion  identity  age.abs  age.rel  "
            "grad     LoRA lr    samples/s   ETA"
        )
    while processed_batches < total_batches:
        if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
            break
        window_size = min(grad_accum_steps, total_batches - processed_batches)
        window = [next(iterator) for _ in range(window_size)]
        sample_counts = [int(batch["source_image"].shape[0]) for batch in window]
        window_samples = sum(sample_counts)
        window_failed = False
        for raw_batch, batch_samples in zip(window, sample_counts):
            if window_failed:
                processed_batches += 1
                processed_samples += batch_samples
                global_step += 1
                continue
            batch = move_batch_to_device(raw_batch, device)
            micro_weight = batch_samples / window_samples
            use_double = _draw_double_prompt(double_prompt_prob, generator)
            first_weight = age_prompt_weight if use_double else 1.0
            second_weight = generic_prompt_weight if use_double else 0.0
            first = run_training_step(
                bundle=bundle, loss_fn=loss_fn, batch=batch, device=device,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                conditioning_dropout_prob=conditioning_dropout_prob,
                timestep_sampling=timestep_sampling,
                min_train_timestep=min_train_timestep, max_train_timestep=max_train_timestep,
                sample_source_posterior=sample_source_posterior,
                sample_target_posterior=sample_target_posterior,
                noise_offset=noise_offset,
                identity_loss_on_image_dropped_samples=identity_loss_on_image_dropped_samples,
                generator=generator, global_step=global_step,
            )
            first_out = first["loss_out"]
            if not torch.isfinite(first_out["loss"]):
                window_failed = True
            else:
                _backward(first_out["loss"] * (micro_weight * first_weight), scaler)
            second_out = None
            if use_double and not window_failed:
                second = run_training_step(
                    bundle=bundle, loss_fn=loss_fn, batch=batch, device=device,
                    prompts=batch["generic_prompt"], prepared=first["prepared"],
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    conditioning_dropout_prob=conditioning_dropout_prob,
                    identity_loss_on_image_dropped_samples=identity_loss_on_image_dropped_samples,
                    generator=generator, global_step=global_step,
                )
                second_out = second["loss_out"]
                if not torch.isfinite(second_out["loss"]):
                    window_failed = True
                else:
                    # Sequential backward: the first U-Net graph is already gone.
                    _backward(second_out["loss"] * (micro_weight * second_weight), scaler)
                    double_prompt_batches += 1
            if not window_failed:
                component_values = _component_values(first_out, second_out, first_weight, second_weight)
                tracker.update(component_values, weight=batch_samples)
                tracker.update(first["diagnostics"], weight=batch_samples)
                tracker.update({"double_prompt_fraction": float(use_double)}, weight=batch_samples)
                _update_gap_metrics(tracker, batch, first_out, loss_fn)
                step_timesteps = first["prepared"]["timesteps"].detach().cpu().double()
                timestep_count += step_timesteps.numel()
                timestep_sum += float(step_timesteps.sum())
                timestep_square_sum += float(step_timesteps.square().sum())
                timestep_min = min(timestep_min, float(step_timesteps.min()))
                timestep_max = max(timestep_max, float(step_timesteps.max()))
                total_timesteps = len(bundle["scheduler_train"].alphas_cumprod)
                quartiles = torch.clamp(step_timesteps.long() * 4 // total_timesteps, max=3)
                timestep_quartile_counts += torch.bincount(quartiles, minlength=4)
            processed_batches += 1
            processed_samples += batch_samples
            global_step += 1
        if window_failed:
            optimizer.zero_grad(set_to_none=True)
            skipped_nonfinite += 1
            consecutive_nonfinite += 1
            if not skip_nonfinite_loss or consecutive_nonfinite >= abort_after_nonfinite_steps:
                raise FloatingPointError(f"Non-finite loss encountered {consecutive_nonfinite} consecutive times")
            continue
        step_report = perform_safe_optimizer_step(
            optimizer, parameters, scaler=scaler, lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm, safe_snapshot=safe_optimizer_step,
        )
        if step_report["applied"]:
            optimizer_step += 1
            consecutive_nonfinite = 0
        else:
            skipped_updates += 1
            consecutive_nonfinite += 1
            if consecutive_nonfinite >= abort_after_nonfinite_steps:
                raise FloatingPointError(f"Optimizer step repeatedly failed: {step_report['reason']}")
        tracker.update({"grad_norm": step_report["grad_norm"], "gradient_clipped_fraction": float(step_report["clipped"])}, weight=window_samples)
        tracker.update(optimizer_group_lrs(optimizer), weight=window_samples)
        if log_every and (processed_batches % log_every == 0 or processed_batches == total_batches):
            current = tracker.averages()
            elapsed_now = max(time.perf_counter() - start, 1e-9)
            batches_per_second = processed_batches / elapsed_now
            eta = (total_batches - processed_batches) / max(batches_per_second, 1e-9)
            progress = 100.0 * processed_batches / max(total_batches, 1)
            print(
                f"   {processed_batches:04d}/{total_batches:04d}  {progress:6.1f}%  {optimizer_step:8d}  "
                f"{current.get('loss_total', float('nan')):8.4f}  "
                f"{current.get('loss_diff', float('nan')):9.4f}  "
                f"{current.get('loss_id', float('nan')):8.4f}  "
                f"{current.get('loss_age', float('nan')):7.3f}  "
                f"{current.get('loss_relative_age', float('nan')):7.3f}  "
                f"{current.get('grad_norm', float('nan')):7.3f}  "
                f"{current.get('lr_adapter', float('nan')):9.2e}  "
                f"{processed_samples / elapsed_now:9.2f}  {_format_eta(eta):>8}"
            )
    elapsed = time.perf_counter() - start
    metrics = tracker.averages(prefix="train")
    if timestep_count:
        mean = timestep_sum / timestep_count
        variance = max(0.0, timestep_square_sum / timestep_count - mean * mean)
        metrics.update({
            "train/timestep_mean": mean,
            "train/timestep_std": math.sqrt(variance),
            "train/timestep_min": timestep_min,
            "train/timestep_max": timestep_max,
            **{f"train/timestep_q{index + 1}_fraction": float(timestep_quartile_counts[index]) / timestep_count for index in range(4)},
        })
    metrics.update({
        "train/skipped_nonfinite": float(skipped_nonfinite),
        "train/skipped_optimizer_updates": float(skipped_updates),
        "train/samples_per_second": processed_samples / elapsed if elapsed else 0.0,
    })
    return {
        "metrics": metrics,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "num_batches": processed_batches,
        "num_samples": processed_samples,
        "optimizer_updates": optimizer_step - initial_optimizer_step,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_optimizer_updates": skipped_updates,
        "double_prompt_batches": double_prompt_batches,
        "duration_seconds": elapsed,
    }
