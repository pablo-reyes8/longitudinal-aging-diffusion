"""Direct, one-step diffusion timestep sampling utilities."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def scheduler_num_train_timesteps(scheduler: Any) -> int:
    config = scheduler.config
    value = config.get("num_train_timesteps") if isinstance(config, Mapping) else getattr(config, "num_train_timesteps")
    total = int(value)
    if total <= 0:
        raise ValueError("scheduler.config.num_train_timesteps must be positive")
    return total


def sample_diffusion_timesteps(
    batch_size: int,
    scheduler: Any,
    device: str | torch.device,
    generator: torch.Generator | None = None,
    strategy: str = "uniform",
    min_timestep: int = 0,
    max_timestep: int | None = None,
) -> torch.Tensor:
    """Sample one timestep per observation; this never runs a diffusion chain."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if strategy != "uniform":
        raise ValueError("V1 supports only strategy='uniform'")
    total = scheduler_num_train_timesteps(scheduler)
    upper = total - 1 if max_timestep is None else int(max_timestep)
    lower = int(min_timestep)
    if lower < 0 or upper >= total or lower > upper:
        raise ValueError(f"Invalid inclusive timestep range [{lower}, {upper}] for T={total}")
    # CPU generators are useful for reproducible CPU/GPU-independent diagnostics.
    generator_device = torch.device(getattr(generator, "device", "cpu")) if generator is not None else torch.device(device)
    sampled = torch.randint(lower, upper + 1, (batch_size,), generator=generator, device=generator_device, dtype=torch.long)
    return sampled.to(device=device)


def deterministic_validation_timesteps(
    batch_size: int,
    scheduler: Any,
    device: str | torch.device,
    *,
    batch_index: int = 0,
    min_timestep: int = 0,
    max_timestep: int | None = None,
) -> torch.Tensor:
    """Cycle through quartile centers so fixed validation covers low-to-high SNR."""
    total = scheduler_num_train_timesteps(scheduler)
    upper = total - 1 if max_timestep is None else int(max_timestep)
    lower = int(min_timestep)
    if lower < 0 or upper >= total or lower > upper:
        raise ValueError(f"Invalid inclusive timestep range [{lower}, {upper}] for T={total}")
    span = upper - lower + 1
    centers = [lower + min(span - 1, int(span * fraction)) for fraction in (0.125, 0.375, 0.625, 0.875)]
    values = [centers[(batch_index * batch_size + index) % 4] for index in range(batch_size)]
    return torch.tensor(values, dtype=torch.long, device=device)


def timestep_statistics(timesteps: torch.Tensor, scheduler: Any) -> dict[str, float]:
    values = timesteps.detach().float()
    total = scheduler_num_train_timesteps(scheduler)
    quartile = torch.clamp((timesteps.detach().long() * 4) // total, max=3)
    result = {
        "timestep_mean": float(values.mean()),
        "timestep_std": float(values.std(unbiased=False)),
        "timestep_min": float(values.min()),
        "timestep_max": float(values.max()),
    }
    result.update({f"timestep_q{index + 1}_fraction": float((quartile == index).float().mean()) for index in range(4)})
    return result
