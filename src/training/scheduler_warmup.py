"""Optimizer-step linear warmup followed by ratio-preserving cosine decay."""

from __future__ import annotations

import math


class WarmupCosineLR:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.1):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            if not (total_steps == 1 and warmup_steps == 0):
                raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_num = 0
        self._set_lrs(0)

    def multiplier_at(self, step: int) -> float:
        step = max(0, min(int(step), self.total_steps))
        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine

    def _set_lrs(self, step: int) -> None:
        multiplier = self.multiplier_at(step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        self.step_num += 1
        self._set_lrs(self.step_num)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("total_steps", "warmup_steps", "min_lr_ratio", "base_lrs", "step_num"):
            if key not in state:
                raise ValueError(f"Scheduler checkpoint missing {key}")
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler parameter-group count mismatch")
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.step_num = int(state["step_num"])
        self._set_lrs(self.step_num)


def estimate_optimizer_steps(num_batches: int, grad_accum_steps: int) -> int:
    if num_batches <= 0 or grad_accum_steps <= 0:
        raise ValueError("num_batches and grad_accum_steps must be positive")
    return math.ceil(num_batches / grad_accum_steps)


def compute_warmup_steps(total_steps: int, warmup_ratio: float = 0.05) -> int:
    if total_steps <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("Invalid scheduler budget or warmup ratio")
    if total_steps == 1 or warmup_ratio == 0:
        return 0
    return min(total_steps - 1, max(1, math.floor(total_steps * warmup_ratio)))
