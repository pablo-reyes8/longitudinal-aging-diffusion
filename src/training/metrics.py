"""Graph-free, sample-weighted training metrics."""

from __future__ import annotations

from collections import defaultdict

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value, weight: int | float = 1) -> None:
        scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
        self.total += scalar * float(weight)
        self.weight += float(weight)

    @property
    def average(self) -> float:
        return self.total / self.weight if self.weight else float("nan")


class MetricsTracker:
    def __init__(self) -> None:
        self.meters: dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, values: dict[str, float], *, weight: int | float = 1) -> None:
        for name, value in values.items():
            if value is not None:
                self.meters[name].update(value, weight)

    def averages(self, prefix: str | None = None) -> dict[str, float]:
        return {(f"{prefix}/{name}" if prefix else name): meter.average for name, meter in self.meters.items()}


AGE_GAP_BINS = (
    (-float("inf"), -30, "reverse_30+"),
    (-29, -20, "reverse_20-29"),
    (-19, -10, "reverse_10-19"),
    (-9, -5, "reverse_5-9"),
    (-4, -1, "reverse_1-4"),
    (0, 0, "0"),
    (1, 4, "1-4"),
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, None, "30+"),
)
AGE_BANDS = ((0, 14, "0-14"), (15, 29, "15-29"), (30, 44, "30-44"), (45, 59, "45-59"), (60, None, "60+"))


def bin_name(value: float, bins) -> str:
    for lower, upper, name in bins:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "outside"


def optimizer_group_lrs(optimizer) -> dict[str, float]:
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("group_name", f"group_{index}")
        result[f"lr_{name}"] = float(group["lr"])
    return result
