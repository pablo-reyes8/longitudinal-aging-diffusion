"""Small graph-free calibration metrics for fixed age-monitoring sweeps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def fit_age_response_calibration(rows: Iterable[Mapping]) -> dict[str, float] | None:
    """Fit predicted delta = intercept + slope * requested delta.

    Invalid rows are ignored. At least two distinct requested deltas are needed.
    No model is called here; the function only consumes existing diagnostics.
    """
    points = []
    for row in rows:
        requested = row.get("target_delta_age")
        predicted = row.get("predicted_delta_age")
        if requested is None or predicted is None:
            continue
        x, y = float(requested), float(predicted)
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if len(points) < 2:
        return None
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    x_variance_sum = sum((x - x_mean) ** 2 for x, _ in points)
    if x_variance_sum <= 1e-12:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / x_variance_sum
    intercept = y_mean - slope * x_mean
    residual_sum = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total_sum = sum((y - y_mean) ** 2 for _, y in points)
    r2 = (
        1.0 if residual_sum <= 1e-12 else 0.0
    ) if total_sum <= 1e-12 else 1.0 - residual_sum / total_sum
    score = abs(intercept) + 10.0 * abs(slope - 1.0)
    return {
        "age_calibration_intercept": float(intercept),
        "age_calibration_slope": float(slope),
        "age_calibration_r2": float(r2),
        "age_calibration_score": float(score),
    }


def compute_directional_age_metrics(rows: Iterable[Mapping]) -> dict[str, float | None]:
    """Summarize signed-delta errors without running additional inference."""
    errors: dict[str, list[float]] = {"forward": [], "reverse": []}
    for row in rows:
        requested = row.get("target_delta_age")
        predicted = row.get("predicted_delta_age")
        if requested is None or predicted is None:
            continue
        requested, predicted = float(requested), float(predicted)
        if not (math.isfinite(requested) and math.isfinite(predicted)):
            continue
        direction = "forward" if requested > 0 else "reverse" if requested < 0 else None
        if direction is not None:
            errors[direction].append(predicted - requested)

    result: dict[str, float | None] = {}
    for direction, values in errors.items():
        result[f"{direction}_mae"] = (
            sum(abs(value) for value in values) / len(values) if values else None
        )
        result[f"{direction}_bias"] = sum(values) / len(values) if values else None
    return result
