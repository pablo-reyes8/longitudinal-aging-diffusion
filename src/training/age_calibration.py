"""Small graph-free calibration metrics for fixed age-monitoring sweeps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


TARGET_INTERCEPT = -3.19
TARGET_SLOPE = 0.84
REAL_IDENTITY_MEAN = 0.532
REAL_IDENTITY_MEDIAN = 0.589


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
    score = (
        abs(intercept - TARGET_INTERCEPT)
        + 10.0 * abs(slope - TARGET_SLOPE)
    )
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


def fit_directional_age_calibrations(rows: Iterable[Mapping]) -> dict[str, float]:
    """Fit independent regressions for positive and negative requested deltas."""
    materialized = list(rows)
    output: dict[str, float] = {}
    for direction, predicate in (
        ("forward", lambda value: value > 0),
        ("reverse", lambda value: value < 0),
    ):
        selected = []
        for row in materialized:
            requested = row.get("target_delta_age")
            if requested is None:
                continue
            requested = float(requested)
            if math.isfinite(requested) and predicate(requested):
                selected.append(row)
        fit = fit_age_response_calibration(selected)
        output[f"{direction}_calibration_intercept"] = (
            fit["age_calibration_intercept"] if fit is not None else math.nan
        )
        output[f"{direction}_calibration_slope"] = (
            fit["age_calibration_slope"] if fit is not None else math.nan
        )
        output[f"{direction}_calibration_r2"] = (
            fit["age_calibration_r2"] if fit is not None else math.nan
        )
    return output


def summarize_age_diagnostics(rows: Iterable[Mapping]) -> dict[str, float]:
    """Return calibration, direction-error, and identity summaries for one sweep."""
    materialized = list(rows)
    calibration = fit_age_response_calibration(materialized)
    if calibration is None:
        calibration = {
            "age_calibration_intercept": math.nan,
            "age_calibration_slope": math.nan,
            "age_calibration_r2": math.nan,
            "age_calibration_score": math.nan,
        }
    direction = compute_directional_age_metrics(materialized)
    direction_finite = {
        key: math.nan if value is None else float(value)
        for key, value in direction.items()
    }
    identities = []
    for row in materialized:
        value = row.get("identity_cosine")
        if value is not None and math.isfinite(float(value)):
            identities.append(float(value))
    return {
        **calibration,
        **fit_directional_age_calibrations(materialized),
        **direction_finite,
        "mean_identity_cosine": (
            sum(identities) / len(identities) if identities else math.nan
        ),
    }
