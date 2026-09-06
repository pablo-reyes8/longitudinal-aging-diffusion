"""Graph-free evaluation of existing age diagnostics by requested delta size."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
import statistics

import pandas as pd


DEFAULT_DELTA_BIN_THRESHOLDS = (5.0, 15.0, 30.0)
DELTA_BIN_COLUMNS = [
    "delta_bin",
    "direction",
    "N",
    "mean_absolute_age_error",
    "age_bias",
    "mean_requested_delta",
    "mean_predicted_delta",
    "mean_identity_cosine",
    "median_identity_cosine",
    "directional_accuracy",
]


def _validated_thresholds(thresholds: Sequence[float] | None) -> tuple[float, float, float]:
    values = DEFAULT_DELTA_BIN_THRESHOLDS if thresholds is None else tuple(thresholds)
    if len(values) != 3:
        raise ValueError("delta_bin_thresholds must contain exactly three values")
    try:
        resolved = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_bin_thresholds must be finite positive numbers") from exc
    if (
        any(not math.isfinite(value) or value <= 0 for value in resolved)
        or not resolved[0] < resolved[1] < resolved[2]
    ):
        raise ValueError("delta_bin_thresholds must be finite, positive, and increasing")
    return resolved


def _bin_name(delta: float, thresholds: tuple[float, float, float]) -> str:
    magnitude = abs(delta)
    if magnitude == 0:
        return "zero"
    if magnitude <= thresholds[0]:
        return "short"
    if magnitude <= thresholds[1]:
        return "medium"
    if magnitude <= thresholds[2]:
        return "long"
    return "very_long"


def _finite(row: Mapping, key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _summarize(rows: list[dict], *, bin_name: str, direction: str) -> dict:
    requested = [row["requested"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    age_errors = [row["age_error"] for row in rows if row["age_error"] is not None]
    identities = [row["identity"] for row in rows if row["identity"] is not None]
    directional = [
        float(req * pred > 0)
        for req, pred in zip(requested, predicted)
        if req != 0
    ]

    def mean(values):
        return sum(values) / len(values) if values else math.nan

    return {
        "delta_bin": bin_name,
        "direction": direction,
        "N": len(rows),
        "mean_absolute_age_error": mean([abs(value) for value in age_errors]),
        "age_bias": mean(age_errors),
        "mean_requested_delta": mean(requested),
        "mean_predicted_delta": mean(predicted),
        "mean_identity_cosine": mean(identities),
        "median_identity_cosine": statistics.median(identities) if identities else math.nan,
        "directional_accuracy": mean(directional),
    }


def evaluate_delta_bins(
    rows: Iterable[Mapping],
    *,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Summarize existing MiVOLO/identity rows without new inference passes."""
    resolved_thresholds = _validated_thresholds(thresholds)
    prepared = []
    for row in rows:
        requested = _finite(row, "target_delta_age")
        predicted = _finite(row, "predicted_delta_age")
        if requested is None or predicted is None:
            continue
        age_error = _finite(row, "age_error")
        if age_error is None:
            target_age = _finite(row, "target_age")
            predicted_age = _finite(row, "predicted_generated_age")
            if target_age is not None and predicted_age is not None:
                age_error = predicted_age - target_age
        prepared.append({
            "requested": requested,
            "predicted": predicted,
            "age_error": age_error,
            "identity": _finite(row, "identity_cosine"),
            "bin": _bin_name(requested, resolved_thresholds),
            "direction": "forward" if requested > 0 else "reverse" if requested < 0 else "zero",
        })

    names = ("zero", "short", "medium", "long", "very_long")
    output = [
        _summarize(
            [row for row in prepared if row["bin"] == name],
            bin_name=name,
            direction="all",
        )
        for name in names
    ]
    has_forward = any(row["direction"] == "forward" for row in prepared)
    has_reverse = any(row["direction"] == "reverse" for row in prepared)
    if has_forward and has_reverse:
        for name in names:
            for direction in ("forward", "reverse"):
                selected = [
                    row for row in prepared
                    if row["bin"] == name and row["direction"] == direction
                ]
                if selected:
                    output.append(
                        _summarize(selected, bin_name=name, direction=direction)
                    )
    frame = pd.DataFrame(output, columns=DELTA_BIN_COLUMNS)
    frame.attrs["thresholds"] = resolved_thresholds
    return frame


def save_delta_bin_evaluation(
    rows: Iterable[Mapping],
    output_path: str | Path,
    *,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Evaluate diagnostic rows and save the requested one-row-per-group CSV."""
    frame = evaluate_delta_bins(rows, thresholds=thresholds)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    frame.attrs["csv_path"] = str(path)
    return frame
