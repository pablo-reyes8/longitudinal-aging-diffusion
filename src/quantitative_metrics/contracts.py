"""Manifest contracts for model-independent face-aging evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "sample_id",
    "source_path",
    "target_path",
    "generated_path",
    "source_age",
    "target_age",
    "model_name",
)


def load_prediction_manifest(predictions) -> pd.DataFrame:
    """Load and validate a prediction manifest without dropping any rows."""
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
    else:
        path = Path(predictions)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError("predictions must be a DataFrame, CSV, or parquet manifest")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Prediction manifest cannot be empty")
    if frame["sample_id"].isna().any():
        raise ValueError("sample_id cannot contain missing values")
    uniqueness = ["sample_id", "model_name"] + [
        column for column in ("ablation_case", "epoch") if column in frame.columns
    ]
    if frame.duplicated(uniqueness).any():
        raise ValueError(f"Rows must be unique by {', '.join(uniqueness)}")

    frame["source_age"] = pd.to_numeric(frame["source_age"], errors="raise")
    frame["target_age"] = pd.to_numeric(frame["target_age"], errors="raise")
    if "delta_age" not in frame:
        frame["delta_age"] = frame["target_age"] - frame["source_age"]
    return frame.reset_index(drop=True)
