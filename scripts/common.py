"""Small, dependency-light helpers shared by project CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {resolved}")
    return content


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return destination


def resolve_dtype(value: str | None) -> torch.dtype | None:
    normalized = str(value or "auto").lower()
    if normalized == "auto":
        return None
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in choices:
        raise ValueError(f"Unsupported dtype {value!r}; use auto, fp32, fp16, or bf16")
    return choices[normalized]


def model_builder_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["dtype"] = resolve_dtype(result.pop("dtype", "auto"))
    if result.get("device") == "auto":
        result["device"] = None
    return result
