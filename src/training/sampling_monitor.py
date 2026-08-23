"""Callback-only deterministic monitoring; no fake inference implementation."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch


def sample_monitoring_images(sample_fn, **kwargs):
    if sample_fn is None:
        return {"status": "NOT RUN", "reason": "sample_fn was not supplied"}
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(sample_fn)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    with torch.inference_mode():
        result = sample_fn(**supported)
    return {"status": "PASSED", "result": result}
