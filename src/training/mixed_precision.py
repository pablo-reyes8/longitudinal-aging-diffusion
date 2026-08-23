"""Device, autocast, and numerically safe optimizer-step helpers."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return resolved


def get_effective_amp_dtype(
    amp_dtype: str | torch.dtype = "auto",
    device: str | torch.device = "auto",
    *,
    amp_enabled: bool = True,
) -> torch.dtype | None:
    if not amp_enabled:
        return None
    resolved = resolve_device(device)
    if isinstance(amp_dtype, torch.dtype):
        requested = amp_dtype
    else:
        name = str(amp_dtype).lower()
        if name == "auto":
            if resolved.type == "cuda":
                requested = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif resolved.type == "cpu":
                requested = torch.bfloat16
            else:
                return None
        elif name in {"bf16", "bfloat16"}:
            requested = torch.bfloat16
        elif name in {"fp16", "float16"}:
            requested = torch.float16
        elif name in {"fp32", "float32", "none"}:
            return None
        else:
            raise ValueError("amp_dtype must be auto, bf16, fp16, or fp32")
    if resolved.type == "cuda":
        if requested == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return torch.float16
        return requested
    if resolved.type == "cpu" and requested == torch.bfloat16:
        return requested
    return None


def autocast_ctx(device, *, enabled: bool = True, amp_dtype: str | torch.dtype = "auto"):
    dtype = get_effective_amp_dtype(amp_dtype, device, amp_enabled=enabled)
    resolved = torch.device(device)
    if dtype is None or resolved.type not in {"cpu", "cuda"}:
        return nullcontext()
    return torch.autocast(device_type=resolved.type, dtype=dtype)


def make_grad_scaler(device, *, amp_enabled: bool = True, amp_dtype: str | torch.dtype = "auto"):
    resolved = torch.device(device)
    dtype = get_effective_amp_dtype(amp_dtype, resolved, amp_enabled=amp_enabled)
    if resolved.type != "cuda" or dtype != torch.float16:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        signature = inspect.signature(torch.amp.GradScaler)
        return torch.amp.GradScaler(device="cuda", enabled=True) if "device" in signature.parameters else torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def setup_device_and_precision(device="auto", *, amp_enabled=True, amp_dtype="auto", scaler=None) -> dict[str, Any]:
    resolved = resolve_device(device)
    effective = get_effective_amp_dtype(amp_dtype, resolved, amp_enabled=amp_enabled)
    actual_scaler = scaler if scaler is not None else make_grad_scaler(resolved, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    return {
        "device": resolved,
        "amp_enabled": effective is not None,
        "amp_dtype_requested": str(amp_dtype),
        "amp_dtype_effective": effective,
        "amp_dtype_name": "fp32" if effective is None else "bf16" if effective == torch.bfloat16 else "fp16",
        "scaler": actual_scaler,
        "use_grad_scaler": actual_scaler is not None,
    }


def move_batch_to_device(value, device: torch.device, *, non_blocking: bool = True):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_batch_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value


def ensure_trainable_parameters_fp32(parameters) -> int:
    converted = 0
    for parameter in parameters:
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
            converted += 1
    return converted


def _all_finite(tensors) -> bool:
    return all(tensor is None or bool(torch.isfinite(tensor).all()) for tensor in tensors)


def safe_optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters,
    *,
    scaler=None,
    lr_scheduler=None,
    max_grad_norm: float | None = 1.0,
    safe_snapshot: bool = True,
) -> dict[str, Any]:
    """Unscale -> finite check -> clip -> step -> parameter check -> schedule."""
    params = list(parameters)
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in params if parameter.grad is not None]
    if not gradients:
        return {"applied": False, "reason": "no_gradients", "grad_norm": 0.0, "clipped": False}
    if not _all_finite(gradients):
        if scaler is not None:
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return {"applied": False, "reason": "nonfinite_gradient", "grad_norm": float("nan"), "clipped": False}
    total_norm = torch.linalg.vector_norm(torch.stack([gradient.detach().float().norm() for gradient in gradients]))
    grad_norm = float(total_norm)
    clipped = max_grad_norm is not None and grad_norm > max_grad_norm
    if max_grad_norm is not None:
        clipped_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if not torch.isfinite(clipped_norm):
            optimizer.zero_grad(set_to_none=True)
            return {"applied": False, "reason": "nonfinite_clip_norm", "grad_norm": grad_norm, "clipped": clipped}
    snapshot = [parameter.detach().clone() for parameter in params] if safe_snapshot else None
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if not _all_finite([parameter.detach() for parameter in params]):
        if snapshot is not None:
            with torch.no_grad():
                for parameter, old in zip(params, snapshot):
                    parameter.copy_(old)
                    optimizer.state.pop(parameter, None)
        optimizer.zero_grad(set_to_none=True)
        return {"applied": False, "reason": "nonfinite_parameter_update", "grad_norm": grad_norm, "clipped": clipped}
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"applied": True, "reason": None, "grad_norm": grad_norm, "clipped": clipped}
