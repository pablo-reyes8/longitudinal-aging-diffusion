"""Optional manual DoRA adapter using the same target discovery as LoRA."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .LoRa import DEFAULT_ATTENTION_TARGETS, find_target_linear_modules, freeze_all_parameters, replace_module


class DoRALinear(nn.Module):
    """Weight-decomposed low-rank adaptation with trainable row magnitudes."""

    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16, dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"DoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.base_layer.requires_grad_(False)
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.magnitude = nn.Parameter(torch.linalg.vector_norm(base_layer.weight.detach().float(), dim=1))

    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_up.weight @ self.lora_down.weight) * self.scale

    def get_effective_weight(self) -> torch.Tensor:
        base = self.base_layer.weight
        direction = base + self.get_delta_weight().to(device=base.device, dtype=base.dtype)
        norm = torch.linalg.vector_norm(direction.float(), dim=1, keepdim=True).clamp_min(self.eps)
        normalized = direction / norm.to(direction.dtype)
        return normalized * self.magnitude.to(device=base.device, dtype=base.dtype).unsqueeze(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        delta = self.get_effective_weight() - self.base_layer.weight
        adapter_output = F.linear(self.dropout(inputs), delta, bias=None)
        return base_output + adapter_output.to(base_output.dtype)

    @property
    def adapter_parameter_count(self) -> int:
        return self.rank * (self.base_layer.in_features + self.base_layer.out_features) + self.base_layer.out_features


def inject_manual_dora_unet(
    unet: nn.Module,
    rank: int = 16,
    alpha: float = 16,
    dropout: float = 0.0,
    target_suffixes: Sequence[str] = DEFAULT_ATTENTION_TARGETS,
    *,
    require_all_targets: bool = True,
    verbose: bool = True,
) -> nn.Module:
    if any(isinstance(module, DoRALinear) for module in unet.modules()):
        raise RuntimeError("DoRA is already injected into this U-Net")
    freeze_all_parameters(unet)
    matches = find_target_linear_modules(unet, target_suffixes)
    counts = Counter(category for _, _, category in matches)
    missing = [target for target in target_suffixes if counts[target] == 0]
    if not matches or (require_all_targets and missing):
        raise RuntimeError(
            f"DoRA target coverage failed: matched={len(matches)}, "
            f"counts={dict(counts)}, missing={missing}"
        )
    names = []
    for name, module, _ in matches:
        wrapper = DoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        wrapper.to(device=module.weight.device, dtype=module.weight.dtype)
        replace_module(unet, name, wrapper)
        names.append(name)
    for name, parameter in unet.named_parameters():
        parameter.requires_grad_(
            ".lora_down." in name or ".lora_up." in name or name.endswith(".magnitude")
        )
    report = {
        "adapter_type": "dora",
        "wrapped_module_names": names,
        "counts_by_target": {target: counts[target] for target in target_suffixes},
        "target_modules": list(target_suffixes),
        "expected_adapter_parameters": sum(
            rank * (m.in_features + m.out_features) + m.out_features for _, m, _ in matches
        ),
    }
    unet._face_aging_adapter_report = report
    if verbose:
        print(f"Injected DoRA into {len(names)} linear layers: {report['counts_by_target']}")
    return unet
