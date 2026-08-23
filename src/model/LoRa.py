"""Dependency-free manual LoRA for Diffusers attention projections."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import torch
from torch import nn


DEFAULT_ATTENTION_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable low-rank residual ``B(A(x))``."""

    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.base_layer.requires_grad_(False)
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        adapter_input = self.dropout(inputs).to(self.lora_down.weight.dtype)
        adapter_output = self.lora_up(self.lora_down(adapter_input)) * self.scale
        return base_output + adapter_output.to(base_output.dtype)

    @property
    def adapter_parameter_count(self) -> int:
        return self.rank * (self.base_layer.in_features + self.base_layer.out_features)


def freeze_all_parameters(model: nn.Module) -> nn.Module:
    model.requires_grad_(False)
    return model


def module_matches_targets(module_name: str, target_suffixes: Sequence[str]) -> bool:
    return any(module_name == suffix or module_name.endswith(f".{suffix}") for suffix in target_suffixes)


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent, child_name = get_parent_module(root, module_name)
    if child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _target_category(module_name: str, targets: Sequence[str]) -> str:
    matches = [target for target in targets if module_name == target or module_name.endswith(f".{target}")]
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous adapter target for {module_name!r}: {matches}")
    return matches[0]


def find_target_linear_modules(
    unet: nn.Module, target_suffixes: Sequence[str] = DEFAULT_ATTENTION_TARGETS
) -> list[tuple[str, nn.Linear, str]]:
    targets = tuple(target_suffixes)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_suffixes must be non-empty and unique")
    found = []
    for name, module in unet.named_modules():
        if module_matches_targets(name, targets) and isinstance(module, nn.Linear):
            found.append((name, module, _target_category(name, targets)))
    return found


def inject_manual_lora_unet(
    unet: nn.Module,
    rank: int = 16,
    alpha: float = 16,
    dropout: float = 0.0,
    target_suffixes: Sequence[str] = DEFAULT_ATTENTION_TARGETS,
    *,
    require_all_targets: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Freeze and inject LoRA into matching U-Net linear layers in-place."""
    if any(isinstance(module, LoRALinear) for module in unet.modules()):
        raise RuntimeError("LoRA is already injected into this U-Net")
    freeze_all_parameters(unet)
    matches = find_target_linear_modules(unet, target_suffixes)
    counts = Counter(category for _, _, category in matches)
    missing = [target for target in target_suffixes if counts[target] == 0]
    if not matches or (require_all_targets and missing):
        raise RuntimeError(
            f"LoRA target coverage failed: matched={len(matches)}, "
            f"counts={dict(counts)}, missing={missing}"
        )
    names = []
    for name, module, _ in matches:
        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        wrapper.to(device=module.weight.device, dtype=module.weight.dtype)
        replace_module(unet, name, wrapper)
        names.append(name)
    for name, parameter in unet.named_parameters():
        parameter.requires_grad_(".lora_down." in name or ".lora_up." in name)
    report = {
        "adapter_type": "lora",
        "wrapped_module_names": names,
        "counts_by_target": {target: counts[target] for target in target_suffixes},
        "target_modules": list(target_suffixes),
        "expected_adapter_parameters": sum(rank * (m.in_features + m.out_features) for _, m, _ in matches),
    }
    unet._face_aging_adapter_report = report
    if verbose:
        print(f"Injected LoRA into {len(names)} linear layers: {report['counts_by_target']}")
    return unet
