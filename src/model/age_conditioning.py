"""Explicit relative-age conditioning for the SD1.x timestep pathway."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class AgeDeltaConditioner(nn.Module):
    """Map normalized scalar age differences to U-Net time-embedding vectors."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1280,
        *,
        age_delta_scale: float = 80.0,
        activation: str = "silu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim != 1:
            raise ValueError("V1 AgeDeltaConditioner supports input_dim=1 (delta only)")
        if hidden_dim <= 0 or output_dim <= 0 or age_delta_scale <= 0:
            raise ValueError("hidden_dim, output_dim, and age_delta_scale must be positive")
        if activation != "silu":
            raise ValueError("V1 AgeDeltaConditioner supports activation='silu'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.age_delta_scale = float(age_delta_scale)
        self.activation = activation
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)

        # Zero delta starts as exactly no perturbation. A tiny final projection
        # keeps non-zero deltas distinguishable without disrupting SD1.5 at init.
        nn.init.zeros_(self.network[0].bias)
        final = self.network[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(self, delta_age: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(delta_age)
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("delta_age must have shape [B] or [B,1]")
        parameter = next(self.parameters())
        normalized = values.to(device=parameter.device, dtype=parameter.dtype) / self.age_delta_scale
        embedding = self.network(normalized)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Age-delta conditioning produced NaN/Inf")
        return embedding

    def get_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "age_delta_scale": self.age_delta_scale,
            "activation": self.activation,
            "use_layernorm": self.use_layernorm,
            "dropout": self.dropout,
            "inputs": "delta_only",
        }


class AgeConditionerV2(nn.Module):
    """Encode source, target and relative age with raw and Fourier features."""

    def __init__(
        self,
        num_fourier_frequencies: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 1280,
        *,
        source_age_scale: float = 100.0,
        target_age_scale: float = 100.0,
        delta_age_scale: float = 80.0,
        activation: str = "silu",
        use_raw_scalars: bool = True,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        use_gate: bool = True,
        initial_gate: float = 1.0,
    ) -> None:
        super().__init__()
        if num_fourier_frequencies <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Fourier frequencies and MLP dimensions must be positive")
        if min(source_age_scale, target_age_scale, delta_age_scale) <= 0:
            raise ValueError("Age normalization scales must be positive")
        if activation != "silu":
            raise ValueError("V2 supports activation='silu'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        input_dim = 3 * 2 * int(num_fourier_frequencies) + (3 if use_raw_scalars else 0)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.register_buffer(
            "fourier_frequencies",
            torch.pow(2.0, torch.arange(num_fourier_frequencies, dtype=torch.float32)) * torch.pi,
            persistent=False,
        )
        self.age_scale = nn.Parameter(torch.tensor(float(initial_gate))) if use_gate else None
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.source_age_scale = float(source_age_scale)
        self.target_age_scale = float(target_age_scale)
        self.delta_age_scale = float(delta_age_scale)
        self.activation = activation
        self.use_raw_scalars = bool(use_raw_scalars)
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)
        self.use_gate = bool(use_gate)
        self.initial_gate = float(initial_gate)

        # Preserve the pretrained U-Net closely while allowing useful gradients
        # through every input feature from the first update.
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _column(values: torch.Tensor, name: str, *, device, dtype) -> torch.Tensor:
        tensor = torch.as_tensor(values, device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 2 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B] or [B,1]")
        return tensor

    def fourier_features(
        self,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        delta_age: torch.Tensor,
    ) -> torch.Tensor:
        parameter = next(self.network.parameters())
        source = self._column(source_age, "source_age", device=parameter.device, dtype=parameter.dtype)
        target = self._column(target_age, "target_age", device=parameter.device, dtype=parameter.dtype)
        delta = self._column(delta_age, "delta_age", device=parameter.device, dtype=parameter.dtype)
        if source.shape[0] != target.shape[0] or source.shape[0] != delta.shape[0]:
            raise ValueError("source_age, target_age, and delta_age batch sizes must match")
        normalized = torch.cat(
            (source / self.source_age_scale, target / self.target_age_scale, delta / self.delta_age_scale),
            dim=1,
        )
        angles = normalized.unsqueeze(-1) * self.fourier_frequencies.to(
            device=parameter.device, dtype=parameter.dtype
        )
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
        features = torch.cat((normalized, fourier), dim=1) if self.use_raw_scalars else fourier
        if not torch.isfinite(features).all():
            raise FloatingPointError("Age Fourier features contain NaN/Inf")
        return features

    def forward(
        self,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        delta_age: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.network(self.fourier_features(source_age, target_age, delta_age))
        if self.age_scale is not None:
            embedding = self.age_scale * embedding
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Age conditioning produced NaN/Inf")
        return embedding

    def get_config(self) -> dict[str, Any]:
        return {
            "version": "v2_fourier",
            "inputs": "source_target_delta",
            "num_fourier_frequencies": self.num_fourier_frequencies,
            "input_dim": self.network[0].in_features,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "source_age_scale": self.source_age_scale,
            "target_age_scale": self.target_age_scale,
            "delta_age_scale": self.delta_age_scale,
            "activation": self.activation,
            "use_raw_scalars": self.use_raw_scalars,
            "use_layernorm": self.use_layernorm,
            "dropout": self.dropout,
            "use_gate": self.use_gate,
            "initial_gate": self.initial_gate,
        }


class AgeConditionedTimeEmbedding(nn.Module):
    """Add an already projected age vector to Diffusers' timestep embedding."""

    def __init__(self, base_time_embedding: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.base_time_embedding = base_time_embedding
        self.output_dim = int(output_dim)

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None):
        try:
            embedding = self.base_time_embedding(sample, None)
        except TypeError:
            embedding = self.base_time_embedding(sample)
        if condition is None:
            return embedding
        if condition.ndim != 2 or condition.shape != (embedding.shape[0], self.output_dim):
            raise ValueError(
                f"Age condition must have shape [{embedding.shape[0]}, {self.output_dim}], "
                f"got {tuple(condition.shape)}"
            )
        return embedding + condition.to(device=embedding.device, dtype=embedding.dtype)


def infer_unet_time_embedding_dim(unet: nn.Module) -> int:
    time_embedding = getattr(unet, "time_embedding", None)
    if time_embedding is None:
        raise TypeError("Age-delta conditioning requires unet.time_embedding")
    if isinstance(time_embedding, AgeConditionedTimeEmbedding):
        return time_embedding.output_dim
    linear_2 = getattr(time_embedding, "linear_2", None)
    if isinstance(linear_2, nn.Linear):
        return int(linear_2.out_features)
    output_dim = getattr(time_embedding, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)
    raise TypeError("Could not infer the U-Net time-embedding output dimension")


def install_age_conditioned_time_embedding(unet: nn.Module, output_dim: int) -> None:
    current = getattr(unet, "time_embedding", None)
    if isinstance(current, AgeConditionedTimeEmbedding):
        if current.output_dim != int(output_dim):
            raise ValueError("Existing age-conditioned time embedding has a different dimension")
        return
    if current is None:
        raise TypeError("Age-delta conditioning requires unet.time_embedding")
    unet.time_embedding = AgeConditionedTimeEmbedding(current, output_dim)


def compute_age_delta_embedding(
    bundle: Mapping[str, Any],
    delta_age: torch.Tensor | None,
    *,
    batch_size: int,
    source_age: torch.Tensor | None = None,
    target_age: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return an explicit U-Net timestep condition, or ``None`` when disabled."""
    if not bundle.get("use_age_delta_conditioning", False):
        return None
    conditioner = bundle.get("age_conditioner")
    if conditioner is None:
        conditioner = bundle.get("age_delta_conditioner")
    if conditioner is None:
        raise RuntimeError("Bundle enables age-delta conditioning but has no conditioner")
    if delta_age is None:
        raise ValueError("delta_age is required when age-delta conditioning is enabled")
    values = torch.as_tensor(delta_age)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError(f"delta_age must have shape [{batch_size}]")
    if isinstance(conditioner, AgeConditionerV2):
        if source_age is None or target_age is None:
            raise ValueError("source_age and target_age are required by AgeConditionerV2")
        return conditioner(source_age, target_age, values)
    return conditioner(values)
