"""Explicit adapters for frozen identity and age networks."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _identity(images: torch.Tensor) -> torch.Tensor:
    return images


def _as_tensor_output(output, component: str) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"{component} must return a Tensor after output_transform; got {type(output)!r}"
        )
    return output


class _FrozenAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        preprocess: TensorTransform | None = None,
        output_transform: Callable | None = None,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.preprocess = preprocess or _identity
        self.output_transform = output_transform or _identity
        self.activation_checkpointing = bool(activation_checkpointing)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _run(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        if self.activation_checkpointing and torch.is_grad_enabled() and processed.requires_grad:
            output = checkpoint(self.model, processed, use_reentrant=False)
        else:
            output = self.model(processed)
        return _as_tensor_output(self.output_transform(output), type(self).__name__)


class IdentityEncoderAdapter(_FrozenAdapter):
    """Turn a frozen face encoder into normalized ``[B, D]`` embeddings."""

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        embeddings = self._run(images_01)
        if embeddings.ndim > 2:
            embeddings = embeddings.flatten(start_dim=1)
        if embeddings.ndim != 2 or embeddings.shape[0] != images_01.shape[0]:
            raise ValueError(f"Identity encoder must return [B, D], got {tuple(embeddings.shape)}")
        stable = embeddings.double() if embeddings.dtype == torch.float64 else embeddings.float()
        return F.normalize(stable, p=2, dim=-1, eps=1e-12)

    @torch.no_grad()
    def encode_reference(self, images_01: torch.Tensor) -> torch.Tensor:
        return self(images_01).detach()


def expected_age_from_logits(logits: torch.Tensor, age_values: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Age logits must have shape [B, K], got {tuple(logits.shape)}")
    logits_fp = logits.double() if logits.dtype == torch.float64 else logits.float()
    if age_values is None:
        values = torch.arange(logits.shape[1], device=logits.device, dtype=logits_fp.dtype)
    else:
        values = torch.as_tensor(age_values, device=logits.device, dtype=logits_fp.dtype)
        if values.ndim != 1 or values.numel() != logits.shape[1]:
            raise ValueError(f"age_values must contain {logits.shape[1]} values")
    return (logits_fp.softmax(dim=-1) * values.unsqueeze(0)).sum(dim=-1)


class AgeEstimatorAdapter(_FrozenAdapter):
    """Convert explicit scalar or classifier outputs into continuous age ``[B]``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        output_type: str = "scalar",
        age_values: torch.Tensor | Sequence[float] | None = None,
        preprocess: TensorTransform | None = None,
        output_transform: Callable | None = None,
        activation_checkpointing: bool = False,
    ) -> None:
        if output_type not in {"scalar", "logits"}:
            raise ValueError("output_type must be 'scalar' or 'logits'")
        super().__init__(model, preprocess, output_transform, activation_checkpointing)
        self.output_type = output_type
        if age_values is not None:
            self.register_buffer("age_values", torch.as_tensor(age_values, dtype=torch.float32), persistent=True)
        else:
            self.age_values = None

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        output = self._run(images_01)
        if self.output_type == "logits":
            ages = expected_age_from_logits(output, self.age_values)
        else:
            if output.ndim == 2 and output.shape[1] == 1:
                output = output[:, 0]
            if output.ndim != 1:
                raise ValueError(f"Scalar age estimator must return [B] or [B,1], got {tuple(output.shape)}")
            ages = output.double() if output.dtype == torch.float64 else output.float()
        if ages.shape[0] != images_01.shape[0]:
            raise ValueError("Age-estimator output batch does not match image batch")
        return ages


def identity_cosine_loss(
    predicted_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if predicted_embeddings.shape != reference_embeddings.shape:
        raise ValueError("Predicted/reference identity embedding shapes must match")
    dtype = torch.float64 if predicted_embeddings.dtype == torch.float64 else torch.float32
    predicted = F.normalize(predicted_embeddings.to(dtype), dim=-1, eps=1e-12)
    reference = F.normalize(reference_embeddings.to(dtype), dim=-1, eps=1e-12)
    per_sample = 1.0 - (predicted * reference).sum(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be 'none' or 'mean'")
