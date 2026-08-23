"""Composite supervised diffusion objective for longitudinal face aging."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .auxiliary_adapters import AgeEstimatorAdapter, IdentityEncoderAdapter, identity_cosine_loss
from .diffusion_utils import (
    compute_diffusion_loss,
    get_diffusion_target,
    get_prediction_type,
    predict_x0_from_model_output,
    sd_image_to_01,
)


def compose_weighted_losses(
    loss_diff: torch.Tensor,
    loss_id: torch.Tensor,
    loss_age: torch.Tensor,
    *,
    diffusion_weight: float,
    identity_weight: float,
    age_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if diffusion_weight <= 0 or identity_weight < 0 or age_weight < 0:
        raise ValueError("diffusion_weight must be > 0 and auxiliary weights must be >= 0")
    weighted_diff = loss_diff * diffusion_weight
    weighted_id = loss_id * identity_weight
    weighted_age = loss_age * age_weight
    return weighted_diff + weighted_id + weighted_age, weighted_diff, weighted_id, weighted_age


class FaceAgingDiffusionLoss(nn.Module):
    """Diffusion MSE plus optional differentiable identity and target-age terms."""

    def __init__(
        self,
        *,
        scheduler: Any,
        vae: nn.Module,
        identity_encoder: IdentityEncoderAdapter | None = None,
        age_estimator: AgeEstimatorAdapter | None = None,
        diffusion_weight: float = 1.0,
        identity_weight: float = 0.1,
        age_weight: float = 0.1,
        identity_reference: str = "target",
        diffusion_loss_type: str = "mse",
        age_loss_type: str = "l1",
        min_snr_gamma: float | None = None,
        auxiliary_every_n_steps: int = 1,
        auxiliary_sample_fraction: float = 1.0,
        auxiliary_max_timestep: int | None = None,
        auxiliary_seed: int = 42,
        clamp_pred_x0: bool = True,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if diffusion_weight <= 0 or identity_weight < 0 or age_weight < 0:
            raise ValueError("diffusion_weight must be > 0 and auxiliary weights must be >= 0")
        if identity_reference not in {"source", "target", "both"}:
            raise ValueError("identity_reference must be 'source', 'target', or 'both'")
        if diffusion_loss_type != "mse":
            raise ValueError("V1 only supports diffusion_loss_type='mse'")
        if age_loss_type not in {"l1", "mse"}:
            raise ValueError("age_loss_type must be 'l1' or 'mse'")
        if min_snr_gamma is not None and min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be positive or None")
        if auxiliary_every_n_steps < 1:
            raise ValueError("auxiliary_every_n_steps must be >= 1")
        if not 0 < auxiliary_sample_fraction <= 1:
            raise ValueError("auxiliary_sample_fraction must be in (0, 1]")
        if auxiliary_max_timestep is not None and auxiliary_max_timestep < 0:
            raise ValueError("auxiliary_max_timestep must be >= 0 or None")
        if identity_weight > 0 and identity_encoder is None:
            raise ValueError("identity_weight > 0 requires identity_encoder")
        if age_weight > 0 and age_estimator is None:
            raise ValueError("age_weight > 0 requires age_estimator")
        get_prediction_type(scheduler)
        self.scheduler = scheduler
        self.vae = vae
        self.identity_encoder = identity_encoder
        self.age_estimator = age_estimator
        self.diffusion_weight = float(diffusion_weight)
        self.identity_weight = float(identity_weight)
        self.age_weight = float(age_weight)
        self.identity_reference = identity_reference
        self.diffusion_loss_type = diffusion_loss_type
        self.age_loss_type = age_loss_type
        self.min_snr_gamma = float(min_snr_gamma) if min_snr_gamma is not None else None
        self.auxiliary_every_n_steps = int(auxiliary_every_n_steps)
        self.auxiliary_sample_fraction = float(auxiliary_sample_fraction)
        self.auxiliary_max_timestep = auxiliary_max_timestep
        self.auxiliary_seed = int(auxiliary_seed)
        # This controls decoded-image conversion clipping only. Latents are never clamped.
        self.clamp_pred_x0 = bool(clamp_pred_x0)
        self.check_finite = bool(check_finite)
        self.vae.requires_grad_(False)
        self.vae.eval()
        if self.identity_encoder is not None:
            self.identity_encoder.eval()
        if self.age_estimator is not None:
            self.age_estimator.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.identity_encoder is not None:
            self.identity_encoder.eval()
        if self.age_estimator is not None:
            self.age_estimator.eval()
        return self

    def get_config(self) -> dict[str, Any]:
        return {
            "diffusion_weight": self.diffusion_weight,
            "identity_weight": self.identity_weight,
            "age_weight": self.age_weight,
            "identity_reference": self.identity_reference,
            "diffusion_loss_type": self.diffusion_loss_type,
            "age_loss_type": self.age_loss_type,
            "min_snr_gamma": self.min_snr_gamma,
            "auxiliary_every_n_steps": self.auxiliary_every_n_steps,
            "auxiliary_sample_fraction": self.auxiliary_sample_fraction,
            "auxiliary_max_timestep": self.auxiliary_max_timestep,
            "auxiliary_seed": self.auxiliary_seed,
            "clamp_pred_x0": self.clamp_pred_x0,
            "check_finite": self.check_finite,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        scheduler: Any,
        vae: nn.Module,
        identity_encoder: IdentityEncoderAdapter | None = None,
        age_estimator: AgeEstimatorAdapter | None = None,
    ) -> "FaceAgingDiffusionLoss":
        return cls(
            scheduler=scheduler,
            vae=vae,
            identity_encoder=identity_encoder,
            age_estimator=age_estimator,
            **dict(config),
        )

    def _check_finite(self, **tensors: torch.Tensor) -> None:
        if not self.check_finite:
            return
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Non-finite values detected in {name}")

    def _validate_core_shapes(
        self,
        model_pred: torch.Tensor,
        noise: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        expected = model_pred.shape
        for name, tensor in (
            ("noise", noise), ("noisy_target_latents", noisy_target_latents),
            ("target_latents", target_latents),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != model_pred shape {tuple(expected)}")
            if tensor.device != model_pred.device:
                raise ValueError(f"{name} and model_pred must be on the same device")
            if tensor.dtype != model_pred.dtype:
                raise ValueError(f"{name} and model_pred must have the same dtype")
        if model_pred.ndim != 4:
            raise ValueError("Latent tensors must have shape [B, C, H, W]")
        if timesteps.ndim != 1 or timesteps.shape[0] != expected[0]:
            raise ValueError(f"timesteps must have shape [{expected[0]}]")
        if timesteps.device != model_pred.device:
            raise ValueError("timesteps and model_pred must be on the same device")

    def _select_auxiliary_indices(self, timesteps: torch.Tensor, global_step: int) -> torch.Tensor:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        if global_step % self.auxiliary_every_n_steps != 0:
            return torch.empty(0, dtype=torch.long, device=timesteps.device)
        eligible = torch.arange(timesteps.shape[0], device=timesteps.device)
        if self.auxiliary_max_timestep is not None:
            eligible = eligible[timesteps <= self.auxiliary_max_timestep]
        if eligible.numel() == 0:
            return eligible
        selected_count = max(1, math.ceil(eligible.numel() * self.auxiliary_sample_fraction))
        if selected_count == eligible.numel():
            return eligible
        generator = torch.Generator(device="cpu").manual_seed(self.auxiliary_seed + int(global_step))
        positions = torch.randperm(eligible.numel(), generator=generator)[:selected_count]
        return eligible.index_select(0, positions.to(eligible.device)).sort().values

    def _normalize_target_ages(
        self,
        target_ages: torch.Tensor | Sequence[int | float] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if target_ages is None:
            raise ValueError("target_ages is required when age loss is active")
        ages = torch.as_tensor(target_ages, device=device, dtype=torch.float32)
        if ages.ndim != 1 or ages.shape[0] != batch_size:
            raise ValueError(f"target_ages must have shape [{batch_size}], got {tuple(ages.shape)}")
        return ages

    def _validate_image_batch(self, name: str, images: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if images is None:
            raise ValueError(f"{name} is required for the configured auxiliary loss")
        if images.ndim != 4 or images.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [B, C, H, W] with B={batch_size}")
        if images.device != device:
            raise ValueError(f"{name} and model_pred must be on the same device")
        return images

    def forward(
        self,
        *,
        model_pred: torch.Tensor,
        noise: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        source_images: torch.Tensor | None = None,
        target_images: torch.Tensor | None = None,
        target_ages: torch.Tensor | Sequence[int | float] | None = None,
        global_step: int = 0,
        return_reconstructions: bool = False,
        return_per_sample: bool = False,
    ) -> dict[str, Any]:
        self._validate_core_shapes(model_pred, noise, noisy_target_latents, target_latents, timesteps)
        self._check_finite(model_pred=model_pred, noise=noise, noisy_target_latents=noisy_target_latents, target_latents=target_latents)
        diffusion_target = get_diffusion_target(self.scheduler, target_latents, noise, timesteps)
        loss_diff, loss_diff_per_sample, snr_weights = compute_diffusion_loss(
            model_pred,
            diffusion_target,
            scheduler=self.scheduler,
            timesteps=timesteps,
            loss_type=self.diffusion_loss_type,
            min_snr_gamma=self.min_snr_gamma,
        )
        zero = loss_diff.new_zeros(())
        loss_id, loss_age = zero, zero
        id_per_sample = loss_diff.new_empty(0)
        age_per_sample = loss_diff.new_empty(0)
        identity_cosine_mean = None
        predicted_age_mean = None
        target_age_mean = None
        pred_x0_latents = None
        pred_x0_images = None

        auxiliary_requested = self.identity_weight > 0 or self.age_weight > 0 or return_reconstructions
        indices = self._select_auxiliary_indices(timesteps, int(global_step)) if auxiliary_requested else torch.empty(0, dtype=torch.long, device=timesteps.device)
        auxiliary_applied = indices.numel() > 0
        if auxiliary_applied:
            selected_pred = model_pred.index_select(0, indices)
            selected_noisy = noisy_target_latents.index_select(0, indices)
            selected_timesteps = timesteps.index_select(0, indices)
            pred_x0_latents = predict_x0_from_model_output(
                selected_pred, selected_noisy, selected_timesteps, self.scheduler
            )
            vae_device = next(self.vae.parameters()).device
            if vae_device != model_pred.device:
                raise ValueError("VAE and model_pred must be on the same device; loss does not move large tensors implicitly")
            scaling_factor = float(getattr(self.vae.config, "scaling_factor"))
            decoded = self.vae.decode(pred_x0_latents.to(next(self.vae.parameters()).dtype) / scaling_factor).sample
            pred_x0_images = sd_image_to_01(decoded, clamp=self.clamp_pred_x0)
            self._check_finite(pred_x0_latents=pred_x0_latents, pred_x0_images=pred_x0_images)

            batch_size = model_pred.shape[0]
            if self.identity_weight > 0:
                target_images_checked = self._validate_image_batch("target_images", target_images, batch_size, model_pred.device)
                pred_embeddings = self.identity_encoder(pred_x0_images)
                reference_losses = []
                if self.identity_reference in {"target", "both"}:
                    target_01 = sd_image_to_01(target_images_checked.index_select(0, indices), clamp=True)
                    target_embeddings = self.identity_encoder.encode_reference(target_01)
                    reference_losses.append(identity_cosine_loss(pred_embeddings, target_embeddings, reduction="none"))
                if self.identity_reference in {"source", "both"}:
                    source_checked = self._validate_image_batch("source_images", source_images, batch_size, model_pred.device)
                    source_01 = sd_image_to_01(source_checked.index_select(0, indices), clamp=True)
                    source_embeddings = self.identity_encoder.encode_reference(source_01)
                    reference_losses.append(identity_cosine_loss(pred_embeddings, source_embeddings, reduction="none"))
                id_per_sample = torch.stack(reference_losses, dim=0).mean(dim=0)
                loss_id = id_per_sample.mean()
                identity_cosine_mean = float((1.0 - id_per_sample.detach()).mean())
            if self.age_weight > 0:
                ages = self._normalize_target_ages(target_ages, model_pred.shape[0], model_pred.device)
                selected_ages = ages.index_select(0, indices)
                predicted_ages = self.age_estimator(pred_x0_images)
                age_per_sample = (
                    (predicted_ages - selected_ages).abs()
                    if self.age_loss_type == "l1"
                    else (predicted_ages - selected_ages).square()
                )
                loss_age = age_per_sample.mean()
                predicted_age_mean = float(predicted_ages.detach().mean())
                target_age_mean = float(selected_ages.detach().mean())

        total_loss, weighted_diff, weighted_id, weighted_age = compose_weighted_losses(
            loss_diff,
            loss_id,
            loss_age,
            diffusion_weight=self.diffusion_weight,
            identity_weight=self.identity_weight,
            age_weight=self.age_weight,
        )
        self._check_finite(loss=total_loss, loss_diff=loss_diff, loss_id=loss_id, loss_age=loss_age)
        diff_denominator = max(abs(float(weighted_diff.detach())), torch.finfo(torch.float32).eps)
        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_diff": float(loss_diff.detach()),
            "loss_id": float(loss_id.detach()),
            "loss_age": float(loss_age.detach()),
            "weighted_diff": float(weighted_diff.detach()),
            "weighted_id": float(weighted_id.detach()),
            "weighted_age": float(weighted_age.detach()),
            "identity_cosine_mean": identity_cosine_mean,
            "pred_age_mean": predicted_age_mean,
            "target_age_mean": target_age_mean,
            "identity_to_diffusion_ratio": abs(float(weighted_id.detach())) / diff_denominator,
            "age_to_diffusion_ratio": abs(float(weighted_age.detach())) / diff_denominator,
            "auxiliary_applied": auxiliary_applied,
            "auxiliary_count": int(indices.numel()),
        }
        output: dict[str, Any] = {
            "loss": total_loss,
            "loss_diff": loss_diff,
            "loss_id": loss_id,
            "loss_age": loss_age,
            "weighted_diff": weighted_diff,
            "weighted_id": weighted_id,
            "weighted_age": weighted_age,
            "auxiliary_applied": auxiliary_applied,
            "auxiliary_indices": indices.detach(),
            "metrics": metrics,
        }
        if snr_weights is not None:
            output["min_snr_weights"] = snr_weights.detach()
        if return_per_sample:
            output.update({
                "loss_diff_per_sample": loss_diff_per_sample,
                "loss_id_per_sample": id_per_sample,
                "loss_age_per_sample": age_per_sample,
            })
        if return_reconstructions and auxiliary_applied:
            output["pred_x0_latents"] = pred_x0_latents
            output["pred_x0_images"] = pred_x0_images
        return output
