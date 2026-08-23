"""Atomic adapter/conv checkpoints with exact optimizer, scheduler, and RNG resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import torch

from src.model import get_bundle_trainable_named_parameters

from .seed import capture_rng_state, restore_rng_state


PROTECTED_BUNDLE_KEYS = (
    "model_id", "adapter_type", "rank", "target_modules",
    "source_conditioning", "unet_in_channels", "unet_cross_attention_dim",
    "identity_model_id", "age_model_id",
    "use_age_delta_conditioning", "age_conditioning_mode", "age_delta_scale",
    "age_condition_hidden_dim", "age_condition_output_dim",
)
PROTECTED_TRAINING_KEYS = (
    "max_train_steps", "total_planned_optimizer_steps", "grad_accum_steps",
    "timestep_sampling", "min_train_timestep", "max_train_timestep",
    "conditioning_dropout_prob", "sample_source_posterior",
    "sample_target_posterior", "noise_offset", "min_snr_gamma",
    "auxiliary_max_timestep", "image_size",
    "use_age_delta_conditioning", "age_conditioning_mode", "age_delta_scale",
    "use_relative_age_loss", "relative_age_weight", "relative_age_loss_type",
)


def get_trainable_state_dict(bundle: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in get_bundle_trainable_named_parameters(bundle)
    }


def load_trainable_state_dict(bundle: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(get_bundle_trainable_named_parameters(bundle))
    if set(parameters) != set(state):
        raise ValueError(f"Trainable checkpoint keys mismatch: missing={sorted(set(parameters)-set(state))}, unexpected={sorted(set(state)-set(parameters))}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
            parameter.copy_(value.to(parameter))


def atomic_torch_save(payload: Any, path: str | Path, *, save_fn: Callable = torch.save) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_fn(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_json_save(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination


def _bundle_config(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return dict(bundle.get("config", {}))


def validate_checkpoint_compatibility(
    bundle, loss_fn, payload, *, strict: bool = True,
    current_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    saved_bundle = payload.get("bundle_config", {})
    current_bundle = _bundle_config(bundle)
    mismatches = {
        key: {"checkpoint": saved_bundle.get(key), "current": current_bundle.get(key)}
        for key in PROTECTED_BUNDLE_KEYS
        if saved_bundle.get(key) != current_bundle.get(key)
    }
    saved_loss = payload.get("loss_config")
    current_loss = loss_fn.get_config() if hasattr(loss_fn, "get_config") else None
    if saved_loss is not None and current_loss is not None and saved_loss != current_loss:
        mismatches["loss_config"] = {"checkpoint": saved_loss, "current": current_loss}
    saved_training = payload.get("training_config", {})
    if current_training_config is not None:
        for key in PROTECTED_TRAINING_KEYS:
            if saved_training.get(key) != current_training_config.get(key):
                mismatches[f"training_config.{key}"] = {
                    "checkpoint": saved_training.get(key),
                    "current": current_training_config.get(key),
                }
    if strict and mismatches:
        raise ValueError(f"Training checkpoint is incompatible: {mismatches}")
    return mismatches


def build_training_payload(
    *, bundle, loss_fn, optimizer, lr_scheduler, scaler,
    epoch: int, batch_position: int, global_step: int, optimizer_step: int,
    best_metric: float | None, best_epoch: int | None,
    history: dict, training_config: dict,
    training_generator_state: torch.Tensor | None = None,
    dataloader_generator_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "face_aging_training_resume",
        "trainable_state_dict": get_trainable_state_dict(bundle),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "batch_position": int(batch_position),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "bundle_config": _bundle_config(bundle),
        "loss_config": loss_fn.get_config() if hasattr(loss_fn, "get_config") else None,
        "training_config": dict(training_config),
        "rng_state": capture_rng_state(),
        "training_generator_state": training_generator_state,
        "dataloader_generator_state": dataloader_generator_state,
    }


def build_inference_payload(bundle, training_config: dict) -> dict[str, Any]:
    adapter_state = get_trainable_state_dict(bundle)
    return {
        "format_version": 1,
        "kind": "face_aging_adapter_inference",
        # Match src.model.load_face_aging_adapter exactly.
        "adapter_state_dict": adapter_state,
        "config": _bundle_config(bundle),
        "bundle_config": _bundle_config(bundle),
        "adapter_config": dict(bundle.get("adapter_config", {})),
        "model_id": bundle.get("model_id"),
        "vae_id": bundle.get("vae_id"),
        "source_conditioning": bundle.get("source_conditioning", "concat"),
        "image_size": training_config.get("image_size"),
        "prompt_configuration": training_config.get("prompt_configuration"),
    }


def load_training_checkpoint(
    path: str | Path,
    *, bundle, loss_fn, optimizer, lr_scheduler=None, scaler=None,
    strict_config: bool = True, restore_rng: bool = True,
    current_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_compatibility(
        bundle, loss_fn, payload, strict=strict_config,
        current_training_config=current_training_config,
    )
    load_trainable_state_dict(bundle, payload["trainable_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if lr_scheduler is not None and payload.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))
    return payload


class TrainingCheckpointManager:
    def __init__(
        self,
        root_dir: str | Path,
        *, monitor: str = "val/loss_total", mode: str = "min",
        save_epoch_checkpoints: bool = True, max_epoch_checkpoints: int = 5,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be min or max")
        if max_epoch_checkpoints < 0:
            raise ValueError("max_epoch_checkpoints must be non-negative")
        self.root_dir = Path(root_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_epoch_checkpoints = save_epoch_checkpoints
        self.max_epoch_checkpoints = max_epoch_checkpoints
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

    def is_improved(self, metric: float) -> bool:
        return self.best_metric is None or (metric < self.best_metric if self.mode == "min" else metric > self.best_metric)

    def _write_pair(self, directory: Path, training_payload, inference_payload) -> dict[str, str]:
        training_path = atomic_torch_save(training_payload, directory / "training_resume.pt")
        inference_path = atomic_torch_save(inference_payload, directory / "adapter_inference.pt")
        return {"training": str(training_path), "inference": str(inference_path)}

    def _prune_epochs(self) -> None:
        if self.max_epoch_checkpoints <= 0:
            keep = []
        else:
            keep = sorted(self.root_dir.glob("epoch_[0-9][0-9][0-9]"))[-self.max_epoch_checkpoints:]
        for directory in sorted(self.root_dir.glob("epoch_[0-9][0-9][0-9]")):
            if directory not in keep:
                for file in directory.iterdir():
                    file.unlink()
                directory.rmdir()

    def save(self, *, training_payload, inference_payload, epoch: int, metric: float) -> dict[str, Any]:
        improved = self.is_improved(metric)
        latest = self._write_pair(self.root_dir / "latest", training_payload, inference_payload)
        best = None
        if improved:
            self.best_metric, self.best_epoch = float(metric), int(epoch)
            best = self._write_pair(self.root_dir / "best", training_payload, inference_payload)
        snapshot = None
        if self.save_epoch_checkpoints:
            snapshot = self._write_pair(self.root_dir / f"epoch_{epoch + 1:03d}", training_payload, inference_payload)
            self._prune_epochs()
        atomic_json_save({
            "monitor": self.monitor, "mode": self.mode,
            "best_metric": self.best_metric, "best_epoch": self.best_epoch,
            "latest_epoch": epoch,
        }, self.root_dir / "checkpoint_state.json")
        return {"latest": latest, "best": best, "snapshot": snapshot, "improved": improved}

    def load_manager_state(self) -> None:
        path = self.root_dir / "checkpoint_state.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            self.best_metric = state.get("best_metric")
            self.best_epoch = state.get("best_epoch")
