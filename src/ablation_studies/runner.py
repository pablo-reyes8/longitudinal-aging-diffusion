"""High-level orchestration for controlled training ablations."""

from __future__ import annotations

import gc
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data import build_face_aging_dataloaders
from src.loss import FaceAgingDiffusionLoss
from src.model import build_face_aging_diffusion_bundle
from src.quantitative_metrics import load_quantitative_metrics
from src.training import TRAIN_AGGING_MODEL, set_seed

from .config import resolve_training_ablation_config
from .evaluation import AblationEpochEvaluator
from .panel import build_fixed_evaluation_panel, persist_fixed_evaluation_panel


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _prepare_resolved_config(
    path: Path, config: Mapping[str, Any], *, resume: bool, overwrite: bool
) -> None:
    normalized = json.loads(json.dumps(config, default=str))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if resume and existing != normalized:
            raise ValueError(
                "Resume configuration differs from the saved resolved configuration; "
                "start a new output directory for a changed experiment"
            )
        if not resume and not overwrite:
            raise FileExistsError(
                f"Ablation output already exists at {path.parent}; "
                "use resume=True or overwrite=True"
            )
    elif resume:
        raise FileNotFoundError(f"Cannot resume: {path} does not exist")
    _write_json(path, normalized)


def _archive_existing_case(case_dir: Path) -> Path | None:
    """Move an old run aside so overwrite never mixes scientific artifacts."""
    if not case_dir.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = case_dir.with_name(f"{case_dir.name}.previous_{stamp}")
    case_dir.rename(archived)
    return archived


def _resolve_device(device: str | torch.device) -> str | torch.device:
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str | torch.dtype | None, device: str | torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is not None:
        normalized = str(dtype).lower().replace("torch.", "")
        choices = {
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        if normalized not in choices:
            raise ValueError("dtype must be float32, float16, bfloat16, or a torch.dtype")
        return choices[normalized]
    if torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _cases(value: int | Sequence[int]) -> list[int]:
    cases = [int(value)] if isinstance(value, int) else [int(case) for case in value]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("case must contain one or more unique case IDs")
    return cases


def _explicit_panel(
    evaluation_pairs,
    *,
    source_image,
    target_image,
    source_age,
    target_age,
    identity_id,
):
    shorthand = [source_image, target_image, source_age, target_age]
    if evaluation_pairs is not None and any(value is not None for value in shorthand):
        raise ValueError("Use evaluation_pairs or the two-image shorthand, not both")
    if evaluation_pairs is not None:
        return evaluation_pairs
    if any(value is not None for value in shorthand):
        if not all(value is not None for value in shorthand):
            raise ValueError(
                "Two-image evaluation requires source/target images and source/target ages"
            )
        return [{
            "sample_id": f"{identity_id}_{int(source_age)}_{int(target_age)}",
            "identity_id": str(identity_id),
            "source_path": str(source_image),
            "target_path": str(target_image),
            "source_age": int(source_age),
            "target_age": int(target_age),
        }]
    return None


def _load_metrics(metrics_config: Mapping[str, Any] | None):
    if metrics_config is None:
        raise ValueError(
            "metrics_config is required for epoch/final ablation evaluation; "
            "provide local AdaFace, DEX, and KID checkpoint paths"
        )
    options = dict(metrics_config)
    evaluation = dict(options.pop("evaluation", {}))
    bundle = options.pop("metrics_bundle", None)
    if bundle is None:
        options.setdefault("device", "cpu")
        options.setdefault("local_files_only", True)
        bundle = load_quantitative_metrics(**options)
    return bundle, evaluation


def ablation_studies(
    case: int | Sequence[int],
    *,
    dataset_root: str | Path,
    kaggle_path: str | Path | None = None,
    monitoring_image=None,
    monitoring_source_age: int | None = None,
    evaluation_pairs: Sequence[Mapping[str, Any]] | None = None,
    evaluation_source_image=None,
    evaluation_target_image=None,
    evaluation_source_age: int | None = None,
    evaluation_target_age: int | None = None,
    evaluation_identity_id: str = "evaluation_identity",
    num_epochs: int | None = None,
    max_train_steps: int | None = None,
    lr_lora: float | None = None,
    lr_conv_in: float | None = None,
    lr_age_conditioner: float | None = None,
    weight_decay: float | None = None,
    warmup_ratio: float | None = None,
    min_lr_ratio: float | None = None,
    grad_accum_steps: int | None = None,
    max_grad_norm: float | None = None,
    seed: int | None = None,
    validation_seed: int | None = None,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype | None = None,
    amp_enabled: bool | None = None,
    gradient_checkpointing: bool | None = None,
    num_inference_steps: int | None = None,
    strength: float | None = None,
    age_guidance_scale: float | None = None,
    text_guidance_scale: float | None = None,
    image_guidance_scale: float | None = None,
    inference_seed: int | None = None,
    monitoring_target_ages: Sequence[int] | None = None,
    monitoring_target_age_strength_map: Mapping[float, float] | None = None,
    evaluate_each_epoch: bool = True,
    evaluate_final: bool = True,
    epoch_eval_size: int = 64,
    final_eval_size: int | None = None,
    metrics_config: Mapping[str, Any] | None = None,
    data_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    loss_overrides: Mapping[str, Any] | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    monitoring_overrides: Mapping[str, Any] | None = None,
    checkpoint_root: str | Path | None = None,
    output_root: str | Path = "output/ablations",
    resume: bool | str | Path = False,
    overwrite: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Train cases 0--5 sequentially under one frozen evaluation protocol."""
    case_ids = _cases(case)
    # Validate the scientific case before loading any large model or metric.
    for case_id in case_ids:
        resolve_training_ablation_config(case_id)
    explicit_pairs = _explicit_panel(
        evaluation_pairs,
        source_image=evaluation_source_image,
        target_image=evaluation_target_image,
        source_age=evaluation_source_age,
        target_age=evaluation_target_age,
        identity_id=evaluation_identity_id,
    )
    if evaluate_each_epoch and int(epoch_eval_size) < 1:
        raise ValueError("epoch_eval_size must be at least 1")
    if final_eval_size is not None and int(final_eval_size) < 1:
        raise ValueError("final_eval_size must be at least 1 or None")

    metrics_bundle = evaluation_options = None
    if evaluate_each_epoch or evaluate_final:
        metrics_bundle, evaluation_options = _load_metrics(metrics_config)

    high_level = {
        "training": {
            "num_epochs": num_epochs,
            "max_train_steps": max_train_steps,
            "lr_lora": lr_lora,
            "lr_conv_in": lr_conv_in,
            "lr_age_conditioner": lr_age_conditioner,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "min_lr_ratio": min_lr_ratio,
            "grad_accum_steps": grad_accum_steps,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "validation_seed": validation_seed,
            "amp_enabled": amp_enabled,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "monitoring": {
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "age_guidance_scale": age_guidance_scale,
            "text_guidance_scale": text_guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "seed": inference_seed,
            "target_ages": list(monitoring_target_ages) if monitoring_target_ages is not None else None,
            "target_age_strength_map": (
                dict(monitoring_target_age_strength_map)
                if monitoring_target_age_strength_map is not None else None
            ),
        },
        "evaluation": {
            "evaluate_each_epoch": evaluate_each_epoch,
            "epoch_eval_size": epoch_eval_size,
            "final_eval_size": final_eval_size,
        },
    }
    section_overrides = {
        "data": dict(data_overrides or {}),
        "model": dict(model_overrides or {}),
        "loss": dict(loss_overrides or {}),
        "training": dict(training_overrides or {}),
        "monitoring": dict(monitoring_overrides or {}),
    }
    for section, values in high_level.items():
        section_overrides.setdefault(section, {})
        section_overrides[section].update({key: value for key, value in values.items() if value is not None})
    section_overrides = {key: value for key, value in section_overrides.items() if value}

    resolved_device = _resolve_device(device)
    model_dtype = _resolve_dtype(dtype, resolved_device)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    reports: dict[int, dict[str, Any]] = {}
    shared_panel_path = root / "fixed_validation_panel.json"

    for case_id in case_ids:
        config = resolve_training_ablation_config(case_id, section_overrides)
        slug = config["ablation"]["slug"]
        case_name = f"case_{case_id:02d}_{slug}"
        case_dir = root / case_name
        if overwrite and not resume:
            _archive_existing_case(case_dir)
        resolved_path = case_dir / "resolved_config.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        _prepare_resolved_config(
            resolved_path, config, resume=bool(resume), overwrite=bool(overwrite)
        )

        # Adapter and age-conditioner weights are random; reset before model
        # construction so every case starts from the same scientific seed.
        set_seed(config["training"]["seed"], deterministic=config["training"]["deterministic"])

        data_config = dict(config["data"])
        data_config["seed"] = config["training"]["seed"]
        if data_config["include_kaggle"]:
            if kaggle_path is None:
                raise ValueError("Canonical ablation data uses FG-NET; provide kaggle_path")
            data_config["kaggle_path"] = kaggle_path
        loaders, data_metadata = build_face_aging_dataloaders(dataset_root, **data_config)

        full_panel = None
        epoch_panel = None
        if evaluate_each_epoch or evaluate_final:
            full_panel = build_fixed_evaluation_panel(
                loaders["val"], size=final_eval_size, explicit_pairs=explicit_pairs
            )
            persist_fixed_evaluation_panel(full_panel, shared_panel_path)
            epoch_panel = full_panel[: min(len(full_panel), int(epoch_eval_size))]

        model_config = dict(config["model"])
        auxiliary_name = model_config.pop("auxiliary_dtype")
        model_config["auxiliary_dtype"] = _resolve_dtype(auxiliary_name, resolved_device)
        bundle = build_face_aging_diffusion_bundle(
            **model_config,
            device=resolved_device,
            dtype=model_dtype,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        loss_fn = FaceAgingDiffusionLoss(
            scheduler=bundle["scheduler_train"],
            vae=bundle["vae"],
            identity_encoder=bundle.get("identity_encoder"),
            age_estimator=bundle.get("age_estimator"),
            **config["loss"],
        )

        inference_config = {
            "mode": "direct",
            "use_inverse_diffusion": config["monitoring"]["use_inverse_diffusion"],
            "num_inference_steps": config["monitoring"]["num_inference_steps"],
            "strength": config["monitoring"]["strength"],
            "text_reference_mode": config["monitoring"]["text_reference_mode"],
            "age_guidance_scale": config["monitoring"]["age_guidance_scale"],
            "text_guidance_scale": config["monitoring"]["text_guidance_scale"],
            "image_guidance_scale": config["monitoring"]["image_guidance_scale"],
            "seed": config["monitoring"]["seed"],
            "image_size": config["data"]["image_size"],
        }
        epoch_evaluator = None
        final_evaluator = None
        if evaluate_each_epoch:
            epoch_evaluator = AblationEpochEvaluator(
                panel=epoch_panel,
                metrics_bundle=metrics_bundle,
                output_dir=case_dir,
                model_name=case_name,
                ablation_case=case_id,
                inference_config=inference_config,
                evaluation_config=evaluation_options,
            )
        if evaluate_final:
            final_evaluator = AblationEpochEvaluator(
                panel=full_panel,
                metrics_bundle=metrics_bundle,
                output_dir=case_dir,
                model_name=case_name,
                ablation_case=case_id,
                inference_config=inference_config,
                evaluation_config=evaluation_options,
            )

        checkpoint_dir = (
            case_dir / "checkpoints"
            if checkpoint_root is None
            else Path(checkpoint_root) / case_name
        )
        resume_from = None
        if resume:
            resume_from = (
                checkpoint_dir / "latest" / "training_resume.pt"
                if resume is True else Path(resume)
            )
            if not Path(resume_from).is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")

        train_options = dict(config["training"])
        train_options.update({
            "use_bidirectional_training": config["data"]["include_bidirectional_pairs"],
            "reverse_pair_prob": config["data"]["reverse_pair_prob"],
            "use_age_delta_conditioning": config["model"]["use_age_delta_conditioning"],
            "age_conditioning_mode": config["model"]["age_conditioning_mode"],
            "use_age_conditioner_v2": config["model"]["use_age_conditioner_v2"],
            "age_conditioning_version": config["model"]["age_conditioning_version"],
            "age_delta_scale": config["model"]["age_delta_scale"],
            "min_snr_gamma": config["loss"]["min_snr_gamma"],
            "auxiliary_max_timestep": config["loss"]["auxiliary_max_timestep"],
            "image_size": config["data"]["image_size"],
            "device": device,
            "checkpoint_dir": checkpoint_dir,
            "resume_from": resume_from,
            "monitor": "val/loss_total",
            "monitor_mode": "min",
            "save_epoch_checkpoints": True,
            "max_epoch_checkpoints": 10,
            "sample_every_epochs": 1 if monitoring_image is not None else 0,
            "monitoring_dir": case_dir / "monitoring",
            "monitoring_image": monitoring_image,
            "monitoring_source_age": monitoring_source_age,
            "monitoring_target_age": config["monitoring"]["target_ages"],
            "monitoring_use_inverse_diffusion": config["monitoring"]["use_inverse_diffusion"],
            "monitoring_num_inference_steps": config["monitoring"]["num_inference_steps"],
            "monitoring_strength": config["monitoring"]["strength"],
            "monitoring_strength_multi": config["monitoring"]["strength_multi"],
            "monitoring_use_adaptive_strength": config["monitoring"]["use_adaptive_strength"],
            "monitoring_target_age_strength_map": config["monitoring"]["target_age_strength_map"],
            "monitoring_delta_bin_thresholds": config["monitoring"]["delta_bin_thresholds"],
            "monitoring_use_delta_dependent_strength": config["monitoring"]["use_delta_dependent_strength"],
            "monitoring_text_reference_mode": config["monitoring"]["text_reference_mode"],
            "monitoring_age_guidance_scale": config["monitoring"]["age_guidance_scale"],
            "monitoring_text_guidance_scale": config["monitoring"]["text_guidance_scale"],
            "monitoring_image_guidance_scale": config["monitoring"]["image_guidance_scale"],
            "monitoring_seed": config["monitoring"]["seed"],
            "monitoring_compute_diagnostics": config["monitoring"]["compute_diagnostics"],
            "epoch_end_callback": epoch_evaluator.on_epoch if epoch_evaluator else None,
        })
        training_state = TRAIN_AGGING_MODEL(
            bundle=bundle,
            loss_fn=loss_fn,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            **train_options,
        )
        final_report = final_evaluator.evaluate_final(bundle=bundle) if final_evaluator else None
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "case": config["ablation"],
            "checkpoint_dir": str(checkpoint_dir),
            "pytorch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "python": platform.python_version(),
            "evaluation_models": (metrics_bundle or {}).get("metadata", {}),
            "fixed_panel": str(shared_panel_path) if full_panel is not None else None,
            "data_summary": data_metadata,
        }
        _write_json(case_dir / "run_metadata.json", metadata)
        reports[case_id] = {
            "case": config["ablation"],
            "output_dir": str(case_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "resolved_config": str(resolved_path),
            "training": {
                "best_epoch": training_state.get("best_epoch"),
                "best_metric": training_state.get("best_metric"),
                "global_step": training_state.get("global_step"),
                "optimizer_step": training_state.get("optimizer_step"),
            },
            "final_evaluation": final_report,
        }
        del training_state, loss_fn, bundle, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports
