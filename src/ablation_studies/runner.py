"""High-level orchestration for controlled training ablations."""

from __future__ import annotations

import gc
import json
import math
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
from src.training import (
    TRAIN_AGGING_MODEL,
    run_training_step,
    set_seed,
    setup_device_and_precision,
    estimate_optimizer_steps,
)
from src.training.mixed_precision import move_batch_to_device
from src.training.train_face_aging import (
    _enable_memory_features,
    _move_training_objects,
)

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
    _validate_resolved_config(path, config, resume=resume, overwrite=overwrite)
    normalized = json.loads(json.dumps(config, default=str))
    _write_json(path, normalized)


def _validate_resolved_config(
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


def _case_data_loaders(config, dataset_root, kaggle_path):
    data_config = dict(config["data"])
    data_config["seed"] = config["training"]["seed"]
    if data_config["include_kaggle"]:
        if kaggle_path is None:
            raise ValueError("Canonical ablation data uses FG-NET; provide kaggle_path")
        data_config["kaggle_path"] = kaggle_path
    return build_face_aging_dataloaders(dataset_root, **data_config)


def _loader_batch_preserving_state(loader):
    generator = getattr(loader, "generator", None)
    generator_state = generator.get_state() if generator is not None else None
    persistent_workers = getattr(loader, "persistent_workers", False)
    iterator = None
    if persistent_workers:
        loader.persistent_workers = False
    try:
        iterator = iter(loader)
        batch = next(iterator, None)
        if batch is None:
            raise ValueError("DataLoader produced no batch")
        return batch
    finally:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            del iterator
        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)
        if persistent_workers:
            loader.persistent_workers = True


def _validate_batch_finite(batch, *, image_size: int) -> int:
    required = ("source_image", "target_image", "source_age", "target_age", "delta_age")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"Training batch is missing required tensors: {missing}")
    source = batch["source_image"]
    target = batch["target_image"]
    if source.ndim != 4 or target.shape != source.shape:
        raise ValueError(
            "source_image and target_image must have matching [B, C, H, W] shapes; "
            f"got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[1:] != (3, int(image_size), int(image_size)):
        raise ValueError(
            f"Images must have shape [B, 3, {image_size}, {image_size}], "
            f"got {tuple(source.shape)}"
        )
    batch_size = int(source.shape[0])
    for key in required[2:]:
        value = batch[key]
        if not torch.is_tensor(value) or value.shape != (batch_size,):
            raise ValueError(f"{key} must be a tensor with shape [{batch_size}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains non-finite values")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("Training images contain non-finite values")
    prompts = batch.get("target_prompt")
    if prompts is None or len(prompts) != batch_size:
        raise ValueError(f"target_prompt must contain {batch_size} prompts")
    return batch_size


def _smoke_test_case(
    *,
    config: Mapping[str, Any],
    loaders,
    device,
    model_dtype,
    local_files_only,
    token,
    revision,
) -> None:
    training = config["training"]
    data = config["data"]
    set_seed(training["seed"], deterministic=training["deterministic"])
    raw_batch = _loader_batch_preserving_state(loaders["train"])
    batch_size = _validate_batch_finite(raw_batch, image_size=data["image_size"])

    model_config = dict(config["model"])
    auxiliary_name = model_config.pop("auxiliary_dtype")
    model_config["auxiliary_dtype"] = _resolve_dtype(auxiliary_name, device)
    bundle = loss_fn = None
    try:
        bundle = build_face_aging_diffusion_bundle(
            **model_config,
            device=device,
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

        precision = setup_device_and_precision(
            device,
            amp_enabled=training["amp_enabled"],
            amp_dtype=training["amp_dtype"],
        )
        _move_training_objects(bundle, loss_fn, precision["device"])
        _enable_memory_features(
            bundle,
            gradient_checkpointing=training["gradient_checkpointing"],
            enable_xformers=training["enable_xformers"],
        )
        loss_fn.min_snr_gamma = config["loss"]["min_snr_gamma"]
        loss_fn.auxiliary_max_timestep = config["loss"]["auxiliary_max_timestep"]
        if len(loaders["val"]) < 1:
            raise ValueError("Validation DataLoader has no batches")
        scheduler = bundle["scheduler_train"]
        total_timesteps = len(scheduler.alphas_cumprod)
        min_timestep = int(training["min_train_timestep"])
        max_timestep = (
            total_timesteps - 1
            if training["max_train_timestep"] is None
            else int(training["max_train_timestep"])
        )
        if min_timestep < 0 or max_timestep < min_timestep or max_timestep >= total_timesteps:
            raise ValueError(
                f"Invalid training timestep range [{min_timestep}, {max_timestep}] "
                f"for scheduler with {total_timesteps} steps"
            )
        smoke_timestep = min_timestep
        moved_batch = move_batch_to_device(raw_batch, precision["device"])
        bundle["unet"].train()
        if bundle.get("age_delta_conditioner") is not None:
            bundle["age_delta_conditioner"].train()
        with torch.no_grad():
            result = run_training_step(
                bundle=bundle,
                loss_fn=loss_fn,
                batch=moved_batch,
                device=precision["device"],
                amp_enabled=precision["amp_enabled"],
                amp_dtype=training["amp_dtype"],
                conditioning_dropout_prob=training["conditioning_dropout_prob"],
                target_prompt_policy=training["target_prompt_policy"],
                generic_prompt_prob=training["generic_prompt_prob"],
                numeric_prompt_prob=training["numeric_prompt_prob"],
                timestep_sampling=training["timestep_sampling"],
                min_train_timestep=min_timestep,
                max_train_timestep=max_timestep,
                sample_source_posterior=training["sample_source_posterior"],
                sample_target_posterior=training["sample_target_posterior"],
                noise_offset=training["noise_offset"],
                identity_loss_on_image_dropped_samples=training[
                    "identity_loss_on_image_dropped_samples"
                ],
                timesteps=torch.full(
                    (batch_size,), smoke_timestep, dtype=torch.long, device=precision["device"]
                ),
                dropout_random_values=torch.ones(batch_size, device=precision["device"]),
                global_step=0,
            )
        loss = result["loss_out"]["loss"]
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Smoke forward produced a non-finite scalar loss: {loss}")
        del result, loss, moved_batch, raw_batch
    finally:
        del loss_fn, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _preflight_cases(
    *,
    case_ids,
    configs,
    dataset_root,
    kaggle_path,
    evaluate_each_epoch,
    evaluate_final,
    epoch_eval_size,
    final_eval_size,
    explicit_pairs,
    device,
    model_dtype,
    local_files_only,
    token,
    revision,
):
    prepared = {}
    shared_panel = None
    for case_id in case_ids:
        print(f"\n[preflight] Validating ablation case {case_id}...")
        config = configs[case_id]
        try:
            set_seed(config["training"]["seed"], deterministic=config["training"]["deterministic"])
            loaders, data_metadata = _case_data_loaders(config, dataset_root, kaggle_path)
            full_panel = epoch_panel = None
            if evaluate_each_epoch or evaluate_final:
                full_panel = build_fixed_evaluation_panel(
                    loaders["val"], size=final_eval_size, explicit_pairs=explicit_pairs
                )
                if shared_panel is None:
                    shared_panel = full_panel
                elif full_panel != shared_panel:
                    raise ValueError("This case produced a different fixed validation panel")
                epoch_panel = full_panel[: min(len(full_panel), int(epoch_eval_size))]
            _smoke_test_case(
                config=config,
                loaders=loaders,
                device=device,
                model_dtype=model_dtype,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            prepared[case_id] = {
                "loaders": loaders,
                "data_metadata": data_metadata,
                "full_panel": full_panel,
                "epoch_panel": epoch_panel,
            }
            print(f"[preflight {case_id}] OK: setup and one-microbatch forward/loss ready")
        except Exception as exc:
            prepared.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"Preflight failed for ablation case {case_id}; no ablation training has started: {exc}"
            ) from exc
    return prepared


def _top_validation_epochs(training_state, checkpoint_dir: Path, *, limit: int = 3):
    """Rank saved epoch snapshots by the checkpoint-selection validation loss."""
    ranked = []
    for record in training_state.get("history", {}).get("epochs", []):
        validation = record.get("val")
        value = validation.get("val/loss_total") if validation else None
        if value is None or not math.isfinite(float(value)):
            continue
        epoch = int(record["epoch"]) + 1
        ranked.append({
            "epoch": epoch,
            "validation_loss": float(value),
            "checkpoint_path": str(
                checkpoint_dir / f"epoch_{epoch:03d}" / "adapter_inference.pt"
            ),
        })
    ranked.sort(key=lambda entry: (entry["validation_loss"], entry["epoch"]))
    return [dict(rank=index + 1, **entry) for index, entry in enumerate(ranked[:limit])]


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
    configs = {
        case_id: resolve_training_ablation_config(case_id, section_overrides)
        for case_id in case_ids
    }

    resolved_device = _resolve_device(device)
    model_dtype = _resolve_dtype(dtype, resolved_device)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    reports: dict[int, dict[str, Any]] = {}
    shared_panel_path = root / "fixed_validation_panel.json"

    # Catch output/resume conflicts across every requested case before a prior
    # case can spend time training and leave later cases unable to start.
    for case_id in case_ids:
        config = configs[case_id]
        case_name = f"case_{case_id:02d}_{config['ablation']['slug']}"
        case_dir = root / case_name
        _validate_resolved_config(
            case_dir / "resolved_config.json",
            config,
            resume=bool(resume),
            overwrite=bool(overwrite),
        )
        if resume:
            resume_checkpoint_dir = (
                case_dir / "checkpoints"
                if checkpoint_root is None
                else Path(checkpoint_root) / case_name
            )
            resume_checkpoint = (
                resume_checkpoint_dir / "latest" / "training_resume.pt"
                if resume is True
                else Path(resume)
            )
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    prepared_cases = {}
    if len(case_ids) > 1:
        print(
            "\nMulti-case preflight is mandatory: each requested ablation gets "
            "one real microbatch forward/loss check before any training starts."
        )
        prepared_cases = _preflight_cases(
            case_ids=case_ids,
            configs=configs,
            dataset_root=dataset_root,
            kaggle_path=kaggle_path,
            evaluate_each_epoch=evaluate_each_epoch,
            evaluate_final=evaluate_final,
            epoch_eval_size=epoch_eval_size,
            final_eval_size=final_eval_size,
            explicit_pairs=explicit_pairs,
            device=resolved_device,
            model_dtype=model_dtype,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if evaluate_each_epoch or evaluate_final:
            persist_fixed_evaluation_panel(
                prepared_cases[case_ids[0]]["full_panel"], shared_panel_path
            )
        print("\nAll ablation preflights passed; starting sequential training.\n")

    for case_id in case_ids:
        config = configs[case_id]
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

        prepared = prepared_cases.pop(case_id, None)
        if prepared is not None:
            loaders = prepared["loaders"]
            data_metadata = prepared["data_metadata"]
            full_panel = prepared["full_panel"]
            epoch_panel = prepared["epoch_panel"]
        else:
            # Adapter and age-conditioner weights are random; reset before
            # model construction so each single-case run starts from its seed.
            set_seed(
                config["training"]["seed"],
                deterministic=config["training"]["deterministic"],
            )
            loaders, data_metadata = _case_data_loaders(config, dataset_root, kaggle_path)
            full_panel = epoch_panel = None
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

        steps_per_epoch = estimate_optimizer_steps(
            len(loaders["train"]), config["training"]["grad_accum_steps"]
        )
        total_epochs = (
            math.ceil(int(config["training"]["max_train_steps"]) / steps_per_epoch)
            if config["training"]["max_train_steps"] is not None
            else int(config["training"]["num_epochs"])
        )

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
            "max_epoch_checkpoints": max(1, total_epochs),
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
        top_epochs = _top_validation_epochs(training_state, Path(checkpoint_dir))
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
            "top_validation_epochs": top_epochs,
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
                "top_epochs": top_epochs,
            },
            "final_evaluation": final_report,
        }
        del training_state, loss_fn, bundle, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nTop 3 epochs per ablation by lowest val/loss_total:")
    for case_id, report in reports.items():
        print(f"  Case {case_id}: {report['case']['slug']}")
        if not report["training"]["top_epochs"]:
            print("    No validation-ranked epoch checkpoints were produced.")
            continue
        for entry in report["training"]["top_epochs"]:
            print(
                f"    #{entry['rank']} epoch {entry['epoch']:03d} | "
                f"val/loss_total={entry['validation_loss']:.6f} | "
                f"{entry['checkpoint_path']}"
            )
    return reports
