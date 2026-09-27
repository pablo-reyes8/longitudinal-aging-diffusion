"""Canonical, isolated configurations for training ablations 0--5."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


CASE_DEFINITIONS = {
    0: ("full", "Full canonical training system"),
    1: ("no_numeric_age", "No continuous numerical age conditioner"),
    2: ("no_dora", "No attention adapter"),
    3: ("no_relative_age", "No relative-age loss"),
    4: ("no_preservation", "No zero-delta pairs or explicit preservation loss"),
    5: ("no_identity_loss", "No ArcFace identity loss"),
}


_CANONICAL = {
    "data": {
        "image_size": 400,
        "batch_size": 4,
        "num_workers": 4,
        "train_pair_strategy": "random_target",
        "eval_pair_strategy": "all",
        "split_ratios": (0.90, 0.05, 0.05),
        "horizontal_flip_prob": 0.20,
        "include_zero_delta_pairs": True,
        "zero_delta_pair_prob": 0.20,
        "include_bidirectional_pairs": True,
        "reverse_pair_prob": 0.40,
        "include_kaggle": True,
        "kaggle_proportion": 0.40,
        "kaggle_reverse_pair_prob": 0.50,
        "train_drop_last": False,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "model": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "adapter_type": "dora",
        "rank": 16,
        "alpha": 16,
        "dropout": 0.0,
        "use_age_delta_conditioning": True,
        "age_conditioning_mode": "delta_mlp",
        "use_age_conditioner_v2": True,
        "age_conditioning_version": "v2_fourier",
        "age_delta_scale": 80.0,
        "age_condition_hidden_dim": 256,
        "age_condition_output_dim": None,
        "num_fourier_frequencies": 8,
        "age_condition_use_raw_scalars": True,
        "age_condition_use_gate": True,
        "load_auxiliary_models": True,
        "identity_model_id": "py-feat/arcface_r50",
        "age_model_id": "iitolstykh/mivolo_v2",
        "auxiliary_dtype": "float32",
        "auxiliary_activation_checkpointing": True,
        "auxiliary_trust_remote_code": True,
    },
    "loss": {
        "diffusion_weight": 1.0,
        "identity_weight": 0.25,
        "age_weight": 0.05,
        "use_relative_age_loss": True,
        "relative_age_weight": 0.08,
        "relative_age_loss_type": "l1",
        "use_directional_relative_weighting": False,
        "reverse_relative_weight": 1.25,
        "use_preservation_loss": True,
        "preservation_weight": 0.08,
        "preservation_loss_type": "l1",
        "preservation_max_delta": 1,
        "use_small_delta_weighting": True,
        "small_delta_threshold": 3,
        "small_delta_weight": 1.5,
        "source_age_prediction_mode": "age_estimator",
        "identity_reference": "target",
        "diffusion_loss_type": "mse",
        "age_loss_type": "l1",
        "min_snr_gamma": 5.0,
        "auxiliary_every_n_steps": 2,
        "auxiliary_sample_fraction": 0.25,
        "auxiliary_max_timestep": 300,
        "clamp_pred_x0": True,
        "check_finite": True,
        "vae_decode_checkpointing": True,
    },
    "training": {
        "num_epochs": 14,
        "max_train_steps": None,
        "lr_lora": 2e-5,
        "lr_conv_in": 3e-6,
        "lr_age_conditioner": 7.5e-5,
        "weight_decay": 1e-2,
        "conv_in_weight_decay": 1e-2,
        "age_conditioner_weight_decay": 1e-2,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.10,
        "grad_accum_steps": 4,
        "max_grad_norm": 1.0,
        "timestep_sampling": "uniform",
        "min_train_timestep": 0,
        "max_train_timestep": None,
        "conditioning_dropout_prob": 0.02,
        "target_prompt_policy": "mixed",
        "generic_prompt_prob": 0.10,
        "numeric_prompt_prob": 0.90,
        "identity_loss_on_image_dropped_samples": False,
        "sample_source_posterior": False,
        "sample_target_posterior": True,
        "noise_offset": 0.0,
        "double_prompt_prob": 0.0,
        "amp_enabled": True,
        "amp_dtype": "auto",
        "gradient_checkpointing": True,
        "enable_xformers": True,
        "validate_every_epochs": 1,
        "deterministic_validation": True,
        "validation_seed": 2026,
        "seed": 42,
        "deterministic": False,
    },
    "monitoring": {
        "target_ages": [8, 12, 26, 35, 45, 55, 65],
        "num_inference_steps": 39,
        "strength": 0.35,
        "use_inverse_diffusion": False,
        "use_delta_dependent_strength": False,
        "use_adaptive_strength": True,
        "target_age_strength_map": {
            8: 0.33, 12: 0.30, 26: 0.04, 35: 0.18,
            45: 0.27, 55: 0.30, 65: 0.32,
        },
        "strength_multi": None,
        "delta_bin_thresholds": [5, 15, 30],
        "text_reference_mode": "source_age",
        "age_guidance_scale": 2.0,
        "text_guidance_scale": 7.0,
        "image_guidance_scale": 1.5,
        "seed": 2026,
        "compute_diagnostics": True,
    },
    "evaluation": {
        "evaluate_each_epoch": True,
        "epoch_eval_size": 64,
        "final_eval_size": None,
    },
}


def canonical_training_ablation_config() -> dict[str, Any]:
    """Return a fresh copy of the frozen notebook baseline."""
    return deepcopy(_CANONICAL)


def _merge_sections(config: dict[str, Any], overrides: Mapping[str, Any] | None) -> None:
    for section, values in dict(overrides or {}).items():
        if section not in config:
            raise KeyError(f"Unknown ablation configuration section: {section}")
        if not isinstance(values, Mapping):
            raise TypeError(f"Overrides for {section!r} must be a mapping")
        unknown = set(values).difference(config[section])
        if unknown:
            raise KeyError(f"Unknown {section} override(s): {sorted(unknown)}")
        config[section].update(deepcopy(dict(values)))


def resolve_training_ablation_config(
    case: int,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one case without ever mutating the canonical baseline."""
    case = int(case)
    if case >= 100:
        raise NotImplementedError(
            "The inference ablations (101-104) are intentionally deferred; "
            "this runner implements training cases 0-5 only."
        )
    if case not in CASE_DEFINITIONS:
        raise ValueError(f"Unknown training ablation case {case}; choose 0, 1, 2, 3, 4, or 5")
    config = canonical_training_ablation_config()
    if case == 1:
        config["model"]["use_age_delta_conditioning"] = False
    elif case == 2:
        config["model"]["adapter_type"] = "none"
    elif case == 3:
        config["loss"].update(use_relative_age_loss=False, relative_age_weight=0.0)
    elif case == 4:
        config["data"].update(include_zero_delta_pairs=False, zero_delta_pair_prob=0.0)
        config["loss"].update(use_preservation_loss=False, preservation_weight=0.0)
    elif case == 5:
        config["loss"]["identity_weight"] = 0.0
    _merge_sections(config, overrides)
    slug, description = CASE_DEFINITIONS[case]
    config["ablation"] = {"case": case, "slug": slug, "description": description}
    _validate_case(config)
    return config


def _validate_case(config: Mapping[str, Any]) -> None:
    case = config["ablation"]["case"]
    if case == 1 and config["model"]["use_age_delta_conditioning"] is not False:
        raise ValueError("Case 1 must disable continuous age conditioning")
    if case == 2 and config["model"]["adapter_type"] != "none":
        raise ValueError("Case 2 must use adapter_type='none'")
    if case == 3 and (config["loss"]["use_relative_age_loss"] or config["loss"]["relative_age_weight"] != 0):
        raise ValueError("Case 3 must fully disable relative-age loss")
    if case == 4:
        if config["data"]["include_zero_delta_pairs"] or config["data"]["zero_delta_pair_prob"] != 0:
            raise ValueError("Case 4 must disable zero-delta pairs")
        if config["loss"]["use_preservation_loss"] or config["loss"]["preservation_weight"] != 0:
            raise ValueError("Case 4 must disable explicit preservation loss")
        if not config["loss"]["use_small_delta_weighting"]:
            raise ValueError("Case 4 must preserve small-delta diffusion weighting")
    if case == 5 and config["loss"]["identity_weight"] != 0:
        raise ValueError("Case 5 must set identity_weight=0")
