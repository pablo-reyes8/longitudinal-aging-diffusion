"""Structured numerical report; strict independent oracles live in the test tier."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .diffusion_utils import (
    compute_diffusion_loss,
    extract_scheduler_coefficients,
    get_diffusion_target,
    get_prediction_type,
    predict_x0_from_model_output,
)


def _error_stats(observed: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict[str, Any]:
    difference = (observed - expected).detach().double()
    absolute = float(difference.abs().max())
    denominator = float(expected.detach().double().norm())
    relative = float(difference.norm()) / max(denominator, torch.finfo(torch.float64).eps)
    return {
        "max_absolute_error": absolute,
        "relative_l2_error": relative,
        "tolerance": tolerance,
        "passed": absolute <= tolerance,
    }


def run_face_aging_loss_validation(
    loss_fn,
    *,
    forward_kwargs: Mapping[str, Any] | None = None,
    random_seed: int = 913,
    tier_c_executed: bool = False,
) -> dict[str, Any]:
    """Run reproducible algebraic checks and optionally a full supplied batch."""
    errors: list[str] = []
    warnings: list[str] = []
    scheduler = loss_fn.scheduler
    device = torch.device("cpu")
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    batch, channels, height, width = 4, 3, 5, 7
    x0 = torch.randn(batch, channels, height, width, generator=generator, dtype=dtype, device=device)
    noise = torch.randn(batch, channels, height, width, generator=generator, dtype=dtype, device=device)
    total_steps = len(scheduler.alphas_cumprod)
    timesteps = torch.tensor([0, 1, total_steps // 2, total_steps - 1], device=device)
    coefficients = extract_scheduler_coefficients(scheduler, timesteps, x0)
    noisy = coefficients["sqrt_alpha_bar"] * x0 + coefficients["sqrt_one_minus_alpha_bar"] * noise
    prediction_type = get_prediction_type(scheduler)
    if prediction_type == "epsilon":
        exact_target = noise
    else:
        exact_target = coefficients["sqrt_alpha_bar"] * noise - coefficients["sqrt_one_minus_alpha_bar"] * x0
    recovered = predict_x0_from_model_output(exact_target, noisy, timesteps, scheduler)
    reconstruction = _error_stats(recovered, x0, tolerance=2e-10)
    if not reconstruction["passed"]:
        errors.append(f"Analytical x0 reconstruction failed: {reconstruction}")
    target = get_diffusion_target(scheduler, x0, noise, timesteps)
    perfect_loss, per_sample, _ = compute_diffusion_loss(
        exact_target, target, scheduler=scheduler, timesteps=timesteps,
        min_snr_gamma=loss_fn.min_snr_gamma,
    )
    perfect = {
        "observed": float(perfect_loss),
        "expected": 0.0,
        "absolute_error": abs(float(perfect_loss)),
        "relative_error": 0.0 if float(perfect_loss) == 0.0 else float("inf"),
        "max_per_sample": float(per_sample.max()),
        "tolerance": 1e-12,
        "passed": abs(float(perfect_loss)) <= 1e-12,
    }
    if not perfect["passed"]:
        errors.append(f"Perfect diffusion prediction is non-zero: {perfect}")
    wrong_reconstruction = noisy - exact_target
    injected_error = float((wrong_reconstruction - x0).abs().max())
    failure_injection = {
        "naive_x0_formula": {
            "expected_detection": True,
            "observed_max_absolute_error": injected_error,
            "detection_threshold": 1e-4,
            "passed": injected_error > 1e-4,
        }
    }
    if not failure_injection["naive_x0_formula"]["passed"]:
        errors.append("Failure injection was not detected: naive x0 formula")
    coefficient_audit = {
        "shape": list(coefficients["alpha_bar"].shape),
        "dtype": str(coefficients["alpha_bar"].dtype),
        "device": str(coefficients["alpha_bar"].device),
        "finite": bool(torch.isfinite(coefficients["alpha_bar"]).all()),
        "range_valid": bool(((coefficients["alpha_bar"] >= 0) & (coefficients["alpha_bar"] <= 1)).all()),
    }
    tier_a_tests = {
        "perfect_prediction": perfect,
        "x0_reconstruction": reconstruction,
        "coefficient_extraction": coefficient_audit,
    }
    tier_a_passed = perfect["passed"] and reconstruction["passed"] and coefficient_audit["finite"] and coefficient_audit["range_valid"]

    integration: dict[str, Any] = {"status": "NOT RUN"}
    gradient_flow: dict[str, Any] = {}
    gradient_scales: dict[str, Any] = {}
    finite_difference: dict[str, Any] = {}
    loss_scales: dict[str, Any] = {}
    identity_report: dict[str, Any] = {"enabled": loss_fn.identity_weight > 0, "status": "NOT RUN"}
    age_report: dict[str, Any] = {"enabled": loss_fn.age_weight > 0, "status": "NOT RUN"}
    tier_b_passed: bool | None = None
    if forward_kwargs is not None:
        first = loss_fn(**dict(forward_kwargs), return_per_sample=True)
        second = loss_fn(**dict(forward_kwargs), return_per_sample=True)
        reproducible = all(torch.equal(first[key], second[key]) for key in ("loss", "loss_diff", "loss_id", "loss_age"))
        prediction = forward_kwargs["model_pred"]
        component_gradients = {}
        for key in ("loss_diff", "loss_id", "loss_age"):
            if first[key].requires_grad:
                component_gradients[key] = torch.autograd.grad(
                    first[key], prediction, retain_graph=True, allow_unused=True
                )[0]
            else:
                component_gradients[key] = None
        gradients = torch.autograd.grad(first["loss"], prediction, retain_graph=True, allow_unused=True)[0]
        gradient_flow = {
            "model_pred_connected": gradients is not None,
            "finite": gradients is not None and bool(torch.isfinite(gradients).all()),
            "norm": float(gradients.detach().float().norm()) if gradients is not None else None,
        }
        def norm(value):
            return float(value.detach().float().norm()) if value is not None else 0.0
        def cosine(left, right):
            if left is None or right is None or norm(left) == 0 or norm(right) == 0:
                return None
            return float(torch.nn.functional.cosine_similarity(left.flatten().float(), right.flatten().float(), dim=0))
        gradient_scales = {
            "model_pred_norms": {key: norm(value) for key, value in component_gradients.items()},
            "cosine_diff_id": cosine(component_gradients["loss_diff"], component_gradients["loss_id"]),
            "cosine_diff_age": cosine(component_gradients["loss_diff"], component_gradients["loss_age"]),
            "cosine_id_age": cosine(component_gradients["loss_id"], component_gradients["loss_age"]),
        }
        identity_values = first["loss_id_per_sample"]
        age_values = first["loss_age_per_sample"]
        identity_report = {
            "enabled": loss_fn.identity_weight > 0,
            "status": "PASSED" if not loss_fn.identity_weight or bool(torch.isfinite(identity_values).all()) else "FAILED",
            "observed_mean": float(first["loss_id"].detach()),
            "per_sample_finite": bool(torch.isfinite(identity_values).all()),
            "gradient_norm": norm(component_gradients["loss_id"]),
            "reference": loss_fn.identity_reference,
        }
        age_report = {
            "enabled": loss_fn.age_weight > 0,
            "status": "PASSED" if not loss_fn.age_weight or bool(torch.isfinite(age_values).all()) else "FAILED",
            "observed_mean": float(first["loss_age"].detach()),
            "per_sample_finite": bool(torch.isfinite(age_values).all()),
            "gradient_norm": norm(component_gradients["loss_age"]),
            "loss_type": loss_fn.age_loss_type,
        }
        if gradients is not None:
            expected_gradient = (
                loss_fn.diffusion_weight * component_gradients["loss_diff"]
                + (loss_fn.identity_weight * component_gradients["loss_id"] if component_gradients["loss_id"] is not None else 0)
                + (loss_fn.age_weight * component_gradients["loss_age"] if component_gradients["loss_age"] is not None else 0)
            )
            decomposition_error = float((gradients - expected_gradient).detach().float().abs().max())
            gradient_scales["weighted_decomposition_max_error"] = decomposition_error
            if decomposition_error > 2e-5:
                errors.append(f"Weighted gradient decomposition failed: max_error={decomposition_error}")

            direction = torch.randn(
                prediction.shape,
                generator=torch.Generator(device="cpu").manual_seed(random_seed + 1),
                dtype=prediction.dtype,
                device="cpu",
            ).to(prediction.device)
            direction = direction / direction.float().norm().to(direction.dtype)
            step = 1e-3 if prediction.dtype != torch.float64 else 1e-6
            plus_kwargs = {**dict(forward_kwargs), "model_pred": prediction.detach() + step * direction}
            minus_kwargs = {**dict(forward_kwargs), "model_pred": prediction.detach() - step * direction}
            plus = loss_fn(**plus_kwargs)["loss"]
            minus = loss_fn(**minus_kwargs)["loss"]
            numerical = float(((plus - minus) / (2 * step)).detach())
            analytical = float((gradients * direction).sum().detach())
            absolute_error = abs(numerical - analytical)
            tolerance = 2e-3 if prediction.dtype != torch.float64 else 3e-6
            finite_difference = {
                "analytical_directional_derivative": analytical,
                "numerical_directional_derivative": numerical,
                "absolute_error": absolute_error,
                "tolerance": tolerance,
                "passed": absolute_error <= tolerance,
            }
            if not finite_difference["passed"]:
                errors.append(f"Composite directional derivative failed: {finite_difference}")
        loss_scales = dict(first["metrics"])
        integration_passed = (
            reproducible
            and gradient_flow["model_pred_connected"]
            and gradient_flow["finite"]
            and finite_difference.get("passed", False)
            and identity_report["status"] == "PASSED"
            and age_report["status"] == "PASSED"
        )
        integration = {
            "status": "PASSED" if integration_passed else "FAILED",
            "reproducible": reproducible,
            "auxiliary_applied": first["auxiliary_applied"],
            "per_sample_shapes": {
                key: list(first[key].shape)
                for key in ("loss_diff_per_sample", "loss_id_per_sample", "loss_age_per_sample")
            },
        }
        tier_b_passed = integration["status"] == "PASSED"
        if not tier_b_passed:
            errors.append(f"Tier B integration failed: {integration}, gradients={gradient_flow}")
    else:
        warnings.append("Tier B composite forward was NOT RUN because forward_kwargs was not supplied")

    tier_c = {
        "passed": True if tier_c_executed and not errors else None,
        "status": "PASSED" if tier_c_executed and not errors else "NOT RUN",
        "tests": {"real_sd15_and_real_auxiliary_encoders": "NOT RUN" if not tier_c_executed else "caller-confirmed"},
    }
    if not tier_c_executed:
        warnings.append("Tier C real SD1.5 + real identity/age encoders was NOT RUN in this environment")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": ["Tier C is never counted as passed unless explicitly executed"],
        "tier_a": {"passed": tier_a_passed, "tests": tier_a_tests},
        "tier_b": {"passed": tier_b_passed, "status": integration["status"], "tests": integration},
        "tier_c": tier_c,
        "diffusion": perfect,
        "x0_reconstruction": reconstruction,
        "identity": identity_report,
        "age": age_report,
        "gradient_flow": gradient_flow,
        "finite_difference": finite_difference,
        "loss_scales": loss_scales,
        "gradient_scales": gradient_scales,
        "numerical_stability": coefficient_audit,
        "failure_injection": failure_injection,
        "integration": integration,
    }
