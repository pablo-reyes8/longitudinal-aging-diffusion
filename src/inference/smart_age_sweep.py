"""Deterministic bias-aware screening of a few inference candidates per age."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from .checkpoint_loading import load_face_aging_adapter_for_inference
from .infer_face_aging import infer_face_aging
from .inference_utils import prepare_inference_image, tensor_to_pil
from .prompt_assistance import (
    resolve_prompt_assistance,
    stack_sweep_variants,
    validate_prompt_assistance_scale,
)
from .prompt_building import build_inference_prompt_pack


DEFAULT_SMART_TARGET_STRENGTH_MAP = {
    8: 0.35,
    12: 0.34,
    26: 0.04,
    35: 0.18,
    45: 0.27,
    55: 0.27,
    65: 0.25,
}

DEFAULT_SOURCE_BIAS_CORRECTION_AGE_MAP = {
    0: 0.25,
    8: 0.30,
    18: 1.00,
    50: 1.00,
    55: 0.40,
    65: 0.00,
    120: 0.00,
}

SMART_TRIAL_COLUMNS = [
    "source_age", "target_age", "requested_delta_age", "trial_idx",
    "direction_policy", "selection_target_age", "acceptable_age_min",
    "acceptable_age_max",
    "base_strength", "strength", "age_guidance_scale", "text_guidance_scale",
    "image_guidance_scale", "pred_source_age", "expected_mivolo_delta",
    "expected_mivolo_target_age", "pred_age", "selection_pred_age",
    "source_mivolo_bias", "source_bias_correction_weight",
    "source_bias_correction_applied", "age_error", "abs_age_error",
    "confidence_margin_years", "outside_confidence_error", "within_confidence_band",
    "identity_cosine",
    "selected_best", "image_path",
]


def adaptive_mivolo_confidence_margin(
    absolute_delta: float,
    *,
    small_delta_margin_years: float = 4.0,
    large_delta_margin_years: float = 2.0,
    full_confidence_delta: float = 25.0,
) -> float:
    """Interpolate from a cautious small-delta band to a tighter large-delta band."""
    values = (
        absolute_delta,
        small_delta_margin_years,
        large_delta_margin_years,
        full_confidence_delta,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("MiVOLO confidence parameters must be finite")
    if absolute_delta < 0 or small_delta_margin_years < 0 or large_delta_margin_years < 0:
        raise ValueError("MiVOLO confidence deltas and margins must be non-negative")
    if large_delta_margin_years > small_delta_margin_years:
        raise ValueError("large-delta MiVOLO margin cannot exceed the small-delta margin")
    if full_confidence_delta <= 0:
        raise ValueError("full_confidence_delta must be positive")
    progress = min(float(absolute_delta) / float(full_confidence_delta), 1.0)
    return float(small_delta_margin_years) + progress * (
        float(large_delta_margin_years) - float(small_delta_margin_years)
    )


def source_specific_mivolo_bias_weight(
    target_age: float,
    *,
    requested_delta: float | None = None,
    correction_age_map: Mapping[float, float] | None = None,
) -> float:
    """Interpolate personalized correction by target age, preserving extremes."""
    if not math.isfinite(float(target_age)) or not 0 <= float(target_age) <= 120:
        raise ValueError("target_age must be finite and in [0, 120]")
    if requested_delta is not None:
        if not math.isfinite(float(requested_delta)):
            raise ValueError("requested_delta must be finite")
        if float(requested_delta) == 0.0:
            return 1.0
    policy = (
        DEFAULT_SOURCE_BIAS_CORRECTION_AGE_MAP
        if correction_age_map is None else correction_age_map
    )
    if not isinstance(policy, Mapping) or len(policy) < 2:
        raise ValueError("correction_age_map must contain at least two age:weight points")
    points = []
    for raw_age, raw_weight in policy.items():
        try:
            age, weight = float(raw_age), float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("correction_age_map ages and weights must be numeric") from exc
        if not math.isfinite(age) or not 0 <= age <= 120:
            raise ValueError("correction_age_map ages must be finite and in [0, 120]")
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("correction_age_map weights must be in [0, 1]")
        points.append((age, weight))
    points.sort()
    age_value = float(target_age)
    if age_value <= points[0][0]:
        return points[0][1]
    if age_value >= points[-1][0]:
        return points[-1][1]
    for (left_age, left_weight), (right_age, right_weight) in zip(points, points[1:]):
        if left_age <= age_value <= right_age:
            fraction = (age_value - left_age) / (right_age - left_age)
            return left_weight + fraction * (right_weight - left_weight)
    raise RuntimeError("Unable to interpolate source-bias correction map")


def _validate_strength_map(mapping: Mapping[float, float] | None) -> dict[float, float]:
    policy = DEFAULT_SMART_TARGET_STRENGTH_MAP if mapping is None else mapping
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("target_age_strength_map must be a non-empty mapping")
    normalized = {}
    for raw_age, raw_strength in policy.items():
        try:
            age, strength = float(raw_age), float(raw_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError("target ages and strengths must be numeric") from exc
        if not math.isfinite(age):
            raise ValueError("target ages must be finite")
        if not math.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError("target strengths must be finite and in (0, 1]")
        normalized[age] = strength
    return normalized


def _initial_strength(target_age: float, mapping: Mapping[float, float] | None) -> float:
    policy = _validate_strength_map(mapping)
    if target_age in policy:
        return policy[target_age]
    nearest_age = min(policy, key=lambda age: (abs(age - target_age), age))
    return policy[nearest_age]


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _candidate_grid(candidates_by_target: list[list[dict]]) -> Image.Image:
    cell_width, cell_height = candidates_by_target[0][0]["result"]["image"].size
    footer = 78
    label_width = 92
    columns = max(len(values) for values in candidates_by_target)
    grid = Image.new(
        "RGB",
        (label_width + columns * cell_width, len(candidates_by_target) * (cell_height + footer)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for row_index, candidates in enumerate(candidates_by_target):
        y = row_index * (cell_height + footer)
        draw.text((6, y + 8), f"Target\n{candidates[0]['row']['target_age']:.0f}", fill="black")
        for column, candidate in enumerate(candidates):
            row = candidate["row"]
            x = label_width + column * cell_width
            grid.paste(candidate["result"]["image"], (x, y))
            label = (
                f"Trial {row['trial_idx']} | s={row['strength']:.3f}\n"
                f"Raw={row['pred_age']:.1f} | corr={row['selection_pred_age']:.1f} "
                f"| err={row['age_error']:+.1f}\n"
                f"ID={row['identity_cosine']:.3f}"
                + (" | SELECTED" if row["selected_best"] else "")
            )
            draw.multiline_text((x + 4, y + cell_height + 4), label, fill="black", spacing=2)
    return grid


def _final_grid(source_image, source_age: int, selected: list[dict], image_size: int) -> Image.Image:
    source = tensor_to_pil(
        prepare_inference_image(source_image, image_size=image_size).div(2).add(0.5)
    )
    ordered = sorted(selected, key=lambda candidate: candidate["row"]["target_age"])
    source_position = sum(
        candidate["row"]["target_age"] < source_age for candidate in ordered
    )
    images = [candidate["result"]["image"] for candidate in ordered]
    labels = []
    for candidate in ordered:
        row = candidate["row"]
        labels.append(
            f"Target {row['target_age']:.0f} | MiVOLO {row['pred_age']:.1f} "
            f"| Corr {row['selection_pred_age']:.1f}\n"
            f"Err {row['age_error']:+.1f} | Strength {row['strength']:.3f}\n"
            f"ID {row['identity_cosine']:.3f} | "
            f"band [{row['acceptable_age_min']:.1f}, {row['acceptable_age_max']:.1f}] | "
            f"{'OK' if row['within_confidence_band'] else 'MISS'}"
        )
    images.insert(source_position, source)
    labels.insert(source_position, f"Original\nAge {source_age}")
    width, height = images[0].size
    footer = 58
    grid = Image.new("RGB", (len(images) * width, height + footer), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        grid.paste(image, (x, 0))
        draw.multiline_text((x + 4, height + 4), label, fill="black", spacing=2)
    return grid


def _select_best(candidates: list[dict], *, identity_tiebreak: bool, age_margin: float) -> dict:
    pool = candidates
    if candidates[0]["row"]["direction_policy"] == "aging_at_or_above_target":
        at_or_above = [
            candidate for candidate in candidates
            if candidate["row"]["selection_pred_age"] >= candidate["row"]["acceptable_age_min"]
        ]
        # Undershooting can win only if the model never reached the requested age.
        if at_or_above:
            pool = at_or_above
    minimum = min(candidate["row"]["outside_confidence_error"] for candidate in pool)
    in_band = [
        candidate for candidate in pool
        if math.isclose(
            candidate["row"]["outside_confidence_error"], 0.0, abs_tol=1e-12
        )
    ]
    eligible = in_band or [
        candidate for candidate in pool
        if candidate["row"]["outside_confidence_error"] <= minimum + age_margin
    ]
    if identity_tiebreak:
        best_identity = max(candidate["row"]["identity_cosine"] for candidate in eligible)
        eligible = [
            candidate for candidate in eligible
            if math.isclose(candidate["row"]["identity_cosine"], best_identity, abs_tol=1e-12)
        ]
    return min(
        eligible,
        key=lambda candidate: (
            abs(candidate["row"]["strength"] - candidate["row"]["base_strength"]),
            abs(candidate["row"]["age_guidance_scale"] - candidate["base_guidance"][0])
            + abs(candidate["row"]["text_guidance_scale"] - candidate["base_guidance"][1])
            + abs(candidate["row"]["image_guidance_scale"] - candidate["base_guidance"][2]),
            candidate["row"]["trial_idx"],
        ),
    )


def _directional_selection_policy(
    *,
    requested_delta: float,
    target_age: float,
    use_personalized_mivolo_bias: bool,
    confidence_aging: float,
    confidence_rejuvenecer: float,
) -> dict[str, float | str]:
    if requested_delta > 0:
        return {
            "direction_policy": "aging_at_or_above_target",
            "selection_target_age": float(target_age),
            "acceptable_age_min": float(target_age),
            "acceptable_age_max": float(target_age) + float(confidence_aging),
            "confidence_margin_years": float(confidence_aging),
        }
    if requested_delta < 0:
        center = float(target_age)
        return {
            "direction_policy": (
                "rejuvenation_source_bias_corrected"
                if use_personalized_mivolo_bias
                else "rejuvenation_raw_mivolo"
            ),
            "selection_target_age": center,
            "acceptable_age_min": center - float(confidence_rejuvenecer),
            "acceptable_age_max": center + float(confidence_rejuvenecer),
            "confidence_margin_years": float(confidence_rejuvenecer),
        }
    return {
        "direction_policy": "zero_delta_source_bias_corrected",
        "selection_target_age": float(target_age),
        "acceptable_age_min": float(target_age),
        "acceptable_age_max": float(target_age),
        "confidence_margin_years": 0.0,
    }


def _distance_to_acceptable_band(
    predicted_age: float, minimum_age: float, maximum_age: float
) -> float:
    if predicted_age < minimum_age:
        return float(minimum_age - predicted_age)
    if predicted_age > maximum_age:
        return float(predicted_age - maximum_age)
    return 0.0


def _strength_search_direction(
    *, requested_delta: float, predicted_age: float,
    minimum_age: float, maximum_age: float,
) -> float:
    if requested_delta > 0:
        return 1.0 if predicted_age < minimum_age else -1.0
    if requested_delta < 0:
        return 1.0 if predicted_age > maximum_age else -1.0
    return -1.0


def diagnose_checkpoint_smart_age_sweep(
    checkpoint_path: str | Path,
    bundle,
    source_image,
    source_age: int,
    target_ages: Iterable[int],
    *,
    output_dir: str | Path,
    target_age_strength_map: Mapping[float, float] | None = None,
    max_trials_per_target: int = 5,
    use_bias_corrected_mivolo_target: bool = True,
    source_bias_correction_age_map: Mapping[float, float] | None = None,
    confidence_aging: float = 3.0,
    confidence_rejuvenecer: float = 2.0,
    enable_guidance_micro_search: bool = False,
    age_guidance_scale: float = 3.0,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    min_strength: float = 0.04,
    max_strength: float = 0.45,
    strength_step_coarse: float = 0.05,
    strength_step_medium: float = 0.03,
    strength_step_fine: float = 0.02,
    identity_tiebreak: bool = True,
    identity_margin_for_tiebreak: float = 1.0,
    save_all_trials: bool = False,
    source_prompt: str | None = None,
    num_inference_steps: int = 50,
    text_reference_mode: str = "source_age",
    negative_prompt: str = "",
    prompt_style: str = "selfage",
    use_cfg: bool = True,
    seed: int = 2026,
    image_size: int = 256,
    strict_config: bool = True,
    generate_assisted_prompt_variant: bool = False,
    prompt_assistance_config=None,
    prompt_assistance_scale: float = 1.0,
    negative_prompt_assistance_scale: float = 1.0,
    source_mouth_state: str = "auto",
    source_expression_state: str = "auto",
    save_prompt_comparison: bool = True,
) -> pd.DataFrame:
    """Search candidates with asymmetric aging and calibrated rejuvenation bands.

    MiVOLO readings are first corrected by the measured source-image bias using
    a target-age map: correction is strongest in the center and light at both
    childhood and old-age extremes. Aging
    accepts only corrected predictions from ``target_age`` through
    ``target_age + confidence_aging``. Rejuvenation uses a symmetric
    ``confidence_rejuvenecer`` band around the real requested target after the
    personalized correction. Zero delta is scored against the real source age.
    """
    if bundle.get("identity_encoder") is None or bundle.get("age_estimator") is None:
        raise ValueError("Smart sweep requires ArcFace and MiVOLO in the inference bundle")
    ages = [int(age) for age in target_ages]
    if not ages or len(set(ages)) != len(ages):
        raise ValueError("target_ages must be a non-empty sequence of unique ages")
    if any(age < 0 or age > 120 for age in ages):
        raise ValueError("target ages must be in [0, 120]")
    if not isinstance(use_bias_corrected_mivolo_target, bool):
        raise ValueError("use_bias_corrected_mivolo_target must be a boolean")
    if isinstance(max_trials_per_target, bool) or not 1 <= int(max_trials_per_target) <= 20:
        raise ValueError("max_trials_per_target must be an integer in [1, 20]")
    max_trials_per_target = int(max_trials_per_target)
    scalar_values = (
        age_guidance_scale, text_guidance_scale, image_guidance_scale,
        min_strength, max_strength,
        strength_step_coarse, strength_step_medium, strength_step_fine,
        confidence_aging, confidence_rejuvenecer, identity_margin_for_tiebreak,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("smart-sweep scalar parameters must be finite")
    if not 0 < min_strength <= max_strength <= 1:
        raise ValueError("strength bounds must satisfy 0 < min_strength <= max_strength <= 1")
    if any(step <= 0 for step in (strength_step_coarse, strength_step_medium, strength_step_fine)):
        raise ValueError("strength search steps must be positive")
    if confidence_aging < 0 or confidence_rejuvenecer < 0:
        raise ValueError("confidence_aging and confidence_rejuvenecer must be non-negative")
    if identity_margin_for_tiebreak < 0:
        raise ValueError("identity tie-break margin must be non-negative")
    source_specific_mivolo_bias_weight(
        float(source_age),
        requested_delta=0.0,
        correction_age_map=source_bias_correction_age_map,
    )
    prompt_assistance_scale = validate_prompt_assistance_scale(
        prompt_assistance_scale, "prompt_assistance_scale"
    )
    negative_prompt_assistance_scale = validate_prompt_assistance_scale(
        negative_prompt_assistance_scale, "negative_prompt_assistance_scale"
    )
    _validate_strength_map(target_age_strength_map)
    checkpoint = Path(checkpoint_path).expanduser()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    load_face_aging_adapter_for_inference(bundle, checkpoint, strict_config=strict_config)

    global_pred_source_age = None
    all_candidates: list[list[dict]] = []
    selected_candidates = []
    trial_rows = []
    for target_age in ages:
        requested_delta = float(target_age - source_age)
        base_strength = (
            0.04 if requested_delta == 0
            else _clip(
                _initial_strength(float(target_age), target_age_strength_map),
                min_strength,
                max_strength,
            )
        )
        current_strength = base_strength
        tested_strengths = set()
        candidates = []
        next_guidance = (age_guidance_scale, text_guidance_scale, image_guidance_scale)
        for trial_idx in range(1, max_trials_per_target + 1):
            current_age_guidance, current_text_guidance, current_image_guidance = next_guidance
            result = infer_face_aging(
                bundle=bundle,
                image=source_image,
                source_age=source_age,
                target_age=target_age,
                source_prompt=source_prompt,
                mode="direct",
                strength=current_strength,
                num_inference_steps=num_inference_steps,
                text_reference_mode=text_reference_mode,
                age_guidance_scale=current_age_guidance,
                text_guidance_scale=current_text_guidance,
                image_guidance_scale=current_image_guidance,
                negative_prompt=negative_prompt,
                prompt_style=prompt_style,
                use_cfg=use_cfg,
                seed=seed,
                image_size=image_size,
                compute_diagnostics=True,
            )
            diagnostics = result.get("diagnostics")
            if diagnostics is None:
                raise RuntimeError("Smart sweep did not receive MiVOLO/ArcFace diagnostics")
            if global_pred_source_age is None:
                global_pred_source_age = float(diagnostics["predicted_source_age"])
            # Kept as explicit diagnostics; population alpha/beta calibration was
            # removed. Selection is now on a source-personalized MiVOLO scale.
            expected_delta = requested_delta
            expected_target = float(target_age)
            policy = _directional_selection_policy(
                requested_delta=requested_delta,
                target_age=float(target_age),
                use_personalized_mivolo_bias=bool(use_bias_corrected_mivolo_target),
                confidence_aging=float(confidence_aging),
                confidence_rejuvenecer=float(confidence_rejuvenecer),
            )
            predicted_age = float(diagnostics["predicted_generated_age"])
            source_bias = global_pred_source_age - float(source_age)
            correction_weight = (
                source_specific_mivolo_bias_weight(
                    float(target_age),
                    requested_delta=requested_delta,
                    correction_age_map=source_bias_correction_age_map,
                )
                if use_bias_corrected_mivolo_target else 0.0
            )
            correction_applied = source_bias * correction_weight
            selection_pred_age = predicted_age - correction_applied
            age_error = selection_pred_age - float(policy["selection_target_age"])
            absolute_error = abs(age_error)
            outside_error = _distance_to_acceptable_band(
                selection_pred_age,
                float(policy["acceptable_age_min"]),
                float(policy["acceptable_age_max"]),
            )
            row = {
                "source_age": float(source_age),
                "target_age": float(target_age),
                "requested_delta_age": requested_delta,
                "trial_idx": trial_idx,
                **policy,
                "base_strength": base_strength,
                "strength": current_strength,
                "age_guidance_scale": float(current_age_guidance),
                "text_guidance_scale": float(current_text_guidance),
                "image_guidance_scale": float(current_image_guidance),
                "pred_source_age": global_pred_source_age,
                "expected_mivolo_delta": expected_delta,
                "expected_mivolo_target_age": expected_target,
                "pred_age": predicted_age,
                "selection_pred_age": selection_pred_age,
                "source_mivolo_bias": source_bias,
                "source_bias_correction_weight": correction_weight,
                "source_bias_correction_applied": correction_applied,
                "age_error": age_error,
                "abs_age_error": absolute_error,
                "outside_confidence_error": outside_error,
                "within_confidence_band": outside_error == 0.0,
                "identity_cosine": float(diagnostics["identity_cosine_source_generated"]),
                "selected_best": False,
                "image_path": None,
            }
            candidate = {
                "row": row,
                "result": result,
                "base_guidance": (
                    float(age_guidance_scale),
                    float(text_guidance_scale),
                    float(image_guidance_scale),
                ),
            }
            candidates.append(candidate)
            trial_rows.append(row)
            tested_strengths.add(round(current_strength, 8))

            if outside_error == 0.0:
                break
            direction = _strength_search_direction(
                requested_delta=requested_delta,
                predicted_age=selection_pred_age,
                minimum_age=float(policy["acceptable_age_min"]),
                maximum_age=float(policy["acceptable_age_max"]),
            )
            if enable_guidance_micro_search and trial_idx >= 3:
                if trial_idx == 3:
                    next_guidance = (
                        float(age_guidance_scale) + direction * 0.25,
                        float(text_guidance_scale),
                        float(image_guidance_scale),
                    )
                else:
                    best_identity = max(item["row"]["identity_cosine"] for item in candidates)
                    if row["identity_cosine"] + 1e-12 < best_identity:
                        next_guidance = (
                            float(age_guidance_scale),
                            float(text_guidance_scale),
                            float(image_guidance_scale) + 0.1,
                        )
                    else:
                        next_guidance = (
                            float(age_guidance_scale),
                            float(text_guidance_scale) + direction * 0.25,
                            float(image_guidance_scale),
                        )
                continue

            step = (
                strength_step_coarse if trial_idx == 1
                else strength_step_medium if trial_idx == 2
                else strength_step_fine
            )
            proposed = _clip(current_strength + direction * step, min_strength, max_strength)
            if round(proposed, 8) in tested_strengths:
                break
            current_strength = proposed
            next_guidance = (age_guidance_scale, text_guidance_scale, image_guidance_scale)

        selected = _select_best(
            candidates,
            identity_tiebreak=identity_tiebreak,
            age_margin=float(identity_margin_for_tiebreak),
        )
        selected["row"]["selected_best"] = True
        selected_candidates.append(selected)
        all_candidates.append(candidates)
        print(
            " Smart sweep | "
            f"target={target_age:3d} | trials={len(candidates)} | "
            f"strength={selected['row']['strength']:.3f} | "
            f"MiVOLO={selected['row']['pred_age']:.2f} | "
            f"corrected={selected['row']['selection_pred_age']:.2f} | "
            f"source_bias={selected['row']['source_mivolo_bias']:+.2f} "
            f"x{selected['row']['source_bias_correction_weight']:.2f} | "
            f"policy={selected['row']['direction_policy']} | "
            f"band=[{selected['row']['acceptable_age_min']:.2f}, "
            f"{selected['row']['acceptable_age_max']:.2f}] | "
            f"status={'OK' if selected['row']['within_confidence_band'] else 'MISS'} | "
            f"ID={selected['row']['identity_cosine']:.3f}"
        )

    if save_all_trials:
        trials_dir = destination / "smart_trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        for candidates in all_candidates:
            for candidate in candidates:
                row = candidate["row"]
                path = trials_dir / (
                    f"target_{int(row['target_age']):03d}_trial_{row['trial_idx']:02d}.png"
                )
                candidate["result"]["image"].save(path)
                row["image_path"] = str(path)
        candidate_grid = _candidate_grid(all_candidates)
        candidate_grid_path = destination / "smart_sweep_all_trials.png"
        candidate_grid.save(candidate_grid_path, format="PNG", optimize=False)
    else:
        candidate_grid_path = None

    assisted_candidates = []
    assisted_rows = []
    prompt_records = []
    if generate_assisted_prompt_variant:
        for selected in selected_candidates:
            base_row = selected["row"]
            target_age = int(base_row["target_age"])
            base_target_prompt = selected["result"].get("target_prompt")
            if not base_target_prompt:
                base_target_prompt = build_inference_prompt_pack(
                    target_age=target_age,
                    source_age=source_age,
                    source_prompt=source_prompt,
                    prompt_style=prompt_style,
                    negative_prompt=negative_prompt,
                )["target_prompt"]
            assistance = resolve_prompt_assistance(
                base_target_prompt=base_target_prompt,
                base_negative_prompt=negative_prompt,
                prompt_assistance_config=prompt_assistance_config,
                source_mouth_state=source_mouth_state,
                source_expression_state=source_expression_state,
            )
            result = infer_face_aging(
                bundle=bundle,
                image=source_image,
                source_age=source_age,
                target_age=target_age,
                source_prompt=source_prompt,
                target_prompt=assistance["target_prompt"],
                prompt_assistance_base_prompt=base_target_prompt,
                prompt_assistance_scale=prompt_assistance_scale,
                mode="direct",
                strength=float(base_row["strength"]),
                num_inference_steps=num_inference_steps,
                text_reference_mode=text_reference_mode,
                age_guidance_scale=float(base_row["age_guidance_scale"]),
                text_guidance_scale=float(base_row["text_guidance_scale"]),
                image_guidance_scale=float(base_row["image_guidance_scale"]),
                negative_prompt=assistance["negative_prompt"],
                negative_prompt_assistance_base_prompt=negative_prompt,
                negative_prompt_assistance_scale=negative_prompt_assistance_scale,
                prompt_style=prompt_style,
                use_cfg=use_cfg,
                seed=seed,
                image_size=image_size,
                compute_diagnostics=True,
            )
            diagnostics = result.get("diagnostics")
            if diagnostics is None:
                raise RuntimeError("Prompt-assisted smart sweep did not receive diagnostics")
            predicted_age = float(diagnostics["predicted_generated_age"])
            assisted_source_bias = (
                float(diagnostics["predicted_source_age"]) - float(source_age)
            )
            correction_weight = float(base_row["source_bias_correction_weight"])
            correction_applied = assisted_source_bias * correction_weight
            selection_pred_age = predicted_age - correction_applied
            age_error = selection_pred_age - float(base_row["selection_target_age"])
            absolute_error = abs(age_error)
            row = {
                **base_row,
                "trial_idx": 1,
                "pred_source_age": float(diagnostics["predicted_source_age"]),
                "pred_age": predicted_age,
                "selection_pred_age": selection_pred_age,
                "source_mivolo_bias": assisted_source_bias,
                "source_bias_correction_weight": correction_weight,
                "source_bias_correction_applied": correction_applied,
                "age_error": age_error,
                "abs_age_error": absolute_error,
                "outside_confidence_error": _distance_to_acceptable_band(
                    selection_pred_age,
                    float(base_row["acceptable_age_min"]),
                    float(base_row["acceptable_age_max"]),
                ),
                "identity_cosine": float(
                    diagnostics["identity_cosine_source_generated"]
                ),
                "selected_best": True,
                "image_path": None,
            }
            row["within_confidence_band"] = row["outside_confidence_error"] == 0.0
            assisted = {
                "row": row,
                "result": result,
                "base_guidance": selected["base_guidance"],
            }
            assisted_candidates.append(assisted)
            assisted_rows.append(row)
            prompt_records.append({"target_age": target_age, **assistance})
            print(
                " Prompt comparison | "
                f"target={target_age:3d} | strength={row['strength']:.3f} | "
                f"base corr={base_row['selection_pred_age']:.2f} "
                f"ID={base_row['identity_cosine']:.3f} | "
                f"assisted corr={row['selection_pred_age']:.2f} "
                f"ID={row['identity_cosine']:.3f}"
            )
        for warning in dict.fromkeys(
            warning for record in prompt_records for warning in record["warnings"]
        ):
            print(f"  Note: {warning}")
        if prompt_records:
            print("  Positive support: " + ", ".join(prompt_records[0]["positive_terms"]))
            print("  Negative prompt: " + prompt_records[0]["negative_prompt"])
            print(
                f"  Embedding scales | positive={float(prompt_assistance_scale):.3f} | "
                f"negative={float(negative_prompt_assistance_scale):.3f}"
            )

    final_grid = _final_grid(source_image, source_age, selected_candidates, image_size)
    final_grid_path = destination / (
        "smart_age_sweep_base.png"
        if generate_assisted_prompt_variant else "smart_age_sweep.png"
    )
    final_grid.save(final_grid_path, format="PNG", optimize=False)
    trials_frame = pd.DataFrame(trial_rows, columns=SMART_TRIAL_COLUMNS)
    trials_path = destination / (
        "smart_sweep_trials_base.csv"
        if generate_assisted_prompt_variant else "smart_sweep_trials.csv"
    )
    trials_frame.to_csv(trials_path, index=False)
    summary_rows = []
    for candidate, candidates in zip(selected_candidates, all_candidates):
        summary_rows.append({**candidate["row"], "trials_run": len(candidates)})
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = destination / (
        "smart_sweep_summary_base.csv"
        if generate_assisted_prompt_variant else "smart_sweep_summary.csv"
    )
    summary_frame.to_csv(summary_path, index=False)
    assisted_frame = None
    assisted_grid_path = None
    assisted_trials_path = None
    assisted_summary_path = None
    comparison_grid_path = None
    if assisted_candidates:
        assisted_grid = _final_grid(
            source_image, source_age, assisted_candidates, image_size
        )
        assisted_grid_path = destination / "smart_age_sweep_assisted.png"
        assisted_grid.save(assisted_grid_path, format="PNG", optimize=False)
        assisted_frame = pd.DataFrame([
            {**candidate["row"], "trials_run": 1}
            for candidate in assisted_candidates
        ])
        assisted_frame["support_prompt_used"] = [
            ", ".join(record["positive_terms"]) for record in prompt_records
        ]
        assisted_frame["negative_prompt_used"] = [
            record["negative_prompt"] for record in prompt_records
        ]
        assisted_frame["source_mouth_state"] = [
            record["source_mouth_state"] for record in prompt_records
        ]
        assisted_frame["source_expression_state"] = [
            record["source_expression_state"] for record in prompt_records
        ]
        assisted_frame["prompt_assistance_scale"] = float(prompt_assistance_scale)
        assisted_frame["negative_prompt_assistance_scale"] = float(
            negative_prompt_assistance_scale
        )
        assisted_trials_path = destination / "smart_sweep_trials_assisted.csv"
        assisted_trials_frame = pd.DataFrame(
            assisted_rows, columns=SMART_TRIAL_COLUMNS
        )
        assisted_trials_frame["support_prompt_used"] = assisted_frame[
            "support_prompt_used"
        ]
        assisted_trials_frame["negative_prompt_used"] = assisted_frame[
            "negative_prompt_used"
        ]
        assisted_trials_frame["source_mouth_state"] = assisted_frame[
            "source_mouth_state"
        ]
        assisted_trials_frame["source_expression_state"] = assisted_frame[
            "source_expression_state"
        ]
        assisted_trials_frame["prompt_assistance_scale"] = assisted_frame[
            "prompt_assistance_scale"
        ]
        assisted_trials_frame["negative_prompt_assistance_scale"] = assisted_frame[
            "negative_prompt_assistance_scale"
        ]
        assisted_trials_frame.to_csv(assisted_trials_path, index=False)
        assisted_summary_path = destination / "smart_sweep_summary_assisted.csv"
        assisted_frame.to_csv(assisted_summary_path, index=False)
        if save_prompt_comparison:
            comparison_grid_path = destination / "smart_age_sweep_comparison.png"
            stack_sweep_variants(final_grid, assisted_grid).save(
                comparison_grid_path, format="PNG", optimize=False
            )
    summary_frame.attrs.update({
        "grid_path": str(final_grid_path),
        "trials_csv_path": str(trials_path),
        "summary_csv_path": str(summary_path),
        "all_trials_grid_path": str(candidate_grid_path) if candidate_grid_path else None,
        "trials": trials_frame,
        "selected_results": [candidate["result"] for candidate in selected_candidates],
        "bias_corrected": bool(use_bias_corrected_mivolo_target),
        "personalized_mivolo_bias": bool(use_bias_corrected_mivolo_target),
        "source_mivolo_bias": float(global_pred_source_age - float(source_age)),
        "source_bias_correction_age_map": dict(
            DEFAULT_SOURCE_BIAS_CORRECTION_AGE_MAP
            if source_bias_correction_age_map is None
            else source_bias_correction_age_map
        ),
        "confidence_aging": float(confidence_aging),
        "confidence_rejuvenecer": float(confidence_rejuvenecer),
    })
    if generate_assisted_prompt_variant:
        summary_frame.attrs.update({
            "assisted": assisted_frame,
            "assisted_grid_path": (
                str(assisted_grid_path) if assisted_grid_path else None
            ),
            "assisted_trials_csv_path": (
                str(assisted_trials_path) if assisted_trials_path else None
            ),
            "assisted_summary_csv_path": (
                str(assisted_summary_path) if assisted_summary_path else None
            ),
            "comparison_grid_path": (
                str(comparison_grid_path) if comparison_grid_path else None
            ),
            "prompt_assistance": prompt_records,
        })
    return summary_frame
