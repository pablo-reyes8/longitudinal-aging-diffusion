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


DEFAULT_SMART_TARGET_STRENGTH_MAP = {
    8: 0.35,
    12: 0.34,
    26: 0.04,
    35: 0.18,
    45: 0.27,
    55: 0.27,
    65: 0.25,
}

SMART_TRIAL_COLUMNS = [
    "source_age", "target_age", "requested_delta_age", "trial_idx",
    "base_strength", "strength", "age_guidance_scale", "text_guidance_scale",
    "image_guidance_scale", "pred_source_age", "expected_mivolo_delta",
    "expected_mivolo_target_age", "pred_age", "age_error", "abs_age_error",
    "confidence_margin_years", "outside_confidence_error", "identity_cosine",
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
                f"Pred={row['pred_age']:.1f} | err={row['age_error']:+.1f}\n"
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
            f"Target {row['target_age']:.0f} | Pred {row['pred_age']:.1f}\n"
            f"Err {row['age_error']:+.1f} | Strength {row['strength']:.3f}\n"
            f"ID {row['identity_cosine']:.3f} | band +/-{row['confidence_margin_years']:.1f}"
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
    minimum = min(candidate["row"]["outside_confidence_error"] for candidate in candidates)
    eligible = [
        candidate for candidate in candidates
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
    mivolo_bias_alpha: float = -3.19,
    mivolo_bias_beta: float = 0.841,
    mivolo_small_delta_confidence_years: float = 4.0,
    mivolo_large_delta_confidence_years: float = 2.0,
    mivolo_full_confidence_delta: float = 25.0,
    enable_guidance_micro_search: bool = False,
    age_guidance_scale: float = 3.0,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    min_strength: float = 0.04,
    max_strength: float = 0.45,
    strength_step_coarse: float = 0.05,
    strength_step_medium: float = 0.03,
    strength_step_fine: float = 0.02,
    age_tolerance_years: float = 2.0,
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
) -> pd.DataFrame:
    """Screen a short strength trajectory per age and save only selected outputs."""
    if bundle.get("identity_encoder") is None or bundle.get("age_estimator") is None:
        raise ValueError("Smart sweep requires ArcFace and MiVOLO in the inference bundle")
    ages = [int(age) for age in target_ages]
    if not ages or len(set(ages)) != len(ages):
        raise ValueError("target_ages must be a non-empty sequence of unique ages")
    if any(age < 0 or age > 120 for age in ages):
        raise ValueError("target ages must be in [0, 120]")
    if isinstance(max_trials_per_target, bool) or not 1 <= int(max_trials_per_target) <= 20:
        raise ValueError("max_trials_per_target must be an integer in [1, 20]")
    max_trials_per_target = int(max_trials_per_target)
    scalar_values = (
        mivolo_bias_alpha, mivolo_bias_beta, age_guidance_scale,
        text_guidance_scale, image_guidance_scale, min_strength, max_strength,
        strength_step_coarse, strength_step_medium, strength_step_fine,
        age_tolerance_years, identity_margin_for_tiebreak,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("smart-sweep scalar parameters must be finite")
    if not 0 < min_strength <= max_strength <= 1:
        raise ValueError("strength bounds must satisfy 0 < min_strength <= max_strength <= 1")
    if any(step <= 0 for step in (strength_step_coarse, strength_step_medium, strength_step_fine)):
        raise ValueError("strength search steps must be positive")
    if age_tolerance_years < 0 or identity_margin_for_tiebreak < 0:
        raise ValueError("age tolerance and identity tie-break margin must be non-negative")
    _validate_strength_map(target_age_strength_map)
    # Validate the adaptive confidence policy once before loading/generating.
    adaptive_mivolo_confidence_margin(
        0,
        small_delta_margin_years=mivolo_small_delta_confidence_years,
        large_delta_margin_years=mivolo_large_delta_confidence_years,
        full_confidence_delta=mivolo_full_confidence_delta,
    )

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
        absolute_delta = abs(requested_delta)
        confidence_margin = max(
            float(age_tolerance_years),
            adaptive_mivolo_confidence_margin(
                absolute_delta,
                small_delta_margin_years=mivolo_small_delta_confidence_years,
                large_delta_margin_years=mivolo_large_delta_confidence_years,
                full_confidence_delta=mivolo_full_confidence_delta,
            ),
        )
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
            expected_delta = (
                float(mivolo_bias_alpha) + float(mivolo_bias_beta) * requested_delta
                if use_bias_corrected_mivolo_target else requested_delta
            )
            expected_target = (
                global_pred_source_age + expected_delta
                if use_bias_corrected_mivolo_target else float(target_age)
            )
            predicted_age = float(diagnostics["predicted_generated_age"])
            age_error = predicted_age - expected_target
            absolute_error = abs(age_error)
            row = {
                "source_age": float(source_age),
                "target_age": float(target_age),
                "requested_delta_age": requested_delta,
                "trial_idx": trial_idx,
                "base_strength": base_strength,
                "strength": current_strength,
                "age_guidance_scale": float(current_age_guidance),
                "text_guidance_scale": float(current_text_guidance),
                "image_guidance_scale": float(current_image_guidance),
                "pred_source_age": global_pred_source_age,
                "expected_mivolo_delta": expected_delta,
                "expected_mivolo_target_age": expected_target,
                "pred_age": predicted_age,
                "age_error": age_error,
                "abs_age_error": absolute_error,
                "confidence_margin_years": confidence_margin,
                "outside_confidence_error": max(0.0, absolute_error - confidence_margin),
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

            if requested_delta == 0 or absolute_error <= confidence_margin:
                break
            direction = -1.0 if age_error > 0 else 1.0
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
            f"pred={selected['row']['pred_age']:.2f} | "
            f"expected={selected['row']['expected_mivolo_target_age']:.2f} +/- "
            f"{selected['row']['confidence_margin_years']:.2f} | "
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

    final_grid = _final_grid(source_image, source_age, selected_candidates, image_size)
    final_grid_path = destination / "smart_age_sweep.png"
    final_grid.save(final_grid_path, format="PNG", optimize=False)
    trials_frame = pd.DataFrame(trial_rows, columns=SMART_TRIAL_COLUMNS)
    trials_path = destination / "smart_sweep_trials.csv"
    trials_frame.to_csv(trials_path, index=False)
    summary_rows = []
    for candidate, candidates in zip(selected_candidates, all_candidates):
        summary_rows.append({**candidate["row"], "trials_run": len(candidates)})
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = destination / "smart_sweep_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    summary_frame.attrs.update({
        "grid_path": str(final_grid_path),
        "trials_csv_path": str(trials_path),
        "summary_csv_path": str(summary_path),
        "all_trials_grid_path": str(candidate_grid_path) if candidate_grid_path else None,
        "trials": trials_frame,
        "selected_results": [candidate["result"] for candidate in selected_candidates],
        "bias_corrected": bool(use_bias_corrected_mivolo_target),
        "mivolo_bias_alpha": float(mivolo_bias_alpha),
        "mivolo_bias_beta": float(mivolo_bias_beta),
    })
    return summary_frame
