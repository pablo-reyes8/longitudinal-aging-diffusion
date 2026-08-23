"""Standalone diagnostic sweeps for saved face-aging checkpoints."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd

from .checkpoint_loading import load_face_aging_adapter_for_inference
from .comparison_helpers import generate_age_sweep
from .infer_face_aging import save_inference_image


DIAGNOSTIC_COLUMNS = [
    "checkpoint", "source_age", "target_age", "predicted_generated_age",
    "age_error", "identity_cosine", "mode", "strength",
    "num_inference_steps", "text_guidance_scale", "image_guidance_scale", "seed",
]


def _require_auxiliaries(bundle) -> None:
    missing = [name for name in ("identity_encoder", "age_estimator") if bundle.get(name) is None]
    if missing:
        raise ValueError(
            "Checkpoint diagnostics require the existing auxiliary adapters in the bundle: "
            + ", ".join(missing)
            + ". Build the inference bundle with load_auxiliary_models=True."
        )


def _checkpoint_label(path: Path) -> str:
    if re.fullmatch(r"epoch_\d+", path.parent.name):
        return path.parent.name
    return str(path)


def diagnose_checkpoint_age_sweep(
    checkpoint_path: str | Path,
    bundle,
    source_image,
    source_age: int | None,
    target_ages: Iterable[int],
    *,
    output_dir: str | Path | None = None,
    mode: str = "direct",
    use_inverse_diffusion: bool | None = None,
    num_inference_steps: int = 50,
    strength: float = 0.45,
    inversion_strength: float = 1.0,
    text_guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
    negative_prompt: str = "",
    prompt_style: str = "selfage",
    use_cfg: bool = True,
    seed: int = 2026,
    image_size: int = 256,
    strict_config: bool = True,
) -> pd.DataFrame:
    """Load one checkpoint, diagnose a fixed-seed age sweep, and optionally save it."""
    _require_auxiliaries(bundle)
    checkpoint = Path(checkpoint_path).expanduser()
    ages = [int(age) for age in target_ages]
    if not ages:
        raise ValueError("target_ages must not be empty")
    load_face_aging_adapter_for_inference(bundle, checkpoint, strict_config=strict_config)

    destination = Path(output_dir).expanduser() if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    sweep = generate_age_sweep(
        bundle=bundle,
        image=source_image,
        ages=ages,
        output_path=destination / "age_sweep.png" if destination is not None else None,
        annotate_diagnostics=True,
        include_source=True,
        source_age=source_age,
        mode=mode,
        use_inverse_diffusion=use_inverse_diffusion,
        num_inference_steps=num_inference_steps,
        strength=strength,
        inversion_strength=inversion_strength,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        negative_prompt=negative_prompt,
        prompt_style=prompt_style,
        use_cfg=use_cfg,
        seed=seed,
        image_size=image_size,
        compute_diagnostics=True,
    )

    rows = []
    for age, result in zip(ages, sweep["results"]):
        diagnostics = result.get("diagnostics")
        if diagnostics is None:
            raise RuntimeError("Auxiliary diagnostics unexpectedly returned None")
        if destination is not None:
            save_inference_image(result, destination / f"age_{age:03d}.png")
        predicted = diagnostics["predicted_generated_age"]
        rows.append({
            "checkpoint": _checkpoint_label(checkpoint),
            "source_age": source_age,
            "target_age": float(age),
            "predicted_generated_age": predicted,
            "age_error": predicted - float(age),
            "identity_cosine": diagnostics["identity_cosine_source_generated"],
            "mode": result["mode"],
            "strength": float(strength),
            "num_inference_steps": int(num_inference_steps),
            "text_guidance_scale": float(text_guidance_scale),
            "image_guidance_scale": float(image_guidance_scale),
            "seed": int(seed),
        })
    frame = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    if destination is not None:
        csv_path = destination / "sampling_diagnostics.csv"
        frame.to_csv(csv_path, index=False)
        frame.attrs.update({
            "output_dir": str(destination),
            "grid_path": str(destination / "age_sweep.png"),
            "csv_path": str(csv_path),
        })
    return frame


def diagnose_checkpoints_age_sweep(
    checkpoint_paths: Sequence[str | Path],
    bundle,
    source_image,
    source_age: int | None,
    target_ages: Iterable[int],
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compare several checkpoints under identical image, ages, seed, and settings."""
    checkpoints = [Path(path).expanduser() for path in checkpoint_paths]
    if not checkpoints:
        raise ValueError("checkpoint_paths must not be empty")
    ages = list(target_ages)
    frames = []
    root = Path(output_dir).expanduser() if output_dir is not None else None
    for index, checkpoint in enumerate(checkpoints):
        label = checkpoint.parent.name if checkpoint.parent.name else checkpoint.stem
        child = root / f"{index:02d}_{label}" if root is not None else None
        frame = diagnose_checkpoint_age_sweep(
            checkpoint_path=checkpoint,
            bundle=bundle,
            source_image=source_image,
            source_age=source_age,
            target_ages=ages,
            output_dir=child,
            **kwargs,
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        combined.to_csv(root / "checkpoints_diagnostics.csv", index=False)
    return combined
