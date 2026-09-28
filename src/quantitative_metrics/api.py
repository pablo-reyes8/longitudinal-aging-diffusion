"""External evaluation helpers for checkpoints and labeled image trees.

This module is deliberately separate from :mod:`src.ablation_studies`.  It
downloads/loads only the frozen evaluation assets and runs inference through a
saved adapter checkpoint; the ablation runner keeps its existing metric path.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from PIL import Image

from .assets import prepare_metrics_config
from .backends import load_quantitative_metrics
from .evaluator import evaluate_aging_outputs
from .image_io import ImageResolver


DEFAULT_METRICS_ROOT = Path("/content/evaluation_models")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical per-target policy used by the external inference notebook.  This is
# intentionally opt-in: the low-level ``evaluate_aging`` API still accepts an
# explicit scalar strength, and ablations keep their own configuration.
DEFAULT_TARGET_AGE_STRENGTH_MAP = {
    8: 0.33,
    15: 0.31,
    26: 0.04,
    35: 0.22,
    44: 0.24,
    55: 0.27,
    65: 0.29,
}


def build_default_metrics_config(
    metrics_root: str | Path = DEFAULT_METRICS_ROOT,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build the same local metric layout used by the Colab notebooks."""
    root = Path(metrics_root).expanduser()
    return {
        "adaface_repo_path": root / "AdaFace",
        "adaface_checkpoint_path": root / "adaface_ir101_webface12m.ckpt",
        "dex_prototxt_path": root / "DEX" / "age.prototxt",
        "dex_checkpoint_path": root / "DEX" / "dex_chalearn_iccv2015.caffemodel",
        "kid_inception_weights_path": root / "KID" / "weights-inception-2015-12-05-6726825d.pth",
        "device": device,
        "local_files_only": True,
    }


def prepare_quantitative_metrics(
    metrics_root: str | Path = DEFAULT_METRICS_ROOT,
    *,
    device: str = "cpu",
    include_kid: bool = True,
    metrics_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Download/validate metric assets and return a runner-ready config.

    ``include_kid=False`` downloads only AdaFace, DEX, and the alignment code;
    this is the lightweight path used by :func:`evaluate_aging`.
    """
    config = build_default_metrics_config(metrics_root, device=device)
    if metrics_config is not None:
        config.update(dict(metrics_config))
    config.setdefault("device", device)
    config["local_files_only"] = True
    return prepare_metrics_config(config, include_kid=include_kid)


def _load_external_metrics(
    *,
    metrics_bundle,
    metrics_config: Mapping[str, Any] | None,
    metrics_root: str | Path | None,
    include_kid: bool,
    download_assets: bool,
):
    if metrics_bundle is not None:
        return metrics_bundle, None
    config = dict(metrics_config or {})
    if not config:
        config = build_default_metrics_config(metrics_root or DEFAULT_METRICS_ROOT)
    config.setdefault("device", "cpu")
    config["local_files_only"] = True
    if download_assets:
        config = prepare_metrics_config(config, include_kid=include_kid)
    return load_quantitative_metrics(**config), config


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def evaluate_aging(
    checkpoint_path: str | Path,
    source_image: str | Path,
    target_image: str | Path,
    source_age: float,
    target_age: float,
    *,
    metrics_bundle=None,
    metrics_config: Mapping[str, Any] | None = None,
    metrics_root: str | Path | None = None,
    output_dir: str | Path = "outputs/quantitative_metrics/checkpoint",
    model_name: str = "checkpoint",
    inference_config: Mapping[str, Any] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    device=None,
    dtype=None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    download_assets: bool = True,
):
    """Generate one target from a ``.pt`` checkpoint and score it.

    This computes AdaFace ID-Source, AdaFace ID-GT, and DEX age MAE. KID is
    intentionally disabled here and is exposed separately by :func:`evaluate_kid`.
    """
    if not Path(checkpoint_path).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    for name, reference in (("source_image", source_image), ("target_image", target_image)):
        if not Path(reference).expanduser().is_file():
            raise FileNotFoundError(f"{name} not found: {reference}")

    metrics, prepared_config = _load_external_metrics(
        metrics_bundle=metrics_bundle,
        metrics_config=metrics_config,
        metrics_root=metrics_root,
        include_kid=False,
        download_assets=download_assets,
    )

    from src.inference import infer_face_aging_direct, load_face_aging_inference_bundle

    bundle = load_face_aging_inference_bundle(
        checkpoint_path,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )
    inference_options = dict(inference_config or {})
    protected = {"bundle", "image", "source_age", "target_age"}
    duplicate = sorted(protected.intersection(inference_options))
    if duplicate:
        raise ValueError(f"inference_config cannot override: {', '.join(duplicate)}")
    inference_options.setdefault("compute_diagnostics", False)
    inference_options.setdefault("output_type", "pil")
    inference_options.setdefault("return_dict", True)
    result = infer_face_aging_direct(
        bundle=bundle,
        image=source_image,
        source_age=source_age,
        target_age=target_age,
        **inference_options,
    )
    generated = result["image"] if isinstance(result, dict) else result
    if not isinstance(generated, Image.Image):
        raise TypeError("Inference did not return a PIL image; keep output_type='pil'")

    output = Path(output_dir).expanduser()
    generated_path = output / "generated" / f"{_safe_name(model_name)}.png"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated.convert("RGB").save(generated_path)
    sample_id = f"{_safe_name(model_name)}__to_{float(target_age):g}"
    manifest = pd.DataFrame([{
        "sample_id": sample_id,
        "source_path": str(Path(source_image).expanduser()),
        "target_path": str(Path(target_image).expanduser()),
        "generated_path": str(generated_path),
        "source_age": float(source_age),
        "target_age": float(target_age),
        "model_name": model_name,
    }])
    options = dict(evaluation_config or {})
    options["compute_kid"] = False
    options.setdefault("output_dir", output / "evaluation")
    evaluation = evaluate_aging_outputs(manifest, metrics_bundle=metrics, **options)
    report = bundle.get("inference_checkpoint_report")
    del bundle
    gc.collect()
    return {
        "generated_path": generated_path,
        "manifest": manifest,
        "manifest_path": evaluation["paths"]["manifest"],
        "checkpoint_report": report,
        "evaluation": evaluation,
        "metrics_config": prepared_config,
    }


def _target_image_lookup(
    target_images: Mapping[float, str | Path] | Sequence[str | Path] | None,
    target_ages: Sequence[float],
) -> dict[float, str | Path | None]:
    if target_images is None:
        return {float(age): None for age in target_ages}
    if isinstance(target_images, Mapping):
        normalized = {}
        for raw_age, path in target_images.items():
            try:
                normalized[float(raw_age)] = path
            except (TypeError, ValueError) as exc:
                raise ValueError("target_images keys must be numeric ages") from exc
        missing = [age for age in target_ages if float(age) not in normalized]
        if missing:
            raise ValueError(
                "target_images is missing target ages: "
                + ", ".join(f"{age:g}" for age in missing)
            )
        return {float(age): normalized[float(age)] for age in target_ages}
    values = list(target_images)
    if len(values) != len(target_ages):
        raise ValueError("target_images sequence must match target_ages length")
    return {float(age): path for age, path in zip(target_ages, values)}


def _validate_target_ages(target_ages: Iterable[float]) -> list[float]:
    ages = []
    for raw_age in target_ages:
        if isinstance(raw_age, bool):
            raise ValueError("target_ages must contain finite numeric ages")
        try:
            age = float(raw_age)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_ages must contain finite numeric ages") from exc
        if not math.isfinite(age) or not 0 <= age <= 120:
            raise ValueError("target_ages must be finite ages in [0, 120]")
        ages.append(age)
    if not ages or len(set(ages)) != len(ages):
        raise ValueError("target_ages must be a non-empty sequence of unique ages")
    return ages


def _load_inference_bundle(
    checkpoint_path: str | Path,
    *,
    inference_bundle,
    device,
    dtype,
    local_files_only: bool,
    token: str | bool | None,
    revision: str | None,
    load_auxiliary_models: bool,
):
    from src.inference import load_face_aging_inference_bundle

    if inference_bundle is not None:
        return inference_bundle, None
    return (
        load_face_aging_inference_bundle(
            checkpoint_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            load_auxiliary_models=load_auxiliary_models,
        ),
        True,
    )


def _evaluate_generated_manifest(
    *,
    manifest: pd.DataFrame,
    metrics,
    output_dir: Path,
    evaluation_config: Mapping[str, Any] | None,
):
    options = dict(evaluation_config or {})
    options["compute_kid"] = False
    options.setdefault("output_dir", output_dir / "evaluation")
    return evaluate_aging_outputs(manifest, metrics_bundle=metrics, **options)


def evaluate_aging_inference(
    checkpoint_path: str | Path,
    source_image: str | Path,
    source_age: float,
    target_ages: Iterable[float],
    *,
    target_images: Mapping[float, str | Path] | Sequence[str | Path] | None = None,
    target_age_strength_map: Mapping[float, float] | None = None,
    metrics_bundle=None,
    metrics_config: Mapping[str, Any] | None = None,
    metrics_root: str | Path | None = None,
    inference_bundle=None,
    output_dir: str | Path = "outputs/quantitative_metrics/inference",
    model_name: str = "checkpoint",
    inference_config: Mapping[str, Any] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    device=None,
    dtype=None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    download_assets: bool = True,
):
    """Evaluate a checkpoint over an exact target-age strength map.

    This is the canonical external inference path: one adapter is loaded once,
    each target age receives its mapped strength, and ID/age metrics are run on
    the resulting images. KID remains in :func:`evaluate_kid`.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    source = Path(source_image).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not source.is_file():
        raise FileNotFoundError(f"source_image not found: {source}")
    ages = _validate_target_ages(target_ages)
    target_lookup = _target_image_lookup(target_images, ages)
    policy = None if target_age_strength_map is None else dict(target_age_strength_map)
    metrics, prepared_config = _load_external_metrics(
        metrics_bundle=metrics_bundle,
        metrics_config=metrics_config,
        metrics_root=metrics_root,
        include_kid=False,
        download_assets=download_assets,
    )
    bundle, owns_bundle = _load_inference_bundle(
        checkpoint,
        inference_bundle=inference_bundle,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        load_auxiliary_models=False,
    )
    from src.inference import generate_aged_face_adaptive_strength

    options = dict(inference_config or {})
    protected = {
        "bundle", "image", "source_age", "target_age", "strength",
        "strength_map", "target_age_strength_map",
    }
    duplicate = sorted(protected.intersection(options))
    if duplicate:
        raise ValueError(f"inference_config cannot override: {', '.join(duplicate)}")
    options.setdefault("mode", "direct")
    options.setdefault("compute_diagnostics", False)
    options.setdefault("output_type", "pil")
    options.setdefault("return_dict", True)
    output = Path(output_dir).expanduser()
    generated_root = output / "generated" / _safe_name(model_name)
    generated_root.mkdir(parents=True, exist_ok=True)
    rows = []
    generated_paths = {}
    try:
        for age in ages:
            adaptive_options = {} if policy is None else {
                "target_age_strength_map": policy,
            }
            result = generate_aged_face_adaptive_strength(
                bundle=bundle,
                image=source,
                source_age=source_age,
                target_age=age,
                **adaptive_options,
                **options,
            )
            generated = result["image"] if isinstance(result, dict) else result
            if not isinstance(generated, Image.Image):
                raise TypeError("Inference did not return a PIL image; keep output_type='pil'")
            generated_path = generated_root / f"target_{age:g}.png"
            generated.convert("RGB").save(generated_path)
            generated_paths[age] = generated_path
            raw_strength = (
                result.get("effective_strength", result.get("metadata", {}).get("effective_strength"))
                if isinstance(result, dict) else None
            )
            rows.append({
                "sample_id": f"{_safe_name(model_name)}__to_{age:g}",
                "source_path": str(source),
                "target_path": (
                    str(Path(target_lookup[age]).expanduser())
                    if target_lookup[age] is not None else None
                ),
                "generated_path": str(generated_path),
                "source_age": float(source_age),
                "target_age": float(age),
                "model_name": model_name,
                "strength": float(raw_strength) if raw_strength is not None else float("nan"),
            })
        manifest = pd.DataFrame(rows)
        evaluation = _evaluate_generated_manifest(
            manifest=manifest,
            metrics=metrics,
            output_dir=output,
            evaluation_config=evaluation_config,
        )
        return {
            "generated_paths": generated_paths,
            "manifest": manifest,
            "manifest_path": evaluation["paths"]["manifest"],
            "evaluation": evaluation,
            "metrics_config": prepared_config,
            "target_age_strength_map": policy,
        }
    finally:
        if owns_bundle:
            del bundle
            gc.collect()


def _picker_candidate_score(row: Mapping[str, Any]) -> tuple[float, float]:
    age_error = row.get("age_error", float("nan"))
    identity = row.get("identity_cosine", float("nan"))
    try:
        age_score = abs(float(age_error)) if math.isfinite(float(age_error)) else float("inf")
    except (TypeError, ValueError):
        age_score = float("inf")
    try:
        identity_score = -float(identity) if math.isfinite(float(identity)) else float("inf")
    except (TypeError, ValueError):
        identity_score = float("inf")
    return age_score, identity_score


def evaluate_aging_inference_picking(
    checkpoint_path: str | Path,
    source_image: str | Path,
    source_age: float,
    target_ages: Iterable[float],
    *,
    target_images: Mapping[float, str | Path] | Sequence[str | Path] | None = None,
    target_age_strength_map: Mapping[float, float] | None = None,
    metrics_bundle=None,
    metrics_config: Mapping[str, Any] | None = None,
    metrics_root: str | Path | None = None,
    inference_bundle=None,
    output_dir: str | Path = "outputs/quantitative_metrics/inference_picking",
    model_name: str = "checkpoint_picked",
    diagnostic_config: Mapping[str, Any] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    device=None,
    dtype=None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    download_assets: bool = True,
):
    """Run adaptive diagnostics, select the best base/assisted image per age, and score it."""
    checkpoint = Path(checkpoint_path).expanduser()
    source = Path(source_image).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not source.is_file():
        raise FileNotFoundError(f"source_image not found: {source}")
    ages = _validate_target_ages(target_ages)
    if any(age != int(age) for age in ages):
        raise ValueError("adaptive picking requires integer target ages")
    target_lookup = _target_image_lookup(target_images, ages)
    metrics, prepared_config = _load_external_metrics(
        metrics_bundle=metrics_bundle,
        metrics_config=metrics_config,
        metrics_root=metrics_root,
        include_kid=False,
        download_assets=download_assets,
    )
    bundle, owns_bundle = _load_inference_bundle(
        checkpoint,
        inference_bundle=inference_bundle,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        load_auxiliary_models=True,
    )
    from src.inference import diagnose_checkpoint_adaptive_age_sweep

    options = dict(diagnostic_config or {})
    protected = {"checkpoint_path", "bundle", "source_image", "source_age", "target_ages"}
    duplicate = sorted(protected.intersection(options))
    if duplicate:
        raise ValueError(f"diagnostic_config cannot override: {', '.join(duplicate)}")
    if target_age_strength_map is not None:
        options.setdefault("target_age_strength_map", dict(target_age_strength_map))
    options.setdefault("output_dir", Path(output_dir).expanduser() / "diagnostic")
    # Picking compares the canonical candidate with the prompt-assisted one by
    # default; callers can explicitly set this to False for a base-only sweep.
    options.setdefault("generate_assisted_prompt_variant", True)
    diagnostic_df = diagnose_checkpoint_adaptive_age_sweep(
        checkpoint_path=checkpoint,
        bundle=bundle,
        source_image=source,
        source_age=source_age,
        target_ages=ages,
        **options,
    )
    base_results = list(diagnostic_df.attrs.get("base_results", []))
    assisted_results = list(diagnostic_df.attrs.get("assisted_results", []))
    assisted_frame = diagnostic_df.attrs.get("assisted")
    if len(base_results) != len(ages):
        raise RuntimeError(
            "Adaptive diagnostics did not expose generated candidates; "
            "use the current checkpoint_diagnostics implementation."
        )
    output = Path(output_dir).expanduser()
    generated_root = output / "generated" / _safe_name(model_name)
    generated_root.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_variants = {}
    try:
        for index, age in enumerate(ages):
            candidates = [{
                "variant": "base",
                "row": diagnostic_df.iloc[index].to_dict(),
                "result": base_results[index],
            }]
            if assisted_results and assisted_frame is not None:
                candidates.append({
                    "variant": "assisted",
                    "row": assisted_frame.iloc[index].to_dict(),
                    "result": assisted_results[index],
                })
            selected = min(candidates, key=lambda item: _picker_candidate_score(item["row"]))
            result = selected["result"]
            generated = result["image"] if isinstance(result, dict) else result
            if not isinstance(generated, Image.Image):
                raise TypeError("Adaptive diagnostics did not return a PIL image")
            generated_path = generated_root / f"target_{age:g}.png"
            generated.convert("RGB").save(generated_path)
            selected_variants[age] = selected["variant"]
            rows.append({
                "sample_id": f"{_safe_name(model_name)}__to_{age:g}",
                "source_path": str(source),
                "target_path": (
                    str(Path(target_lookup[age]).expanduser())
                    if target_lookup[age] is not None else None
                ),
                "generated_path": str(generated_path),
                "source_age": float(source_age),
                "target_age": float(age),
                "model_name": model_name,
                "selected_variant": selected["variant"],
                "strength": float(selected["row"].get("effective_strength", selected["row"].get("strength"))),
                "age_error": float(selected["row"].get("age_error", float("nan"))),
                "identity_cosine": float(selected["row"].get("identity_cosine", float("nan"))),
            })
        manifest = pd.DataFrame(rows)
        evaluation = _evaluate_generated_manifest(
            manifest=manifest,
            metrics=metrics,
            output_dir=output,
            evaluation_config=evaluation_config,
        )
        return {
            "diagnostic": diagnostic_df,
            "generated_paths": {age: Path(row["generated_path"]) for age, row in zip(ages, rows)},
            "selected_variants": selected_variants,
            "manifest": manifest,
            "manifest_path": evaluation["paths"]["manifest"],
            "evaluation": evaluation,
            "metrics_config": prepared_config,
        }
    finally:
        if owns_bundle:
            del bundle
            gc.collect()


def _parse_age(stem: str) -> int | None:
    normalized = re.sub(r"^id_[A-Za-z0-9-]+[_-]?", "", stem, flags=re.IGNORECASE)
    marked = re.search(
        r"(?:target|to|age)[_-]?(\d{1,3})(?!\d)", normalized, flags=re.IGNORECASE
    )
    match = marked or re.search(r"^(\d{1,3})(?:[_-]\d+)?$", normalized)
    if match is None:
        return None
    age = int(match.group(1))
    return age if 0 <= age <= 122 else None


def _identity_for_path(root: Path, path: Path) -> str | None:
    relative = path.relative_to(root)
    for component in relative.parts[:-1]:
        if component.startswith("id_"):
            return component
    match = re.search(r"(id_[A-Za-z0-9-]+)", path.stem)
    return match.group(1) if match else None


def index_labeled_image_tree(root: str | Path) -> dict[tuple[str, int], Path]:
    """Index ``id_*/<age>.<ext>`` images, keeping the first duplicate."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image root not found: {root}")
    indexed: dict[tuple[str, int], Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        identity = _identity_for_path(root, path)
        age = _parse_age(path.stem)
        if identity is None or age is None:
            continue
        indexed.setdefault((identity, age), path)
    if not indexed:
        raise ValueError(
            f"No labeled images found under {root}; expected id_*/<age>.<jpg|png>"
        )
    return indexed


def select_kid_pairs(
    generated_root: str | Path,
    reference_root: str | Path,
    *,
    num_ids: int | None = None,
    num_transitions: int | None = None,
    seed: int = 2026,
) -> list[dict[str, str | int]]:
    """Select deterministic generated/reference pairs by identity and target age."""
    generated = index_labeled_image_tree(generated_root)
    reference = index_labeled_image_tree(reference_root)
    common = sorted(set(generated).intersection(reference))
    if not common:
        raise ValueError("Generated and reference trees have no common (identity, age) labels")
    identities = sorted({identity for identity, _ in common})
    rng = random.Random(int(seed))
    if num_ids is not None:
        if int(num_ids) < 1:
            raise ValueError("num_ids must be positive")
        if int(num_ids) < len(identities):
            identities = sorted(rng.sample(identities, int(num_ids)))
    candidates = [
        {
            "identity_id": identity,
            "target_age": age,
            "real_path": str(reference[(identity, age)]),
            "generated_path": str(generated[(identity, age)]),
        }
        for identity, age in common
        if identity in identities
    ]
    if num_transitions is not None:
        if int(num_transitions) < 1:
            raise ValueError("num_transitions must be positive")
        rng.shuffle(candidates)
        candidates = candidates[: int(num_transitions)]
    if len(candidates) < 2:
        raise ValueError("KID requires at least two generated/reference image pairs")
    return candidates


def evaluate_kid(
    generated_root: str | Path,
    reference_root: str | Path,
    *,
    metrics_bundle=None,
    metrics_config: Mapping[str, Any] | None = None,
    metrics_root: str | Path | None = None,
    output_dir: str | Path = "outputs/quantitative_metrics/kid",
    num_ids: int | None = None,
    num_transitions: int | None = None,
    seed: int = 2026,
    subsets: int = 100,
    subset_size: int | None = None,
    download_assets: bool = True,
):
    """Compute KID between two ``id_*/age.*`` image trees."""
    pairs = select_kid_pairs(
        generated_root,
        reference_root,
        num_ids=num_ids,
        num_transitions=num_transitions,
        seed=seed,
    )
    metrics, prepared_config = _load_external_metrics(
        metrics_bundle=metrics_bundle,
        metrics_config=metrics_config,
        metrics_root=metrics_root,
        include_kid=True,
        download_assets=download_assets,
    )
    resolver = ImageResolver()
    real_images = [resolver.load(pair["real_path"])[0] for pair in pairs]
    generated_images = [resolver.load(pair["generated_path"])[0] for pair in pairs]
    resolved_subset_size = min(50, len(pairs)) if subset_size is None else min(int(subset_size), len(pairs))
    kid_mean, kid_std = metrics["kid_metric"].compute(
        real_images,
        generated_images,
        subsets=int(subsets),
        subset_size=resolved_subset_size,
        seed=int(seed),
    )
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    selection = pd.DataFrame(pairs)
    selection_path = output / "kid_selection.csv"
    summary_path = output / "kid_summary.json"
    selection.to_csv(selection_path, index=False)
    summary = {
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "n_pairs": len(pairs),
        "num_ids": num_ids,
        "num_transitions": num_transitions,
        "subsets": int(subsets),
        "subset_size": resolved_subset_size,
        "seed": int(seed),
        "metrics": metrics.get("metadata", {}),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        **summary,
        "selection": selection,
        "selection_path": selection_path,
        "summary_path": summary_path,
        "metrics_config": prepared_config,
    }


evaluate_aging_checkpoint = evaluate_aging
evaluate_kid_dataset = evaluate_kid
