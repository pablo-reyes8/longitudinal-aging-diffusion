"""Longitudinal test-set comparison using the existing model and metric APIs."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image
import torch

from .generate_images import age_image
from .load_models import load_aging_models
from src.quantitative_metrics import (
    evaluate_aging_outputs,
    evaluate_aging_inference_picking,
    load_quantitative_metrics,
    prepare_quantitative_metrics,
)


MODEL_NAMES = ("SAM", "FRAN", "FADING", "CUSP", "HRFAE", "Cradle2Cane")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def select_test_transitions(test_root, *, max_transitions_per_id=5, direction="aging"):
    """Select largest age gaps within each identity, deterministically.

    ``35_1.jpg`` is an additional photograph at age 35. Prefer ``35.jpg``
    when present and select one image per age to avoid duplicate age pairs.
    ``both`` targets 70% aging / 30% rejuvenation across the selected panel.
    Rounding is balanced across identities (e.g. 4/1 then 3/2 for five pairs).
    Keep both directions per identity when the cap and available pairs permit.
    """
    root = Path(test_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {root}")
    if isinstance(max_transitions_per_id, bool) or not isinstance(max_transitions_per_id, int) or max_transitions_per_id < 1:
        raise ValueError("max_transitions_per_id must be a positive integer")
    if direction not in {"aging", "rejuvenation", "both"}:
        raise ValueError("direction must be 'aging', 'rejuvenation' or 'both'")
    rows = []
    aging_selected = 0
    for folder in sorted(root.glob("id_*")):
        if not folder.is_dir():
            continue
        observations = {}
        paths = sorted(folder.iterdir(), key=lambda p: ("_" in p.stem, p.name))
        for path in paths:
            match = re.fullmatch(r"(\d{1,3})(?:_\d+)?", path.stem)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and match:
                age = int(match[1])
                if 0 <= age <= 120:
                    observations.setdefault(age, path)
        pairs = [
            (source, target) for source in sorted(observations) for target in sorted(observations)
            if source != target
            and (direction == "both" or (target > source if direction == "aging" else target < source))
        ]
        pairs.sort(key=lambda pair: (-abs(pair[1] - pair[0]), pair[0], pair[1]))
        if direction == "both" and pairs:
            aging = [pair for pair in pairs if pair[1] > pair[0]]
            rejuvenation = [pair for pair in pairs if pair[1] < pair[0]]
            count = min(max_transitions_per_id, len(pairs))
            # Balance integer rounding over the whole panel rather than giving
            # every five-pair identity 4/1 (80%) or every identity 3/2 (60%).
            target_aging = (7 * (len(rows) + count) + 5) // 10
            minimum = max(1 if count >= 2 else 0, count - len(rejuvenation))
            maximum = min(count - 1 if count >= 2 else count, len(aging))
            aging_count = max(minimum, min(maximum, target_aging - aging_selected))
            pairs = aging[:aging_count] + rejuvenation[:count - aging_count]
            pairs.sort(key=lambda pair: (-abs(pair[1] - pair[0]), pair[0], pair[1]))
            aging_selected += aging_count
        for source, target in pairs[:max_transitions_per_id]:
            rows.append({
                "sample_id": f"{folder.name}__{source}_to_{target}",
                "identity_id": folder.name,
                "source_age": source, "target_age": target, "delta_age": target - source,
                "source_path": str(observations[source]),
                "target_path": str(observations[target]),
            })
    if not rows:
        raise ValueError("No valid longitudinal transitions; expected id_*/<age>.<jpg|png>")
    return pd.DataFrame(rows)


def _offload(bundle):
    if bundle is not None:
        for key in ("unet", "vae", "text_encoder", "age_delta_conditioner"):
            model = bundle.get(key)
            if model is not None:
                model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _smoke_test_metrics(samples, metrics_bundle, output_dir, *, seed, batch_size=32):
    """Exercise real metric forwards before any model generation.

    Real input photographs stand in for generated images only in this smoke
    report. Its numbers are never included in the comparison or paper CSVs.
    Two distinct photographs allow KID to run even for a one-transition test.
    """
    print("[metrics smoke] Checking AdaFace, DEX and KID before generation...", flush=True)
    references = {}
    for sample in samples.to_dict("records"):
        for key in ("source_path", "target_path"):
            references.setdefault(sample[key], sample)
        if len(references) >= 2:
            break
    rows = [
        {**sample, "sample_id": f"metrics_smoke_{index}",
         "generated_path": path, "model_name": "__metrics_smoke__"}
        for index, (path, sample) in enumerate(list(references.items())[:2])
    ]
    report = evaluate_aging_outputs(
        pd.DataFrame(rows), metrics_bundle=metrics_bundle, output_dir=output_dir,
        compute_kid=True, kid_subsets=1, kid_subset_size=2, kid_seed=seed,
        batch_size=batch_size,
    )
    columns = ["adaface_cosine_source", "adaface_cosine_gt", "dex_age_abs_error"]
    individual_ok = all(
        pd.notna(value) and math.isfinite(float(value))
        for value in report["per_sample"][columns].to_numpy().flat
    )
    summary = report["summary"].iloc[0]
    kid_ok = summary["kid_status"] == "ok" and math.isfinite(float(summary["kid_mean"]))
    if not individual_ok or not kid_ok:
        errors = report["failures"].head(3).to_dict("records")
        raise RuntimeError(
            f"Metrics smoke failed before any generation: {errors}. "
            f"See {output_dir} (KID status: {summary['kid_status']})."
        )
    print("[metrics smoke] OK: AdaFace ID-Source / ID-GT, DEX and KID", flush=True)


def evaluate_test_set(
    test_root,
    checkpoint_path,
    output_dir,
    *,
    models=("SAM", "FRAN", "FADING", "ours"),
    comparison_bundle=None,
    model_loader_config=None,
    metrics_bundle=None,
    metrics_config=None,
    metrics_root="/content/evaluation_models",
    max_transitions_per_id=5,
    direction="aging",
    gender="male",
    gender_by_id=None,
    seed=2026,
    device=None,
    local_files_only=False,
    our_mode="both",
    picking_max_trials=8,
    our_inference_config=None,
    target_age_strength_map=None,
    diagnostic_config=None,
    evaluation_config=None,
    resume=True,
):
    """Compare literature models and an adapter on one shared longitudinal panel.

    Model setup reuses ``load_aging_models`` (including its compatibility fixes),
    or accepts its existing bundle. Metrics load on CPU by default. ``ours``
    expands ``ours`` to two reported models by default: ``ours_strength_X`` (configured
    direct inference) and ``ours_picking`` (iterative smart strength search,
    at most eight trials per transition). Both generate at 400px by default.
    ``our_mode='direct'`` or ``'picking'`` evaluates just that variant.
    Supply gender_by_id for models needing gender; labels are never inferred.

    A mandatory metric smoke test runs on real input photographs before any
    generation, including actual AdaFace/DEX and KID forwards. Then, up to two *real* transitions run through
    every selected model, AdaFace/DEX and a small KID check. Completed generations persist immediately;
    resume only reuses them when inputs and configuration match. KID is computed
    once per model over the complete common set, never per identity.
    """
    lookup = {name.lower(): name for name in MODEL_NAMES}
    lookup["ours"] = "ours"
    lookup["ours_picking"] = "ours_picking"
    requested = [models] if isinstance(models, str) else list(models)
    if not requested or any(str(name).lower() not in lookup for name in requested):
        raise ValueError(f"models must select from {(*MODEL_NAMES, 'ours', 'ours_picking')}")
    selected = list(dict.fromkeys(lookup[str(name).lower()] for name in requested))
    if our_mode not in {"direct", "picking", "both"}:
        raise ValueError("our_mode must be 'direct', 'picking' or 'both'")
    if isinstance(picking_max_trials, bool) or not isinstance(picking_max_trials, int) or not 1 <= picking_max_trials <= 20:
        raise ValueError("picking_max_trials must be an integer in [1, 20]")
    expanded = []
    for name in selected:
        if name == "ours":
            expanded.extend(["ours", "ours_picking"] if our_mode == "both" else
                            ["ours_picking"] if our_mode == "picking" else ["ours"])
        else:
            expanded.append(name)
    selected = list(dict.fromkeys(expanded))
    own_names = {"ours", "ours_picking"}
    options = dict(evaluation_config or {})
    options.update(compute_id_source=True, compute_id_gt=True, compute_age_mae=True, compute_kid=True)
    options.setdefault("kid_seed", seed)
    options.pop("output_dir", None)
    # Reject invalid metric options before any expensive generation work.
    inspect.signature(evaluate_aging_outputs).bind(
        pd.DataFrame(), metrics_bundle=metrics_bundle, output_dir=output_dir, **options,
    )
    if int(options.get("batch_size", 32)) < 1 or int(options.get("kid_subsets", 100)) < 1:
        raise ValueError("batch_size and kid_subsets must be positive")
    if options.get("kid_subset_size") is not None and int(options["kid_subset_size"]) < 2:
        raise ValueError("kid_subset_size must be at least two")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if own_names.intersection(selected) and not checkpoint.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {checkpoint}")
    samples = select_test_transitions(
        test_root, max_transitions_per_id=max_transitions_per_id, direction=direction,
    )
    genders = {row.identity_id: (gender_by_id or {}).get(row.identity_id, gender) for row in samples.itertuples()}
    if any(value not in {"male", "female"} for value in genders.values()):
        raise ValueError("gender and gender_by_id values must be 'male' or 'female'")
    inference = {"image_size": 400, "num_inference_steps": 40, "strength": 0.35, "seed": seed}
    inference.update(our_inference_config or {})
    if not math.isfinite(float(inference["strength"])) or not 0 < float(inference["strength"]) <= 1:
        raise ValueError("our_inference_config strength must be finite and in (0, 1]")
    protected = {"bundle", "image", "source_age", "target_age", "device", "compute_diagnostics", "return_dict", "output_type"}
    if protected.intersection(inference):
        raise ValueError(f"our_inference_config cannot override {sorted(protected.intersection(inference))}")
    if target_age_strength_map is not None:
        policy = {float(age): float(value) for age, value in target_age_strength_map.items()}
        missing = set(samples.target_age).difference(policy)
        if missing or any(not 0 < value <= 1 for value in policy.values()):
            raise ValueError(f"Invalid strength map or missing target ages: {sorted(missing)}")
    else:
        policy = None
    direct_label = (
        "ours_strength_map" if policy is not None else
        "ours_strength_adaptive" if inference.get("use_delta_dependent_strength", False) else
        f"ours_strength_{float(inference['strength']):g}"
    )
    report_names = {name: direct_label if name == "ours" else name for name in selected}
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Include file fingerprints so resume cannot silently mix different inputs.
    files = sorted(set(samples.source_path) | set(samples.target_path))
    if own_names.intersection(selected):
        files.append(str(checkpoint))
    settings = {
        "samples": samples.to_dict("records"), "models": selected,
        "files": [(name, Path(name).stat().st_size, Path(name).stat().st_mtime_ns) for name in files],
        "inference": inference, "policy": policy, "mode": our_mode,
        "picking_max_trials": picking_max_trials,
        "picking_search_method": "smart",
        "diagnostic": diagnostic_config, "genders": genders, "seed": seed,
        "loader": model_loader_config,
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    run_file = output / "run_config.json"
    if resume and run_file.exists() and json.loads(run_file.read_text())["signature"] != signature:
        raise ValueError("Output directory contains another protocol; choose a new output_dir or resume=False")
    run_file.write_text(json.dumps({"signature": signature, "settings": settings}, indent=2, default=str))
    samples.to_csv(output / "transitions.csv", index=False)
    external = [name for name in selected if name not in own_names]
    if metrics_bundle is None and comparison_bundle is not None:
        metrics_bundle = comparison_bundle.get("quantitative_metrics")
    if metrics_bundle is None:
        config = prepare_quantitative_metrics(metrics_root, device="cpu", metrics_config=metrics_config)
        metrics_bundle = load_quantitative_metrics(**config)
    _smoke_test_metrics(
        samples, metrics_bundle, output / "preflight" / "metrics",
        seed=options["kid_seed"], batch_size=options.get("batch_size", 32),
    )
    if external and comparison_bundle is None:
        loader_options = dict(model_loader_config or {})
        loader_options.update(models=external, load_metrics=False)
        comparison_bundle = load_aging_models(**loader_options)
    for name in external:
        if comparison_bundle.get(name) is None:
            raise RuntimeError(f"{name} was not prepared by load_aging_models; check its setup log")

    from src.inference import infer_face_aging_direct, load_face_aging_inference_bundle

    own_bundle = None
    manifest_rows = []
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate(name, sample):
        nonlocal own_bundle
        report_name = report_names[name]
        destination = output / "generated" / report_name / sample["identity_id"] / f"{sample['sample_id']}.png"
        marker = destination.with_suffix(".json")
        valid = False
        if resume and destination.is_file() and marker.is_file():
            valid = json.loads(marker.read_text()).get("signature") == signature
            if valid:
                with Image.open(destination) as opened:
                    opened.verify()
        if not valid:
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{report_name}] {sample['sample_id']}", flush=True)
            if name in own_names:
                if own_bundle is None:
                    own_bundle = load_face_aging_inference_bundle(
                        checkpoint, device="cpu", local_files_only=local_files_only,
                        load_auxiliary_models=False,
                    )
                options = dict(inference)
                strength = policy[float(sample["target_age"])] if policy is not None else options["strength"]
                options["strength"] = strength
                if name == "ours_picking":
                    common_keys = {
                        "image_size", "num_inference_steps", "seed", "text_reference_mode",
                        "age_guidance_scale", "text_guidance_scale", "image_guidance_scale",
                        "negative_prompt", "prompt_style", "use_cfg", "source_prompt",
                    }
                    options = {key: value for key, value in options.items() if key in common_keys}
                    options.update(diagnostic_config or {})
                    # Budget is explicit and applies independently per transition.
                    options["max_trials_per_target"] = picking_max_trials
                    options.setdefault("generate_assisted_prompt_variant", False)
                    # Keep the entire owned bundle together on its chosen device.
                    for key in ("unet", "vae", "text_encoder", "age_delta_conditioner"):
                        if own_bundle.get(key) is not None:
                            own_bundle[key].to(resolved_device)
                    result = evaluate_aging_inference_picking(
                        checkpoint, sample["source_path"], sample["source_age"], [sample["target_age"]],
                        target_images={sample["target_age"]: sample["target_path"]},
                        target_age_strength_map={sample["target_age"]: strength},
                        metrics_bundle=metrics_bundle, inference_bundle=own_bundle,
                        diagnostic_backend="metrics", diagnostic_config=options,
                        search_method="smart", model_name=name,
                        output_dir=output / "picking" / sample["sample_id"], download_assets=False,
                    )
                    shutil.copyfile(next(iter(result["generated_paths"].values())), destination)
                else:
                    result = infer_face_aging_direct(
                        bundle=own_bundle, image=sample["source_path"],
                        source_age=sample["source_age"], target_age=sample["target_age"],
                        device=resolved_device, compute_diagnostics=False, output_type="pil", return_dict=True,
                        **options,
                    )
                    result["image"].save(destination)
            else:
                work = output / "literature" / name / sample["sample_id"]
                result = age_image(
                    comparison_bundle, image=sample["source_path"], source_age=sample["source_age"],
                    target_ages=[sample["target_age"]], models=[name], gender=genders[sample["identity_id"]],
                    seed=seed, show=False, strict=True, metrics=False, output_dir=str(work),
                )
                generated = result.get(name, {}).get(sample["target_age"])
                if not isinstance(generated, Image.Image):
                    raise RuntimeError(f"{name} produced no image for {sample['sample_id']}: {result.get('status')}")
                generated.save(destination)
            marker.write_text(json.dumps({"signature": signature}))
        row = {**sample, "model_name": report_name, "generated_path": str(destination)}
        manifest_rows.append(row)
        pd.DataFrame(manifest_rows).to_csv(output / "prediction_manifest.csv", index=False)
        return row

    try:
        records = samples.to_dict("records")
        # KID needs two samples. Test its real backend too, rather than waiting
        # for the end of a potentially expensive run to discover missing assets.
        probe_size = min(2, len(records))
        for name in selected:
            probe_rows = [generate(name, sample) for sample in records[:probe_size]]
            probe = evaluate_aging_outputs(
                pd.DataFrame(probe_rows), metrics_bundle=metrics_bundle,
                output_dir=output / "preflight" / name, compute_kid=(probe_size >= 2),
                kid_subsets=1, kid_subset_size=2, kid_seed=seed,
            )
            values = probe["per_sample"]
            if not values[[
                "adaface_cosine_source", "adaface_cosine_gt", "dex_age_abs_error",
            ]].notna().all().all():
                raise RuntimeError(f"Preflight metrics failed for {name}; see {output / 'preflight' / name}")
            if probe_size >= 2 and probe["summary"].iloc[0]["kid_status"] != "ok":
                raise RuntimeError(f"Preflight KID failed for {name}; see {output / 'preflight' / name}")
            _offload(own_bundle)
        for name in selected:
            for sample in records[probe_size:]:
                generate(name, sample)
            _offload(own_bundle)
        evaluation = evaluate_aging_outputs(
            pd.DataFrame(manifest_rows), metrics_bundle=metrics_bundle,
            output_dir=output / "metrics", **options,
        )
        metric_columns = ["adaface_cosine_source", "adaface_cosine_gt", "dex_predicted_age", "dex_age_abs_error"]
        wide = evaluation["per_sample"].pivot(index="sample_id", columns="model_name", values=metric_columns)
        wide.columns = [f"{model}__{metric}" for metric, model in wide.columns]
        transitions = samples.merge(wide.reset_index(), on="sample_id", validate="one_to_one")
        transitions.to_csv(output / "transition_metrics.csv", index=False)
        evaluation["summary"].to_csv(output / "paper_metrics.csv", index=False)
        print(evaluation["summary"].to_string(index=False))
        return {
            "transitions": samples, "per_transition": transitions,
            "summary": evaluation["summary"], "evaluation": evaluation,
            "paths": {"transitions": output / "transition_metrics.csv", "paper": output / "paper_metrics.csv"},
        }
    finally:
        _offload(own_bundle)
