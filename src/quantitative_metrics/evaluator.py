"""Manifest-first quantitative evaluator."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import load_prediction_manifest
from .image_io import ImageResolver


def _cosine(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _safe_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _json_records(frame: pd.DataFrame):
    return json.loads(frame.to_json(orient="records"))


def evaluate_aging_outputs(
    predictions,
    *,
    metrics_bundle,
    output_dir=None,
    run_name=None,
    compute_id_source=True,
    compute_id_gt=True,
    compute_age_mae=True,
    compute_kid=True,
    batch_size=32,
    device="auto",
    kid_subsets=100,
    kid_subset_size=None,
    kid_seed=2026,
    cache_embeddings=True,
    overwrite=False,
):
    """Evaluate saved generated images and persist reproducible reports."""
    del overwrite  # Reserved for persistent-cache policy without changing the public API.
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    manifest = load_prediction_manifest(predictions)
    output = Path(output_dir or "outputs/quantitative_metrics").expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "prediction_manifest.csv", index=False)

    required = set()
    if compute_id_source or compute_id_gt or compute_age_mae:
        required.add("aligner")
    if compute_id_source or compute_id_gt:
        required.add("identity_encoder")
    if compute_age_mae:
        required.add("age_estimator")
    if compute_kid:
        required.add("kid_metric")
    missing = sorted(required.difference(metrics_bundle))
    if missing:
        raise ValueError(f"metrics_bundle is missing: {', '.join(missing)}")

    resolver = ImageResolver()
    aligner = metrics_bundle.get("aligner")
    identity = metrics_bundle.get("identity_encoder")
    age_estimator = metrics_bundle.get("age_estimator")
    aligned_cache = {}
    embedding_cache = {}
    age_cache = {}
    failures = []

    def resolve_aligned(reference, sample_id, role):
        if reference is None or (not isinstance(reference, (str, Path)) and pd.isna(reference)):
            if role != "target":
                failures.append(
                    {"sample_id": sample_id, "stage": f"{role}_missing", "error": "missing_reference"}
                )
            return None, None, None
        try:
            image, digest = resolver.load(reference)
            aligned = None
            if aligner is not None:
                if digest not in aligned_cache:
                    aligned_cache[digest] = aligner(image)
                aligned = aligned_cache[digest]
                if aligned is None:
                    failures.append(
                        {"sample_id": sample_id, "stage": f"{role}_alignment", "error": "face_not_detected"}
                    )
            return aligned, digest, image
        except Exception as error:
            failures.append({"sample_id": sample_id, "stage": f"{role}_load", "error": str(error)})
            return None, None, None

    prepared = []
    identity_inputs = {}
    age_inputs = {}
    digest_samples = {}
    for row_index, source_row in enumerate(manifest.to_dict(orient="records")):
        sample_id = str(source_row["sample_id"])
        source, source_hash, source_raw = resolve_aligned(
            source_row["source_path"], sample_id, "source"
        )
        generated, generated_hash, generated_raw = resolve_aligned(
            source_row["generated_path"], sample_id, "generated"
        )
        target, target_hash, target_raw = resolve_aligned(
            source_row.get("target_path"), sample_id, "target"
        )
        for digest in (source_hash, generated_hash, target_hash):
            if digest is not None:
                digest_samples.setdefault(digest, set()).add(sample_id)
        prepared.append(
            {
                "index": row_index,
                "row": source_row,
                "source": source,
                "source_hash": source_hash,
                "generated": generated,
                "generated_hash": generated_hash,
                "target": target,
                "target_hash": target_hash,
                "source_raw": source_raw,
                "generated_raw": generated_raw,
                "target_raw": target_raw,
            }
        )
        if compute_id_source:
            for image, digest in ((source, source_hash), (generated, generated_hash)):
                if image is not None:
                    identity_inputs.setdefault(digest, image)
        if compute_id_gt:
            for image, digest in ((source, source_hash), (target, target_hash), (generated, generated_hash)):
                if image is not None:
                    identity_inputs.setdefault(digest, image)
        if compute_age_mae and generated is not None:
            age_inputs.setdefault(generated_hash, generated)

    def run_backend(items, backend_fn, stage, destination, convert):
        def run_chunk(chunk):
            try:
                values = backend_fn([image for _, image in chunk])
                if len(values) != len(chunk):
                    raise RuntimeError("backend returned a different number of outputs than inputs")
                for (digest, _), value in zip(chunk, values):
                    destination[digest] = convert(value)
            except Exception as error:
                if len(chunk) > 1:
                    middle = len(chunk) // 2
                    run_chunk(chunk[:middle])
                    run_chunk(chunk[middle:])
                    return
                digest = chunk[0][0]
                for failed_sample_id in sorted(digest_samples.get(digest, {"unknown"})):
                    failures.append(
                        {"sample_id": failed_sample_id, "stage": stage, "error": str(error)}
                    )

        for start in range(0, len(items), batch_size):
            run_chunk(items[start:start + batch_size])

    run_backend(
        list(identity_inputs.items()), identity.embed_batch if identity is not None else lambda _: [],
        "identity_backend", embedding_cache, lambda value: np.asarray(value, dtype=np.float64)
    )
    run_backend(
        list(age_inputs.items()), age_estimator.predict_batch if age_estimator is not None else lambda _: [],
        "age_backend", age_cache, float
    )

    rows = []
    kid_images = {}
    for item in prepared:
        row_index = item["index"]
        source_row = item["row"]
        source, source_hash = item["source"], item["source_hash"]
        generated, generated_hash = item["generated"], item["generated_hash"]
        target, target_hash = item["target"], item["target_hash"]
        generated_raw, target_raw = item["generated_raw"], item["target_raw"]
        identity_ok = source is not None and generated is not None
        age_ok = generated is not None
        record = dict(source_row)
        record.update(
            adaface_cosine_source=np.nan,
            adaface_cosine_gt=np.nan,
            adaface_real_source_target_cosine=np.nan,
            dex_predicted_age=np.nan,
            dex_age_abs_error=np.nan,
            identity_face_detected=identity_ok,
            age_face_detected=age_ok,
            source_sha256=source_hash,
            target_sha256=target_hash,
            generated_sha256=generated_hash,
        )
        if compute_id_source and identity_ok and source_hash in embedding_cache and generated_hash in embedding_cache:
            record["adaface_cosine_source"] = _cosine(
                embedding_cache[source_hash], embedding_cache[generated_hash]
            )
        if (
            compute_id_gt and target is not None and generated is not None
            and target_hash in embedding_cache and generated_hash in embedding_cache
        ):
            record["adaface_cosine_gt"] = _cosine(
                embedding_cache[target_hash], embedding_cache[generated_hash]
            )
            if source is not None and source_hash in embedding_cache:
                record["adaface_real_source_target_cosine"] = _cosine(
                    embedding_cache[source_hash], embedding_cache[target_hash]
                )
        if compute_age_mae and age_ok and generated_hash in age_cache:
            predicted = age_cache[generated_hash]
            record["dex_predicted_age"] = predicted
            record["dex_age_abs_error"] = abs(predicted - float(source_row["target_age"]))
        if compute_kid and target_raw is not None and generated_raw is not None:
            kid_images[row_index] = (target_raw, generated_raw)
        rows.append(record)

    per_sample = pd.DataFrame(rows)
    group_columns = ["model_name"] + [
        name for name in ("ablation_case", "epoch") if name in per_sample.columns
    ]
    grouped = list(per_sample.groupby(group_columns, dropna=False, sort=False))
    target_sets = []
    target_signatures = []
    for _, group in grouped:
        has_target = group["target_path"].notna() & group["target_path"].astype(str).ne("")
        target_sets.append(frozenset(group.loc[has_target, "sample_id"].astype(str)))
        target_signatures.append(
            frozenset(
                zip(
                    group.loc[has_target, "sample_id"].astype(str),
                    group.loc[has_target, "target_sha256"].astype(str),
                )
            )
        )
    comparable_target_sets = len(set(target_sets)) <= 1 and len(set(target_signatures)) <= 1
    summary_rows = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = dict(zip(group_columns, keys))
        pairs = [kid_images[index] for index in group.index if index in kid_images]
        has_target = group["target_path"].notna() & group["target_path"].astype(str).ne("")
        n_target_expected = int(has_target.sum())
        n_target_available = int(group.loc[has_target, "target_sha256"].notna().sum())
        kid_mean = kid_std = np.nan
        resolved_subset_size = None
        kid_status = "disabled" if not compute_kid else "insufficient_samples"
        if compute_kid and not comparable_target_sets:
            kid_status = "incomparable_target_set"
        elif compute_kid and n_target_available < n_target_expected:
            kid_status = "incomplete_target_set"
        elif compute_kid and len(pairs) < n_target_available and n_target_available >= 2:
            kid_status = "incomplete_generated_set"
        elif compute_kid and len(pairs) >= 2:
            real, generated = zip(*pairs)
            resolved_subset_size = min(50, len(pairs)) if kid_subset_size is None else min(
                int(kid_subset_size), len(pairs)
            )
            try:
                kid_mean, kid_std = metrics_bundle["kid_metric"].compute(
                    list(real), list(generated), subsets=kid_subsets,
                    subset_size=resolved_subset_size, seed=kid_seed
                )
                kid_status = "ok"
            except Exception as error:
                kid_status = "backend_error"
                for failed_sample_id in group["sample_id"].astype(str):
                    failures.append(
                        {"sample_id": failed_sample_id, "stage": "kid_backend", "error": str(error)}
                    )
        summary.update(
            n_samples=len(group),
            n_valid_identity=int(group["adaface_cosine_source"].notna().sum()),
            n_valid_identity_gt=int(group["adaface_cosine_gt"].notna().sum()),
            n_valid_age=int(group["dex_age_abs_error"].notna().sum()),
            n_valid_kid=len(pairs),
            n_expected_kid=n_target_expected,
            adaface_id_source_mean=group["adaface_cosine_source"].mean(),
            adaface_id_gt_mean=group["adaface_cosine_gt"].mean(),
            dex_age_mae=group["dex_age_abs_error"].mean(),
            kid_mean=kid_mean,
            kid_std=kid_std,
            kid_status=kid_status,
            kid_subset_size=resolved_subset_size,
        )
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    failure_frame = pd.DataFrame(failures, columns=["sample_id", "stage", "error"])

    per_sample.to_csv(output / "per_sample_metrics.csv", index=False)
    summary_frame.to_csv(output / "metrics_summary.csv", index=False)
    failure_frame.to_csv(output / "failure_log.csv", index=False)
    (output / "metrics_summary.json").write_text(
        json.dumps(_json_records(summary_frame), indent=2), encoding="utf-8"
    )
    sample_hash = hashlib.sha256(
        "\n".join(per_sample["sample_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    metadata = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": metrics_bundle.get("metadata", {}),
        "n_samples": len(per_sample),
        "sample_ids_sha256": sample_hash,
        "same_identity_target_contract": True,
        "kid": {"subsets": kid_subsets, "subset_size": kid_subset_size, "seed": kid_seed},
        "cache_embeddings": bool(cache_embeddings),
        "batch_size": batch_size,
        "requested_device": str(device),
        "python": platform.python_version(),
        "pytorch": _safe_version("torch"),
        "torchmetrics": _safe_version("torchmetrics"),
        "git_commit": _git_commit(),
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "per_sample": per_sample,
        "summary": summary_frame,
        "failures": failure_frame,
        "metadata": metadata,
        "output_dir": output,
        "paths": {
            "manifest": output / "prediction_manifest.csv",
            "per_sample": output / "per_sample_metrics.csv",
            "summary_csv": output / "metrics_summary.csv",
            "summary_json": output / "metrics_summary.json",
            "failures": output / "failure_log.csv",
            "metadata": output / "evaluation_metadata.json",
        },
    }
