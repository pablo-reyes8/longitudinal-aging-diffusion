# Quantitative Evaluation Design

## Purpose

Add an evaluator independent from training and candidate selection. It reports exactly four paper metrics: AdaFace ID-Source, AdaFace ID-GT, DEX Age MAE, and KID. ArcFace and MiVOLO are not used by this module.

## Architecture

- `src.quantitative_metrics` accepts a DataFrame or manifest path and does not depend on a diffusion model.
- The comparison API builds the same manifest from generated outputs and optionally evaluates it through `age_image(..., metrics=True)`.
- AdaFace and DEX are loaded from explicit local paths. No evaluator weights are downloaded by project code.
- One shared AdaFace-compatible face alignment is applied to source, target, and generated images before both identity and age evaluation.
- Backends are injectable so numerical contracts can be tested without heavyweight checkpoints.

## Public API

`load_quantitative_metrics(...)` loads frozen AdaFace and DEX backends plus a lazy KID backend.

`evaluate_aging_outputs(predictions, metrics_bundle, output_dir, ...)` validates a standard manifest, computes per-sample fields and aggregate rows, records failures, caches by image SHA-256, and writes CSV/JSON metadata.

`load_aging_models(..., load_metrics=True, metrics_config={...})` stores the evaluator bundle under `bundle["quantitative_metrics"]`.

`age_image(..., metrics=True, target_images={age: path_or_url})` builds `prediction_manifest.csv`, invokes the independent evaluator, and returns output paths/tables in `results["metrics"]`. Missing target images leave ID-GT and KID unavailable with explicit counts/status rather than silently dropping samples.

## Scientific and failure rules

- `target_images` must be real longitudinal images of the same identity; this is a caller contract and is recorded in metadata.
- AdaFace uses the official 112×112 aligned BGR, `(x / 255 - 0.5) / 0.5` preprocessing.
- DEX uses the official Caffe checkpoint, softmax over ages 0–100, and expected age.
- KID uses TorchMetrics Inception features on exactly matched generated/real-target rows; fewer than two valid pairs returns unavailable status.
- Every failed load/alignment/inference remains represented in per-sample output and `failure_log.csv`.
- Output includes manifests, per-sample CSV, summary CSV/JSON, failure log, and reproducibility metadata with hashes and versions.

