# Training Ablation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible API for training ablations 0–5 that reuses the existing training stack and independently evaluates a fixed longitudinal panel after every epoch.

**Architecture:** A canonical immutable configuration is resolved per case, then a runner constructs the existing loaders, diffusion bundle, loss, and `TRAIN_AGGING_MODEL`. The trainer gains one optional epoch-end callback; an ablation evaluator uses it to generate a fixed panel and call `src.quantitative_metrics` without duplicating metrics or training logic.

**Tech Stack:** Python, PyTorch, pandas, existing diffusion/training/inference APIs, pytest, Jupyter.

**Spec:** `planing/17_ABLATION_STUDIES_SPEC.md`

## Global Constraints

- Implement training cases `0–5`; reject inference cases `101–104` as deferred.
- Preserve the canonical 400px, DoRA, age-conditioner-v2, bidirectional, FG-NET baseline unless a case or explicit high-level override changes it.
- Reuse one fixed deterministic validation panel, independent AdaFace/DEX/KID evaluator, inference seed, and inference settings across cases.
- Never download models in tests or notebook construction.
- Save the resolved config, metadata, prediction manifests, epoch metrics, and final metrics.

## Review Focus

- Case 2 must contain no adapter parameters or empty optimizer group.
- Case configs must be independent deep copies and change only their intended mechanism.
- A resumed/multi-case run must reject a different fixed panel.
- Inference or metric failure must remain visible in manifests rather than silently dropping a sample.
- One-pair evaluation must report unavailable/noisy KID honestly rather than failing the training run.

---

### Task 1: Native no-adapter model path

**Files:** Modify `src/model/load_diffusion_models.py`; test `tests/test_model_adapters.py`.

**Interfaces:** `adapter_type="none"` keeps expanded `conv_in` and the optional age conditioner trainable, freezes original attention, and omits an adapter optimizer group.

- [ ] Write and run failing structural/optimizer tests.
- [ ] Implement the no-adapter branch and run model tests.

### Task 2: Epoch-end training hook

**Files:** Modify `src/training/train_face_aging.py`; test `tests/test_training_sampling_and_dropout.py`.

**Interfaces:** `train_model(..., epoch_end_callback=None)` invokes the callback after validation and normal monitoring with the live bundle and epoch context, and stores its serializable report in epoch history.

- [ ] Write and run a failing callback test.
- [ ] Implement the hook and run training tests.

### Task 3: Ablation configuration and fixed panel

**Files:** Create `src/ablation_studies/config.py`, `src/ablation_studies/panel.py`, `src/ablation_studies/__init__.py`; test `tests/test_training_ablations.py`.

**Interfaces:** `canonical_training_ablation_config()`, `resolve_training_ablation_config(case, overrides)`, and `build_fixed_evaluation_panel(...)` return independent, JSON-safe values and validate same-person pairs.

- [ ] Write and run failing case/panel tests.
- [ ] Implement configs, scientific guards, and panel persistence; run tests.

### Task 4: Epoch evaluator and high-level runner

**Files:** Create `src/ablation_studies/evaluation.py`, `src/ablation_studies/runner.py`; modify `src/__init__.py`; test `tests/test_training_ablations.py`.

**Interfaces:** `ablation_studies(case, ..., metrics_config, evaluation_pairs, ...)` runs cases sequentially through existing APIs, appends `epoch_metrics.csv`, performs final evaluation, saves metadata, and returns lightweight reports.

- [ ] Write and run failing evaluator/runner tests with only external model loading mocked.
- [ ] Implement orchestration, output isolation, resume guards, and sequential cleanup; run tests.

### Task 5: Notebook and verification

**Files:** Create `notebooks/training_ablations.ipynb`; test `tests/test_training_ablations_notebook.py`.

**Interfaces:** The notebook imports the public API and demonstrates a smoke run and cases 0–5 using two same-person longitudinal paths/URLs plus local metric checkpoints.

- [ ] Write and run a failing notebook contract test.
- [ ] Create the notebook, run focused and full test suites, then request a fresh code review.
