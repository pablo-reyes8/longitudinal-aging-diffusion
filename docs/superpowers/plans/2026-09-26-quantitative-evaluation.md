# Quantitative Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build independent AdaFace/DEX/KID evaluation and connect it to comparison inference, CLI, and notebooks.

**Architecture:** A manifest-first evaluator owns metric computation and persisted reports. Comparison inference only generates a manifest and calls it through an optional metrics bundle loaded from explicit local model paths.

**Tech Stack:** Python, PyTorch, pandas, Pillow, OpenCV DNN, TorchMetrics, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-quantitative-evaluation-design.md`

## Global Constraints

- Never download evaluator weights in project code or tests.
- Use AdaFace/DEX, not ArcFace/MiVOLO, for headline evaluation.
- Preserve failed and missing-ground-truth rows with explicit status.
- Keep the evaluator independent of diffusion inference.

## Review Focus

- Partial target-image mappings must not corrupt sample counts or KID matching.
- URL and local image inputs must hash and cache consistently.
- Detection failures must become NaN/status rows rather than disappearing.
- DEX probabilities must produce the exact expected age over classes 0–100.
- Comparison runs with `metrics=False` must remain backward compatible.

---

### Task 1: Manifest and numerical evaluator core

**Files:** Create `src/quantitative_metrics/*`; test `tests/test_quantitative_metrics.py`.

**Interfaces:** Produces `evaluate_aging_outputs`, `load_quantitative_metrics`, and manifest validation.

- [ ] Write failing numerical, partial-GT, failure-recording, and persistence tests.
- [ ] Run focused tests and verify failure because the package is absent.
- [ ] Implement injected backends, caching, AdaFace/DEX/KID adapters, summaries, and metadata.
- [ ] Run focused tests to green.

### Task 2: Comparison and CLI integration

**Files:** Modify `src/comparisong/load_models.py`, `src/comparisong/generate_images.py`, `scripts/compare_models.py`; test `tests/test_comparison_metrics_integration.py`.

**Interfaces:** Consumes Task 1 API; produces `load_metrics`, `metrics`, and `target_images` options.

- [ ] Write failing integration and CLI parsing tests.
- [ ] Verify red, implement thin manifest/evaluation integration, then verify green.

### Task 3: User documentation and notebooks

**Files:** Create `notebooks/quantitative_metrics.ipynb`; modify `notebooks/Model_comparisong.ipynb`, `pyproject.toml`, and README where needed.

- [ ] Add notebook-structure tests before notebook edits.
- [ ] Document local checkpoint paths, standalone evaluation, integrated metrics, and partial GT behavior.
- [ ] Run notebook JSON/API tests and the full suite.

