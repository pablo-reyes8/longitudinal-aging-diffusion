from __future__ import annotations

import json
from pathlib import Path


def _source(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_standalone_metrics_notebook_documents_manifest_evaluation():
    source = _source("notebooks/quantitative_metrics.ipynb")
    assert "load_quantitative_metrics" in source
    assert "evaluate_aging_outputs" in source
    assert "prediction_manifest.csv" in source
    assert "same person" in source.lower()


def test_comparison_notebook_is_sweeps_only_and_paper_metrics_are_separate():
    source = _source("notebooks/Model_comparisong.ipynb")
    assert "load_metrics=False" in source
    assert "metrics=False" in source
    assert "evaluate_test_set(" not in source
    paper = _source("notebooks/Paper_comparison_bias_v2.ipynb")
    assert "run_paper_analysis" in paper and "run_sd15_before_after" in paper
    # Closed module-1 bundles use observed target errors, not the old D/pi
    # anchor gate. The notebook checks real zones before any generation.
    assert "preflight_zone_protocol" in paper


def test_ablation_metrics_notebook_keeps_training_resolution_for_inference():
    source = _source("output/abliations/Abliations_metrics_paper.ipynb")
    assert '"image_size": 400' in source


def test_ablation_master_runs_baseline_phase_before_ablations():
    source = _source("output/abliations/Abliations_metrics_paper.ipynb")
    assert "baseline_first=True" in source
    assert "all_manifests.extend(_run_baseline_phase" in source


def test_ablation_comparison_keeps_metric_backends_on_cpu():
    source = _source("output/abliations/Abliations_metrics_paper.ipynb")
    assert 'device="cpu"' in source
