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


def test_comparison_notebook_enables_integrated_metrics():
    source = _source("notebooks/Model_comparisong.ipynb")
    assert "load_metrics=True" in source
    assert "metrics_config=METRICS_CONFIG" in source
    assert "metrics=True" in source
    assert "target_images=TARGET_IMAGES" in source
