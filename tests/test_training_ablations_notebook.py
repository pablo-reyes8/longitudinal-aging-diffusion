from __future__ import annotations

import json
from pathlib import Path


def test_training_ablation_notebook_is_a_thin_public_api_example():
    path = Path("notebooks/training_ablations.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "from src.ablation_studies import ablation_studies" in source
    assert "case=[0, 1, 2, 3, 4, 5]" in source
    assert "evaluation_source_image" in source and "evaluation_target_image" in source
    assert "adaface_checkpoint_path" in source and "dex_checkpoint_path" in source
    assert "KID" in source and ">=2" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
