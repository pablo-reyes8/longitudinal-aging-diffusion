from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

import torch

from scripts.common import load_yaml, model_builder_kwargs, resolve_dtype
from scripts.infer import build_parser as build_infer_parser
from scripts.prepare_data import build_parser as build_data_parser
from scripts.train import build_parser as build_train_parser


ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_and_required_quality_files_exist():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "longitudinal-face-aging"
    assert set(metadata["project"]["scripts"]) == {"aging-data", "aging-train", "aging-infer"}
    for name in ("LICENSE", "CONTRIBUTING.md", "Dockerfile", "compose.yaml", ".dockerignore"):
        assert (ROOT / name).is_file()


def test_all_default_yaml_configs_are_mappings_and_model_dtype_is_resolved():
    for path in sorted((ROOT / "config").rglob("*.yaml")):
        assert load_yaml(path)
    model = model_builder_kwargs(load_yaml("config/models/sd15_lora.yaml"))
    assert model["dtype"] is None and model["device"] is None
    assert model["auxiliary_dtype"] is None
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16


def test_cli_parsers_accept_minimal_documented_invocations():
    data_args = build_data_parser().parse_args(["--dataset-root", "/data"])
    assert data_args.dataset_root == "/data"
    train_args = build_train_parser().parse_args(["--dataset-root", "/data"])
    assert train_args.dataset_root == "/data"
    infer_args = build_infer_parser().parse_args([
        "--checkpoint", "model.pt", "--image", "face.jpg",
        "--target-age", "65", "--output", "aged.png",
    ])
    assert infer_args.target_age == 65 and infer_args.target_prompt is None
