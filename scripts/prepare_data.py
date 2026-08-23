"""Build and validate longitudinal data indexes from the command line."""
# ruff: noqa: E402 -- direct-file execution bootstraps the project root.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_face_aging_dataloaders, run_data_pipeline_validation

from scripts.common import load_yaml, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing one folder per identity")
    parser.add_argument("--config", default="config/data/default.yaml")
    parser.add_argument("--output-dir", default="artifacts/data", help="Manifest, splits, and report directory")
    parser.add_argument("--skip-image-validation", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.update({
        "manifest_path": output_dir / "manifest.csv",
        "split_path": output_dir / "splits.csv",
    })
    loaders, metadata = build_face_aging_dataloaders(args.dataset_root, **config)
    report = run_data_pipeline_validation(
        args.dataset_root,
        seed=config["seed"],
        split_ratios=config["split_ratios"],
        min_age_gap=config["min_age_gap"],
        max_age_gap=config.get("max_age_gap"),
        validate_images=not args.skip_image_validation,
        manifest_path=config["manifest_path"],
        split_path=config["split_path"],
    )
    report["loader_batches"] = {name: len(loader) for name, loader in loaders.items()}
    report["dataset_items"] = {name: len(metadata["datasets"][name]) for name in loaders}
    report_path = save_json(report, output_dir / "validation_report.json")
    print(f"Data validation: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"Report: {report_path.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
