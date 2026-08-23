"""Train the source-conditioned SD1.5 face-aging adapter from YAML configs."""
# ruff: noqa: E402 -- direct-file execution bootstraps the project root.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_face_aging_dataloaders
from src.loss import FaceAgingDiffusionLoss
from src.model import build_face_aging_diffusion_bundle
from src.training import TRAIN_AGGING_MODEL

from scripts.common import load_yaml, model_builder_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--data-config", default="config/data/default.yaml")
    parser.add_argument("--model-config", default="config/models/sd15_lora.yaml")
    parser.add_argument("--training-config", default="config/training/photo_editing.yaml")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint_dir from YAML")
    parser.add_argument("--resume-from", help="Override resume_from from YAML")
    parser.add_argument("--monitoring-image", help="Fixed image rendered at each sampling epoch")
    parser.add_argument("--local-files-only", action="store_true", help="Never access Hugging Face over network")
    return parser


def run(args: argparse.Namespace):
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)
    if args.local_files_only:
        model_config["local_files_only"] = True
    if args.checkpoint_dir:
        training_config["checkpoint_dir"] = args.checkpoint_dir
    if args.resume_from:
        training_config["resume_from"] = args.resume_from
    if args.monitoring_image:
        training_config["monitoring_image"] = args.monitoring_image

    loaders, metadata = build_face_aging_dataloaders(args.dataset_root, **data_config)
    bundle = build_face_aging_diffusion_bundle(**model_builder_kwargs(model_config))
    loss_config = dict(training_config.pop("loss", {}))
    if loss_config.get("identity_weight", 0) or loss_config.get("age_weight", 0):
        raise ValueError(
            "The generic CLI has no auxiliary checkpoint factories. Keep identity_weight and "
            "age_weight at zero, or construct their adapters through notebooks/training.ipynb."
        )
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=bundle["scheduler_train"], vae=bundle["vae"], **loss_config
    )
    training_config.setdefault("image_size", data_config.get("image_size"))
    training_config.setdefault("prompt_configuration", {
        "style": data_config.get("prompt_style"),
        "dynamic_person_word": data_config.get("dynamic_person_word", False),
    })
    print(f"Dataset: {metadata['root_dir']}")
    return TRAIN_AGGING_MODEL(
        bundle=bundle,
        loss_fn=loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        **training_config,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
