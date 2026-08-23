"""Run the full offline data/model/loss audit without downloading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data import build_face_aging_dataloaders
from model_fakes import make_fake_components
from src.loss import (
    AgeEstimatorAdapter,
    FaceAgingDiffusionLoss,
    IdentityEncoderAdapter,
    run_face_aging_loss_validation,
)
from src.model import assemble_face_aging_diffusion_bundle, prepare_face_aging_forward


class OfflineIdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class OfflineAgeEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).mean((1, 2, 3)) * 10.0 + 35.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", nargs="?", type=Path, default=REPOSITORY_ROOT / "data" / "sample")
    args = parser.parse_args()
    torch.manual_seed(913)
    loaders, _ = build_face_aging_dataloaders(
        args.root_dir,
        image_size=32,
        batch_size=2,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
    )
    batch = next(iter(loaders["train"]))
    bundle = assemble_face_aging_diffusion_bundle(
        make_fake_components(),
        model_id="offline-fake/sd15-architecture",
        rank=2,
        alpha=2,
        verbose=False,
    )
    prepared = prepare_face_aging_forward(
        bundle,
        batch["source_image"],
        batch["target_image"],
        batch["target_prompt"],
    )
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=bundle["scheduler_train"],
        vae=bundle["vae"],
        identity_encoder=IdentityEncoderAdapter(OfflineIdentityEncoder()),
        age_estimator=AgeEstimatorAdapter(OfflineAgeEstimator()),
    )
    forward_kwargs = {
        "model_pred": prepared["noise_pred"],
        "noise": prepared["noise"],
        "noisy_target_latents": prepared["noisy_target_latents"],
        "target_latents": prepared["target_latents"],
        "timesteps": prepared["timesteps"],
        "source_images": batch["source_image"],
        "target_images": batch["target_image"],
        "target_ages": batch["target_age"],
        "global_step": 0,
    }
    report = run_face_aging_loss_validation(loss_fn, forward_kwargs=forward_kwargs)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
