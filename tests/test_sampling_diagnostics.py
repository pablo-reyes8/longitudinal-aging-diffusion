from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch import nn

from src.inference import (
    compute_face_aging_diagnostics,
    diagnose_checkpoint_age_sweep,
    infer_face_aging_direct,
)
from src.loss import AgeEstimatorAdapter, IdentityEncoderAdapter
from src.training import atomic_torch_save, build_inference_payload
from src.training.sampling_monitor import run_face_aging_monitor
from training_fakes import make_training_bundle


class ChannelIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        return images.mean(dim=(2, 3)) * self.scale


class MeanAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(100.0))

    def forward(self, images):
        return images.mean(dim=(1, 2, 3)) * self.scale


def attach_diagnostics(bundle):
    bundle["identity_encoder"] = IdentityEncoderAdapter(ChannelIdentity())
    bundle["age_estimator"] = AgeEstimatorAdapter(MeanAge())
    return bundle


def test_diagnostic_returns_scalar_age_and_controlled_identity_ordering():
    bundle = attach_diagnostics(make_training_bundle())
    red = torch.zeros(1, 3, 20, 20)
    red[:, 0] = 1
    green = torch.zeros_like(red)
    green[:, 1] = 1
    same = compute_face_aging_diagnostics(bundle, red, red, 50, image_size=16)
    different = compute_face_aging_diagnostics(bundle, red, green, 50, image_size=16)
    assert isinstance(same["predicted_generated_age"], float)
    assert same["target_age"] == 50.0
    assert same["identity_cosine_source_generated"] > different["identity_cosine_source_generated"]


def test_inference_diagnostics_do_not_change_generated_output():
    bundle = attach_diagnostics(make_training_bundle(seed=880))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    kwargs = dict(
        bundle=bundle, image=image, target_age=65, source_age=30,
        num_inference_steps=4, strength=0.5, image_size=32, seed=91,
    )
    plain = infer_face_aging_direct(**kwargs, compute_diagnostics=False)
    diagnosed = infer_face_aging_direct(**kwargs, compute_diagnostics=True)
    assert torch.equal(plain["image_tensor"], diagnosed["image_tensor"])
    assert plain["diagnostics"] is None
    assert set(diagnosed["diagnostics"]) == {
        "target_age", "predicted_generated_age", "identity_cosine_source_generated"
    }


def test_training_sweep_writes_epoch_rows_and_appends_history(tmp_path):
    bundle = attach_diagnostics(make_training_bundle(seed=881))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    for epoch in (0, 1):
        report = run_face_aging_monitor(
            bundle=bundle, image=image, epoch=epoch, output_dir=tmp_path,
            target_age=[30, 50, 65], source_age=25,
            num_inference_steps=3, strength=0.5, seed=2026, image_size=32,
        )
        epoch_csv = tmp_path / f"epoch_{epoch + 1:03d}" / f"sampling_diagnostics_epoch_{epoch + 1:03d}.csv"
        frame = pd.read_csv(epoch_csv)
        assert len(frame) == 3
        assert frame["target_age"].tolist() == [30.0, 50.0, 65.0]
        assert report["diagnostics_csv"] == str(epoch_csv)
    history = pd.read_csv(tmp_path / "sampling_diagnostics_history.csv")
    assert len(history) == 6
    assert history.groupby("epoch").size().to_dict() == {1: 3, 2: 3}


def test_checkpoint_diagnostic_age_error_and_generation_match_normal_inference(tmp_path):
    original = make_training_bundle(seed=882)
    checkpoint = atomic_torch_save(
        build_inference_payload(original, {"image_size": 32}),
        tmp_path / "epoch_005" / "adapter_inference.pt",
    )
    rebuilt = attach_diagnostics(make_training_bundle(seed=882))
    image = Image.new("RGB", (38, 32), (110, 75, 55))
    output_dir = tmp_path / "diagnostic"
    frame = diagnose_checkpoint_age_sweep(
        checkpoint_path=checkpoint,
        bundle=rebuilt,
        source_image=image,
        source_age=26,
        target_ages=[35, 65],
        output_dir=output_dir,
        mode="direct",
        num_inference_steps=3,
        strength=0.5,
        seed=77,
        image_size=32,
    )
    assert frame.columns.tolist() == [
        "checkpoint", "source_age", "target_age", "predicted_generated_age",
        "age_error", "identity_cosine", "mode", "strength",
        "num_inference_steps", "text_guidance_scale", "image_guidance_scale", "seed",
    ]
    assert torch.equal(
        torch.tensor(frame["age_error"].to_numpy()),
        torch.tensor((frame["predicted_generated_age"] - frame["target_age"]).to_numpy()),
    )
    normal = infer_face_aging_direct(
        bundle=rebuilt, image=image, target_age=35, source_age=26,
        num_inference_steps=3, strength=0.5, seed=77, image_size=32,
        compute_diagnostics=True,
    )
    with Image.open(output_dir / "age_035.png") as saved:
        assert list(saved.getdata()) == list(normal["image"].getdata())
    assert (output_dir / "age_sweep.png").exists()
    assert (output_dir / "sampling_diagnostics.csv").exists()
