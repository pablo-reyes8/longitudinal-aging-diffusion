from __future__ import annotations

import torch
from PIL import Image

from data import build_face_aging_dataloaders
from src.inference import (
    infer_face_aging_direct,
    load_face_aging_adapter_for_inference,
)
from src.model import build_face_aging_optimizer
from src.training import (
    TRAIN_AGGING_MODEL,
    WarmupCosineLR,
    atomic_torch_save,
    build_inference_payload,
    build_training_payload,
)
from training_fakes import make_training_bundle, make_training_loss


def make_saved_checkpoints(tmp_path):
    bundle = make_training_bundle(seed=602)
    loss_fn = make_training_loss(bundle)
    optimizer = build_face_aging_optimizer(bundle)
    scheduler = WarmupCosineLR(optimizer, total_steps=4, warmup_steps=1)
    inference_path = atomic_torch_save(
        build_inference_payload(bundle, {"image_size": 32}), tmp_path / "inference.pt"
    )
    training_path = atomic_torch_save(
        build_training_payload(
            bundle=bundle, loss_fn=loss_fn, optimizer=optimizer,
            lr_scheduler=scheduler, scaler=None,
            epoch=0, batch_position=0, global_step=0, optimizer_step=0,
            best_metric=None, best_epoch=None, history={}, training_config={"image_size": 32},
        ),
        tmp_path / "training.pt",
    )
    return bundle, inference_path, training_path


def test_inference_and_training_resume_checkpoints_load_without_optimizer_dependency(tmp_path):
    original, inference_path, training_path = make_saved_checkpoints(tmp_path)
    image = Image.new("RGB", (32, 32), (90, 60, 30))
    expected = infer_face_aging_direct(
        bundle=original, image=image, target_age=60,
        num_inference_steps=4, strength=0.5, image_size=32,
        seed=7, return_latents=True,
    )
    for checkpoint in (inference_path, training_path):
        # The frozen backbone is reconstructed from the same base checkpoint;
        # only LoRA + conv_in come from the adapter checkpoint.
        rebuilt = make_training_bundle(seed=602)
        report = load_face_aging_adapter_for_inference(rebuilt, checkpoint)
        observed = infer_face_aging_direct(
            bundle=rebuilt, image=image, target_age=60,
            num_inference_steps=4, strength=0.5, image_size=32,
            seed=7, return_latents=True,
        )
        assert report["loaded_tensors"] == len(original["trainable_param_names"])
        assert torch.equal(expected["latents"], observed["latents"])
        assert torch.equal(expected["image_tensor"], observed["image_tensor"])


def test_training_builtin_inverse_monitor_uses_same_image_and_writes_each_epoch(tiny_root, tmp_path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
    )
    bundle = make_training_bundle(seed=710)
    loss_fn = make_training_loss(bundle)
    monitor_image = Image.new("RGB", (40, 32), (120, 80, 50))
    result = TRAIN_AGGING_MODEL(
        bundle=bundle, loss_fn=loss_fn,
        train_loader=loaders["train"], val_loader=loaders["val"],
        num_epochs=2, grad_accum_steps=2, max_train_batches=2, max_val_batches=1,
        lr_lora=2e-3, lr_conv_in=5e-4,
        amp_enabled=False, device="cpu", gradient_checkpointing=False, enable_xformers=False,
        min_snr_gamma=None, sample_target_posterior=False, log_every=0,
        checkpoint_dir=tmp_path / "checkpoints", sample_every_epochs=1,
        monitoring_image=monitor_image, monitoring_target_age=65, monitoring_source_age=30,
        monitoring_use_inverse_diffusion=True, monitoring_num_inference_steps=3,
        monitoring_seed=444, image_size=32,
    )
    monitoring = tmp_path / "checkpoints" / "monitoring"
    assert (monitoring / "epoch_001.png").exists() and (monitoring / "epoch_002.png").exists()
    reports = [epoch["sampling"] for epoch in result["history"]["epochs"]]
    assert all(report["status"] == "PASSED" for report in reports)
    assert all(report["result"]["mode"] == "inverse" for report in reports)
    assert all(report["result"]["seed"] == 444 for report in reports)
