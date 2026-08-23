from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model import build_face_aging_optimizer
from src.training import (
    TrainingCheckpointManager,
    WarmupCosineLR,
    atomic_torch_save,
    build_inference_payload,
    build_training_payload,
    load_training_checkpoint,
)
from training_fakes import make_training_bundle, make_training_loss


def deterministic_optimizer_step(bundle, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    objective = sum(parameter.square().sum() for parameter in bundle["unet"].parameters() if parameter.requires_grad)
    objective.backward(); optimizer.step(); scheduler.step()


def make_stack(seed=801):
    bundle = make_training_bundle(seed)
    loss_fn = make_training_loss(bundle)
    optimizer = build_face_aging_optimizer(bundle, lr_lora=5e-4, lr_conv_in=1e-4)
    scheduler = WarmupCosineLR(optimizer, total_steps=20, warmup_steps=2, min_lr_ratio=0.1)
    return bundle, loss_fn, optimizer, scheduler


def test_checkpoint_roundtrip_and_next_update_exactness(tmp_path):
    bundle, loss_fn, optimizer, scheduler = make_stack()
    for _ in range(7):
        deterministic_optimizer_step(bundle, optimizer, scheduler)
    payload = build_training_payload(
        bundle=bundle, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=scheduler,
        scaler=None, epoch=2, batch_position=0, global_step=21, optimizer_step=7,
        best_metric=0.7, best_epoch=2, history={"epochs": [1, 2]}, training_config={"seed": 42},
    )
    path = atomic_torch_save(payload, tmp_path / "resume.pt")
    rebuilt, rebuilt_loss, rebuilt_optimizer, rebuilt_scheduler = make_stack()
    restored = load_training_checkpoint(
        path, bundle=rebuilt, loss_fn=rebuilt_loss, optimizer=rebuilt_optimizer,
        lr_scheduler=rebuilt_scheduler,
    )
    assert (restored["epoch"], restored["global_step"], restored["optimizer_step"]) == (2, 21, 7)
    assert rebuilt_scheduler.step_num == scheduler.step_num == 7
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        ((n, p) for n, p in bundle["unet"].named_parameters() if p.requires_grad),
        ((n, p) for n, p in rebuilt["unet"].named_parameters() if p.requires_grad),
    ):
        assert name_a == name_b and torch.equal(parameter_a, parameter_b)
    deterministic_optimizer_step(bundle, optimizer, scheduler)
    deterministic_optimizer_step(rebuilt, rebuilt_optimizer, rebuilt_scheduler)
    for parameter_a, parameter_b in zip(bundle["unet"].parameters(), rebuilt["unet"].parameters()):
        assert torch.equal(parameter_a, parameter_b)
    assert scheduler.get_last_lr() == rebuilt_scheduler.get_last_lr()


def test_checkpoint_config_mismatch_is_rejected(tmp_path):
    bundle, loss_fn, optimizer, scheduler = make_stack()
    payload = build_training_payload(
        bundle=bundle, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=scheduler,
        scaler=None, epoch=0, batch_position=0, global_step=0, optimizer_step=0,
        best_metric=None, best_epoch=None, history={}, training_config={},
    )
    path = atomic_torch_save(payload, tmp_path / "resume.pt")
    incompatible, incompatible_loss, incompatible_optimizer, incompatible_scheduler = make_stack()
    incompatible["config"]["rank"] = 99
    with pytest.raises(ValueError, match="incompatible"):
        load_training_checkpoint(
            path, bundle=incompatible, loss_fn=incompatible_loss,
            optimizer=incompatible_optimizer, lr_scheduler=incompatible_scheduler,
        )


def test_atomic_failure_preserves_previous_checkpoint(tmp_path):
    path = atomic_torch_save({"version": 1}, tmp_path / "latest.pt")
    original = path.read_bytes()
    def fail_save(payload, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("injected write failure")
    with pytest.raises(OSError, match="injected"):
        atomic_torch_save({"version": 2}, path, save_fn=fail_save)
    assert path.read_bytes() == original
    assert torch.load(path, weights_only=True)["version"] == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "mode,values,expected_epochs",
    [("min", [0.8, 0.7, 0.75, 0.6], [0, 1, 3]), ("max", [0.2, 0.3, 0.25, 0.4], [0, 1, 3])],
)
def test_best_logic_and_epoch_retention(tmp_path, mode, values, expected_epochs):
    manager = TrainingCheckpointManager(
        tmp_path, mode=mode, save_epoch_checkpoints=True, max_epoch_checkpoints=3
    )
    improved_epochs = []
    for epoch, value in enumerate(values + [values[-1], values[-1], values[-1]]):
        report = manager.save(
            training_payload={"epoch": epoch}, inference_payload={"epoch": epoch},
            epoch=epoch, metric=value,
        )
        if report["improved"]:
            improved_epochs.append(epoch)
    assert improved_epochs == expected_epochs
    assert {path.name for path in tmp_path.glob("epoch_*" )} == {"epoch_005", "epoch_006", "epoch_007"}
    assert (tmp_path / "latest" / "training_resume.pt").exists()
    assert (tmp_path / "best" / "training_resume.pt").exists()
