from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training import train_one_epoch


class RegressionDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    def __len__(self): return len(self.xs)
    def __getitem__(self, index):
        return {
            "x": self.xs[index:index+1], "y": self.ys[index:index+1],
            "source_image": torch.zeros(1, 1, 1),
            "source_age": torch.tensor(20), "target_age": torch.tensor(30), "delta_age": torch.tensor(10),
            "generic_prompt": "photo of a person",
        }


class DummyLoss(nn.Module):
    diffusion_weight = 1.0
    identity_weight = 0.0
    age_weight = 0.0
    age_loss_type = "l1"


def fake_step(*, bundle, batch, **kwargs):
    prediction = bundle["unet"](batch["x"])
    per_sample = (prediction - batch["y"]).square().flatten(1).mean(1)
    loss = per_sample.mean()
    zero = loss * 0
    output = {
        "loss": loss, "loss_diff": loss, "loss_id": zero, "loss_age": zero,
        "weighted_diff": loss, "weighted_id": zero, "weighted_age": zero,
        "loss_diff_per_sample": per_sample,
        "loss_id_per_sample": per_sample.new_empty(0), "loss_age_per_sample": per_sample.new_empty(0),
        "identity_indices": torch.empty(0, dtype=torch.long), "auxiliary_indices": torch.empty(0, dtype=torch.long),
        "metrics": {"identity_cosine_mean": None},
    }
    return {"loss_out": output, "prepared": {"timesteps": torch.zeros(batch["x"].shape[0], dtype=torch.long)}, "diagnostics": {"timestep_mean": 0.0}}


def make_bundle(weight=0.4):
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    model.weight.data.fill_(weight)
    return {
        "unet": model, "vae": nn.Identity(), "text_encoder": nn.Identity(),
        "scheduler_train": SimpleNamespace(alphas_cumprod=torch.ones(1)),
    }


@pytest.fixture(autouse=True)
def patch_training_step(monkeypatch):
    module = importlib.import_module("src.training.train_one_epoch")
    monkeypatch.setattr(module, "run_training_step", fake_step)


def run_epoch(batch_size, accumulation, xs=(1, 2, 3, 4), ys=(2, 1, 4, 3), **kwargs):
    bundle = make_bundle()
    optimizer = torch.optim.SGD(bundle["unet"].parameters(), lr=0.1)
    result = train_one_epoch(
        bundle=bundle, loss_fn=DummyLoss(),
        train_loader=DataLoader(RegressionDataset(xs, ys), batch_size=batch_size, shuffle=False),
        optimizer=optimizer, lr_scheduler=None, device=torch.device("cpu"), epoch=0,
        amp_enabled=False, grad_accum_steps=accumulation, max_grad_norm=1e9, log_every=0,
        **kwargs,
    )
    return bundle["unet"].weight.detach().clone(), result


def test_gradient_accumulation_matches_large_batch_exactly():
    large, result_large = run_epoch(4, 1)
    accumulated, result_accumulated = run_epoch(2, 2)
    assert torch.allclose(large, accumulated, atol=1e-14, rtol=1e-14)
    assert result_large["optimizer_updates"] == result_accumulated["optimizer_updates"] == 1


def test_partial_accumulation_window_is_correct_and_steps_scheduler_twice():
    class Counter:
        def __init__(self): self.calls = 0
        def step(self): self.calls += 1
    bundle = make_bundle()
    optimizer = torch.optim.SGD(bundle["unet"].parameters(), lr=0.1)
    scheduler = Counter()
    xs, ys = (1, 2, 3, 4, 5), (2, 1, 4, 3, 7)
    result = train_one_epoch(
        bundle=bundle, loss_fn=DummyLoss(), train_loader=DataLoader(RegressionDataset(xs, ys), batch_size=1),
        optimizer=optimizer, lr_scheduler=scheduler, device=torch.device("cpu"), epoch=0,
        amp_enabled=False, grad_accum_steps=4, max_grad_norm=1e9, log_every=0,
    )
    reference = make_bundle()
    reference_optimizer = torch.optim.SGD(reference["unet"].parameters(), lr=0.1)
    for indices in ((0, 1, 2, 3), (4,)):
        x = torch.tensor([xs[i] for i in indices], dtype=torch.float64).view(-1, 1)
        y = torch.tensor([ys[i] for i in indices], dtype=torch.float64).view(-1, 1)
        ((reference["unet"](x) - y).square().mean()).backward()
        reference_optimizer.step(); reference_optimizer.zero_grad(set_to_none=True)
    assert torch.allclose(bundle["unet"].weight, reference["unet"].weight, atol=1e-14, rtol=1e-14)
    assert result["optimizer_updates"] == scheduler.calls == 2


def test_double_prompt_half_plus_half_does_not_double_gradient():
    ordinary, ordinary_result = run_epoch(4, 1, double_prompt_prob=0.0)
    doubled, doubled_result = run_epoch(
        4, 1, double_prompt_prob=1.0,
        age_prompt_weight=0.5, generic_prompt_weight=0.5,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.allclose(ordinary, doubled, atol=1e-14, rtol=1e-14)
    assert ordinary_result["optimizer_updates"] == doubled_result["optimizer_updates"] == 1
    assert doubled_result["double_prompt_batches"] == 1


def test_nonfinite_loss_skips_entire_window_without_step(monkeypatch):
    module = importlib.import_module("src.training.train_one_epoch")
    def nan_step(*, bundle, batch, **kwargs):
        loss = bundle["unet"].weight.sum() * float("nan")
        zero = loss * 0
        output = {
            "loss": loss, "loss_diff": loss, "loss_id": zero, "loss_age": zero,
            "weighted_diff": loss, "weighted_id": zero, "weighted_age": zero,
            "loss_diff_per_sample": loss.expand(batch["x"].shape[0]),
            "loss_id_per_sample": loss.new_empty(0), "loss_age_per_sample": loss.new_empty(0),
            "identity_indices": torch.empty(0, dtype=torch.long), "auxiliary_indices": torch.empty(0, dtype=torch.long),
            "metrics": {"identity_cosine_mean": None},
        }
        return {"loss_out": output, "prepared": {}, "diagnostics": {}}
    monkeypatch.setattr(module, "run_training_step", nan_step)
    bundle = make_bundle()
    before = bundle["unet"].weight.detach().clone()
    optimizer = torch.optim.SGD(bundle["unet"].parameters(), lr=1)
    result = train_one_epoch(
        bundle=bundle, loss_fn=DummyLoss(),
        train_loader=DataLoader(RegressionDataset((1, 2), (2, 3)), batch_size=1),
        optimizer=optimizer, lr_scheduler=None, device=torch.device("cpu"), epoch=0,
        amp_enabled=False, grad_accum_steps=2, log_every=0,
    )
    assert result["skipped_nonfinite"] == 1 and result["optimizer_updates"] == 0
    assert torch.equal(bundle["unet"].weight, before)
