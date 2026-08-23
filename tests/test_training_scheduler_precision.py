from __future__ import annotations

import math

import pytest
import torch

from src.training import (
    WarmupCosineLR,
    get_effective_amp_dtype,
    make_grad_scaler,
    safe_optimizer_step,
)


def oracle_multiplier(step, total, warmup, floor):
    if warmup and step <= warmup:
        return step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def test_warmup_cosine_matches_independent_formula_and_preserves_lr_ratio():
    first, second = torch.nn.Parameter(torch.tensor(1.0)), torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([
        {"params": [first], "lr": 5e-5},
        {"params": [second], "lr": 1e-5},
    ])
    scheduler = WarmupCosineLR(optimizer, total_steps=100, warmup_steps=5, min_lr_ratio=0.1)
    observed = {0: scheduler.get_last_lr()}
    for step in range(1, 101):
        optimizer.step(); scheduler.step()
        if step in {1, 3, 5, 6, 52, 100}:
            observed[step] = scheduler.get_last_lr()
    for step, lrs in observed.items():
        multiplier = oracle_multiplier(step, 100, 5, 0.1)
        assert lrs == pytest.approx([5e-5 * multiplier, 1e-5 * multiplier], rel=1e-13, abs=1e-15)
        if step:
            assert lrs[0] / lrs[1] == pytest.approx(5.0, rel=1e-12)


def test_scheduler_state_resume_does_not_restart_warmup():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = WarmupCosineLR(optimizer, 20, 4, 0.1)
    for _ in range(9):
        optimizer.step(); scheduler.step()
    state, expected_lr = scheduler.state_dict(), scheduler.get_last_lr()
    rebuilt_parameter = torch.nn.Parameter(torch.tensor(1.0))
    rebuilt_optimizer = torch.optim.SGD([rebuilt_parameter], lr=1e-3)
    rebuilt = WarmupCosineLR(rebuilt_optimizer, 20, 4, 0.1)
    rebuilt.load_state_dict(state)
    assert rebuilt.step_num == 9 and rebuilt.get_last_lr() == expected_lr
    rebuilt_optimizer.step(); rebuilt.step()
    assert rebuilt.get_last_lr()[0] == pytest.approx(1e-3 * oracle_multiplier(10, 20, 4, 0.1))


def test_gradient_clipping_oracle_and_below_threshold_identity():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    parameter.grad = torch.tensor([6.0, 8.0])
    report = safe_optimizer_step(optimizer, [parameter], max_grad_norm=1.0)
    assert report["applied"] and report["clipped"] and report["grad_norm"] == pytest.approx(10.0)
    parameter.grad = torch.tensor([0.3, 0.4])
    report = safe_optimizer_step(optimizer, [parameter], max_grad_norm=1.0)
    assert report["applied"] and not report["clipped"] and report["grad_norm"] == pytest.approx(0.5)


def test_nonfinite_gradient_skips_optimizer_and_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=1)
    parameter.grad = torch.tensor(float("nan"))
    class Counter:
        calls = 0
        def step(self): self.calls += 1
    scheduler = Counter()
    report = safe_optimizer_step(optimizer, [parameter], lr_scheduler=scheduler)
    assert not report["applied"] and report["reason"] == "nonfinite_gradient"
    assert parameter.item() == 2 and scheduler.calls == 0


def test_corrupt_optimizer_update_is_rolled_back():
    class CorruptingSGD(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            self.param_groups[0]["params"][0].data.fill_(float("inf"))
            return result
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = CorruptingSGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.tensor(1.0)
    report = safe_optimizer_step(optimizer, [parameter], safe_snapshot=True)
    assert not report["applied"] and report["reason"] == "nonfinite_parameter_update"
    assert parameter.item() == 2.0 and parameter not in optimizer.state


def test_bf16_cpu_uses_autocast_without_grad_scaler_and_fp16_cpu_falls_back():
    assert get_effective_amp_dtype("bf16", "cpu") == torch.bfloat16
    assert make_grad_scaler("cpu", amp_enabled=True, amp_dtype="bf16") is None
    assert get_effective_amp_dtype("fp16", "cpu") is None


def test_fp16_scaler_order_is_unscale_clip_step_update_schedule(monkeypatch):
    events = []
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(2.0)
    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("optimizer_step")
            return super().step(closure)
    optimizer = RecordingSGD([parameter], lr=0.1)
    original_clip = torch.nn.utils.clip_grad_norm_
    def recording_clip(parameters, max_norm):
        events.append("clip")
        return original_clip(parameters, max_norm)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    class Scaler:
        def unscale_(self, optimizer): events.append("unscale")
        def step(self, optimizer):
            events.append("scaler_step")
            optimizer.step()
        def update(self): events.append("scaler_update")
    class Scheduler:
        def step(self): events.append("scheduler_step")
    report = safe_optimizer_step(
        optimizer, [parameter], scaler=Scaler(), lr_scheduler=Scheduler(), max_grad_norm=1.0
    )
    assert report["applied"]
    assert events == ["unscale", "clip", "scaler_step", "optimizer_step", "scaler_update", "scheduler_step"]
