from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.model import (
    DEFAULT_ATTENTION_TARGETS,
    DoRALinear,
    LoRALinear,
    assemble_face_aging_diffusion_bundle,
    build_face_aging_optimizer,
    expand_unet_conv_in_for_source_conditioning,
    inject_manual_lora_unet,
    inject_manual_dora_unet,
)
from model_fakes import FakeUNet, make_fake_components


def _unet_output(unet, sample):
    batch = sample.shape[0]
    return unet(sample, torch.zeros(batch, dtype=torch.long), torch.randn(batch, 5, 6), return_dict=True).sample


def test_conv_in_expansion_preserves_base_function_and_zeros_source():
    torch.manual_seed(1)
    base = FakeUNet().eval()
    expanded = copy.deepcopy(base).eval()
    old_weight = expanded.conv_in.weight.detach().clone()
    old_bias = expanded.conv_in.bias.detach().clone()
    report = expand_unet_conv_in_for_source_conditioning(expanded)
    target = torch.randn(2, 4, 5, 5)
    source_a, source_b = torch.randn_like(target), torch.randn_like(target)
    hidden = torch.randn(2, 5, 6)
    timestep = torch.tensor([1, 9])
    original = base(target, timestep, hidden).sample
    output_a = expanded(torch.cat([target, source_a], dim=1), timestep, hidden).sample
    output_b = expanded(torch.cat([target, source_b], dim=1), timestep, hidden).sample
    assert torch.equal(expanded.conv_in.weight[:, :4], old_weight)
    assert torch.count_nonzero(expanded.conv_in.weight[:, 4:]) == 0
    assert torch.equal(expanded.conv_in.bias, old_bias)
    assert report["expanded_in_channels"] == 8
    assert torch.allclose(original, output_a, atol=1e-7)
    assert torch.allclose(output_a, output_b, atol=1e-7)


def test_lora_initial_equivalence_coverage_and_formula():
    torch.manual_seed(2)
    unet = FakeUNet().eval()
    sample, hidden = torch.randn(2, 4, 4, 4), torch.randn(2, 5, 6)
    before = unet(sample, torch.tensor([2, 3]), hidden).sample
    inject_manual_lora_unet(unet, rank=3, alpha=3, verbose=False)
    unet.eval()
    after = unet(sample, torch.tensor([2, 3]), hidden).sample
    assert torch.allclose(before, after, atol=1e-7)
    report = unet._face_aging_adapter_report
    assert set(report["counts_by_target"]) == set(DEFAULT_ATTENTION_TARGETS)
    assert all(count == 1 for count in report["counts_by_target"].values())
    actual = sum(p.numel() for n, p in unet.named_parameters() if ".lora_" in n)
    assert actual == report["expected_adapter_parameters"]


def test_dora_module_initial_effective_weight_and_output():
    torch.manual_seed(3)
    base = nn.Linear(7, 5)
    inputs = torch.randn(4, 7)
    expected = base(inputs)
    dora = DoRALinear(base, rank=2, alpha=2)
    assert torch.allclose(dora.get_effective_weight(), base.weight, atol=1e-7)
    assert torch.allclose(dora(inputs), expected, atol=1e-7)


def test_dora_full_unet_initial_equivalence():
    torch.manual_seed(33)
    unet = FakeUNet().eval()
    sample, timestep, hidden = torch.randn(2, 4, 4, 4), torch.tensor([2, 8]), torch.randn(2, 5, 6)
    before = unet(sample, timestep, hidden).sample
    inject_manual_dora_unet(unet, rank=2, alpha=2, verbose=False)
    unet.eval()
    after = unet(sample, timestep, hidden).sample
    assert torch.allclose(before, after, atol=1e-7)


@pytest.mark.parametrize("adapter_type", ["lora", "dora"])
def test_bundle_trainable_policy_and_optimizer_groups(adapter_type):
    bundle = assemble_face_aging_diffusion_bundle(
        make_fake_components(), model_id="fake/sd15", adapter_type=adapter_type,
        rank=2, alpha=2, verbose=False,
    )
    names = bundle["trainable_param_names"]
    assert any(name.startswith("conv_in.") for name in names)
    assert all(
            name.startswith("conv_in.") or ".lora_" in name or name.endswith(".magnitude")
            or name.startswith("age_delta_conditioner.") or name.startswith("age_conditioner.")
        for name in names
    )
    assert all(p.dtype == torch.float32 for p in bundle["trainable_params"])
    assert not any(p.requires_grad for p in bundle["vae"].parameters())
    assert not any(p.requires_grad for p in bundle["text_encoder"].parameters())
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-4, lr_conv_in=2e-5)
    assert [group["group_name"] for group in optimizer.param_groups] == ["adapter", "conv_in", "age_conditioner"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-5, 1e-4]


def test_adapter_target_failure_is_loud():
    with pytest.raises(RuntimeError, match="coverage failed"):
        inject_manual_lora_unet(FakeUNet(), target_suffixes=("does_not_exist",), verbose=False)


def test_model_train_eval_does_not_change_requires_grad():
    bundle = assemble_face_aging_diffusion_bundle(
        make_fake_components(), model_id="fake/sd15", rank=2, alpha=2, verbose=False
    )
    before = {name: p.requires_grad for name, p in bundle["unet"].named_parameters()}
    bundle["unet"].eval()
    bundle["unet"].train()
    after = {name: p.requires_grad for name, p in bundle["unet"].named_parameters()}
    assert before == after
