from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from data import build_face_aging_dataloaders
from model_fakes import make_fake_components
from src.loss import (
    AgeEstimatorAdapter,
    FaceAgingDiffusionLoss,
    IdentityEncoderAdapter,
    predict_x0_from_model_output,
    run_face_aging_loss_validation,
)
from src.model import (
    assemble_face_aging_diffusion_bundle,
    build_face_aging_optimizer,
    prepare_face_aging_forward,
)


class IntegrationIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, images):
        return self.pool(torch.tanh(self.conv(images))).flatten(1)


class IntegrationAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, images):
        return self.conv(images).mean((1, 2, 3)) * 10 + 35


def build_integration(tiny_root, seed=701):
    torch.manual_seed(seed)
    bundle = assemble_face_aging_diffusion_bundle(
        make_fake_components(), model_id="offline/sd15-shaped", rank=2, alpha=2, verbose=False
    )
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False, seed=42,
    )
    batch = next(iter(loaders["train"]))
    prepared = prepare_face_aging_forward(
        bundle, batch["source_image"], batch["target_image"], batch["target_prompt"]
    )
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=bundle["scheduler_train"],
        vae=bundle["vae"],
        identity_encoder=IdentityEncoderAdapter(IntegrationIdentity()),
        age_estimator=AgeEstimatorAdapter(IntegrationAge()),
        identity_weight=0.1,
        age_weight=0.1,
    )
    kwargs = {
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
    return bundle, batch, prepared, loss_fn, kwargs


def test_real_loader_complete_loss_backward_and_structured_report(tiny_root):
    bundle, batch, prepared, loss_fn, kwargs = build_integration(tiny_root)
    report = run_face_aging_loss_validation(loss_fn, forward_kwargs=kwargs)
    assert report["passed"], report["errors"]
    assert report["tier_a"]["passed"]
    assert report["tier_b"]["passed"]
    assert report["tier_c"]["status"] == "NOT RUN" and report["tier_c"]["passed"] is None
    assert any("Tier C" in warning for warning in report["warnings"])
    output = loss_fn(**kwargs, return_per_sample=True)
    output["loss"].backward()
    trainable = [(name, p) for name, p in bundle["unet"].named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
    assert any(name.startswith("conv_in") and p.grad.norm() > 0 for name, p in trainable)
    assert any("lora_up" in name and p.grad.norm() > 0 for name, p in trainable)
    assert all(p.grad is None for p in loss_fn.vae.parameters())
    assert all(p.grad is None for p in loss_fn.identity_encoder.parameters())
    assert all(p.grad is None for p in loss_fn.age_estimator.parameters())
    assert output["metrics"]["loss_diff"] > 0
    assert output["metrics"]["loss_id"] >= 0
    assert output["metrics"]["loss_age"] >= 0


def test_conv_and_lora_weighted_gradient_decomposition(tiny_root):
    bundle, _, _, loss_fn, kwargs = build_integration(tiny_root, seed=702)
    output = loss_fn(**kwargs)
    named = dict(bundle["unet"].named_parameters())
    selected = {
        "conv_in": named["conv_in.weight"],
        "lora_up": next(p for name, p in named.items() if "lora_up.weight" in name),
    }
    for group, parameter in selected.items():
        total = torch.autograd.grad(output["loss"], parameter, retain_graph=True)[0]
        diff = torch.autograd.grad(output["loss_diff"], parameter, retain_graph=True)[0]
        identity = torch.autograd.grad(output["loss_id"], parameter, retain_graph=True)[0]
        age = torch.autograd.grad(output["loss_age"], parameter, retain_graph=True)[0]
        expected = diff + 0.1 * identity + 0.1 * age
        assert torch.allclose(total, expected, atol=2e-7, rtol=2e-5), group
        assert torch.isfinite(total).all()


def test_frozen_models_unchanged_and_trainables_mutate_after_steps(tiny_root):
    bundle, batch, _, loss_fn, _ = build_integration(tiny_root, seed=703)
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-2, lr_conv_in=1e-2, weight_decay=0)
    frozen_modules = [bundle["vae"], bundle["text_encoder"], loss_fn.identity_encoder, loss_fn.age_estimator]
    frozen_before = [[p.detach().clone() for p in module.parameters()] for module in frozen_modules]
    unet_frozen_before = {name: p.detach().clone() for name, p in bundle["unet"].named_parameters() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in bundle["unet"].named_parameters() if p.requires_grad}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prepared = prepare_face_aging_forward(bundle, batch["source_image"], batch["target_image"], batch["target_prompt"])
        output = loss_fn(
            model_pred=prepared["noise_pred"], noise=prepared["noise"],
            noisy_target_latents=prepared["noisy_target_latents"], target_latents=prepared["target_latents"],
            timesteps=prepared["timesteps"], source_images=batch["source_image"], target_images=batch["target_image"],
            target_ages=batch["target_age"], global_step=step,
        )
        output["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in bundle["trainable_params"] if p.grad is not None)
        optimizer.step()
    for module, old_parameters in zip(frozen_modules, frozen_before):
        assert all(torch.equal(parameter, old) for parameter, old in zip(module.parameters(), old_parameters))
    assert all(torch.equal(parameter, unet_frozen_before[name]) for name, parameter in bundle["unet"].named_parameters() if not parameter.requires_grad)
    changed = {name for name, parameter in bundle["unet"].named_parameters() if parameter.requires_grad and not torch.equal(parameter, trainable_before[name])}
    assert any(name.startswith("conv_in") for name in changed)
    assert any("lora_up" in name for name in changed)
    assert any("lora_down" in name for name in changed)


def test_real_batch_perfect_epsilon_x0_and_decode_reconstruction(tiny_root):
    bundle, _, prepared, loss_fn, kwargs = build_integration(tiny_root, seed=704)
    exact = predict_x0_from_model_output(
        prepared["noise"], prepared["noisy_target_latents"], prepared["timesteps"], bundle["scheduler_train"]
    )
    assert torch.allclose(exact, prepared["target_latents"], atol=2e-6, rtol=2e-5)
    scale = bundle["vae"].config.scaling_factor
    decoded_exact = bundle["vae"].decode(exact / scale).sample
    decoded_target = bundle["vae"].decode(prepared["target_latents"] / scale).sample
    assert torch.allclose(decoded_exact, decoded_target, atol=2e-6, rtol=2e-5)
    perfect_kwargs = {**kwargs, "model_pred": prepared["noise"]}
    output = loss_fn(**perfect_kwargs, return_reconstructions=True)
    assert output["loss_diff"].item() == 0.0


def test_extreme_timesteps_are_finite_and_report_amplification(tiny_root):
    bundle, batch, prepared, loss_fn, kwargs = build_integration(tiny_root, seed=705)
    norms = []
    for timestep in (0, 1, 98, 99):
        target = prepared["target_latents"][:1]
        noise = torch.randn_like(target)
        steps = torch.tensor([timestep])
        noisy = bundle["scheduler_train"].add_noise(target, noise, steps)
        prediction = (noise + 0.2).detach().requires_grad_(True)
        output = loss_fn(
            model_pred=prediction, noise=noise, noisy_target_latents=noisy,
            target_latents=target, timesteps=steps,
            source_images=batch["source_image"][:1], target_images=batch["target_image"][:1],
            target_ages=batch["target_age"][:1], return_reconstructions=True,
        )
        gradient = torch.autograd.grad(output["loss"], prediction)[0]
        assert torch.isfinite(output["loss"]) and torch.isfinite(output["pred_x0_latents"]).all()
        assert torch.isfinite(output["pred_x0_images"]).all() and torch.isfinite(gradient).all()
        norms.append(float(gradient.norm()))
    assert all(norm >= 0 for norm in norms)
