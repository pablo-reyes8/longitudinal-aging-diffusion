from __future__ import annotations

import torch
import pytest

from data import build_face_aging_dataloaders
from src.model import (
    assemble_face_aging_diffusion_bundle,
    build_conditioned_unet_input,
    build_face_aging_optimizer,
    encode_images_to_latents,
    encode_prompts,
    load_face_aging_adapter,
    prepare_face_aging_forward,
    prepare_source_target_latents,
    run_face_aging_model_validation,
    save_face_aging_adapter,
)
from model_fakes import make_fake_components


def make_bundle(seed=10, model_id="fake/sd15"):
    torch.manual_seed(seed)
    return assemble_face_aging_diffusion_bundle(
        make_fake_components(), model_id=model_id, rank=2, alpha=2, dropout=0, verbose=False
    )


def test_prompt_latent_scaling_and_conditioned_input():
    bundle = make_bundle()
    images = torch.randn(2, 3, 32, 32)
    raw = bundle["vae"].encode(images).latent_dist.mean
    expected = raw * bundle["vae"].config.scaling_factor
    actual = encode_images_to_latents(bundle, images)
    assert torch.allclose(actual, expected)
    latents = prepare_source_target_latents(bundle, images, images.flip(-1))
    assert latents["source_latents"].shape == (2, 4, 4, 4)
    conditioned = build_conditioned_unet_input(latents["target_latents"], latents["source_latents"])
    assert conditioned.shape == (2, 8, 4, 4)
    prompts = ["photo of a person as 25-year-old", "photo of a person as 60-year-old"]
    assert torch.equal(encode_prompts(bundle, prompts), encode_prompts(bundle, prompts))
    assert not torch.equal(encode_prompts(bundle, prompts)[0], encode_prompts(bundle, prompts)[1])


def test_conditioned_input_rejects_mismatch():
    with pytest.raises(ValueError, match="incompatible"):
        build_conditioned_unet_input(torch.randn(2, 4, 4, 4), torch.randn(2, 4, 8, 8))
    with pytest.raises(ValueError, match="same dtype"):
        build_conditioned_unet_input(torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4).double())


def test_real_dataloader_batch_full_offline_validation(tiny_root):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
    )
    batch = next(iter(loaders["train"]))
    bundle = make_bundle()
    report = run_face_aging_model_validation(bundle, batch=batch)
    assert report["passed"], report["errors"]
    assert report["latent_tests"]["conditioned_shape"] == [2, 8, 4, 4]
    assert report["latent_tests"]["noise_prediction_shape"] == [2, 4, 4, 4]


def test_source_becomes_causally_active_after_update():
    bundle = make_bundle()
    unet = bundle["unet"]
    hidden, timestep = torch.randn(1, 12, 6), torch.tensor([5])
    target = torch.randn(1, 4, 4, 4)
    source_a, source_b = torch.randn_like(target), torch.randn_like(target)
    def output(source):
        return unet(torch.cat([target, source], dim=1), timestep, hidden).sample
    assert torch.allclose(output(source_a), output(source_b), atol=1e-7)
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-2, lr_conv_in=1e-2, weight_decay=0)
    loss = output(source_a).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.allclose(output(source_a), output(source_b), atol=1e-8)


def test_two_step_gradient_nuance_and_parameter_changes():
    bundle = make_bundle()
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-2, lr_conv_in=1e-2, weight_decay=0)
    unet = bundle["unet"]
    frozen_before = {n: p.detach().clone() for n, p in unet.named_parameters() if not p.requires_grad}
    trainable_before = {n: p.detach().clone() for n, p in unet.named_parameters() if p.requires_grad}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = unet(torch.randn(2, 8, 4, 4), torch.tensor([3, 7]), torch.randn(2, 12, 6)).sample
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bundle["trainable_params"])
        optimizer.step()
    changed = {n for n, p in unet.named_parameters() if p.requires_grad and not torch.equal(p, trainable_before[n])}
    assert any("lora_up" in name for name in changed)
    assert any("lora_down" in name for name in changed)
    assert any(name.startswith("conv_in") for name in changed)
    assert all(torch.equal(p, frozen_before[n]) for n, p in unet.named_parameters() if not p.requires_grad)


def test_adapter_checkpoint_roundtrip_and_wrong_backbone(tmp_path):
    bundle = make_bundle(seed=22)
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-2, lr_conv_in=1e-2)
    sample, timestep, hidden = torch.randn(1, 8, 4, 4), torch.tensor([2]), torch.randn(1, 12, 6)
    loss = bundle["unet"](sample, timestep, hidden).sample.square().mean()
    loss.backward(); optimizer.step()
    expected = bundle["unet"](sample, timestep, hidden).sample.detach()
    checkpoint = save_face_aging_adapter(bundle, tmp_path / "adapter.pt")
    rebuilt = make_bundle(seed=22)
    loaded = load_face_aging_adapter(rebuilt, checkpoint)
    actual = rebuilt["unet"](sample, timestep, hidden).sample.detach()
    assert loaded["loaded_tensors"] == len(bundle["trainable_param_names"])
    assert torch.allclose(expected, actual, atol=1e-7)
    wrong = make_bundle(seed=22, model_id="fake/other")
    with pytest.raises(ValueError, match="incompatible"):
        load_face_aging_adapter(wrong, checkpoint)


@pytest.mark.parametrize("resolution,latent_size", [(256, 32), (512, 64)])
def test_resolution_independence(resolution, latent_size):
    bundle = make_bundle()
    latent = encode_images_to_latents(bundle, torch.randn(1, 3, resolution, resolution))
    assert latent.shape == (1, 4, latent_size, latent_size)


def test_batch_permutation_equivariance():
    bundle = make_bundle()
    source, target = torch.randn(3, 3, 32, 32), torch.randn(3, 3, 32, 32)
    prompts = ["age 20", "age 40", "age 60"]
    latents = prepare_source_target_latents(bundle, source, target)
    noise = torch.randn_like(latents["target_latents"])
    timesteps = torch.tensor([3, 20, 70])
    original = prepare_face_aging_forward(bundle, source, target, prompts, noise=noise, timesteps=timesteps)["noise_pred"]
    permutation = torch.tensor([2, 0, 1])
    permuted = prepare_face_aging_forward(
        bundle, source[permutation], target[permutation], [prompts[i] for i in permutation],
        noise=noise[permutation], timesteps=timesteps[permutation],
    )["noise_pred"]
    inverse = torch.argsort(permutation)
    assert torch.allclose(original, permuted[inverse], atol=1e-6)


def test_scheduler_determinism_boundaries_and_text_sensitivity():
    bundle = make_bundle()
    scheduler = bundle["scheduler_train"]
    latents, noise = torch.randn(3, 4, 4, 4), torch.randn(3, 4, 4, 4)
    timesteps = torch.tensor([0, scheduler.config.num_train_timesteps // 2, scheduler.config.num_train_timesteps - 1])
    first = scheduler.add_noise(latents, noise, timesteps)
    second = scheduler.add_noise(latents, noise, timesteps)
    assert torch.equal(first, second)
    assert first.shape == latents.shape and first.dtype == latents.dtype
    assert torch.isfinite(first).all()

    conditioned = torch.randn(2, 8, 4, 4)
    conditioned[1] = conditioned[0]
    text = encode_prompts(bundle, ["photo as 25-year-old", "photo as 60-year-old"])
    output = bundle["unet"](conditioned, torch.tensor([10, 10]), text).sample
    assert not torch.equal(output[0], output[1])
