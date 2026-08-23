from __future__ import annotations

import torch

from src.model import encode_images_to_latents
from src.training import run_training_step, train_one_epoch, validate_one_epoch
from training_fakes import clone_module_parameters, make_training_bundle, make_training_loss


def synthetic_batch(batch_size=3):
    torch.manual_seed(551)
    source = torch.full((batch_size, 3, 32, 32), -0.7)
    target = torch.full((batch_size, 3, 32, 32), 0.6)
    return {
        "source_image": source,
        "target_image": target,
        "source_age": torch.arange(20, 20 + batch_size),
        "target_age": torch.arange(40, 40 + batch_size),
        "delta_age": torch.full((batch_size,), 20),
        "target_prompt": [f"photo of a person as {40+i}-year-old" for i in range(batch_size)],
        "generic_prompt": ["photo of a person"] * batch_size,
    }


def test_one_step_calls_add_noise_once_and_has_no_source_target_leakage():
    bundle = make_training_bundle(counting_scheduler=True)
    loss_fn = make_training_loss(bundle, auxiliaries=False)
    batch = synthetic_batch(3)
    with torch.no_grad():
        source_latents = encode_images_to_latents(bundle, batch["source_image"], sample_posterior=False)
        target_latents = encode_images_to_latents(bundle, batch["target_image"], sample_posterior=False)
    noise = torch.full_like(target_latents, 0.25)
    timesteps = torch.tensor([0, 40, 99])
    result = run_training_step(
        bundle=bundle, loss_fn=loss_fn, batch=batch, device=torch.device("cpu"),
        amp_enabled=False, conditioning_dropout_prob=0,
        sample_target_posterior=False, noise=noise, timesteps=timesteps,
        return_debug_tensors=True,
    )
    prepared, model_input = result["prepared"], result["debug"]["model_input"]
    assert bundle["scheduler_train"].add_noise_calls == 1
    assert torch.equal(prepared["source_latents"], source_latents)
    assert torch.equal(prepared["target_latents"], target_latents)
    assert torch.equal(model_input[:, :4], prepared["noisy_target_latents"])
    assert torch.equal(model_input[:, 4:], source_latents)
    assert not torch.equal(model_input[:, 4:], target_latents)


def test_identity_excludes_image_dropped_samples_but_age_keeps_all():
    bundle = make_training_bundle()
    loss_fn = make_training_loss(bundle)
    batch = synthetic_batch(4)
    result = run_training_step(
        bundle=bundle, loss_fn=loss_fn, batch=batch, device=torch.device("cpu"),
        amp_enabled=False, conditioning_dropout_prob=0.2,
        dropout_random_values=torch.tensor([0.1, 0.3, 0.5, 0.9]),
        sample_target_posterior=False,
        identity_loss_on_image_dropped_samples=False,
    )
    output = result["loss_out"]
    assert output["metrics"]["identity_count"] == 2
    assert output["metrics"]["age_count"] == 4
    assert output["identity_indices"].tolist() == [0, 3]
    assert output["auxiliary_indices"].tolist() == [0, 1, 2, 3]


def test_cpu_bf16_smoke_backward_keeps_objective_and_gradients_finite():
    bundle = make_training_bundle()
    loss_fn = make_training_loss(bundle)
    result = run_training_step(
        bundle=bundle, loss_fn=loss_fn, batch=synthetic_batch(2), device=torch.device("cpu"),
        amp_enabled=True, amp_dtype="bf16", sample_target_posterior=False,
        return_debug_tensors=True,
    )
    output = result["loss_out"]
    output["loss"].backward()
    gradients = [p.grad for p in bundle["unet"].parameters() if p.requires_grad]
    age_gradients = [p.grad for p in bundle["age_delta_conditioner"].parameters()]
    assert result["debug"]["model_pred_compute_dtype"] == torch.bfloat16
    assert torch.isfinite(output["loss"])
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in age_gradients)


def test_validation_is_deterministic_and_mutates_no_parameter(tiny_root):
    from data import build_face_aging_dataloaders
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
    )
    bundle = make_training_bundle()
    loss_fn = make_training_loss(bundle)
    before = {
        "unet": clone_module_parameters(bundle["unet"]),
        "vae": clone_module_parameters(bundle["vae"]),
        "text": clone_module_parameters(bundle["text_encoder"]),
        "identity": clone_module_parameters(loss_fn.identity_encoder),
        "age": clone_module_parameters(loss_fn.age_estimator),
    }
    first = validate_one_epoch(
        bundle=bundle, loss_fn=loss_fn, val_loader=loaders["val"], device=torch.device("cpu"),
        amp_enabled=False, deterministic_validation=True, validation_seed=2026,
    )
    second = validate_one_epoch(
        bundle=bundle, loss_fn=loss_fn, val_loader=loaders["val"], device=torch.device("cpu"),
        amp_enabled=False, deterministic_validation=True, validation_seed=2026,
    )
    assert first["metrics"] == second["metrics"]
    assert torch.equal(first["timesteps"], second["timesteps"])
    assert first["timesteps"].min() < 25
    assert ((first["timesteps"] >= 25) & (first["timesteps"] < 75)).any()
    assert first["timesteps"].max() >= 50
    for key, module in (("unet", bundle["unet"]), ("vae", bundle["vae"]), ("text", bundle["text_encoder"]), ("identity", loss_fn.identity_encoder), ("age", loss_fn.age_estimator)):
        assert all(torch.equal(parameter, before[key][name]) for name, parameter in module.named_parameters())


def test_real_loader_train_updates_lora_and_conv_only(tiny_root):
    from data import build_face_aging_dataloaders
    from src.model import build_face_aging_optimizer
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
    )
    bundle = make_training_bundle()
    loss_fn = make_training_loss(bundle)
    frozen_before = {name: p.detach().clone() for name, p in bundle["unet"].named_parameters() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in bundle["unet"].named_parameters() if p.requires_grad}
    vae_before = clone_module_parameters(bundle["vae"])
    text_before = clone_module_parameters(bundle["text_encoder"])
    optimizer = build_face_aging_optimizer(bundle, lr_lora=1e-2, lr_conv_in=1e-2, weight_decay=0)
    result = train_one_epoch(
        bundle=bundle, loss_fn=loss_fn, train_loader=loaders["train"], optimizer=optimizer,
        lr_scheduler=None, device=torch.device("cpu"), epoch=0, amp_enabled=False,
        grad_accum_steps=2, max_batches=4, log_every=0,
        sample_target_posterior=False, generator=torch.Generator().manual_seed(99),
    )
    changed = {name for name, p in bundle["unet"].named_parameters() if p.requires_grad and not torch.equal(p, trainable_before[name])}
    assert result["optimizer_updates"] == 2
    assert any(name.startswith("conv_in") for name in changed)
    assert any("lora_" in name for name in changed)
    assert all(torch.equal(p, frozen_before[name]) for name, p in bundle["unet"].named_parameters() if not p.requires_grad)
    assert all(torch.equal(p, vae_before[name]) for name, p in bundle["vae"].named_parameters())
    assert all(torch.equal(p, text_before[name]) for name, p in bundle["text_encoder"].named_parameters())
