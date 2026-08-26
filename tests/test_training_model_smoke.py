from __future__ import annotations

import json

import pytest
import torch

from data import build_face_aging_dataloaders
from src.model import load_face_aging_adapter
from src.training import TRAIN_AGGING_MODEL
from training_fakes import clone_module_parameters, make_training_bundle, make_training_loss


def test_train_agging_model_real_loader_mixed_precision_checkpoint_and_validation(tiny_root, tmp_path):
    loaders, _ = build_face_aging_dataloaders(
        tiny_root, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
        include_zero_delta_pairs=True, zero_delta_pair_prob=0.20,
        include_bidirectional_pairs=True, reverse_pair_prob=0.20,
    )
    bundle = make_training_bundle(seed=990)
    loss_fn = make_training_loss(bundle)
    loss_fn.use_relative_age_loss = True
    loss_fn.relative_age_weight = 0.05
    frozen = {
        "vae": clone_module_parameters(bundle["vae"]),
        "text": clone_module_parameters(bundle["text_encoder"]),
        "identity": clone_module_parameters(loss_fn.identity_encoder),
        "age": clone_module_parameters(loss_fn.age_estimator),
    }
    trainable = clone_module_parameters(bundle["unet"])
    conditioner_before = clone_module_parameters(bundle["age_delta_conditioner"])
    result = TRAIN_AGGING_MODEL(
        bundle=bundle, loss_fn=loss_fn,
        train_loader=loaders["train"], val_loader=loaders["val"],
        max_train_steps=2, num_epochs=10,
        grad_accum_steps=2, max_train_batches=4, max_val_batches=2,
        lr_lora=1e-2, lr_conv_in=1e-2, weight_decay=0,
        amp_enabled=True, amp_dtype="bf16", device="cpu",
        gradient_checkpointing=True, enable_xformers=True,
        min_snr_gamma=5.0, checkpoint_dir=tmp_path / "checkpoints",
        log_every=0, sample_every_epochs=0,
        sample_target_posterior=False, deterministic=True,
        use_bidirectional_training=True, reverse_pair_prob=0.20,
    )
    assert result["optimizer_step"] == 2
    assert result["precision"]["amp_dtype_name"] == "bf16" and result["scaler"] is None
    assert len(result["history"]["train"]) == len(result["history"]["val"]) == 1
    assert torch.isfinite(torch.tensor(result["history"]["val"][0]["val/loss_total"]))
    for split, prefix in (("train", "train"), ("val", "val")):
        metrics = result["history"][split][0]
        assert f"{prefix}/loss_relative_age" in metrics
        assert f"{prefix}/weighted_relative_age" in metrics
        assert torch.isfinite(torch.tensor(metrics[f"{prefix}/loss_relative_age"]))
    train_metrics = result["history"]["train"][0]
    assert train_metrics["train/numeric_prompt_count"] + train_metrics["train/generic_prompt_count"] == 8
    assert train_metrics["train/numeric_prompt_fraction"] + train_metrics["train/generic_prompt_fraction"] == pytest.approx(1.0)
    assert "train/age_conditioner_scale" in train_metrics
    assert train_metrics["train/forward_pair_count"] > 0
    assert train_metrics["train/reverse_pair_count"] > 0
    assert train_metrics["train/zero_pair_count"] > 0
    assert sum(
        train_metrics[key]
        for key in (
            "train/forward_pair_fraction",
            "train/reverse_pair_fraction",
            "train/zero_pair_fraction",
        )
    ) == pytest.approx(1.0)
    assert result["memory_features"] == {"gradient_checkpointing": "unavailable", "xformers": "unavailable"}
    changed = {name for name, p in bundle["unet"].named_parameters() if p.requires_grad and not torch.equal(p, trainable[name])}
    assert any(name.startswith("conv_in") for name in changed)
    assert any("lora_" in name for name in changed)
    assert any(
        not torch.equal(parameter, conditioner_before[name])
        for name, parameter in bundle["age_delta_conditioner"].named_parameters()
    )
    for key, module in (("vae", bundle["vae"]), ("text", bundle["text_encoder"]), ("identity", loss_fn.identity_encoder), ("age", loss_fn.age_estimator)):
        assert all(torch.equal(parameter, frozen[key][name]) for name, parameter in module.named_parameters())
    root = tmp_path / "checkpoints"
    assert (root / "latest" / "training_resume.pt").exists()
    assert (root / "latest" / "adapter_inference.pt").exists()
    assert (root / "best" / "training_resume.pt").exists()
    history = json.loads((root / "history.json").read_text())
    assert history["optimizer_step"] if "optimizer_step" in history else result["optimizer_step"] == 2
    inference = torch.load(root / "latest" / "adapter_inference.pt", weights_only=True)
    resume_payload = torch.load(root / "latest" / "training_resume.pt", weights_only=False)
    assert resume_payload["training_config"]["use_bidirectional_training"] is True
    assert resume_payload["training_config"]["include_bidirectional_pairs"] is True
    assert resume_payload["training_config"]["reverse_pair_prob"] == 0.20
    assert set(inference["adapter_state_dict"]) == set(bundle["trainable_param_names"])
    assert inference["model_id"] == bundle["model_id"]
    reloaded_bundle = make_training_bundle(seed=991)
    load_report = load_face_aging_adapter(
        reloaded_bundle, root / "latest" / "adapter_inference.pt"
    )
    assert load_report["loaded_tensors"] == len(bundle["trainable_param_names"])
    for name, parameter in bundle["unet"].named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, dict(reloaded_bundle["unet"].named_parameters())[name])


def test_interrupted_resume_matches_uninterrupted_trajectory(tiny_root, tmp_path):
    def loaders():
        return build_face_aging_dataloaders(
            tiny_root, image_size=32, batch_size=2, num_workers=0,
            train_drop_last=False, train_shuffle=False,
        )[0]
    common = dict(
        num_epochs=2, grad_accum_steps=2, max_train_batches=2, max_val_batches=1,
        lr_lora=2e-3, lr_conv_in=5e-4,
        amp_enabled=False, device="cpu", gradient_checkpointing=False, enable_xformers=False,
        min_snr_gamma=None, log_every=0, sample_target_posterior=False,
        deterministic=True, seed=77,
    )
    uninterrupted_bundle = make_training_bundle(seed=1234)
    uninterrupted_loss = make_training_loss(uninterrupted_bundle)
    uninterrupted_loaders = loaders()
    uninterrupted = TRAIN_AGGING_MODEL(
        bundle=uninterrupted_bundle, loss_fn=uninterrupted_loss,
        train_loader=uninterrupted_loaders["train"], val_loader=uninterrupted_loaders["val"],
        checkpoint_dir=tmp_path / "uninterrupted", sample_every_epochs=0,
        **common,
    )

    interrupted_bundle = make_training_bundle(seed=1234)
    interrupted_loss = make_training_loss(interrupted_bundle)
    interrupted_loaders = loaders()
    def interrupt_after_checkpoint(**kwargs):
        raise KeyboardInterrupt
    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        TRAIN_AGGING_MODEL(
            bundle=interrupted_bundle, loss_fn=interrupted_loss,
            train_loader=interrupted_loaders["train"], val_loader=interrupted_loaders["val"],
            checkpoint_dir=interrupted_root, sample_fn=interrupt_after_checkpoint,
            sample_every_epochs=1, **common,
        )
    emergency = interrupted_root / "interrupted_training_resume.pt"
    assert emergency.exists()

    resumed_bundle = make_training_bundle(seed=1234)
    resumed_loss = make_training_loss(resumed_bundle)
    resumed_loaders = loaders()
    resumed = TRAIN_AGGING_MODEL(
        bundle=resumed_bundle, loss_fn=resumed_loss,
        train_loader=resumed_loaders["train"], val_loader=resumed_loaders["val"],
        checkpoint_dir=tmp_path / "resumed", resume_from=emergency,
        sample_every_epochs=0, **common,
    )
    assert uninterrupted["optimizer_step"] == resumed["optimizer_step"] == 2
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        ((name, parameter) for name, parameter in uninterrupted_bundle["unet"].named_parameters() if parameter.requires_grad),
        ((name, parameter) for name, parameter in resumed_bundle["unet"].named_parameters() if parameter.requires_grad),
    ):
        assert name_a == name_b
        assert torch.equal(parameter_a, parameter_b), name_a
    for parameter_a, parameter_b in zip(
        uninterrupted_bundle["age_delta_conditioner"].parameters(),
        resumed_bundle["age_delta_conditioner"].parameters(),
    ):
        assert torch.equal(parameter_a, parameter_b)
    assert uninterrupted["lr_scheduler"].get_last_lr() == resumed["lr_scheduler"].get_last_lr()
