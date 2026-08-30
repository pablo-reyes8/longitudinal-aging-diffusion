from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.inference import infer_face_aging_direct
from src.model import (
    AgeConditionerV2,
    AgeDeltaConditioner,
    assemble_face_aging_diffusion_bundle,
    build_face_aging_optimizer,
    compute_age_delta_embedding,
    load_face_aging_adapter,
    save_face_aging_adapter,
)
from model_fakes import make_fake_components
from src.training import run_training_step
from training_fakes import make_training_bundle, make_training_loss


def synthetic_batch(batch_size=2):
    return {
        "source_image": torch.full((batch_size, 3, 32, 32), -0.5),
        "target_image": torch.full((batch_size, 3, 32, 32), 0.4),
        "source_age": torch.tensor([20, 30][:batch_size]),
        "target_age": torch.tensor([35, 65][:batch_size]),
        "delta_age": torch.tensor([15, 35][:batch_size]),
        "target_prompt": ["photo of a person as 35-year-old", "photo of a person as 65-year-old"][:batch_size],
        "generic_prompt": ["photo of a person"] * batch_size,
    }


def test_age_delta_conditioner_shape_determinism_variation_zero_and_negative():
    torch.manual_seed(70)
    conditioner = AgeDeltaConditioner(hidden_dim=16, output_dim=32, age_delta_scale=80)
    one_dimensional = conditioner(torch.tensor([0.0, 20.0, -20.0]))
    column = conditioner(torch.tensor([[0.0], [20.0], [-20.0]]))
    assert one_dimensional.shape == (3, 32)
    assert torch.equal(one_dimensional, column)
    assert torch.count_nonzero(one_dimensional[0]) == 0
    assert not torch.equal(one_dimensional[1], one_dimensional[2])
    assert torch.isfinite(one_dimensional).all()


def test_v2_contract_survives_notebook_style_class_reload():
    class ReloadedAgeConditionerV2(torch.nn.Module):
        """V2-compatible object intentionally failing local isinstance checks."""

        def __init__(self):
            super().__init__()
            self.inner = AgeConditionerV2(
                num_fourier_frequencies=2, hidden_dim=8, output_dim=16
            )

        def get_config(self):
            return self.inner.get_config()

        def forward(self, source_age, target_age, delta_age):
            return self.inner(source_age, target_age, delta_age)

    conditioner = ReloadedAgeConditionerV2()
    assert not isinstance(conditioner, AgeConditionerV2)
    bundle = {
        "use_age_delta_conditioning": True,
        "age_conditioner": conditioner,
        "age_conditioning_version": "v2_fourier",
    }
    output = compute_age_delta_embedding(
        bundle,
        torch.tensor([-20.0, 0.0, 20.0]),
        batch_size=3,
        source_age=torch.tensor([50.0, 30.0, 20.0]),
        target_age=torch.tensor([30.0, 30.0, 40.0]),
    )
    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_age_conditioner_v2_fourier_shapes_determinism_and_age_sensitivity():
    torch.manual_seed(71)
    conditioner = AgeConditionerV2(
        num_fourier_frequencies=8, hidden_dim=32, output_dim=64,
        use_raw_scalars=True, use_gate=True,
    )
    source = torch.tensor([10.0, 26.0, 80.0, 50.0])
    target = torch.tensor([10.0, 30.0, 110.0, 30.0])
    delta = torch.tensor([0.0, 4.0, 30.0, -20.0])
    features = conditioner.fourier_features(source, target, delta)
    repeated = conditioner.fourier_features(source[:, None], target[:, None], delta[:, None])
    output = conditioner(source, target, delta)
    assert features.shape == (4, 3 + 3 * 2 * 8)
    assert torch.equal(features, repeated)
    assert output.shape == (4, 64) and torch.isfinite(output).all()
    assert not torch.equal(
        conditioner(torch.tensor([20.0]), torch.tensor([30.0]), torch.tensor([10.0])),
        conditioner(torch.tensor([20.0]), torch.tensor([60.0]), torch.tensor([40.0])),
    )
    assert not torch.equal(
        conditioner(torch.tensor([20.0]), torch.tensor([30.0]), torch.tensor([10.0])),
        conditioner(torch.tensor([40.0]), torch.tensor([50.0]), torch.tensor([10.0])),
    )
    signed = conditioner(
        torch.tensor([40.0, 40.0, 40.0]),
        torch.tensor([60.0, 40.0, 20.0]),
        torch.tensor([20.0, 0.0, -20.0]),
    )
    assert signed.shape == (3, 64) and torch.isfinite(signed).all()
    assert not torch.equal(signed[0], signed[1])
    assert not torch.equal(signed[0], signed[2])
    assert not torch.equal(signed[1], signed[2])


def test_age_conditioner_v2_gate_is_trainable_and_zero_recovers_base_timestep_path():
    bundle = make_training_bundle(seed=904)
    conditioner = bundle["age_conditioner"]
    assert conditioner.age_scale is not None and conditioner.age_scale.requires_grad
    sample = torch.randn(2, 8, 4, 4)
    timesteps = torch.tensor([3, 7])
    hidden = torch.randn(2, 12, 6)
    with torch.no_grad():
        conditioner.age_scale.zero_()
    condition = conditioner(
        torch.tensor([20.0, 30.0]), torch.tensor([40.0, 65.0]), torch.tensor([20.0, 35.0])
    )
    assert torch.count_nonzero(condition) == 0
    with_condition = bundle["unet"](sample, timesteps, hidden, timestep_cond=condition).sample
    without_condition = bundle["unet"](sample, timesteps, hidden).sample
    assert torch.equal(with_condition, without_condition)
    with torch.no_grad():
        conditioner.age_scale.fill_(1.0)
    conditioner.zero_grad(set_to_none=True)
    conditioner(
        torch.tensor([20.0, 30.0]), torch.tensor([40.0, 65.0]), torch.tensor([20.0, 35.0])
    ).square().mean().backward()
    assert conditioner.age_scale.grad is not None and torch.isfinite(conditioner.age_scale.grad)


def test_bundle_delta_embedding_changes_unet_output_and_receives_optimizer_update():
    bundle = make_training_bundle(seed=901)
    loss_fn = make_training_loss(bundle, auxiliaries=False)
    batch = synthetic_batch(2)
    before = {name: value.detach().clone() for name, value in bundle["age_delta_conditioner"].named_parameters()}
    optimizer = build_face_aging_optimizer(
        bundle, lr_lora=1e-2, lr_conv_in=1e-2, lr_age_conditioner=1e-2, weight_decay=0,
    )
    result = run_training_step(
        bundle=bundle, loss_fn=loss_fn, batch=batch, device=torch.device("cpu"),
        amp_enabled=False, sample_target_posterior=False,
    )
    result["loss_out"]["loss"].backward()
    assert all(parameter.grad is not None for parameter in bundle["age_delta_conditioner"].parameters())
    optimizer.step()
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in bundle["age_delta_conditioner"].named_parameters()
    )

    source = torch.tensor([20.0, 30.0])
    first = compute_age_delta_embedding(
        bundle, torch.tensor([10.0, 10.0]), batch_size=2,
        source_age=source, target_age=source + 10,
    )
    second = compute_age_delta_embedding(
        bundle, torch.tensor([40.0, 40.0]), batch_size=2,
        source_age=source, target_age=source + 40,
    )
    assert not torch.equal(first, second)


def test_conditioner_checkpoint_roundtrip(tmp_path):
    bundle = make_training_bundle(seed=902)
    with torch.no_grad():
        for parameter in bundle["age_delta_conditioner"].parameters():
            parameter.add_(0.123)
    checkpoint = save_face_aging_adapter(bundle, tmp_path / "adapter.pt")
    rebuilt = make_training_bundle(seed=902)
    report = load_face_aging_adapter(rebuilt, checkpoint)
    assert report["loaded_tensors"] == len(bundle["trainable_param_names"])
    for (name_a, value_a), (name_b, value_b) in zip(
        bundle["age_delta_conditioner"].state_dict().items(),
        rebuilt["age_delta_conditioner"].state_dict().items(),
    ):
        assert name_a == name_b and torch.equal(value_a, value_b)


def test_legacy_v1_checkpoint_without_v2_metadata_still_loads(tmp_path):
    def legacy_bundle(seed):
        torch.manual_seed(seed)
        return assemble_face_aging_diffusion_bundle(
            make_fake_components(), model_id="offline/legacy-v1", rank=2, alpha=2,
            use_age_conditioner_v2=False, age_conditioning_version="v1_delta",
            age_condition_hidden_dim=128, verbose=False,
        )
    original = legacy_bundle(905)
    checkpoint = save_face_aging_adapter(original, tmp_path / "legacy.pt")
    payload = torch.load(checkpoint, weights_only=True)
    for key in (
        "use_age_conditioner_v2", "age_conditioning_version", "num_fourier_frequencies",
        "age_condition_use_raw_scalars", "age_condition_use_gate",
    ):
        payload["config"].pop(key, None)
    torch.save(payload, checkpoint)
    rebuilt = legacy_bundle(906)
    report = load_face_aging_adapter(rebuilt, checkpoint)
    assert report["loaded_tensors"] == len(original["trainable_param_names"])


def test_v2_checkpoint_rejects_fourier_configuration_mismatch(tmp_path):
    original = make_training_bundle(seed=906)
    checkpoint = save_face_aging_adapter(original, tmp_path / "v2.pt")
    torch.manual_seed(906)
    incompatible = assemble_face_aging_diffusion_bundle(
        make_fake_components(), model_id="offline/sd15-shaped", rank=2, alpha=2,
        num_fourier_frequencies=4, verbose=False,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_face_aging_adapter(incompatible, checkpoint)


def test_inference_requires_source_age_and_same_prompt_responds_to_delta():
    bundle = make_training_bundle(seed=903)
    image = Image.new("RGB", (32, 32), (100, 80, 60))
    with pytest.raises(ValueError, match="source_age and target_age"):
        infer_face_aging_direct(
            bundle=bundle, image=image, target_age=50,
            num_inference_steps=3, strength=0.5, image_size=32,
        )
    common = dict(
        bundle=bundle, image=image, source_age=30,
        target_prompt="photo of a person",
        num_inference_steps=3, strength=0.5, image_size=32, seed=19,
        compute_diagnostics=False,
    )
    small_delta = infer_face_aging_direct(**common, target_age=35)
    large_delta = infer_face_aging_direct(**common, target_age=65)
    assert small_delta["target_prompt"] == large_delta["target_prompt"]
    assert small_delta["metadata"]["delta_age"] == 5
    assert large_delta["metadata"]["delta_age"] == 35
    assert not torch.equal(small_delta["image_tensor"], large_delta["image_tensor"])
