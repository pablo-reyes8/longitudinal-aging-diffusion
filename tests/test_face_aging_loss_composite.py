from __future__ import annotations

import copy
import itertools

import pytest
import torch

from loss_fakes import (
    CountingAgeModel,
    CountingIdentityModel,
    NumericalScheduler,
    TinyAgeModel,
    TinyIdentityModel,
    TinyVAE,
)
from src.loss import (
    AgeEstimatorAdapter,
    FaceAgingDiffusionLoss,
    IdentityEncoderAdapter,
    compose_weighted_losses,
)


def make_loss(
    *,
    prediction_type="epsilon",
    dtype=torch.float64,
    identity_weight=0.1,
    age_weight=0.1,
    identity_reference="target",
    age_loss_type="l1",
    **kwargs,
):
    scheduler = NumericalScheduler(prediction_type, steps=100, dtype=dtype)
    vae = TinyVAE().to(dtype)
    identity = IdentityEncoderAdapter(TinyIdentityModel().to(dtype)) if identity_weight > 0 else None
    age = AgeEstimatorAdapter(TinyAgeModel().to(dtype)) if age_weight > 0 else None
    loss = FaceAgingDiffusionLoss(
        scheduler=scheduler,
        vae=vae,
        identity_encoder=identity,
        age_estimator=age,
        identity_weight=identity_weight,
        age_weight=age_weight,
        identity_reference=identity_reference,
        age_loss_type=age_loss_type,
        clamp_pred_x0=False,
        **kwargs,
    )
    return loss


def make_inputs(loss_fn, batch=4, dtype=torch.float64, requires_grad=True):
    torch.manual_seed(200 + batch)
    x0 = torch.randn(batch, 1, 2, 2, dtype=dtype) * 0.2
    noise = torch.randn_like(x0) * 0.3
    timesteps = torch.linspace(0, 99, batch).long()
    noisy = loss_fn.scheduler.add_noise(x0, noise, timesteps)
    model_pred = (noise + 0.07 * torch.randn_like(noise)).requires_grad_(requires_grad)
    return {
        "model_pred": model_pred,
        "noise": noise,
        "noisy_target_latents": noisy,
        "target_latents": x0,
        "timesteps": timesteps,
        "source_images": torch.randn(batch, 1, 2, 2, dtype=dtype).clamp(-1, 1),
        "target_images": torch.randn(batch, 1, 2, 2, dtype=dtype).clamp(-1, 1),
        "target_ages": torch.linspace(20, 50, batch),
        "global_step": 0,
    }


def test_weighted_composition_exact_oracle_and_negative_rejection():
    total, diff, identity, age = compose_weighted_losses(
        torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0),
        diffusion_weight=1.0, identity_weight=0.1, age_weight=0.2,
    )
    assert total.item() == pytest.approx(3.1, abs=1e-7)
    assert (diff.item(), identity.item(), age.item()) == pytest.approx((2.0, 0.3, 0.8))
    with pytest.raises(ValueError):
        compose_weighted_losses(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), diffusion_weight=1, identity_weight=-1, age_weight=0)


@pytest.mark.parametrize("identity_weight,age_weight", itertools.product((0.0, 0.1), repeat=2))
def test_component_independence_and_exact_total(identity_weight, age_weight):
    loss_fn = make_loss(identity_weight=identity_weight, age_weight=age_weight)
    output = loss_fn(**make_inputs(loss_fn))
    expected = output["weighted_diff"] + output["weighted_id"] + output["weighted_age"]
    assert torch.equal(output["loss"], expected)
    assert (output["loss_id"].item() == 0) == (identity_weight == 0)
    assert (output["loss_age"].item() == 0) == (age_weight == 0)


def test_zero_auxiliary_weights_do_not_decode_or_call_models():
    class ExplodingVAE(TinyVAE):
        def decode(self, latents):
            raise AssertionError("decode should not be called")
    loss_fn = FaceAgingDiffusionLoss(
        scheduler=NumericalScheduler(), vae=ExplodingVAE().double(),
        identity_weight=0, age_weight=0,
    )
    output = loss_fn(**make_inputs(loss_fn))
    assert not output["auxiliary_applied"]


def test_auxiliary_cadence_exact_for_twenty_steps():
    loss_fn = make_loss(auxiliary_every_n_steps=4)
    inputs = make_inputs(loss_fn)
    observed = []
    for step in range(20):
        inputs["global_step"] = step
        observed.append(loss_fn(**inputs)["auxiliary_applied"])
    assert observed == [step % 4 == 0 for step in range(20)]


def test_auxiliary_subset_reproducibility_alignment_and_fraction_mean():
    first = make_loss(auxiliary_sample_fraction=0.5, auxiliary_seed=123)
    second = make_loss(auxiliary_sample_fraction=0.5, auxiliary_seed=123)
    third = make_loss(auxiliary_sample_fraction=0.5, auxiliary_seed=999)
    inputs = make_inputs(first, batch=10)
    indices_a = first(**inputs, return_per_sample=True)["auxiliary_indices"]
    indices_b = second(**inputs, return_per_sample=True)["auxiliary_indices"]
    indices_c = third(**inputs, return_per_sample=True)["auxiliary_indices"]
    assert torch.equal(indices_a, indices_b)
    assert not torch.equal(indices_a, indices_c)
    output = first(**inputs, return_per_sample=True)
    assert len(indices_a) == 5
    assert torch.equal(output["loss_id"], output["loss_id_per_sample"].mean())
    assert torch.equal(output["loss_age"], output["loss_age_per_sample"].mean())


def test_auxiliary_max_timestep_filters_only_auxiliary_not_diffusion():
    loss_fn = make_loss(auxiliary_max_timestep=30)
    inputs = make_inputs(loss_fn, batch=4)
    output = loss_fn(**inputs, return_per_sample=True)
    assert torch.all(inputs["timesteps"][output["auxiliary_indices"]] <= 30)
    assert output["loss_diff_per_sample"].shape == (4,)


@pytest.mark.parametrize("identity_reference", ["source", "target", "both"])
def test_identity_reference_modes_against_explicit_adapter_oracle(identity_reference):
    loss_fn = make_loss(identity_reference=identity_reference, age_weight=0)
    inputs = make_inputs(loss_fn, batch=3)
    output = loss_fn(**inputs, return_reconstructions=True, return_per_sample=True)
    pred_embeddings = loss_fn.identity_encoder(output["pred_x0_images"])
    references = []
    if identity_reference in {"source", "both"}:
        references.append(loss_fn.identity_encoder.encode_reference((inputs["source_images"] / 2 + 0.5).clamp(0, 1)))
    if identity_reference in {"target", "both"}:
        references.append(loss_fn.identity_encoder.encode_reference((inputs["target_images"] / 2 + 0.5).clamp(0, 1)))
    oracle = torch.stack([1 - (pred_embeddings * ref).sum(-1) for ref in references]).mean(0)
    assert torch.allclose(output["loss_id_per_sample"], oracle, atol=1e-12)


@pytest.mark.parametrize("age_loss_type", ["l1", "mse"])
@pytest.mark.parametrize("age_dtype", [torch.int64, torch.int32, torch.float32])
def test_age_loss_type_and_target_dtype_normalization(age_loss_type, age_dtype):
    loss_fn = make_loss(identity_weight=0, age_loss_type=age_loss_type)
    inputs = make_inputs(loss_fn, batch=3)
    inputs["target_ages"] = torch.tensor([20, 30, 40], dtype=age_dtype)
    output = loss_fn(**inputs, return_reconstructions=True, return_per_sample=True)
    predicted = loss_fn.age_estimator(output["pred_x0_images"])
    difference = predicted - inputs["target_ages"].float()
    oracle = difference.abs() if age_loss_type == "l1" else difference.square()
    assert torch.allclose(output["loss_age_per_sample"], oracle, atol=1e-12)


def test_target_age_python_list_matches_tensor():
    loss_fn = make_loss(identity_weight=0)
    inputs = make_inputs(loss_fn, batch=3)
    inputs["target_ages"] = [20, 30, 40]
    list_loss = loss_fn(**inputs)["loss_age"]
    inputs["target_ages"] = torch.tensor([20, 30, 40])
    tensor_loss = loss_fn(**inputs)["loss_age"]
    assert torch.equal(list_loss, tensor_loss)


def test_per_sample_reduction_duplicate_and_batch_size_invariance():
    loss_fn = make_loss(identity_weight=0, age_weight=0)
    single = make_inputs(loss_fn, batch=1)
    expected = loss_fn(**single, return_per_sample=True)["loss_diff"]
    for batch_size in (2, 4, 8):
        duplicated = {
            key: (value.repeat((batch_size,) + (1,) * (value.ndim - 1)) if isinstance(value, torch.Tensor) and value.ndim else value)
            for key, value in single.items()
        }
        duplicated["timesteps"] = single["timesteps"].repeat(batch_size)
        duplicated["target_ages"] = single["target_ages"].repeat(batch_size)
        observed = loss_fn(**duplicated, return_per_sample=True)
        assert torch.allclose(observed["loss_diff_per_sample"], expected.repeat(batch_size))
        assert torch.equal(observed["loss_diff"], expected)


def test_batch_permutation_invariance_and_per_sample_alignment():
    loss_fn = make_loss(auxiliary_sample_fraction=1)
    inputs = make_inputs(loss_fn, batch=5)
    original = loss_fn(**inputs, return_per_sample=True)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_inputs = {
        key: (value[permutation] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 5 else value)
        for key, value in inputs.items()
    }
    permuted = loss_fn(**permuted_inputs, return_per_sample=True)
    inverse = torch.argsort(permutation)
    assert torch.allclose(original["loss"], permuted["loss"], atol=1e-12)
    for key in ("loss_diff_per_sample", "loss_id_per_sample", "loss_age_per_sample"):
        assert torch.allclose(original[key], permuted[key][inverse], atol=1e-12)


def test_gradient_decomposition_identity_and_non_detached_auxiliary_paths():
    loss_fn = make_loss(diffusion_weight=1.3, identity_weight=0.2, age_weight=0.05, age_loss_type="mse")
    inputs = make_inputs(loss_fn, batch=3)
    output = loss_fn(**inputs)
    prediction = inputs["model_pred"]
    g_total = torch.autograd.grad(output["loss"], prediction, retain_graph=True)[0]
    g_diff = torch.autograd.grad(output["loss_diff"], prediction, retain_graph=True)[0]
    g_id = torch.autograd.grad(output["loss_id"], prediction, retain_graph=True)[0]
    g_age = torch.autograd.grad(output["loss_age"], prediction)[0]
    expected = 1.3 * g_diff + 0.2 * g_id + 0.05 * g_age
    assert torch.allclose(g_total, expected, atol=2e-12, rtol=2e-11)
    assert g_id.norm() > 0 and g_age.norm() > 0


def test_total_loss_coordinate_and_directional_finite_differences():
    loss_fn = make_loss(identity_reference="both", age_loss_type="mse")
    base_inputs = make_inputs(loss_fn, batch=1)
    base = base_inputs["model_pred"].detach().clone().requires_grad_(True)
    def objective(value):
        kwargs = {**base_inputs, "model_pred": value}
        return loss_fn(**kwargs)["loss"]
    analytical = torch.autograd.grad(objective(base), base)[0]
    h = 1e-6
    for index in range(base.numel()):
        plus, minus = base.detach().clone(), base.detach().clone()
        plus.view(-1)[index] += h; minus.view(-1)[index] -= h
        numerical = (objective(plus) - objective(minus)) / (2 * h)
        assert analytical.view(-1)[index].item() == pytest.approx(numerical.item(), rel=3e-5, abs=3e-7)
    torch.manual_seed(91)
    for _ in range(5):
        direction = torch.randn_like(base); direction /= direction.norm()
        numerical = (objective(base.detach() + h * direction) - objective(base.detach() - h * direction)) / (2 * h)
        directional = (analytical * direction).sum()
        assert directional.item() == pytest.approx(numerical.item(), rel=3e-5, abs=3e-7)


def test_reproducibility_no_graph_retention_and_output_tensor_policy():
    loss_fn = make_loss(auxiliary_sample_fraction=0.5, auxiliary_seed=55)
    inputs = make_inputs(loss_fn, batch=6)
    first = loss_fn(**inputs, return_per_sample=True)
    second = loss_fn(**inputs, return_per_sample=True)
    for key in ("loss", "loss_diff", "loss_id", "loss_age"):
        assert torch.equal(first[key], second[key])
    assert torch.equal(first["auxiliary_indices"], second["auxiliary_indices"])
    assert "pred_x0_images" not in first and "pred_x0_latents" not in first
    first["loss"].backward()
    inputs = make_inputs(loss_fn, batch=6)
    loss_fn(**inputs)["loss"].backward()


def test_config_roundtrip_preserves_all_scientific_hyperparameters():
    original = make_loss(
        identity_reference="both", age_loss_type="mse", min_snr_gamma=5,
        auxiliary_every_n_steps=3, auxiliary_sample_fraction=0.5,
        auxiliary_max_timestep=50, auxiliary_seed=987,
    )
    config = original.get_config()
    rebuilt = FaceAgingDiffusionLoss.from_config(
        config,
        scheduler=original.scheduler,
        vae=TinyVAE().double(),
        identity_encoder=IdentityEncoderAdapter(TinyIdentityModel().double()),
        age_estimator=AgeEstimatorAdapter(TinyAgeModel().double()),
    )
    assert rebuilt.get_config() == config


@pytest.mark.parametrize(
    "override,match",
    [
        ({"diffusion_weight": 0}, "diffusion_weight"),
        ({"identity_weight": -1}, "weights"),
        ({"age_weight": -1}, "weights"),
        ({"identity_reference": "wrong"}, "identity_reference"),
        ({"diffusion_loss_type": "l1"}, "diffusion_loss_type"),
        ({"age_loss_type": "huber"}, "age_loss_type"),
        ({"min_snr_gamma": 0}, "min_snr_gamma"),
        ({"auxiliary_every_n_steps": 0}, "auxiliary_every_n_steps"),
        ({"auxiliary_sample_fraction": 0}, "auxiliary_sample_fraction"),
        ({"auxiliary_sample_fraction": 1.1}, "auxiliary_sample_fraction"),
        ({"auxiliary_max_timestep": -1}, "auxiliary_max_timestep"),
    ],
)
def test_unknown_invalid_scientific_config_rejected(override, match):
    kwargs = dict(scheduler=NumericalScheduler(), vae=TinyVAE(), identity_weight=0, age_weight=0)
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        FaceAgingDiffusionLoss(**kwargs)


def test_shape_contracts_and_nonfinite_debug_detection():
    loss_fn = make_loss()
    inputs = make_inputs(loss_fn, batch=2)
    bad = dict(inputs); bad["noise"] = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="noise shape"):
        loss_fn(**bad)
    bad = dict(inputs); bad["timesteps"] = torch.tensor([1])
    with pytest.raises(ValueError, match="timesteps"):
        loss_fn(**bad)
    bad = dict(inputs); bad["target_ages"] = torch.tensor([[20.0, 30.0]])
    with pytest.raises(ValueError, match="target_ages"):
        loss_fn(**bad)
    bad = dict(inputs); bad["source_images"] = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    source_loss = make_loss(identity_reference="source")
    with pytest.raises(ValueError, match="source_images"):
        source_loss(**bad)
    for tensor_name in ("model_pred", "noise", "target_latents"):
        bad = dict(inputs); bad[tensor_name] = bad[tensor_name].clone(); bad[tensor_name].view(-1)[0] = float("nan")
        with pytest.raises(FloatingPointError, match=tensor_name):
            loss_fn(**bad)


def test_auxiliary_models_are_frozen_eval_and_unchanged():
    loss_fn = make_loss()
    before = {
        "vae": [p.detach().clone() for p in loss_fn.vae.parameters()],
        "identity": [p.detach().clone() for p in loss_fn.identity_encoder.parameters()],
        "age": [p.detach().clone() for p in loss_fn.age_estimator.parameters()],
    }
    loss_fn.train()
    assert not loss_fn.vae.training and not loss_fn.identity_encoder.training and not loss_fn.age_estimator.training
    inputs = make_inputs(loss_fn)
    loss_fn(**inputs)["loss"].backward()
    for key, module in (("vae", loss_fn.vae), ("identity", loss_fn.identity_encoder), ("age", loss_fn.age_estimator)):
        assert all(p.grad is None and not p.requires_grad for p in module.parameters())
        assert all(torch.equal(p, old) for p, old in zip(module.parameters(), before[key]))


def test_v_prediction_composite_uses_velocity_not_noise():
    loss_fn = make_loss(prediction_type="v_prediction", identity_weight=0, age_weight=0)
    inputs = make_inputs(loss_fn, batch=4)
    exact_velocity = loss_fn.scheduler.get_velocity(inputs["target_latents"], inputs["noise"], inputs["timesteps"])
    inputs["model_pred"] = exact_velocity.requires_grad_(True)
    output = loss_fn(**inputs)
    assert output["loss_diff"].item() == 0.0
    inputs["model_pred"] = inputs["noise"].clone().requires_grad_(True)
    assert loss_fn(**inputs)["loss_diff"] > 0


def test_sample_isolation_changes_only_modified_observation():
    loss_fn = make_loss(identity_weight=0, age_weight=0)
    inputs = make_inputs(loss_fn, batch=4)
    original = loss_fn(**inputs, return_per_sample=True)["loss_diff_per_sample"]
    changed_inputs = dict(inputs)
    changed_inputs["model_pred"] = inputs["model_pred"].detach().clone()
    changed_inputs["model_pred"][2] += 1.0
    changed = loss_fn(**changed_inputs, return_per_sample=True)["loss_diff_per_sample"]
    assert torch.equal(original[[0, 1, 3]], changed[[0, 1, 3]])
    assert original[2] != changed[2]


def test_wrong_target_age_and_wrong_identity_reference_are_detectable():
    loss_fn = make_loss(identity_reference="target")
    inputs = make_inputs(loss_fn, batch=3)
    correct = loss_fn(**inputs)
    wrong_age_inputs = dict(inputs)
    wrong_age_inputs["target_ages"] = inputs["target_ages"] - 30
    wrong_age = loss_fn(**wrong_age_inputs)
    assert not torch.equal(correct["loss_age"], wrong_age["loss_age"])
    source_loss = make_loss(identity_reference="source", age_weight=0)
    target_loss = make_loss(identity_reference="target", age_weight=0)
    # Share identical encoder weights so only the selected reference differs.
    target_loss.identity_encoder.model.load_state_dict(source_loss.identity_encoder.model.state_dict())
    source_value = source_loss(**inputs)["loss_id"]
    target_value = target_loss(**inputs)["loss_id"]
    assert not torch.equal(source_value, target_value)


def test_one_dimensional_diffusion_gradient_points_toward_target():
    loss_fn = make_loss(identity_weight=0, age_weight=0)
    inputs = make_inputs(loss_fn, batch=1)
    prediction = torch.tensor([[[[3.0]]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    inputs.update(
        model_pred=prediction, noise=target, noisy_target_latents=torch.zeros_like(target),
        target_latents=torch.zeros_like(target), timesteps=torch.tensor([0]),
        source_images=torch.zeros_like(target), target_images=torch.zeros_like(target),
        target_ages=torch.tensor([20]),
    )
    gradient = torch.autograd.grad(loss_fn(**inputs)["loss"], prediction)[0]
    assert gradient.item() > 0  # gradient descent therefore decreases prediction toward 1
