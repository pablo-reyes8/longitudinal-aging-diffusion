from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.loss as loss_api
import src.loss.pretrained_auxiliary as auxiliary_loading
import src.model.load_diffusion_models as model_loading
from src.model import get_bundle_trainable_named_parameters
from model_fakes import make_fake_components
from src.loss import (
    AGE_MODEL_ID,
    ArcFaceR50InputAdapter,
    IDENTITY_MODEL_ID,
    IdentityEncoderAdapter,
    MiVOLOFaceOnlyAgeModel,
)
from src.loss.pretrained_auxiliary import _load_mivolo


class RecordingArcFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_input = None

    def forward(self, images):
        self.last_input = images.detach().clone()
        return images.mean(dim=(2, 3)) * self.scale


class RecordingMiVOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.faces = None
        self.bodies = None

    def forward(self, *, faces_input, body_input, return_dict=True):
        self.faces = faces_input.detach().clone()
        self.bodies = body_input.detach().clone()
        age = faces_input.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * self.scale
        return SimpleNamespace(age_output=age)


def test_arcface_bridge_resizes_preserves_gradient_and_freezes_weights():
    raw = RecordingArcFace().half()
    adapter = IdentityEncoderAdapter(
        ArcFaceR50InputAdapter(raw), activation_checkpointing=True
    )
    images = torch.rand(2, 3, 32, 40, requires_grad=True)
    embeddings = adapter(images)
    embeddings.sum().backward()
    assert raw.last_input.shape == (2, 3, 112, 112)
    assert raw.last_input.dtype == torch.float32
    assert next(raw.parameters()).dtype == torch.float32
    assert float(raw.last_input.min()) >= 0 and float(raw.last_input.max()) <= 1
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in adapter.parameters())


def test_arcface_bridge_is_fp32_inside_an_outer_cpu_autocast():
    raw = RecordingArcFace().to(dtype=torch.bfloat16)
    adapter = IdentityEncoderAdapter(ArcFaceR50InputAdapter(raw))
    images = torch.rand(2, 3, 16, 16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        embeddings = adapter(images)
    embeddings.sum().backward()
    assert raw.last_input.dtype == torch.float32
    assert next(raw.parameters()).dtype == torch.float32
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_mivolo_bridge_uses_384_face_and_official_missing_body_value():
    raw = RecordingMiVOLO().half()
    bridge = MiVOLOFaceOnlyAgeModel(raw)
    images = torch.rand(2, 3, 48, 32, requires_grad=True)
    ages = bridge(images)
    ages.sum().backward()
    assert ages.shape == (2, 1)
    assert raw.faces.shape == raw.bodies.shape == (2, 3, 384, 384)
    assert raw.faces.dtype == raw.bodies.dtype == torch.float32
    assert next(raw.parameters()).dtype == torch.float32
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(raw.bodies, (-mean / std).expand_as(raw.bodies))
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_mivolo_bridge_is_fp32_inside_an_outer_cpu_autocast():
    raw = RecordingMiVOLO().to(dtype=torch.bfloat16)
    bridge = MiVOLOFaceOnlyAgeModel(raw)
    images = torch.rand(2, 3, 24, 24, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ages = bridge(images)
    ages.sum().backward()
    assert raw.faces.dtype == raw.bodies.dtype == torch.float32
    assert next(raw.parameters()).dtype == torch.float32
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_activation_checkpointing_recomputes_frozen_auxiliary_forward():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return images.flatten(1) * self.weight

    model = CountingModel()
    adapter = IdentityEncoderAdapter(model, activation_checkpointing=True)
    images = torch.rand(2, 1, 2, 2, requires_grad=True)
    adapter(images).sum().backward()
    assert model.calls == 2
    assert images.grad is not None
    assert model.weight.grad is None and not model.weight.requires_grad


def test_mivolo_remote_code_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="trust_remote_code"):
        _load_mivolo(
            "iitolstykh/mivolo_v2",
            dtype=torch.float32,
            revision=None,
            token=None,
            cache_dir=None,
            local_files_only=True,
            trust_remote_code=False,
        )


def test_real_auxiliary_loader_forces_both_normalization_models_to_fp32(monkeypatch):
    observed = {}

    monkeypatch.setattr(
        auxiliary_loading,
        "_load_arcface",
        lambda *args, **kwargs: RecordingArcFace().half(),
    )

    def fake_load_mivolo(*args, **kwargs):
        observed["load_dtype"] = kwargs["dtype"]
        return RecordingMiVOLO().half()

    monkeypatch.setattr(auxiliary_loading, "_load_mivolo", fake_load_mivolo)
    loaded = auxiliary_loading.load_pretrained_auxiliary_models(
        device="cpu",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert observed["load_dtype"] == torch.float32
    assert loaded["requested_dtype"] == torch.bfloat16
    assert loaded["identity_dtype"] == loaded["age_dtype"] == torch.float32
    assert next(loaded["identity_encoder"].parameters()).dtype == torch.float32
    assert next(loaded["age_estimator"].parameters()).dtype == torch.float32


def test_bundle_option_attaches_auxiliaries_without_making_them_trainable(monkeypatch):
    identity = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    age = nn.Sequential(nn.Flatten(), nn.Linear(12, 1))
    identity.requires_grad_(False)
    age.requires_grad_(False)
    observed = {}

    def fake_auxiliary_loader(**kwargs):
        observed.update(kwargs)
        return {
            "identity_encoder": identity,
            "age_estimator": age,
            "identity_model_id": kwargs["identity_model_id"],
            "age_model_id": kwargs["age_model_id"],
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "activation_checkpointing": True,
            "mivolo_body_input": "normalized_black_missing_crop",
        }

    monkeypatch.setattr(model_loading, "load_diffusion_components", lambda *args, **kwargs: make_fake_components())
    monkeypatch.setattr(loss_api, "load_pretrained_auxiliary_models", fake_auxiliary_loader)
    bundle = model_loading.build_face_aging_diffusion_bundle(
        model_id="fake/sd15",
        rank=2,
        alpha=2,
        load_auxiliary_models=True,
        auxiliary_trust_remote_code=True,
        verbose=False,
    )
    assert bundle["identity_encoder"] is identity
    assert bundle["age_estimator"] is age
    assert observed["device"] == torch.device("cpu")
    assert observed["trust_remote_code"] is True
    assert set(bundle["trainable_param_names"]) == {
        name for name, _ in get_bundle_trainable_named_parameters(bundle)
    }
    assert all(not parameter.requires_grad for parameter in identity.parameters())
    assert all(not parameter.requires_grad for parameter in age.parameters())


def test_recommended_yaml_uses_bounded_auxiliary_memory_policy():
    from scripts.common import load_yaml

    model_config = load_yaml("config/models/sd15_lora.yaml")
    train_config = load_yaml("config/training/photo_editing.yaml")
    assert model_config["load_auxiliary_models"] is True
    assert model_config["auxiliary_activation_checkpointing"] is True
    assert train_config["loss"]["auxiliary_every_n_steps"] == 4
    assert train_config["loss"]["auxiliary_sample_fraction"] == 0.25
    assert train_config["loss"]["vae_decode_checkpointing"] is True
    assert train_config["auxiliary_max_timestep"] == 400
    assert IDENTITY_MODEL_ID == model_config["identity_model_id"]
    assert AGE_MODEL_ID == model_config["age_model_id"]
