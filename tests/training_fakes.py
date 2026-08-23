from __future__ import annotations

import copy

import torch
from torch import nn

from loss_fakes import TinyAgeModel, TinyIdentityModel
from model_fakes import FakeScheduler, make_fake_components
from src.loss import AgeEstimatorAdapter, FaceAgingDiffusionLoss, IdentityEncoderAdapter
from src.model import assemble_face_aging_diffusion_bundle


class CountingScheduler(FakeScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_noise_calls = 0

    def add_noise(self, latents, noise, timesteps):
        self.add_noise_calls += 1
        return super().add_noise(latents, noise, timesteps)


class RGBIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Tanh(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())

    def forward(self, images):
        return self.features(images)


class RGBAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 1, 1)

    def forward(self, images):
        return self.projection(images).mean((1, 2, 3)) * 10 + 35


def make_training_bundle(seed=123, *, counting_scheduler=False):
    torch.manual_seed(seed)
    components = make_fake_components()
    if counting_scheduler:
        components["scheduler_train"] = CountingScheduler()
    return assemble_face_aging_diffusion_bundle(
        components, model_id="offline/sd15-shaped", rank=2, alpha=2, verbose=False
    )


def make_training_loss(bundle, *, auxiliaries=True):
    return FaceAgingDiffusionLoss(
        scheduler=bundle["scheduler_train"],
        vae=bundle["vae"],
        identity_encoder=IdentityEncoderAdapter(RGBIdentity()) if auxiliaries else None,
        age_estimator=AgeEstimatorAdapter(RGBAge()) if auxiliaries else None,
        identity_weight=0.1 if auxiliaries else 0.0,
        age_weight=0.1 if auxiliaries else 0.0,
    )


def clone_module_parameters(module):
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
