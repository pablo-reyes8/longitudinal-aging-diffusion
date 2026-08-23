from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class NumericalScheduler:
    def __init__(self, prediction_type="epsilon", steps=1000, dtype=torch.float64):
        self.config = SimpleNamespace(prediction_type=prediction_type, num_train_timesteps=steps)
        betas = torch.linspace(0.0001, 0.02, steps, dtype=dtype)
        self.alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    def add_noise(self, x0, noise, timesteps):
        alpha = self.alphas_cumprod.to(x0)[timesteps].reshape((-1,) + (1,) * (x0.ndim - 1))
        return alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise

    def get_velocity(self, x0, noise, timesteps):
        alpha = self.alphas_cumprod.to(x0)[timesteps].reshape((-1,) + (1,) * (x0.ndim - 1))
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * x0


class TinyVAE(nn.Module):
    def __init__(self, channels=1, scaling_factor=0.25):
        super().__init__()
        self.decoder = nn.Conv2d(channels, channels, 1, bias=True)
        with torch.no_grad():
            self.decoder.weight.fill_(0.7)
            self.decoder.bias.fill_(0.1)
        self.config = SimpleNamespace(scaling_factor=scaling_factor)

    def decode(self, latents):
        return SimpleNamespace(sample=self.decoder(latents))

    def encode(self, images):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=(images - self.decoder.bias.view(1, -1, 1, 1)) / self.decoder.weight.diagonal().view(1, -1, 1, 1)))


class TinyIdentityModel(nn.Module):
    def __init__(self, channels=1, embedding_dim=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.linear = nn.Linear(channels * 4, embedding_dim, bias=True)

    def forward(self, images):
        return self.linear(self.pool(images).flatten(1))


class TinyAgeModel(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.2, 0.5, channels))
        self.bias = nn.Parameter(torch.tensor(20.0))

    def forward(self, images):
        return (images.mean(dim=(2, 3)) * self.weight).sum(dim=1) + self.bias


class CountingIdentityModel(TinyIdentityModel):
    def __init__(self, channels=1):
        super().__init__(channels=channels)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return super().forward(images)


class CountingAgeModel(TinyAgeModel):
    def __init__(self, channels=1):
        super().__init__(channels=channels)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return super().forward(images)
