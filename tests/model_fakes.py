"""Tiny SD-shaped doubles: no Diffusers dependency or network access."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class TokenBatch(dict):
    def __getattr__(self, name):
        return self[name]


class FakeTokenizer:
    model_max_length = 48
    pad_token_id = 0

    def __call__(self, prompts, padding="max_length", max_length=None, truncation=True, return_tensors="pt"):
        if isinstance(prompts, str):
            prompts = [prompts]
        max_length = max_length or self.model_max_length
        rows = []
        for prompt in prompts:
            ids = [1] + [3 + (ord(char) % 60) for char in prompt][: max_length - 2] + [2]
            if padding == "max_length":
                ids += [0] * (max_length - len(ids))
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        return TokenBatch(input_ids=input_ids, attention_mask=(input_ids != 0).long())

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{token_id}" for token_id in ids]


class FakeTextEncoder(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_attention_mask=True)
        self.embedding = nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakePosterior:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, generator=None):
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + 0.01 * noise


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=0.18215)

    def encode(self, images):
        latent = F.avg_pool2d(self.encoder(images), kernel_size=8)
        return SimpleNamespace(latent_dist=FakePosterior(latent))

    def decode(self, latents):
        image = self.decoder(F.interpolate(latents, scale_factor=8, mode="nearest"))
        return SimpleNamespace(sample=image)


class FakeAttention(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size), nn.Dropout(0.0)])

    def forward(self, hidden):
        combined = self.to_q(hidden) + self.to_k(hidden).mean(dim=1, keepdim=True) + self.to_v(hidden).mean(dim=1, keepdim=True)
        return self.to_out[0](torch.tanh(combined))


class FakeTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = FakeAttention()


class FakeAttentionContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeTransformerBlock()])


class FakeDownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([FakeAttentionContainer()])


class FakeTimeEmbedding(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.linear_1 = nn.Linear(1, output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, sample, condition=None):
        embedding = self.linear_2(F.silu(self.linear_1(sample)))
        return embedding if condition is None else embedding + condition


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=4, out_channels=4, cross_attention_dim=6, sample_size=4)
        self.conv_in = nn.Conv2d(4, 8, 3, padding=1)
        self.time_embedding = FakeTimeEmbedding(8)
        self.context_proj = nn.Linear(6, 8)
        self.down_blocks = nn.ModuleList([FakeDownBlock()])
        self.conv_out = nn.Conv2d(8, 4, 3, padding=1)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None, return_dict=True):
        hidden = self.conv_in(sample)
        batch, channels, height, width = hidden.shape
        timestep_values = torch.as_tensor(timestep, device=hidden.device, dtype=hidden.dtype)
        if timestep_values.ndim == 0:
            timestep_values = timestep_values.expand(batch)
        time_embedding = self.time_embedding(timestep_values.reshape(batch, 1), timestep_cond)
        hidden = hidden + time_embedding.to(hidden).view(batch, channels, 1, 1)
        tokens = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        context = self.context_proj(encoder_hidden_states.mean(dim=1)).unsqueeze(1)
        tokens = tokens + context
        tokens = self.down_blocks[0].attentions[0].transformer_blocks[0].attn1(tokens)
        hidden = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return SimpleNamespace(sample=self.conv_out(hidden))


class FakeScheduler:
    def __init__(self, prediction_type="epsilon", num_train_timesteps=100):
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )
        betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.num_inference_steps = None
        self.timesteps = None

    def add_noise(self, latents, noise, timesteps):
        alpha = self.alphas_cumprod.to(latents)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * latents + (1 - alpha).sqrt() * noise

    def get_velocity(self, latents, noise, timesteps):
        alpha = self.alphas_cumprod.to(latents)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * latents

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = int(num_inference_steps)
        ratio = self.config.num_train_timesteps // self.num_inference_steps
        self.timesteps = (torch.arange(self.num_inference_steps) * ratio).flip(0).long().to(device=device)

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, eta=0.0, generator=None):
        timestep = int(timestep)
        previous = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_t = self.alphas_cumprod.to(sample)[timestep]
        alpha_previous = self.alphas_cumprod.to(sample)[previous] if previous >= 0 else self.final_alpha_cumprod.to(sample)
        if self.config.prediction_type == "epsilon":
            epsilon = model_output
            x0 = (sample - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
        else:
            x0 = alpha_t.sqrt() * sample - (1 - alpha_t).sqrt() * model_output
            epsilon = alpha_t.sqrt() * model_output + (1 - alpha_t).sqrt() * sample
        previous_sample = alpha_previous.sqrt() * x0 + (1 - alpha_previous).sqrt() * epsilon
        return SimpleNamespace(prev_sample=previous_sample, pred_original_sample=x0)


def make_fake_components():
    return {
        "vae": FakeVAE(),
        "tokenizer": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "unet": FakeUNet(),
        "scheduler_train": FakeScheduler(),
        "scheduler_infer": FakeScheduler(),
        "device": torch.device("cpu"),
        "weight_dtype": torch.float32,
        "uses_external_vae": False,
    }
