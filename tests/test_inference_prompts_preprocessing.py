from __future__ import annotations

import warnings

import pytest
import torch
from PIL import Image

from data.dataset import _load_image
from src.inference import (
    build_inference_prompt_pack,
    decode_latents_to_tensor,
    encode_image_to_latent,
    prepare_inference_image,
)
from src.model import encode_images_to_latents
from training_fakes import make_training_bundle


def test_prompt_builder_exact_training_templates_and_fallbacks():
    selfage = build_inference_prompt_pack(target_age=65, source_age=25, prompt_style="selfage")
    fading = build_inference_prompt_pack(target_age=65, source_age=25, prompt_style="fading")
    generic = build_inference_prompt_pack(target_age=65)
    assert selfage["target_prompt"] == "photo of a person as 65-year-old"
    assert selfage["source_prompt"] == "photo of a person as 25-year-old"
    assert fading["target_prompt"] == "photo of a 65 year old person"
    assert fading["source_prompt"] == "photo of a 25 year old person"
    assert generic["source_prompt"] == "photo of a person"


def test_explicit_prompt_wins_but_age_mismatch_is_not_silent():
    with pytest.warns(UserWarning, match="encodes age 70"):
        pack = build_inference_prompt_pack(
            target_prompt="photo of a person as 70-year-old", target_age=60
        )
    assert pack["target_prompt"].endswith("70-year-old")
    assert pack["target_age"] == 60 and pack["warnings"]


def test_inference_preprocessing_exactly_matches_training_contract(tmp_path):
    path = tmp_path / "rectangular.png"
    Image.new("RGB", (41, 29), (130, 70, 20)).save(path)
    training = _load_image(path, image_size=32, flip=False)
    inference = prepare_inference_image(path, image_size=32)
    assert inference.shape == (1, 3, 32, 32)
    assert torch.equal(inference[0], training)
    assert inference.min() >= -1 and inference.max() <= 1


def test_tensor_preprocessing_center_crop_resize_and_contract_errors():
    tensor = torch.linspace(-1, 1, 3 * 20 * 30).reshape(3, 20, 30)
    prepared = prepare_inference_image(tensor, image_size=16)
    assert prepared.shape == (1, 3, 16, 16) and torch.isfinite(prepared).all()
    with pytest.raises(ValueError, match="normalized"):
        prepare_inference_image(torch.full((3, 10, 10), 255.0), image_size=16)
    with pytest.raises(ValueError, match="shape"):
        prepare_inference_image(torch.zeros(1, 10, 10), image_size=16)


def test_inference_vae_encoding_scaling_matches_model_training_helper():
    bundle = make_training_bundle()
    images = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    production = encode_image_to_latent(bundle, images, sample_posterior=False)
    training = encode_images_to_latents(bundle, images, sample_posterior=False)
    assert torch.equal(production, training)
    decoded = decode_latents_to_tensor(bundle, production)
    manual = bundle["vae"].decode(production / bundle["vae"].config.scaling_factor).sample
    manual = (manual / 2 + 0.5).clamp(0, 1)
    assert torch.equal(decoded, manual)
