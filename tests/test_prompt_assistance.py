from __future__ import annotations

import pytest
import torch

from src.inference import resolve_prompt_assistance
from src.inference.infer_face_aging import _encode_prompt_with_assistance_scale
from src.model import encode_prompts
from training_fakes import make_training_bundle


def test_prompt_assistance_is_concise_deduplicated_and_state_aware():
    resolved = resolve_prompt_assistance(
        base_target_prompt="photo of a person as 65-year-old",
        base_negative_prompt="low quality",
        source_mouth_state="closed",
        source_expression_state="neutral",
    )
    assert resolved["target_prompt"].startswith("photo of a person as 65-year-old")
    assert "keep lips closed" in resolved["target_prompt"]
    assert "open mouth" in resolved["negative_prompt"]
    assert resolved["negative_prompt"].startswith("low quality")
    assert len(resolved["positive_terms"]) <= 6
    assert len(resolved["negative_terms"]) <= 6
    assert len({term.casefold() for term in resolved["negative_terms"]}) == len(
        resolved["negative_terms"]
    )


def test_custom_term_lists_replace_defaults_and_auto_falls_back_to_unknown():
    resolved = resolve_prompt_assistance(
        base_target_prompt="base age prompt",
        prompt_assistance_config={
            "positive_global_terms": ["custom face", "custom face"],
            "negative_global_terms": ["custom artifact"],
            "use_eye_rules": False,
        },
    )
    assert resolved["positive_terms"] == ["custom face"]
    assert resolved["negative_terms"] == ["custom artifact"]
    assert "realistic facial features" not in resolved["target_prompt"]
    assert resolved["source_mouth_state"] == "unknown"
    assert resolved["source_expression_state"] == "unknown"
    assert resolved["warnings"]

    with pytest.raises(ValueError, match="Unknown prompt assistance keys"):
        resolve_prompt_assistance(
            base_target_prompt="base", prompt_assistance_config={"typo": []}
        )


def test_prompt_assistance_scale_interpolates_only_embedding_delta():
    bundle = make_training_bundle(seed=941)
    device = torch.device("cpu")
    base_prompt = "photo of a person as 40-year-old"
    assisted_prompt = base_prompt + ", preserve facial identity"
    base = encode_prompts(bundle, [base_prompt], device=device)
    assisted = encode_prompts(bundle, [assisted_prompt], device=device)

    zero = _encode_prompt_with_assistance_scale(
        bundle,
        assisted_prompt=assisted_prompt,
        base_prompt=base_prompt,
        assistance_scale=0.0,
        batch_size=1,
        device=device,
    )
    partial = _encode_prompt_with_assistance_scale(
        bundle,
        assisted_prompt=assisted_prompt,
        base_prompt=base_prompt,
        assistance_scale=0.35,
        batch_size=1,
        device=device,
    )
    full = _encode_prompt_with_assistance_scale(
        bundle,
        assisted_prompt=assisted_prompt,
        base_prompt=base_prompt,
        assistance_scale=1.0,
        batch_size=1,
        device=device,
    )

    assert torch.equal(zero, base)
    assert torch.equal(full, assisted)
    assert torch.allclose(partial, base + 0.35 * (assisted - base))
