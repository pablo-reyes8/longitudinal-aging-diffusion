from __future__ import annotations

import pytest

from src.inference import resolve_prompt_assistance


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
