"""Prompt preparation consistent with the longitudinal training dataset."""

from __future__ import annotations

import re
import warnings

from data.prompts import build_prompts


_AGE_PATTERNS = (
    re.compile(r"\b(\d{1,3})\s*[- ]?year[- ]old\b", re.IGNORECASE),
    re.compile(r"\bas\s+(\d{1,3})\b", re.IGNORECASE),
)


def extract_prompt_age(prompt: str) -> int | None:
    for pattern in _AGE_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return int(match.group(1))
    return None


def build_inference_prompt_pack(
    *,
    target_prompt: str | None = None,
    target_age: int | None = None,
    source_prompt: str | None = None,
    source_age: int | None = None,
    prompt_style: str = "selfage",
    negative_prompt: str = "",
) -> dict[str, str | int | None | list[str]]:
    if target_prompt is None and target_age is None:
        raise ValueError("Provide target_prompt or target_age")
    if target_age is not None and not 0 <= int(target_age) <= 120:
        raise ValueError("target_age must be in [0, 120]")
    if source_age is not None and not 0 <= int(source_age) <= 120:
        raise ValueError("source_age must be in [0, 120]")
    if prompt_style not in {"selfage", "fading"}:
        raise ValueError("prompt_style must be 'selfage' or 'fading'")
    if target_prompt is None:
        # build_prompts guarantees exact parity with the training templates.
        source_for_builder = int(source_age) if source_age is not None else int(target_age)
        target_prompt = build_prompts(source_for_builder, int(target_age), prompt_style=prompt_style)["target_prompt"]
    if source_prompt is None:
        source_prompt = (
            build_prompts(int(source_age), int(source_age), prompt_style=prompt_style)["source_prompt"]
            if source_age is not None else "photo of a person"
        )
    emitted_warnings: list[str] = []
    encoded_age = extract_prompt_age(target_prompt)
    if target_age is not None and encoded_age is not None and encoded_age != int(target_age):
        message = (
            f"Explicit target_prompt encodes age {encoded_age}, but target_age={target_age}; "
            "the explicit prompt is used and target_age remains metadata only."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        emitted_warnings.append(message)
    return {
        "target_prompt": str(target_prompt),
        "source_prompt": str(source_prompt),
        "generic_prompt": "photo of a person",
        "null_prompt": str(negative_prompt),
        "source_age": int(source_age) if source_age is not None else None,
        "target_age": int(target_age) if target_age is not None else encoded_age,
        "warnings": emitted_warnings,
    }
