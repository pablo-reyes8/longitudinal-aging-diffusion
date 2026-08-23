"""Prompt templates kept independent from model-specific tokenizers."""

from __future__ import annotations


def get_person_word(age: int, gender: str | None, dynamic: bool = False) -> str:
    if not dynamic or gender not in {"male", "female"}:
        return "person"
    if age < 5:
        return "baby"
    if age < 15:
        return "boy" if gender == "male" else "girl"
    if age < 65:
        return "man" if gender == "male" else "woman"
    return "elderly"


def _age_prompt(age: int, person_word: str, style: str) -> str:
    if style == "selfage":
        return f"photo of a {person_word} as {age}-year-old"
    if style == "fading":
        return f"photo of a {age} year old {person_word}"
    raise ValueError("prompt_style must be 'selfage' or 'fading'")


def build_prompts(
    source_age: int,
    target_age: int,
    *,
    prompt_style: str = "selfage",
    gender: str | None = None,
    dynamic_person_word: bool = False,
) -> dict[str, str]:
    source_word = get_person_word(source_age, gender, dynamic_person_word)
    target_word = get_person_word(target_age, gender, dynamic_person_word)
    return {
        "source_prompt": _age_prompt(source_age, source_word, prompt_style),
        "target_prompt": _age_prompt(target_age, target_word, prompt_style),
        "generic_prompt": "photo of a person",
        "source_person_word": source_word,
        "target_person_word": target_word,
    }
