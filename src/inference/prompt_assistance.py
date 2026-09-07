"""Concise configurable prompt blocks for artifact-suppression comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

from PIL import Image, ImageDraw


DEFAULT_PROMPT_ASSISTANCE_CONFIG = {
    "enabled": True,
    "append_positive_support_prompt": True,
    "append_negative_prompt": True,
    "positive_global_terms": [
        "realistic facial features",
        "natural eyes",
        "realistic lips",
        "preserve facial identity",
        "preserve natural expression",
        "high facial realism",
    ],
    "negative_global_terms": [
        "deformed teeth",
        "extra teeth",
        "malformed mouth",
        "distorted lips",
        "unnatural smile",
        "deformed eyes",
        "cross-eyed",
        "asymmetric eyes",
        "dark hollow eyes",
        "exaggerated eye bags",
        "plastic skin",
    ],
    "use_mouth_rules": True,
    "use_eye_rules": True,
    "use_expression_rules": True,
    "closed_mouth_positive_terms": [
        "keep lips closed",
        "preserve the same mouth shape",
        "preserve the mouth expression",
        "no visible teeth",
    ],
    "closed_mouth_negative_terms": ["open mouth", "visible teeth", "toothy smile"],
    "visible_teeth_positive_terms": [
        "preserve natural teeth visibility",
        "preserve natural smile",
        "realistic teeth",
    ],
    "visible_teeth_negative_terms": [
        "deformed teeth",
        "extra teeth",
        "unnatural teeth",
    ],
    "eye_positive_terms": [
        "natural eyes",
        "preserve eye shape",
        "preserve gaze direction",
        "avoid exaggerated under-eye shadows",
    ],
    "eye_negative_terms": [
        "dark hollow eyes",
        "heavy black eye circles",
        "deformed eyes",
        "asymmetric eyes",
        "cross-eyed",
    ],
    "max_support_terms_per_block": 6,
}

_MOUTH_STATES = {"auto", "closed", "visible_teeth", "unknown"}
_EXPRESSION_STATES = {"auto", "neutral", "smiling", "unknown"}


def validate_prompt_assistance_scale(value, name: str) -> float:
    """Accept non-negative finite interpolation/extrapolation scales."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


def _terms(value, name: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    output = []
    seen = set()
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        term = " ".join(raw.strip().split())
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            output.append(term)
    return output


def _append_unique(destination: list[str], values: Sequence[str], limit: int) -> None:
    known = {value.casefold() for value in destination}
    for value in values:
        if value.casefold() not in known and len(destination) < limit:
            destination.append(value)
            known.add(value.casefold())


def resolve_prompt_assistance(
    *,
    base_target_prompt: str,
    base_negative_prompt: str = "",
    prompt_assistance_config: Mapping | None = None,
    source_mouth_state: str = "auto",
    source_expression_state: str = "auto",
) -> dict:
    """Build one concise assisted positive/negative prompt pair."""
    if source_mouth_state not in _MOUTH_STATES:
        raise ValueError(f"source_mouth_state must be one of {sorted(_MOUTH_STATES)}")
    if source_expression_state not in _EXPRESSION_STATES:
        raise ValueError(
            f"source_expression_state must be one of {sorted(_EXPRESSION_STATES)}"
        )
    config = deepcopy(DEFAULT_PROMPT_ASSISTANCE_CONFIG)
    if prompt_assistance_config is not None:
        if not isinstance(prompt_assistance_config, Mapping):
            raise ValueError("prompt_assistance_config must be a mapping")
        unknown = set(prompt_assistance_config) - set(config)
        if unknown:
            raise ValueError(f"Unknown prompt assistance keys: {sorted(unknown)}")
        config.update(prompt_assistance_config)
    boolean_keys = (
        "enabled",
        "append_positive_support_prompt",
        "append_negative_prompt",
        "use_mouth_rules",
        "use_eye_rules",
        "use_expression_rules",
    )
    for key in boolean_keys:
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean")
    limit = config["max_support_terms_per_block"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_support_terms_per_block must be a positive integer")
    term_keys = [key for key in config if key.endswith("_terms")]
    normalized = {key: _terms(config[key], key) for key in term_keys}
    warnings = []
    resolved_mouth = "unknown" if source_mouth_state == "auto" else source_mouth_state
    resolved_expression = (
        "unknown" if source_expression_state == "auto" else source_expression_state
    )
    if source_mouth_state == "auto" or source_expression_state == "auto":
        warnings.append(
            "No landmark classifier is configured; automatic mouth/expression state fell back to unknown."
        )

    positive: list[str] = []
    negative: list[str] = []
    global_budget = max(1, limit // 2)
    _append_unique(positive, normalized["positive_global_terms"], global_budget)
    _append_unique(negative, normalized["negative_global_terms"], global_budget)
    if config["use_eye_rules"]:
        eye_budget = min(limit, global_budget + 2)
        _append_unique(positive, normalized["eye_positive_terms"], eye_budget)
        _append_unique(negative, normalized["eye_negative_terms"], eye_budget)
    if config["use_mouth_rules"] and resolved_mouth == "closed":
        _append_unique(positive, normalized["closed_mouth_positive_terms"], limit)
        _append_unique(negative, normalized["closed_mouth_negative_terms"], limit)
    elif config["use_mouth_rules"] and resolved_mouth == "visible_teeth":
        _append_unique(positive, normalized["visible_teeth_positive_terms"], limit)
        _append_unique(negative, normalized["visible_teeth_negative_terms"], limit)
    if config["use_expression_rules"] and resolved_expression == "smiling":
        _append_unique(positive, ["preserve natural smile"], limit)
        _append_unique(negative, ["forced smile"], limit)
    elif config["use_expression_rules"] and resolved_expression == "neutral":
        _append_unique(positive, ["preserve neutral expression"], limit)
        _append_unique(negative, ["forced smile"], limit)

    assisted_target = base_target_prompt
    if config["enabled"] and config["append_positive_support_prompt"] and positive:
        assisted_target = ", ".join([base_target_prompt, *positive])
    negative_parts = _terms([base_negative_prompt], "base_negative_prompt") if base_negative_prompt.strip() else []
    if config["enabled"] and config["append_negative_prompt"]:
        _append_unique(negative_parts, negative, len(negative_parts) + len(negative))
    return {
        "target_prompt": assisted_target,
        "negative_prompt": ", ".join(negative_parts),
        "positive_terms": positive,
        "negative_terms": negative,
        "source_mouth_state": resolved_mouth,
        "source_expression_state": resolved_expression,
        "warnings": warnings,
        "config": config,
    }


def stack_sweep_variants(
    base: Image.Image,
    assisted: Image.Image,
    *,
    base_label: str = "BASE",
    assisted_label: str = "ASSISTED",
) -> Image.Image:
    """Stack two otherwise equivalent strips with compact variant headers."""
    width = max(base.width, assisted.width)
    header = 26
    canvas = Image.new("RGB", (width, base.height + assisted.height + 2 * header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), base_label, fill="black")
    canvas.paste(base, (0, header))
    second_header_y = header + base.height
    draw.text((8, second_header_y + 6), assisted_label, fill="black")
    canvas.paste(assisted, (0, second_header_y + header))
    return canvas
