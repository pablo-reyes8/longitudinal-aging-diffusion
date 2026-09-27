"""Public API for controlled longitudinal training ablations."""

from .config import (
    CASE_DEFINITIONS,
    canonical_training_ablation_config,
    resolve_training_ablation_config,
)
from .panel import build_fixed_evaluation_panel, persist_fixed_evaluation_panel
from .evaluation import AblationEpochEvaluator
from .runner import ablation_studies

__all__ = [
    "CASE_DEFINITIONS",
    "AblationEpochEvaluator",
    "build_fixed_evaluation_panel",
    "ablation_studies",
    "canonical_training_ablation_config",
    "persist_fixed_evaluation_panel",
    "resolve_training_ablation_config",
]
