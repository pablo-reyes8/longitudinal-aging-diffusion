"""Population references and aging-bias evaluation (independent of legacy APIs)."""
from .config import FeatureConfig, ReferenceConfig, WeightingConfig, PaperV2Config, SD15Config


def build_reference_bundle(*args, **kwargs):
    from .references import build_reference_bundle as run
    return run(*args, **kwargs)


def describe_population_aging(*args, **kwargs):
    from .population_aging import describe_population_aging as run
    return run(*args, **kwargs)


def run_self_checks(*args, **kwargs):
    from .validation import run_self_checks as run
    return run(*args, **kwargs)
