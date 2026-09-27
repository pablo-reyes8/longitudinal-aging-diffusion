"""Public API for running external face-aging baseline comparisons."""

from .generate_images import age_image
from .load_models import load_aging_models

__all__ = ["load_aging_models", "age_image"]
