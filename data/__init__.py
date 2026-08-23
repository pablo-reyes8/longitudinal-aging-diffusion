"""Public API for the longitudinal face-aging data pipeline."""

from .dataloaders import build_face_aging_dataloaders
from .dataset import FaceAgingDataset, collate_face_aging_batch
from .diagnostics import inspect_batch, plot_pair_grid
from .indexing import (
    ImageRecord,
    PairRecord,
    build_identity_splits,
    build_image_manifest,
    build_pair_index,
    parse_age_filename,
)
from .prompts import build_prompts, get_person_word
from .validation import run_data_pipeline_validation

__all__ = [
    "FaceAgingDataset",
    "ImageRecord",
    "PairRecord",
    "build_face_aging_dataloaders",
    "build_identity_splits",
    "build_image_manifest",
    "build_pair_index",
    "build_prompts",
    "collate_face_aging_batch",
    "get_person_word",
    "inspect_batch",
    "parse_age_filename",
    "plot_pair_grid",
    "run_data_pipeline_validation",
]
