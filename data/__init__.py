"""Public API for the longitudinal face-aging data pipeline."""

from .dataloaders import build_face_aging_dataloaders
from .dataset import CombinedFaceAgingDataset, FaceAgingDataset, collate_face_aging_batch
from .fgnet import (
    build_fgnet_manifest,
    parse_fgnet_filename,
    select_complementary_fgnet_pairs,
    transition_cell,
)
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
    "CombinedFaceAgingDataset",
    "ImageRecord",
    "PairRecord",
    "build_face_aging_dataloaders",
    "build_fgnet_manifest",
    "build_identity_splits",
    "build_image_manifest",
    "build_pair_index",
    "build_prompts",
    "collate_face_aging_batch",
    "get_person_word",
    "inspect_batch",
    "parse_age_filename",
    "parse_fgnet_filename",
    "plot_pair_grid",
    "select_complementary_fgnet_pairs",
    "transition_cell",
    "run_data_pipeline_validation",
]
