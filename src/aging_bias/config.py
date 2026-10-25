"""Explicit, serializable configuration; importing this module loads no models."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    crop_size: int = 256
    harmonize_sigma: float = 1.0
    gray_tol: float = 2.0
    version: str = 'handcrafted-v1'
    landmark_model_path: str | Path | None = None
    landmark_backend_id: str = 'mediapipe-face-landmarker-v1'
    # Opt-in controls; the reference notebook enables them for new bundles.
    max_abs_yaw: float | None = None
    max_smile_score: float | None = None
    max_jaw_open_score: float | None = None
    normalize_color_illumination: bool = False
    new_attributes: tuple = ()
    resolution_matching: bool = False
    resolution_target_percentile: float = 10.
    target_iod_px: float | None = None
    flip_tta: bool = False
    zone_transform: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.resolution_target_percentile <= 100:raise ValueError('Invalid resolution percentile')
        if self.target_iod_px is not None and self.target_iod_px<=0:raise ValueError('target_iod_px must be positive')
        if self.max_abs_yaw is not None and not 0 < self.max_abs_yaw <= 90:
            raise ValueError('max_abs_yaw must be in (0, 90]')
        for value in [self.max_smile_score,self.max_jaw_open_score]:
            if value is not None and not 0 <= value <= 1:
                raise ValueError('Expression score limits must be in [0, 1]')


@dataclass
class WeightingConfig:
    scheme_primary: str = 'bin_balanced'
    bins: tuple = (1, 10, 20, 30)
    identity_balance: bool = True
    operational_target: tuple | list | None = None
    min_bin_n: int = 10


@dataclass
class FFHQSamplingConfig:
    all_groups_at_or_above: float = 40.


@dataclass
class ReferenceConfig:
    colombia_root: str | Path
    ffhq_aging_root: str | Path
    output_dir: str | Path
    train_manifest_path: str | Path | None = None
    split_path: str | Path | None = None
    test_manifest_path: str | Path | None = None
    excluded_test_ids: tuple = ()
    # Explicit acknowledgement required if a train-only directory has no split CSV.
    train_only: bool = False
    ffhq_labels_path: str | Path | None = None
    feature_backend: str = 'handcrafted'
    features: FeatureConfig = field(default_factory=FeatureConfig)
    knots: tuple = (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80)
    min_icc: float = .5
    min_retest_pairs: int = 10
    ffhq_min_conf: float = .6
    ffhq_max_per_group: int | None = 500
    use_sex: bool = False
    min_sex_identities: int = 20
    seed: int = 2026
    resume: bool = True
    retest_min_age: float | None = None
    retest_max_expression_delta: float | None = None
    min_homog_icc: float | None = None
    weight_attributes_by_icc: bool = False
    # DINO PCA is fitted only on these training/proxy images, never test.
    embedding_components: int = 16
    embedding_fit_max_images: int = 2000
    embedding_batch_size: int = 16
    # Optional cross-sectional supplement; never synthetic longitudinal pairs.
    fairface_root: str | Path | None = None
    module1_close: bool = False
    ffhq_sampling: FFHQSamplingConfig | dict | None = None
    zone_indices: bool = False
    previous_reference_dir: str | Path | None = None
    module1_allow_download: bool = False
    module1_assets_root: str | Path = '/content/evaluation_models_v2'

    def __post_init__(self):
        if self.retest_min_age is not None and not 0 <= self.retest_min_age <= 120:
            raise ValueError('retest_min_age must be in [0, 120]')
        if self.retest_max_expression_delta is not None and not 0 <= self.retest_max_expression_delta <= 1:
            raise ValueError('retest_max_expression_delta must be in [0, 1]')


@dataclass
class PaperV2Config:
    test_manifest_path: str | Path
    generated_roots: dict
    models: list
    reference_bundle_path: str | Path
    output_dir: str | Path
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    n_boot: int = 1000
    seed: int = 2026
    resume: bool = True
    d_min: float = .10
    max_cond: float = 1e3
    max_corr: float = .95
    independent_age_name: str = 'FairFace'
    robustness_backend: str | None = None
    include_dex_appendix: bool = True
    include_kid: bool = True
    batch_size: int = 16
    source_root: str | Path | None = None
    oracle_d_bounds: tuple = (.7, 1.3)
    oracle_ratio_tolerance: float = .15
    oracle_homog_tolerance: float = .15
    cache_dir: str | Path | None = None


@dataclass
class SD15Config:
    checkpoint_path: str | Path
    test_manifest_path: str | Path
    output_dir: str | Path
    validation_path: str | Path
    anchors_reviewed: bool = False
    strengths: tuple = (.10, .20, .30, .40, .50)
    inference_config: dict = field(default_factory=lambda: {'image_size':400,'num_inference_steps':40})
    seed: int = 2026
    device: str = 'cuda'
    local_files_only: bool = True
    resume: bool = True
    retry_failed: bool = False
    max_pairs: int | None = None
    source_root: str | Path | None = None
