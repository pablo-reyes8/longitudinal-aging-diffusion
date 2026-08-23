"""Public supervised diffusion-loss API."""

from .auxiliary_adapters import (
    AgeEstimatorAdapter,
    IdentityEncoderAdapter,
    expected_age_from_logits,
    identity_cosine_loss,
)
from .diffusion_utils import (
    compute_diffusion_loss,
    compute_min_snr_weights,
    extract_scheduler_coefficients,
    get_diffusion_target,
    predict_x0_from_model_output,
    sd_image_to_01,
)
from .face_aging_loss import FaceAgingDiffusionLoss, compose_weighted_losses
from .pretrained_auxiliary import (
    AGE_MODEL_ID,
    ArcFaceR50InputAdapter,
    DEFAULT_AGE_MODEL_ID,
    DEFAULT_IDENTITY_MODEL_ID,
    IDENTITY_MODEL_ID,
    MiVOLOFaceOnlyAgeModel,
    load_pretrained_auxiliary_models,
)
from .validation import run_face_aging_loss_validation

__all__ = [name for name in globals() if not name.startswith("_")]
