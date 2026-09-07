"""Public face-aging inference API."""

from .cfg_guidance import (
    combine_referenced_text_cfg,
    combine_referenced_three_way_cfg,
    combine_three_way_cfg,
    predict_three_way_cfg,
)
from .checkpoint_loading import (
    load_face_aging_adapter_for_inference,
    load_face_aging_inference_bundle,
)
from .checkpoint_diagnostics import (
    diagnose_checkpoint_adaptive_age_sweep,
    diagnose_checkpoint_age_sweep,
    diagnose_checkpoint_strength_sweep,
    diagnose_checkpoints_age_sweep,
)
from .comparison_helpers import (
    compare_inference_modes,
    generate_adaptive_age_sweep,
    generate_age_sweep,
    generate_strength_age_sweep,
)
from .delta_bin_evaluation import (
    DEFAULT_DELTA_BIN_THRESHOLDS,
    DELTA_BIN_COLUMNS,
    evaluate_delta_bins,
    save_delta_bin_evaluation,
)
from .conditioning_isolation import (
    CONDITIONING_DIAGNOSTIC_COLUMNS,
    diagnose_checkpoint_conditioning_sources,
    diagnose_conditioning_sources,
)
from .ddim_inversion import (
    ddim_forward_step,
    ddim_invert_source_image,
    edit_from_inverted_latent,
    model_output_to_x0_epsilon,
)
from .diagnostics import compute_face_aging_diagnostics
from .infer_face_aging import (
    DEFAULT_STRENGTH_MAP,
    generate_aged_face_adaptive_strength,
    infer_face_aging,
    infer_face_aging_direct,
    infer_face_aging_inverse,
    resolve_inference_strength,
    resolve_adaptive_strength,
    save_inference_image,
)
from .inference_utils import (
    create_inference_scheduler,
    decode_latents_to_tensor,
    encode_image_to_latent,
    prepare_inference_image,
    tensor_to_pil,
)
from .prompt_building import build_inference_prompt_pack, extract_prompt_age
from .prompt_assistance import (
    DEFAULT_PROMPT_ASSISTANCE_CONFIG,
    resolve_prompt_assistance,
    stack_sweep_variants,
    validate_prompt_assistance_scale,
)
from .smart_age_sweep import (
    DEFAULT_SOURCE_BIAS_CORRECTION_AGE_MAP,
    DEFAULT_SMART_TARGET_STRENGTH_MAP,
    SMART_TRIAL_COLUMNS,
    adaptive_mivolo_confidence_margin,
    diagnose_checkpoint_smart_age_sweep,
    source_specific_mivolo_bias_weight,
)
from .validation_inference_pipeline import run_inference_pipeline_validation

__all__ = [name for name in globals() if not name.startswith("_")]
