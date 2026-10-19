# Module 1: reliability and capture controls

The public reference-building call is unchanged. The reference notebook enables
these settings explicitly; library defaults retain the previous ICC threshold
and do not silently enable quality exclusions for existing callers.

- `retest_min_age=30`: both photos must meet the adult cutoff. The report also
  includes all-age ICC after the same quality controls. Insufficient adult
  repeats remain unknown; there is no fallback to younger people.
- `retest_max_expression_delta=0.10`: near-age photos must have compatible
  expression scores. Duplicate image content within identity is counted once.
  Pair counts are correlated, so distinct retest identity counts are reported.
- MediaPipe supplies transformation matrices and blendshape scores. The notebook
  requires absolute yaw below 15 degrees, smile score at most .25 and jaw-open
  score at most .15. These are operational cutoffs, not calibrated probabilities.
  Missing requested quality measurements fail the filter, not silently pass.
  Smile/open-mouth exclusions affect expression-sensitive classical attributes;
  embedding coordinates cannot be separated that way and are all masked.
- Local illumination correction is applied before ITA/hair-gray measurements.
  It preserves chromatic ratios and the image's median luminance scale; it does
  not white-balance, whiten skin, identify global exposure, or fix heuristic hair
  masks. Color descriptors remain exploratory.

The notebook prespecifies ICC .35 for the structural prior-attribution fit and
.50 for a separate, stricter homogenization fit. Eligible coordinates receive
ICC weights. The reference bundle persists this policy; the v2 evaluator reads
it automatically, including in identity-bootstrap draws. The paper table uses
`homog_excess = homog_model - homog_oracle`, on the same panel and using paired
bootstrap draws. Raw estimates remain available for real-anchor validation.
Unidentified/missing oracle corrections remain NaN. A stricter feature subset
can itself lose identification; that is reported, not bypassed.

Adult reliability is not evidence of reliability in children. Evaluation rows
and population contrasts flag age ranges outside the reliability domain.
Independent outcome noise and regressor measurement error are different:
ICC weighting and oracle subtraction do not guarantee unbiased beta_C/beta_P
when source measurements or estimated reference regressors are noisy.

`capture_sensitivity.csv` compares within-person contrasts on exactly the same
training data, with and without grayscale, original resolution and blur controls.
It marks sign reversals but does not infer that capture period caused them.
Age-period-cohort confounding remains. `capture_coefficients.csv` and
`reference_quality.csv` expose the adjustments and exclusions.

DINOv2-small is optional robustness, not a replacement for the main descriptors.
`prepare_embedding_reference_backend` checks train/test identity separation
before any model setup, fits 16 PCA coordinates on at most 2,000 train/proxy
crops, and retains encoder/PCA/vector-cache provenance. Everything is CPU and
bounded-batch; resumed runs reuse compatible PCA/vectors. References and
reliability reports are separate under `dinov2/`. Higher ICC is not assumed.
No diffusion model needs regeneration when recomputing these measurements.
