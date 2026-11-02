# Module 2 observed longitudinal zone error

**Scope:** module 1 is frozen/read-only. Preserve public calls, transition panel,
inference and ablation modules. No local downloads/neural runs.

**Design:** for each frozen train-standardized, age-oriented attribute/zone,
model change = generated − source; real change = target − source;
error = generated − target. Report signed error, MAE, RMSE with identity-cluster
bootstrap, shared per-zone valid panels, matched model differences, all/aging/
rejuvenation and unmerged ≥20-year subsets. Adult reliability support is explicit.
Components are reported alongside grouped zones; FFHQ alignment is descriptive,
not causal demographic identification. DINO latent coordinates are not face zones.

**Implementation (native):**
- [x] Tests first: exact residuals/signs, perfect oracle, missing data, weighted
  cluster bootstrap and paired SD15 strength contrasts.
- [x] Add read-only module-2 zone report service; route closed module-1 bundles
  after descriptor extraction without obsolete structural-regression gating.
- [x] Reuse frozen DINO PCA/curves, never refit/write module 1 or require train
  images; preserve legacy analysis path for older bundles.
- [x] Direct metric preflight + separate SD15 generation/evaluation chunk,
  restoring pristine conv_in and paired seed/strength settings as before.
- [x] Inventory long test-pair candidates without sampling or modifying test.
- [x] Short coherent notebook with reference paths from supplied completed run,
  no accidental picking regeneration, figures/tables automatically displayed.
- [x] Synthetic tests and legacy checks; document limitations and unrun real runs.

Real generation, model downloads and result interpretation remain intentionally
unexecuted locally. Notebook workflow tested with heavy boundaries replaced by
synthetic fakes; a failed preflight prevents any SD generation call.
