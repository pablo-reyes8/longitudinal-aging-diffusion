# Long-gap counterfactual aging

**Goal:** A separate Colab notebook generates approximately 350 train-source aging transitions and measures signed regional residuals against frozen module-1 expectations, without a real target photo.

**Design authorized by the user:** sources 18–35, requested ages 60/65/70, exclusion margin ±10 years confirmed in the final clarification. Exclude any identity with an observed age in [50,80]. Inventory all known photos before eligibility filtering. Choose one source per identity and balance target ages. Keep existing observed-target evaluation strict and unchanged in meaning. Existing SAM/FRAN/HRFAE batching, grouped FADING, direct inference and picking are reused.

**Architecture:** independent `long_gap_panel.py` selects and audits the panel; `long_gap_evaluation.py` extracts frozen descriptors and computes expected-change residuals, common-panel summaries, identity bootstrap intervals and structural diagnostics. A new notebook orchestrates selection, generation and full diagnostic reporting. Generation alone permits blank target paths; observed evaluation continues to require real targets.

**Scientific contract:** primary residual = standardized generated-minus-source change minus Colombian longitudinal module-1 expected change. FFHQ cross-sectional change and Latino cross-sectional change are separate comparators. No oracle, target similarity or target-based KID is invented. Source-noedit is a genuine zero-change anchor. Report source/reference overlap, pose/expression exclusions, age support and conditional-on-frozen-reference uncertainty. π is reported only if existing structural rank/conditioning/correlation/dose gates pass; do not clip it or promise identification.

## Work and validation

- [x] Deterministic panel selector, exclusion inventory, scarcity checks and provenance tests.
- [x] Generation-only nullable target support, resume roundtrip test and observed-target rejection test.
- [x] Frozen expected-change evaluator, per-photo signs, common panels, bootstrap, operational age/identity and structural diagnostics with toy numerical tests.
- [x] Standalone Spanish Colab notebook with editable models/paths, complete inventory audit, generation resume, summaries/plots/contact sheets and saved diagnostics.
- [x] Verified with `C:/ProgramData/miniconda3/envs/deep_learning/python.exe`: all 95 tests in `tests/comparisong_v2` passed; notebook integration/syntax tests passed again after final diagnostic presentation changes. Independent review complete; relocated DINO landmarks retain frozen byte-hash validation. GPU inference/data-dependent diagnostics run in the user's Colab, not locally.
