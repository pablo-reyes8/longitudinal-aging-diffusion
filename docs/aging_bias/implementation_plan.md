# Aging bias v2 — implementation plan

Spec: `planing/GUIA_AGENTE_metrica_sesgo_envejecimiento.md` (supplied by user).
Execution: inline; CPU synthetic tests only. No model downloads or training.

- [x] Freeze legacy evaluator + ablation caller with deterministic snapshot;
  record baseline full-suite failures before changes.
- [x] Add isolated dataclasses, preprocessing, classical features and ICC;
  test F1–F6, R1 with synthetic landmarks and images.
- [x] Add train-only within-FE references, FFHQ label adapter, persistence and
  population plots/bootstrap; test P1–P5 and leakage/support checks.
- [x] Add bin/identity weights, structural WLS, cluster bootstrap and simulator;
  test W1–W7, T1–T13 (coverage test explicitly slow), self-checks.
- [x] Add provenance, independent FairFace loader and cached paper evaluator;
  test I1–I7 with deterministic backends, missing images and resume.
- [x] Add isolated SD1.5 counterfactual, exact restoration and checkpoint-once
  inference; test S1–S4, resume and matched-D reporting. Real execution gated
  on reviewed real anchors, never fabricate that validation locally.
- [x] Add reporting and two runnable notebooks; keep existing notebook sweeps
  only. Verify old signatures/files unchanged and run complete synthetic suite.

Review focus: missing colors remain NaN; unavailable ICC excludes features;
rank deficient designs never produce headline ratios; cluster resampling keeps
duplicate draws as distinct sampled clusters; cache changes with backend and
config/content; source-noedit is retained but oracle is validation only.

## Verification ledger

- Baseline suite before implementation: 380 passed, 2 failed, 1 skipped.
  Existing failures are documented in README; no production legacy file changed.
- New fast tests: 40 tests, one explicit slow test skipped. Includes
  end-to-end w=.7 recovery through image IO/cache, FE/cohort recovery, exact SD
  restoration, serialized dtype loading, stale-image quarantine and failures.
- Both new notebooks and the sweeps-only notebook: Python code cells parsed,
  NOT executed. No model weights/downloads/inference/training performed.
- Independent read-only review found serialized dtype, numeric ID normalization,
  support/sex clipping, stale PNG after failed refresh and missing source abort.
  All have regression tests; reproduced defects RED then GREEN.
- Ruling: real phase-4 validation cannot run under the user's no-heavy-execution
  constraint. SD1.5 code and toy tests are delivered, but real execution requires
  passing real anchors and explicit anchors_reviewed=True in the notebook.
- Ruling: when a bootstrap draw loses an operational bin, only the operational
  replicate is invalid. Unweighted/bin-balanced fits remain available.
- Ruling: use the pretrained conv_in snapshot, not the trained first four
  channels/bias, for the vanilla SD1.5 counterfactual. Otherwise the before/after
  interpretation would be wrong even with DoRA disabled.
- Ruling: conditional-reference bootstrap intervals and descriptive matched-D
  interpolation are labeled explicitly; no unimplemented joint uncertainty claim.
- Optional retrieval gallery is deferred (optional in the supplied spec).
- Remaining real work: run notebooks with actual train/proxy/test assets; review
  ICC, identification and real anchors; only then generate/evaluate the SD sweep.
- Final full-suite verification: 420 passed, 2 preexisting failures, 2 skipped
  in 30.36 s. New fast tests: 40 passed. No new regression failures.
  Compile/AST checks passed for all new sources and all notebook code cells.
