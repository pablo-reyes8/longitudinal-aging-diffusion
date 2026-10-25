# Module 1 closure implementation plan

**Goal:** implement `planing/GUIA_AGENTE_modulo1_cierre.md` without inference,
model downloads, or changes to ablations/module 2.

**Architecture:** opt-in feature/reference controls; a closure orchestrator behind
the existing reference builder. Old configuration/hash behavior remains usable.
Notebook enables closure in a new versioned directory; DINO remains GPU-preferred.

**Constraints:** no fabricated longitudinal FairFace pairs, no test fitting,
no legacy bundle overwrite, no outcome-dependent sign/consistency rules.
Real landmark overlay is produced on the user's Colab run, not fabricated locally.

## Tasks (native implementation)

- [x] Synthetic tests first: geometry M1–M4; resolution R1–R3; TTA;
  zones Z1–Z4; age sampling S1; consistency and mean-map MF/FZ cases.
- [x] Optional preprocessing/features: signed cheek geometry, physical IOD
  matching, verified used-landmark mirror permutation, TTA and original values.
- [x] Train-only learned zone transform, own ICC, persisted parameters and hash.
- [x] Optional all-older FFHQ sampler; bootstrap cross-sectional contrasts,
  fixed consistency rules, descriptive means/HF maps and bilingual face diagram.
- [x] Closure orchestrator: provenance, train-only percentile, TTA/resolution
  comparisons, internal split placebo, explicit incomplete/empty statuses.
- [x] Minimal notebook activation, documented indices and generated overlay,
  synthetic integration/legacy tests; report unrun real checks honestly.

User amendment: previous output is strictly read-only; a new folder is required.
Only FFHQ 50–69 receives model age estimates for descriptive fine bands;
FairFace keeps the user's CSV and reference curves retain official FFHQ ages.
Implementation/tests complete. Real Colab run and anatomical overlay review
remain intentionally unexecuted (no model/download authorization locally).

## Review focus

- Missing cheek landmarks remain NaN; no fabricated coordinates.
- A singleton supplement cannot establish longitudinal support or retest ICC.
- New default-false fields must not invalidate old feature hashes.
- No-active zones and underfilled mean cells are reported, never silently filled.
- Reruns reject an existing unrelated reference bundle; sensitivity keeps signs
  frozen and placebo stays under internal/, outside paper artifacts.
