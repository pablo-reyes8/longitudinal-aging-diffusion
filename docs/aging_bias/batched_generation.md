# Paper prediction generation

`generate_paper_predictions` keeps the existing manifest, generated roots,
sample IDs, PNG names, and per-image resume markers. It still selects aging
transitions by default. `run_paper_analysis` consumes the same outputs.

New optional arguments: `batch_size=32`, `progress_every=100`.

## Resume without a full Drive scan

The default `resume_validation='fast'` is for an unchanged dataset and run.
`generation/resume_state.json` stores each model's parameters, recorded rows,
and completion status. Completed phases skip immediately without listing
identities, hashing sources, opening JSON sidecars, or reopening PNGs. Partial
phases retain their recorded rows and process only unknown transitions.

Existing runs without this file are imported once using PNG+JSON filename
pairs, with concurrent per-ID directory listings (`resume_workers=16`). No PNG
or successful sidecar content is opened in this import. Failed sidecars without
PNGs are read to retain the existing `retry_failed` behavior. This migration
trusts that the legacy outputs belong to the unchanged run, as requested.
The completed model is then checkpointed before loading any subsequent model.
Do not delete resume_state.json when moving/resuming results.

Fast resume intentionally trusts the immutable inputs and saved progress; it
does not detect an external edit/corruption of an already recorded file. It
does check the run parameters and transition metadata in memory. A missing
prediction root blocks a recorded phase rather than declaring it complete.
If data or generation parameters were changed, use
`resume_validation='strict'`: it validates source fingerprints and prediction
signatures, and verifies newly seen/changed PNGs. A persistent per-model index
avoids reopening unchanged PNG/JSON content on later strict scans. Neither
mode changes the PNG paths or existing prediction signatures.

- SAM uses tensor batches, its original age transform, FP32 inputs, and
  `randomize_noise=False, resize=False`. Its historical RGB/JPEG quality-95
  conversion before alignment is preserved. Aligned 256-pixel input tensors
  are cached by source fingerprint on disk and in a bounded memory cache.
- FRAN keeps the loaded repository's `process_image` preprocessing and final
  compositing. A validated AST transformation suspends that function at its
  image sliding-window call, concatenates the inputs, runs the repository's
  existing batched sliding-window function, and resumes each compositor.
  The face detector results and decoded sources have a bounded in-memory cache.
  An unsupported upstream function layout fails explicitly.
- CUSP previously had no executable benchmark path. The generation-only route
  now reads the official local `G_ema` Image2Image pickle through its repository
  loader. The generator receives one-hot labels: FFHQ-LS uses ten official
  age bins; FFHQ-RR uses 101 ages (0–100). AgeNumber is the discriminator's
  mapping, not the generator's input. Sources are resized by the
  short edge and center cropped to the checkpoint's resolution. Guided input
  gradients remain enabled for the structural mask, without parameter updates.
  Random structural blur is seeded independently per transition and call, so
  splitting/resuming a batch preserves the policy. Decoder noise uses the
  checkpoint's fixed noise maps (`noise_mode='const'`) instead of uncontrolled
  random noise. This policy is recorded in generation_config.json.
  CUSP uses its own signature
  version because the previous CLI stub never produced valid CUSP predictions.
- FADING retains its upstream multi-target mechanism: one inversion per source,
  then all pending ages for that source. This is source-level reuse, not a
  tensor batch of independent null-text inversions.
- HRFAE batches pending sources by target age inside its resident isolated
  process. The original test script supplies setup, normalization, histogram
  correction, clipping and per-image JPEG encoding. Only its image loop is
  replaced. Source preprocessing is cached. Each JPEG is decoded and saved as
  the same canonical PNG expected by analysis. Its spectral normalization is
  frozen with `trainer.eval()`; otherwise outputs change with batch history.
  This corrected mode uses `hrfae-batched-eval-v1` signatures: compatible HRFAE
  predictions resume, but old HRFAE CLI predictions are regenerated. SAM/FRAN/
  FADING signatures remain unchanged. OOM recursively splits batches and clears
  the trainer's retained `content_code_a` before retries and after each batch.
  Worker setup errors stop the phase; individual image errors are persisted.
- The direct ours path retains its current generator. Picking retains its
  adaptive search.

Tensor adapters reduce the batch size automatically on OOM; preprocessing
errors affect only the corresponding sample. Failed tensor batches are split
to isolate errors. No AMP, inference-step, resolution, or model substitution
is used to accelerate SAM/FRAN. Floating-point batch execution need not be
bitwise equal to singleton execution on every CUDA backend; schemas and resume
signatures for existing SAM/FRAN/FADING outputs remain unchanged.

Generation-only residency scopes skip legacy sweep figures and redundant PNGs.
Ordinary `age_image` sweep calls retain those artifacts. Legacy per-image stdout
is suppressed in the generation path. Model setup messages remain visible.
Progress prints every 100 newly completed transitions and at phase completion,
including the already recorded count, remaining count, and failed count.

Each result is committed as a PNG plus JSON marker. Progress CSVs are flushed
every 100 transitions, at phase end, and during orderly interruption. Abrupt
process death can leave the CSV/progress file behind the markers; resume imports
the extra PNG+JSON pairs. Later model checkpoints are retained even when an
earlier phase is resumed or interrupted. Empty initialization attempts do not
overwrite the previous prediction CSV.
Unfinished batches may be computed again if their images were never saved.

Verification uses only toy CPU networks, tiny images, injected OOM/errors,
original-function stand-ins, and resume markers. Real SAM/FRAN/CUSP weights and
CUDA output equivalence were not executed locally.
