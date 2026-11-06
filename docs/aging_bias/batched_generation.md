# Paper prediction generation

`generate_paper_predictions` keeps the existing manifest, generated roots,
sample IDs, PNG names, and per-image resume markers. It still selects aging
transitions by default. `run_paper_analysis` consumes the same outputs.

New optional arguments: `batch_size=32`, `progress_every=100`.

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
  tensor batch of independent null-text inversions. HRFAE and the direct ours
  path retain their current generators. Picking retains its adaptive search.

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
process death can leave the CSV behind the markers; resume reconstructs it.
Unfinished batches may be computed again if their images were never saved.

Verification uses only toy CPU networks, tiny images, injected OOM/errors,
original-function stand-ins, and resume markers. Real SAM/FRAN/CUSP weights and
CUDA output equivalence were not executed locally.
