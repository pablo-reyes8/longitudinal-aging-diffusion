# Supervised Longitudinal Face Aging

PyTorch data pipeline for supervised forward face aging with real longitudinal images. Each identity has its own directory and image filenames begin with the observed age:

```text
dataset_root/
└── id_0001/
    ├── 18.jpg
    ├── 38.jpg
    ├── 38_1.jpg
    └── 52.png
```

The pipeline creates identity-disjoint train/validation/test splits and only pairs images from the same person when `target_age > source_age`. It returns normalized source/target tensors, exact ages, age difference, prompts, paths, and identity metadata. Suspicious labels such as `501.jpg` are audited and normalized to age `51`.

## Quick start

Use the existing Conda environment:

```bash
conda activate deep_learning
```

```python
from data import build_face_aging_dataloaders, inspect_batch

loaders, metadata = build_face_aging_dataloaders(
    root_dir="/path/to/dataset_root",
    image_size=256,
    batch_size=4,
    num_workers=0,
    train_drop_last=False,
)

batch = next(iter(loaders["train"]))
inspect_batch(batch)
```

For a ready-to-run Jupyter example, open `notebooks/data_loader.ipynb` and change only `DATASET_ROOT`.

## Validation

```bash
pytest -q
python tests/run_data_pipeline_validation.py /path/to/dataset_root
```

The validation report checks identity/path leakage, forward-pair correctness against a brute-force oracle, prompt consistency, image decoding, split coverage, and pair concentration. The final diffusion training loop remains intentionally out of scope.

## Model construction

`src/model` builds one SD1.5-compatible bundle with a frozen VAE and CLIP text encoder, an 8-channel source-conditioned U-Net, and manual LoRA (or optional DoRA) attention adapters. The maintained default checkpoint is [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Model weights are not included in this repository.

On the training server, install `diffusers`, `transformers`, `accelerate`, and `safetensors` in the existing environment, then follow `notebooks/model.ipynb`. Offline model tests use small architecture-compatible doubles and never download weights.

## Training loss

`src/loss` provides the composite supervised aging objective: diffusion MSE for
epsilon or velocity prediction, optional Min-SNR weighting, differentiable VAE
reconstruction, frozen identity preservation, and frozen differentiable age
regression. Auxiliary losses support cadence, deterministic subsampling, and a
maximum diffusion timestep without changing the primary diffusion objective.

Run the strict numerical and integration suite with:

```bash
pytest -q tests/test_loss_*.py tests/test_face_aging_loss_composite.py
python tests/run_face_aging_loss_validation.py /path/to/dataset_root
```

The tests use independent float64 mathematical oracles, randomized property
checks, finite differences, gradient decomposition, failure injection, and a
real loader-to-model-to-loss backward pass. Validation against installed real
Diffusers classes is optional and never downloads a checkpoint.

## Training

`src/training` implements the single-model longitudinal training pipeline with
direct random-timestep corruption, source/text conditioning dropout, exact
sample-weighted gradient accumulation, BF16/FP16 autocast, safe clipping,
warmup-cosine scheduling, deterministic validation, and atomic adapter +
`conv_in` checkpoints with exact RNG/optimizer resume.

The recommended photo-editing baseline intentionally uses conservative learning
rates (`5e-5` LoRA, `1e-5` `conv_in`), Min-SNR 5, full timestep support, and
5% conditioning dropout. See `notebooks/training.ipynb` for the documented
server call to `TRAIN_AGGING_MODEL`.

```bash
pytest -q tests/test_training_*.py
python tests/run_training_pipeline_validation.py data/sample
```

## Inference

`src/inference` exposes deterministic direct img2img and DDIM-inversion editing
through `infer_face_aging`. It supports a numeric target age or explicit prompt,
three-way text/image classifier-free guidance, inference and training-resume
checkpoints, direct-vs-inverse grids, age sweeps, and training-time monitoring
of the same fixed photograph across epochs.

Use `notebooks/inference.ipynb` for the complete checkpoint-to-image server
workflow. The main switch is `use_inverse_diffusion=True`.

```bash
pytest -q tests/test_inference_*.py
python tests/run_inference_pipeline_validation.py
```
