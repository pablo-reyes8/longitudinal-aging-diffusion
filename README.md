# Longitudinal Face Aging

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-267%20passed-brightgreen)

Supervised photo aging from real longitudinal observations. The project adapts
Stable Diffusion 1.5 with source-image conditioning, explicit absolute/relative
age conditioning, and lightweight LoRA/DoRA attention adapters, while keeping
the VAE and CLIP text encoder frozen. A Fourier age conditioner and referenced
CFG prevent the numeric age signal from being eclipsed by the CLIP prompt.

> Status: research pipeline. Data loading, model adaptation, numerical loss,
> mixed-precision training, checkpointing, direct inference, and deterministic
> DDIM inversion are implemented and covered by offline tests. Model weights
> and private face data are never included.

## Why longitudinal supervision?

Every training pair contains the same person at two different observed ages.
This gives the model a direct aging target while identity-disjoint splits prevent
the same person from appearing in training and evaluation.

```text
source photograph + regularized target-age prompt
                          │
       VAE / CLIP / Fourier age conditioner V2
         (source age, target age, signed delta)
                          │
       source latent + noisy target latent
                          │
            SD1.5 U-Net + LoRA/DoRA
                          │
 diffusion + identity + absolute/relative age objectives
                          │
            direct or DDIM-inverse edit
```

## Repository layout

```text
config/               YAML presets for data, models, training, and inference
data/                 Longitudinal indexing, pairing, datasets, and loaders
notebooks/            Guided server workflows
scripts/              Data, training, and inference CLI entry points
src/model/            SD1.5 loading, image/age conditioning, LoRA and DoRA
src/loss/             Composite supervised diffusion objective
src/training/         Mixed-precision loop, validation, monitoring, checkpoints
src/inference/        Direct editing, three-way CFG, and DDIM inversion
tests/                Strict numerical, structural, and integration tests
```

## Dataset convention

Each identity has a directory. The numeric filename prefix is the observed age;
suffixes distinguish multiple photographs at the same age.

```text
dataset_root/
└── id_0001/
    ├── 18.jpg
    ├── 38.jpg
    ├── 38_1.jpg
    └── 52.png
```

Only forward pairs from the same identity are generated (`target_age >
source_age`). Labels such as `501.jpg` are audited and normalized to age `51`.
The `data/sample` directory is intentionally ignored by Git.

## Installation

Activate the existing environment and install the project in editable mode:

```bash
conda activate deep_learning
python -m pip install -e ".[auxiliary,dev,notebooks]"
```

The ArcFace auxiliary extra requires Python 3.11 or newer. Core data, model,
loss, training, and inference modules remain compatible with Python 3.10.

On a GPU server, install the PyTorch build matching its CUDA version first.
`xformers` is optional and must match both PyTorch and CUDA:

```bash
python -m pip install -e ".[xformers]"
```

The project lazily imports Diffusers and Transformers, so all offline structural
tests can run without downloading SD1.5. The maintained backbone identifier is
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

## Command-line workflows

All commands read versioned YAML presets from [`config`](config). Values that
identify private resources—dataset paths, checkpoints, and source images—are
provided at runtime.

### 1. Prepare and validate data

```bash
aging-data \
  --dataset-root /server/data/longitudinal_faces \
  --config config/data/default.yaml \
  --output-dir artifacts/data
```

This writes `manifest.csv`, `splits.csv`, and `validation_report.json`. It checks
age parsing, corrupt images, identity/path leakage, forward pairs, and split
coverage. Without editable installation, use:

```bash
python scripts/prepare_data.py --dataset-root /server/data/longitudinal_faces
```

### 2. Train

```bash
aging-train \
  --dataset-root /server/data/longitudinal_faces \
  --data-config config/data/default.yaml \
  --model-config config/models/sd15_lora.yaml \
  --training-config config/training/photo_editing.yaml \
  --checkpoint-dir /server/checkpoints/face_aging \
  --monitoring-image /server/monitor/fixed_face.jpg \
  --monitoring-source-age 26
```

The baseline uses conservative photo-editing settings: LoRA `3e-5`, expanded input convolution `5e-6`, age conditioner `1e-4`, Min-SNR `5`, and 5% conditioning dropout. Loss weights are `identity=0.20`, `absolute_age=0.05`, and `relative_age=0.05`.

Age Conditioner V2 encodes source age, target age, and signed age delta using Fourier features. Training mixes numeric target-age prompts (70%) with generic aging prompts (30%) and uses frozen ArcFace and MiVOLO models for identity and age supervision.

The loader supports:
- **Zero-delta self-pairs:** 20% train-only self-pairs with additional preservation loss and higher diffusion weighting near zero age change.
- **Bidirectional augmentation:** 20% of non-self training observations are reversed to expose the model to negative age deltas.
- **Optional FG-NET augmentation:** enabled with `include_kaggle=True`. FG-NET supplements, but never replaces, Colombian training observations. Validation and test remain Colombian.

To reduce VRAM usage, ArcFace and MiVOLO losses are evaluated every fourth microbatch on a subset of eligible samples, with activation checkpointing through the VAE and auxiliary networks.

### Monitoring

Training can monitor a single target age or an ordered sweep, e.g.

```python
monitoring_source_age = 26
monitoring_target_age = [16, 18, 24, 30, 35, 40, 50, 65]
```

Each monitored epoch saves individual generations, an age_sweep.png, and sampling diagnostics containing target age, predicted age, identity cosine similarity, and age-calibration statistics (intercept, slope, R²). The grid is chronological: rejuvenation targets appear left of the original and aging targets appear right.

Forward and reverse calibration fits are tracked independently. A default multi-strength diagnostic (`0.20`, `0.27`, `0.35`, `0.40`) is saved as one native-resolution PNG plus one compact summary CSV; it does not create per-strength image trees. Optional reverse-only relative-loss weighting and delta-dependent inference strength remain disabled by default.

Checkpoint selection keeps two independent best models: `best/` follows validation loss, while `best_calibration_checkpoint/` minimizes `|intercept| + 10|slope - 1|` from the existing monitoring sweep.

The source image, seed, and guidance settings remain fixed across epochs to make visual comparisons meaningful.

See notebooks/training.ipynb for the complete training setup and configuration details.

Resume exactly from a training checkpoint with:

```bash
aging-train \
  --dataset-root /server/data/longitudinal_faces \
  --resume-from /server/checkpoints/face_aging/latest/training_resume.pt
```

### 3. Infer

```bash
aging-infer \
  --checkpoint /server/checkpoints/face_aging/best/adapter_inference.pt \
  --image /server/input/person.jpg \
  --source-age 26 \
  --target-age 65 \
  --mode direct \
  --output /server/outputs/person_age_65.png
```

An explicit prompt can replace `--target-age`:

```bash
aging-infer \
  --checkpoint checkpoint.pt \
  --image person.jpg \
  --source-age 26 \
  --target-prompt "photo of a person as 70-year-old" \
  --mode direct \
  --output aged.png
```

New V2 checkpoints require both source and target ages at inference; their
numerical condition includes both absolute ages and the signed difference.
Direct inference defaults to source-age referenced CFG (`age_guidance_scale=3`),
while `text_reference_mode="null"` remains an exact legacy-CFG compatibility
mode. Inverse diffusion deliberately keeps the legacy null-reference behavior.
Inference accepts both lightweight adapter checkpoints and full training-resume
checkpoints; optimizer, loader, and GradScaler state are not needed.
`diagnose_checkpoint_age_sweep` loads either checkpoint format and returns a
pandas DataFrame with predicted age, signed age error, and identity cosine while
optionally saving the individual images, annotated grid, and CSV.

## Python and notebooks

The same public APIs used by the CLI remain available for experiments:

```python
from data import build_face_aging_dataloaders
from src.inference import infer_face_aging

loaders, metadata = build_face_aging_dataloaders(
    "/path/to/dataset_root", image_size=256, batch_size=4,
    include_zero_delta_pairs=True, zero_delta_pair_prob=0.20,
    include_bidirectional_pairs=True, reverse_pair_prob=0.20,
    include_kaggle=True, kaggle_path="/path/to/FGNET/images",
    kaggle_proportion=0.40, kaggle_reverse_pair_prob=0.50,
)

result = infer_face_aging(
    bundle=bundle,
    image="person.jpg",
    source_age=26,
    target_age=65,
    use_inverse_diffusion=False,
    text_reference_mode="source_age",
    age_guidance_scale=3.0,
    seed=42,
)
```

- [`notebooks/data_loader.ipynb`](notebooks/data_loader.ipynb): build and inspect loaders.
- [`notebooks/kaggle_dataset.ipynb`](notebooks/kaggle_dataset.ipynb): audit FG-NET and preview complementary selection.
- [`notebooks/model.ipynb`](notebooks/model.ipynb): load SD1.5 and verify trainable parameters.
- [`notebooks/training.ipynb`](notebooks/training.ipynb): full server training workflow.
- [`notebooks/inference.ipynb`](notebooks/inference.ipynb): checkpoint-to-image inference.

## Docker

The image uses a CUDA-enabled PyTorch runtime. Datasets, checkpoints, outputs,
and the Hugging Face cache are mounted rather than copied into the image.

```bash
cp .env.example .env
docker compose build
docker compose run --rm face-aging python -m scripts.train --help
```

Set host paths and `HF_TOKEN` in `.env`. NVIDIA Container Toolkit is required
for GPU access. To launch a real command, override the Compose help command and
use container paths such as `/workspace/data` and `/workspace/checkpoints`.

## Tests and validation

```bash
pytest -q
python tests/run_data_pipeline_validation.py /path/to/dataset_root
python tests/run_face_aging_model_validation.py /path/to/dataset_root
python tests/run_face_aging_loss_validation.py /path/to/dataset_root
python tests/run_training_pipeline_validation.py /path/to/dataset_root
python tests/run_inference_pipeline_validation.py
```

The suite uses architecture-compatible doubles and independent mathematical
oracles. Real SD1.5/GPU qualitative validation is intentionally reported as
`NOT RUN` when weights are unavailable—it is never silently replaced by a fake
success.

## Privacy and responsible use

Face images are sensitive biometric data. Use data with appropriate consent and
legal authority, minimize retention, restrict checkpoint access, and document
demographic coverage and known limitations. Generated ages are synthetic visual
edits, not medical predictions or verified future appearances.

The pretrained `py-feat/arcface_r50` weights inherit InsightFace's
non-commercial-research restriction; the repository's MIT license does not
override that model license. MiVOLO V2 is loaded with Hugging Face
`trust_remote_code=True`; review and pin its repository revision for controlled
or long-lived training runs.

## Contributing and license

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and scientific-testing
requirements, [`SECURITY.md`](SECURITY.md) for responsible disclosure, and
[`CHANGELOG.md`](CHANGELOG.md) for notable changes. This project is released
under the [`MIT License`](LICENSE).
