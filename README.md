# Longitudinal Face Aging

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-233%20passed-brightgreen)

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

The baseline uses conservative photo-editing hyperparameters: LoRA `3e-5`,
expanded input convolution `5e-6`, age conditioner `1e-4`, Min-SNR 5, 5%
conditioning dropout, a fixed monitoring seed, direct monitoring at strength
`0.35`, and loss weights `identity=0.20`, `absolute_age=0.05`,
`relative_age=0.05`. Age Conditioner V2 encodes source age, target age, and
signed delta with eight Fourier frequency bands, a 256-unit hidden layer, and
a trainable output gate initialized to `1.0`. Training independently mixes 70%
numeric target-age prompts with 30% generic aging prompts. It loads frozen
`py-feat/arcface_r50` and `iitolstykh/mivolo_v2` auxiliaries. The zero-delta
baseline injects 20% train-only self-pairs, adds decoded L1 preservation with
weight `0.10` for `|delta_age| <= 2`, and weights diffusion samples `2x` for
`|delta_age| <= 5`. Validation and test retain only real longitudinal pairs.
To control VRAM, the ArcFace and MiVOLO losses run every fourth training
microbatch on 25% of eligible samples and use
activation checkpointing through the VAE and auxiliary networks. See
[`notebooks/training.ipynb`](notebooks/training.ipynb) for the complete setup.
Training images should already be face-centered or aligned: a non-differentiable
face detector is intentionally not inserted into the loss graph.

Training monitoring accepts either one age or an ordered sweep. With
`monitoring_target_age=[30, 35, 40, 50, 65]`, every sampled epoch writes
`age_030.png`, `age_035.png`, `age_040.png`, `age_050.png`, `age_065.png`, and
`age_sweep.png` below `monitoring/epoch_NNN/`. The source image, seed, mode, and
guidance settings remain fixed across ages and epochs. Direct monitoring uses
the source-age prompt as the referenced-CFG baseline with
`age_guidance_scale=3.0`; selecting `text_reference_mode="null"` reproduces the
legacy CFG equation exactly.
When ArcFace and MiVOLO are attached, monitoring also annotates the grid with
target age, predicted age, and source/generated identity cosine. Each epoch
writes `sampling_diagnostics_epoch_NNN.csv`, while
`monitoring/sampling_diagnostics_history.csv` appends the complete trajectory.

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
