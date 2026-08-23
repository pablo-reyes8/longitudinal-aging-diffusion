# Longitudinal Face Aging

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-188%20passed-brightgreen)

Supervised photo aging from real longitudinal observations. The project adapts
Stable Diffusion 1.5 with source-image conditioning and lightweight LoRA/DoRA
attention adapters, while keeping the VAE and CLIP text encoder frozen.

> Status: research pipeline. Data loading, model adaptation, numerical loss,
> mixed-precision training, checkpointing, direct inference, and deterministic
> DDIM inversion are implemented and covered by offline tests. Model weights
> and private face data are never included.

## Why longitudinal supervision?

Every training pair contains the same person at two different observed ages.
This gives the model a direct aging target while identity-disjoint splits prevent
the same person from appearing in training and evaluation.

```text
source photograph + target-age prompt
                 │
          VAE / CLIP encoding
                 │
  source latent + noisy target latent
                 │
       SD1.5 U-Net + LoRA/DoRA
                 │
  diffusion + identity + age objectives
                 │
       direct or DDIM-inverse edit
```

## Repository layout

```text
config/               YAML presets for data, models, training, and inference
data/                 Longitudinal indexing, pairing, datasets, and loaders
notebooks/            Guided server workflows
scripts/              Data, training, and inference CLI entry points
src/model/            SD1.5 loading, 8-channel conditioning, LoRA and DoRA
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
  --monitoring-image /server/monitor/fixed_face.jpg
```

The baseline uses conservative photo-editing hyperparameters: LoRA `5e-5`,
expanded input convolution `1e-5`, Min-SNR 5, 5% conditioning dropout, fixed
monitoring seed, and DDIM-inverse monitoring. It loads frozen
`py-feat/arcface_r50` and `iitolstykh/mivolo_v2` auxiliaries. To control VRAM,
their losses run every fourth training microbatch on 25% of eligible samples and use
activation checkpointing through the VAE and auxiliary networks. See
[`notebooks/training.ipynb`](notebooks/training.ipynb) for the complete setup.
Training images should already be face-centered or aligned: a non-differentiable
face detector is intentionally not inserted into the loss graph.

Training monitoring accepts either one age or an ordered sweep. With
`monitoring_target_age=[30, 40, 50, 65]`, every sampled epoch writes
`age_030.png`, `age_040.png`, `age_050.png`, `age_065.png`, and
`age_sweep.png` below `monitoring/epoch_NNN/`. The source image, seed, mode, and
guidance settings remain fixed across ages and epochs.

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
  --target-age 65 \
  --mode inverse \
  --output /server/outputs/person_age_65.png
```

An explicit prompt can replace `--target-age`:

```bash
aging-infer \
  --checkpoint checkpoint.pt \
  --image person.jpg \
  --target-prompt "photo of a person as 70-year-old" \
  --mode direct \
  --output aged.png
```

Inference accepts both lightweight adapter checkpoints and full training-resume
checkpoints; optimizer, loader, and GradScaler state are not needed.

## Python and notebooks

The same public APIs used by the CLI remain available for experiments:

```python
from data import build_face_aging_dataloaders
from src.inference import infer_face_aging

loaders, metadata = build_face_aging_dataloaders(
    "/path/to/dataset_root", image_size=256, batch_size=4
)

result = infer_face_aging(
    bundle=bundle,
    image="person.jpg",
    target_age=65,
    use_inverse_diffusion=True,
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
