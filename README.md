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
