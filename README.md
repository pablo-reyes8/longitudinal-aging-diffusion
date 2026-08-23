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

The validation report checks identity/path leakage, forward-pair correctness against a brute-force oracle, prompt consistency, image decoding, split coverage, and pair concentration. This repository currently implements only the data layer; model and training code are intentionally out of scope.
