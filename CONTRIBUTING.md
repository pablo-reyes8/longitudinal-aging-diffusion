# Contributing

Thank you for helping improve the longitudinal face-aging pipeline.

## Development setup

Use Python 3.10 or newer. The project was developed with the existing
`deep_learning` Conda environment; creating a new environment is not required.

```bash
conda activate deep_learning
python -m pip install -e ".[auxiliary,dev,notebooks]"
pytest -q
```

For a CUDA server, install the PyTorch wheel matching its CUDA runtime before
installing this project. `xformers` is optional because compatible wheels depend
on the exact PyTorch and CUDA versions.

## Workflow

1. Keep changes focused and preserve identity-disjoint data splitting.
2. Add or update tests for every behavioral change.
3. Run `pytest -q` and `ruff check .` before opening a pull request.
4. Never commit datasets, face images, model weights, credentials, or generated checkpoints.
5. Document configuration or public API changes in the README and example YAML files.

## Scientific changes

Changes to age parsing, pair construction, noise schedules, CFG, DDIM inversion,
checkpoint compatibility, or loss numerics require a small independent oracle
test. Qualitative claims about identity preservation should include the fixed
source image, seed, checkpoint, prompt, and inference settings used.

## Commit and pull-request guidance

Use short imperative commits such as `Add deterministic age sweep CLI`. A pull
request should state the motivation, list validation commands and results, and
call out any real-model or GPU checks that were not run.

By contributing, you agree that your contribution is licensed under the MIT License.
