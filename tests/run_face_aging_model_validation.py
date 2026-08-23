"""Run the model/data integration audit offline without downloading SD weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data import build_face_aging_dataloaders
from model_fakes import make_fake_components
from src.model import assemble_face_aging_diffusion_bundle, run_face_aging_model_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", nargs="?", type=Path, default=REPOSITORY_ROOT / "data" / "sample")
    args = parser.parse_args()
    loaders, _ = build_face_aging_dataloaders(
        args.root_dir,
        image_size=32,
        batch_size=2,
        num_workers=0,
        train_shuffle=False,
        train_drop_last=False,
    )
    batch = next(iter(loaders["train"]))
    bundle = assemble_face_aging_diffusion_bundle(
        make_fake_components(),
        model_id="offline-fake/sd15-architecture",
        rank=2,
        alpha=2,
        verbose=False,
    )
    report = run_face_aging_model_validation(bundle, batch=batch)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
