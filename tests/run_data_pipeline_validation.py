"""CLI audit: python tests/run_data_pipeline_validation.py /path/to/dataset"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data import run_data_pipeline_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--skip-image-validation", action="store_true")
    args = parser.parse_args()
    report = run_data_pipeline_validation(
        args.root_dir, validate_images=not args.skip_image_validation
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
