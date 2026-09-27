"""Run a face-aging sweep with the external comparison baselines."""
# ruff: noqa: E402 -- direct-file execution bootstraps the project root.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.comparisong import age_image, load_aging_models


DEFAULT_MODELS = ["SAM", "FRAN", "FADING", "CUSP", "HRFAE"]


def parse_target_images(assignments):
    """Parse repeatable AGE=PATH_OR_URL CLI values."""
    parsed = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError("--target-image values must use AGE=PATH_OR_URL")
        age, reference = assignment.split("=", 1)
        if not reference:
            raise ValueError("--target-image values must use AGE=PATH_OR_URL")
        parsed[int(age)] = reference
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Local image path or HTTP(S) URL")
    parser.add_argument("--source-age", required=True, type=int)
    parser.add_argument("--target-ages", required=True, nargs="+", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/content/aging_benchmarks")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS.copy())
    parser.add_argument("--gender", choices=("male", "female"), default="male")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument(
        "--target-image", action="append", default=[], metavar="AGE=PATH_OR_URL",
        help="Real same-person longitudinal target; repeat once per available age",
    )
    parser.add_argument("--metrics-output-dir")
    parser.add_argument("--adaface-repo-path")
    parser.add_argument("--adaface-checkpoint-path")
    parser.add_argument("--dex-prototxt-path")
    parser.add_argument("--dex-checkpoint-path")
    parser.add_argument("--kid-inception-weights-path")

    parser.add_argument("--fading-specialized-path")
    parser.add_argument("--cusp-variant", choices=("ffhq_ls", "ffhq_rr"), default="ffhq_ls")
    parser.add_argument("--cusp-checkpoint-path")
    parser.add_argument("--hrfae-checkpoint-path")
    parser.add_argument(
        "--download-cusp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--download-hrfae", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--download-cradle", action="store_true")
    return parser


def run(args: argparse.Namespace):
    metrics_config = None
    if args.metrics:
        metric_paths = {
            "adaface_repo_path": args.adaface_repo_path,
            "adaface_checkpoint_path": args.adaface_checkpoint_path,
            "dex_prototxt_path": args.dex_prototxt_path,
            "dex_checkpoint_path": args.dex_checkpoint_path,
            "kid_inception_weights_path": args.kid_inception_weights_path,
        }
        missing = [name for name, value in metric_paths.items() if not value]
        if missing:
            raise ValueError(
                "--metrics requires local model paths: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        metrics_config = metric_paths
    bundle = load_aging_models(
        root=args.root,
        models=tuple(args.models),
        fading_specialized_path=args.fading_specialized_path,
        cusp_variant=args.cusp_variant,
        cusp_checkpoint_path=args.cusp_checkpoint_path,
        download_cusp=args.download_cusp,
        hrfae_checkpoint_path=args.hrfae_checkpoint_path,
        download_hrfae=args.download_hrfae,
        download_cradle=args.download_cradle,
        load_metrics=args.metrics,
        metrics_config=metrics_config,
    )
    return age_image(
        bundle=bundle,
        image=args.image,
        source_age=args.source_age,
        target_ages=args.target_ages,
        gender=args.gender,
        models=args.models,
        seed=args.seed,
        show=args.show,
        output_dir=args.output_dir,
        strict=args.strict,
        metrics=args.metrics,
        target_images=parse_target_images(args.target_image),
        metrics_output_dir=args.metrics_output_dir,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
