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
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
