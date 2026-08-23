"""Generate one aged photograph from a trained adapter checkpoint."""
# ruff: noqa: E402 -- direct-file execution bootstraps the project root.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import infer_face_aging, load_face_aging_inference_bundle, save_inference_image

from scripts.common import load_yaml, resolve_dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="adapter_inference.pt or training_resume.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-age", type=int)
    target.add_argument("--target-prompt")
    parser.add_argument("--source-age", type=int)
    parser.add_argument("--source-prompt")
    parser.add_argument("--config", default="config/inference/default.yaml")
    parser.add_argument("--mode", choices=("direct", "inverse"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace):
    config = load_yaml(args.config)
    if args.mode is not None:
        config["mode"] = args.mode
        config["use_inverse_diffusion"] = args.mode == "inverse"
    if args.seed is not None:
        config["seed"] = args.seed
    device = None if args.device == "auto" else args.device
    bundle = load_face_aging_inference_bundle(
        args.checkpoint,
        device=device,
        dtype=resolve_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    result = infer_face_aging(
        bundle=bundle,
        image=args.image,
        target_age=args.target_age,
        target_prompt=args.target_prompt,
        source_age=args.source_age,
        source_prompt=args.source_prompt,
        **config,
    )
    destination = save_inference_image(result, args.output)
    print(f"Saved {result['mode']} edit to: {destination.resolve()}")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
