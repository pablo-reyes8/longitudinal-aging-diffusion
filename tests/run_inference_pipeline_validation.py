"""Run direct/inverse offline inference smoke without downloading SD1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.inference import infer_face_aging, run_inference_pipeline_validation, save_inference_image
from training_fakes import make_training_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    args = parser.parse_args()
    image_path = args.image
    if image_path is None:
        candidates = sorted((REPOSITORY_ROOT / "data" / "sample").glob("*/*"))
        if not candidates:
            raise FileNotFoundError("No sample image found; provide an image path")
        image_path = candidates[0]
    bundle = make_training_bundle(seed=2005)
    report = run_inference_pipeline_validation()
    with tempfile.TemporaryDirectory(prefix="face-aging-inference-smoke-") as output_dir:
        outputs = {}
        for mode in ("direct", "inverse"):
            first = infer_face_aging(
                bundle=bundle, image=image_path, target_age=65, source_age=30,
                mode=mode, num_inference_steps=8, strength=0.45,
                image_size=32, seed=42, return_latents=True,
            )
            second = infer_face_aging(
                bundle=bundle, image=image_path, target_age=65, source_age=30,
                mode=mode, num_inference_steps=8, strength=0.45,
                image_size=32, seed=42, return_latents=True,
            )
            path = save_inference_image(first, Path(output_dir) / f"{mode}.png")
            outputs[mode] = {
                "status": "PASSED",
                "deterministic": bool(torch.equal(first["latents"], second["latents"])),
                "finite": bool(torch.isfinite(first["latents"]).all()),
                "shape": list(first["image_tensor"].shape),
                "temporary_output_written": path.exists(),
            }
        report["offline_sd15_shaped_smoke"] = outputs
        if not all(item["deterministic"] and item["finite"] for item in outputs.values()):
            report["passed"] = False
            report["errors"].append("Offline direct/inverse determinism or finiteness failed")
    report["warnings"].append(
        "Real trained SD1.5 checkpoint + GPU qualitative inversion reconstruction remains NOT RUN"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
