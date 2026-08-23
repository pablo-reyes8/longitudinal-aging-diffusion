"""Run the offline training preflight and smoke pipeline without downloading SD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data import build_face_aging_dataloaders
from src.training import TRAIN_AGGING_MODEL, run_training_pipeline_validation
from training_fakes import make_training_bundle, make_training_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", nargs="?", type=Path, default=REPOSITORY_ROOT / "data" / "sample")
    args = parser.parse_args()
    loaders, _ = build_face_aging_dataloaders(
        args.root_dir, image_size=32, batch_size=2, num_workers=0,
        train_drop_last=False, train_shuffle=False,
    )
    bundle = make_training_bundle(seed=440)
    loss_fn = make_training_loss(bundle)
    report = run_training_pipeline_validation(bundle["scheduler_train"])
    with tempfile.TemporaryDirectory(prefix="face-aging-training-smoke-") as checkpoint_dir:
        state = TRAIN_AGGING_MODEL(
            bundle=bundle, loss_fn=loss_fn,
            train_loader=loaders["train"], val_loader=loaders["val"],
            max_train_steps=2, grad_accum_steps=2,
            max_train_batches=4, max_val_batches=2,
            lr_lora=1e-2, lr_conv_in=1e-2,
            amp_enabled=True, amp_dtype="bf16", device="cpu",
            gradient_checkpointing=False, enable_xformers=False,
            sample_target_posterior=False, min_snr_gamma=5,
            checkpoint_dir=checkpoint_dir, sample_every_epochs=0, log_every=0,
        )
        report["offline_sd15_shaped_smoke"] = {
            "status": "PASSED",
            "optimizer_steps": state["optimizer_step"],
            "validation_loss": state["history"]["val"][-1]["val/loss_total"],
            "checkpoint_written": (Path(checkpoint_dir) / "latest" / "training_resume.pt").exists(),
        }
    report["warnings"].append(
        "Actual SD1.5 + real identity/age encoders + GPU smoke remains NOT RUN until executed on the server"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
