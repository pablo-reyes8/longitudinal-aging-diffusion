"""Per-epoch inference and independent quantitative evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.inference import infer_face_aging
from src.inference.source_image_loading import load_sweep_source_image
from src.quantitative_metrics import run_inference_and_evaluate


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _metric_text(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}" if value is not None else "n/a"
    except (TypeError, ValueError):
        return "n/a"


class AblationEpochEvaluator:
    """Generate one immutable panel with the live model and evaluate it."""

    def __init__(
        self,
        *,
        panel: Sequence[Mapping[str, Any]],
        metrics_bundle: Mapping[str, Any],
        output_dir: str | Path,
        model_name: str,
        ablation_case: int,
        inference_config: Mapping[str, Any],
        evaluation_config: Mapping[str, Any] | None = None,
        infer_backend=infer_face_aging,
        source_loader=load_sweep_source_image,
    ) -> None:
        self.panel = [dict(row) for row in panel]
        self.metrics_bundle = metrics_bundle
        self.output_dir = Path(output_dir)
        self.model_name = str(model_name)
        self.ablation_case = int(ablation_case)
        self.inference_config = dict(inference_config)
        self.evaluation_config = dict(evaluation_config or {})
        self.infer_backend = infer_backend
        self.source_loader = source_loader
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, *, bundle, epoch: int | None, destination: Path) -> dict[str, Any]:
        samples = []
        for row in self.panel:
            samples.append({
                **row,
                "model_name": self.model_name,
                "ablation_case": self.ablation_case,
                "epoch": epoch,
                "seed": int(self.inference_config.get("seed", 2026)),
            })

        def generate(*, source_path, source_age, target_age, **inference_options):
            image_size = int(inference_options.get("image_size", 400))
            resolved_source = self.source_loader(
                source_path, image_size=image_size, allow_lanczos_upscale=False
            )
            return self.infer_backend(
                bundle=bundle,
                image=resolved_source,
                source_age=int(source_age),
                target_age=int(target_age),
                output_type="pil",
                return_dict=True,
                compute_diagnostics=False,
                **inference_options,
            )

        evaluation_options = dict(self.evaluation_config)
        evaluation_options["metrics_bundle"] = self.metrics_bundle
        result = run_inference_and_evaluate(
            generate,
            samples,
            self.inference_config,
            evaluation_options,
            destination,
            model_name=self.model_name,
        )
        summary = result["evaluation"]["summary"].copy()
        summary_records = _json_records(summary)
        return {
            "manifest": result["manifest"],
            "manifest_path": Path(result["manifest_path"]),
            "summary": summary,
            "summary_records": summary_records,
            "n_samples": len(result["manifest"]),
            "output_dir": destination,
        }

    def on_epoch(self, *, bundle, epoch: int, **_context) -> dict[str, Any]:
        epoch_number = int(epoch) + 1
        destination = self.output_dir / "epoch_predictions" / f"epoch_{epoch_number:03d}"
        result = self._run(bundle=bundle, epoch=epoch_number, destination=destination)
        cumulative_path = self.output_dir / "epoch_metrics.csv"
        current = result["summary"].copy()
        current["epoch"] = epoch_number
        current["ablation_case"] = self.ablation_case
        if cumulative_path.exists():
            previous = pd.read_csv(cumulative_path)
            if {"epoch", "ablation_case"}.issubset(previous.columns):
                keep = ~(
                    (previous["epoch"] == epoch_number)
                    & (previous["ablation_case"] == self.ablation_case)
                )
                previous = previous.loc[keep]
            current = pd.concat([previous, current], ignore_index=True)
        current.to_csv(cumulative_path, index=False)
        first = result["summary_records"][0] if result["summary_records"] else {}
        print(
            " Ablation metrics  |  "
            f"case={self.ablation_case:02d}  epoch={epoch_number:03d}  "
            f"ID(source)={_metric_text(first.get('adaface_id_source_mean'), 4)}  "
            f"ID(GT)={_metric_text(first.get('adaface_id_gt_mean'), 4)}  "
            f"DEX MAE={_metric_text(first.get('dex_age_mae'), 3)}  "
            f"KID={_metric_text(first.get('kid_mean'), 5)}"
        )
        return {
            "manifest_path": str(result["manifest_path"]),
            "metrics_path": str(cumulative_path),
            "summary": result["summary_records"],
            "n_samples": result["n_samples"],
        }

    def evaluate_final(self, *, bundle) -> dict[str, Any]:
        destination = self.output_dir / "final_predictions"
        result = self._run(bundle=bundle, epoch=None, destination=destination)
        final_manifest = self.output_dir / "final_predictions.csv"
        result["manifest"].to_csv(final_manifest, index=False)
        final_metrics = self.output_dir / "final_metrics.json"
        final_metrics.write_text(
            json.dumps(result["summary_records"], indent=2), encoding="utf-8"
        )
        return {
            "manifest_path": str(final_manifest),
            "metrics_path": str(final_metrics),
            "summary": result["summary_records"],
            "n_samples": result["n_samples"],
        }
