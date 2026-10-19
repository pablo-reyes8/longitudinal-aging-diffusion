"""Pin the legacy public evaluator and the actual ablation evaluation path."""
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path


class Identity:
    def embed_batch(self, images):
        return np.array([np.asarray(im).mean(axis=(0, 1)) for im in images])


class Age:
    def predict_batch(self, images):
        return [float(np.asarray(im)[..., 0].mean()) for im in images]


def test_legacy_evaluator_and_ablation_snapshot(tmp_path):
    from src.quantitative_metrics import evaluate_aging_outputs
    from src.ablation_studies.evaluation import AblationEpochEvaluator

    paths = []
    for name, color in [('source', (10, 0, 0)), ('target', (0, 10, 0)), ('generated', (10, 10, 0))]:
        p = tmp_path / f'{name}.png'
        Image.new('RGB', (16, 16), color).save(p)
        paths.append(str(p))
    source, target, generated = paths
    panel = [dict(sample_id='pair', identity_id='id_1', source_path=source,
                  target_path=target, source_age=5, target_age=20)]
    bundle = dict(aligner=lambda im: im, identity_encoder=Identity(), age_estimator=Age())
    manifest = pd.DataFrame([{**panel[0], 'generated_path': generated, 'model_name': 'M'}])
    direct = evaluate_aging_outputs(manifest, metrics_bundle=bundle,
        output_dir=tmp_path / 'direct', compute_kid=False)
    evaluator = AblationEpochEvaluator(panel=panel, metrics_bundle=bundle,
        output_dir=tmp_path / 'ablation', model_name='M', ablation_case=1,
        inference_config={}, evaluation_config={'compute_kid': False},
        source_loader=lambda *a, **k: Image.open(source),
        infer_backend=lambda **kw: {'image': Image.open(generated)})
    result = evaluator._run(bundle={}, epoch=1, destination=tmp_path / 'ablation' / 'epoch')
    snapshot=json.loads((Path(__file__).parent/'snapshots/evaluator.json').read_text())
    columns = snapshot['summary_columns']
    expected = np.array(snapshot['summary_values'])
    np.testing.assert_allclose(direct['summary'][columns], expected, atol=1e-12)
    np.testing.assert_allclose(result['summary'][columns], expected, atol=1e-12)
    assert list(direct['per_sample'].columns) == [*manifest.columns,*snapshot['per_sample_metric_columns']]
