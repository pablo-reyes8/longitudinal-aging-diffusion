"""Execute the notebook's generation/evaluation cells with toy images."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def notebook_cells():
    notebook=json.loads(Path('notebooks/Paper_comparison_bias_long_gap.ipynb').read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']


def test_notebook_code_syntax_and_clean_outputs():
    notebook=json.loads(Path('notebooks/Paper_comparison_bias_long_gap.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        assert cell.get('id')
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            source=''.join(cell['source'])
            if not source.lstrip().startswith('%'):ast.parse(source)


@pytest.mark.parametrize('run_dino',[False,True])
def test_notebook_generation_and_metrics_use_same_targetless_panel(tmp_path,monkeypatch,run_dino):
    from src.aging_bias.config import FeatureConfig,PaperV2Config
    from src.aging_bias.references import ReferenceBundle
    from src.comparisong.long_gap_evaluation import LongGapEvaluationConfig,evaluate_long_gap
    import src.comparisong.long_gap_experiment as experiment
    from dataclasses import asdict
    import src.comparisong.generate_images as generator
    import src.comparisong.load_models as loader
    import src.comparisong.evaluation_backends_v2 as backends
    import src.comparisong.frozen_dino as frozen
    source=tmp_path/'train'/'id1'/'19.png';source.parent.mkdir(parents=True)
    Image.new('RGB',(8,8)).save(source)
    reference=ReferenceBundle(knots=np.array([18.,80.]),f_coeff=np.array([[0.],[6.2]]),
        mu_c=np.zeros((2,1)),p_ages=np.array([18.,80.]),mu_p=np.array([[0.],[12.4]]),
        mean=np.array([0.]),sd=np.array([1.]),attributes=['zone_upper'],icc=np.array([1.]),
        feature_config_hash='frozen',support=(18.,80.),
        metadata={'config':{'features':asdict(FeatureConfig())}})
    reference.save(tmp_path/'ref')
    class Store:
        config_hash='frozen'
        def extract(self,path):return {'phi__zone_upper':10. if str(path)==str(source) else 14.1}
    monkeypatch.setattr(loader,'load_aging_models',lambda **kwargs:{'SAM':{'model':None}})
    monkeypatch.setattr(generator,'age_image',lambda *a,**kw:{'SAM':{60:Image.new('RGB',(8,8))}})
    monkeypatch.setattr(backends,'prepare_paper_bundle',lambda *a,**kw:{})
    monkeypatch.setattr(experiment,'prepare_paper_bundle',lambda *a,**kw:{'landmark_backend':object()})
    monkeypatch.setattr(frozen,'validate_frozen_dino_files',lambda *a:tmp_path/'ref'/'dinov2')
    monkeypatch.setattr(frozen,'load_frozen_dino_backend',lambda *a,**kw:(reference,{}))
    namespace=dict(Path=Path,pd=pd,np=np,gc=__import__('gc'),display=lambda *a:None,
        OUTPUT=tmp_path,MODELS=['SAM'],TRAIN_ROOT=tmp_path/'train',
        CONFIG=dict(n_transitions=1,target_ages=(60,),n_boot=0,inventory_roots=(),allow_download=False,run_dino=run_dino),
        generated_roots={'SAM':tmp_path/'generated'/'SAM'},REFERENCE_DIR=tmp_path/'ref',
        CHECKPOINT_PATH=tmp_path/'unused.pt',BATCH_SIZE=4,PROGRESS_EVERY=100,SEED=2026,
        ASSETS=tmp_path/'assets',BENCHMARK_ROOT=tmp_path/'bench',FADING_SPECIALIZED_PATH=None,
        GENDER_CSV=tmp_path/'missing.csv',genders={},features=FeatureConfig(),PaperV2Config=PaperV2Config,
        ALLOW_DOWNLOAD=False,N_BOOT=0,LongGapEvaluationConfig=LongGapEvaluationConfig,references=reference)
    def evaluate(config,**kwargs):
        assert config.manifest_path==tmp_path/'transitions.csv'
        return evaluate_long_gap(config,feature_store=Store(),**kwargs)
    monkeypatch.setattr(experiment,'evaluate_long_gap',evaluate)
    cells=notebook_cells()
    run=next(c for c in cells if 'result = run_long_gap_experiment(' in c)
    exec(compile(run,'notebook_run','exec'),namespace)
    result=namespace['result']
    assert result['failures'].empty
    measure=result['per_measure'].set_index('model_name')
    assert measure.loc['SAM','error']==pytest.approx(0.)
    assert measure.loc['source_noedit','error']==pytest.approx(-4.1)
    assert result['per_sample'].target_path.fillna('').eq('').all()
    assert (tmp_path/'per_measure.csv').is_file()
    assert (tmp_path/'selection_audit.json').is_file()
    assert (tmp_path/'generated'/'SAM').is_dir()
    assert (tmp_path/'.state').is_dir()
    assert (result['dinov2'] is not None)==run_dino
    if run_dino:
        assert (tmp_path/'dinov2_per_measure.csv').is_file()
    assert (tmp_path/'experiment_summary.json').is_file()
    assert len([c for c in cells])==4
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.setattr('IPython.display.display',lambda *a:None)
    experiment.display_long_gap_analysis(result,n_images=1)
    assert (tmp_path/'signed_residuals_by_photo.csv').is_file()
    assert (tmp_path/'signed_regional_residuals.png').is_file()
    assert (tmp_path/'sign_counts.csv').is_file()
    plt.close('all')
