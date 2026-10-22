import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def test_analysis_requires_dino_and_preserves_picking_images(tmp_path,monkeypatch):
    from test_evaluator import setup
    import src.comparisong.paper_analysis as api
    from src.aging_bias.references import ReferenceBundle
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    reference=ReferenceBundle.load(cfg.reference_bundle_path,load_training_data=False)
    before={p:p.read_bytes() for p in tmp_path.glob('M*/*/*.png')}
    monkeypatch.setattr(api,'refit_cached_references',lambda *a,**k:reference)
    monkeypatch.setattr(api,'preflight_dino_inputs',lambda *a,**k:None)
    moves=[]
    bundle['feature_backend'].encoder=SimpleNamespace(to=lambda device:moves.append(device))
    def dino(*args,**kwargs):
        assert kwargs['device']=='auto'
        return reference,dict(feature_config=cfg.features,feature_backend=bundle['feature_backend'])
    monkeypatch.setattr(api,'build_cached_embedding_references',dino)
    result=api.run_paper_analysis(cfg,bundle=bundle,allow_download=False)
    assert set(result['analyses'])=={'handcrafted_primary','handcrafted_icc030','dinov2'}
    assert result['analysis_status']['complete']
    assert moves==['cpu']  # one offload, not one per image
    for p,data in before.items():assert p.read_bytes()==data
    for folder in [cfg.output_dir,cfg.output_dir/'sensitivity_icc030',cfg.output_dir/'dinov2']:
        assert (folder/'diagnostics/direction_diagnostics.csv').is_file()
        assert (folder/'diagnostics/oracle_full.csv').is_file()
        assert (folder/'diagnostics/checks_report.csv').is_file()
    def fail(*a,**k):raise FileNotFoundError('missing FFHQ images for DINO')
    monkeypatch.setattr(api,'build_cached_embedding_references',fail)
    with pytest.raises(FileNotFoundError,match='FFHQ'):
        api.run_paper_analysis(cfg,bundle=bundle,allow_download=False)
    status=json.loads((cfg.output_dir/'diagnostics/analysis_status.json').read_text())
    assert not status['complete'] and not status['passed']


def test_refit_icc_sensitivity_uses_saved_rows_not_original_images(tmp_path):
    from src.aging_bias.references import fit_reference_feature_frames
    from src.aging_bias.config import ReferenceConfig
    from src.aging_bias.cached_references import refit_cached_references
    cfg=ReferenceConfig('/missing/train','/missing/proxy',tmp_path/'refs',train_only=True,min_retest_pairs=3)
    c=pd.DataFrame([dict(identity_id=f'id_{i}',age=age,image_sha256=f'{i}_{age}_{r}',
        image_path='/missing/photo.png',feature_config_hash='cached',phi__a=i+age/10+r*.001)
        for i in range(10) for age in [30,50] for r in [0,1]])
    p=pd.DataFrame(dict(age=[30.,50.],phi__a=[2.,4.],feature_config_hash='cached',image_path='/missing/proxy.png'))
    fit_reference_feature_frames(cfg,c,p,'cached')
    result=refit_cached_references(cfg.output_dir,tmp_path/'sensitivity',min_icc=.3)
    assert result.metadata['config']['min_icc']==.3
    assert (tmp_path/'sensitivity/attribute_reliability.csv').exists()
    with pytest.raises(ValueError,match='leakage'):
        refit_cached_references(cfg.output_dir,tmp_path/'bad',excluded_test_ids=['id_0'])
