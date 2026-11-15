"""Generation accepts absent targets without weakening observed evaluation."""
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from src.comparisong.evaluate_paper_v2 import load_test_panel


def test_missing_target_is_explicitly_generation_only_and_roundtrips(tmp_path):
    manifest=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id='cf1',identity_id='id1',source_age=19,target_age=65,
        source_path='19.png',target_path=None)]).to_csv(manifest,index=False)
    with pytest.raises(ValueError,match='missing'):
        load_test_panel(manifest)
    panel=load_test_panel(manifest,tmp_path,require_target=False)
    assert panel.target_path.tolist()==['']
    assert panel.source_path.tolist()==[str(tmp_path/'id1'/'19.png')]
    panel.to_csv(manifest,index=False)
    assert load_test_panel(manifest,require_target=False).target_path.tolist()==['']


def test_counterfactual_generation_fast_resume(tmp_path,monkeypatch):
    from src.comparisong.generate_paper_predictions import generate_paper_predictions
    import src.comparisong.generate_images as generator
    import src.comparisong.load_models as loader
    image=tmp_path/'19.png';Image.new('RGB',(8,8)).save(image)
    manifest=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id='cf1',identity_id='id1',source_age=19,target_age=65,
        source_path=str(image),target_path=None)]).to_csv(manifest,index=False)
    config=SimpleNamespace(test_manifest_path=manifest,source_root=None,models=['SAM'],
        generated_roots={'SAM':tmp_path/'generated'/'SAM'},output_dir=tmp_path/'metrics',seed=2026,resume=True)
    calls=[]
    def generate(*args,**kwargs):
        calls.append(kwargs['target_ages'])
        return {'SAM':{65:Image.new('RGB',(8,8))}}
    monkeypatch.setattr(generator,'age_image',generate)
    monkeypatch.setattr(loader,'load_aging_models',lambda **kwargs:{'SAM':{'model':None}})
    first=generate_paper_predictions(config)
    second=generate_paper_predictions(config)
    assert calls==[[65]]
    assert first['manifest'].inference_status.tolist()==['ok']
    assert second['manifest'].target_path.tolist()==['']
    with pytest.raises(ValueError,match='missing'):
        load_test_panel(manifest)
