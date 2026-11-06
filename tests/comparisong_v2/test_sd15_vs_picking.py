import json
from pathlib import Path
import pandas as pd
import pytest
from src.aging_bias.config import SD15Config
from src.comparisong import sd15_vs_picking as mechanism


def test_three_variants_share_aging_panel_and_resume_settings(tmp_path,monkeypatch):
    panel=pd.DataFrame([dict(sample_id=f'p{i}',identity_id='id_1',source_age=a,target_age=b,
        source_path=f'{a}.png',target_path=f'{b}.png') for i,(a,b) in enumerate([(20,50),(50,20),(20,25)])])
    manifest=tmp_path/'input.csv';panel.to_csv(manifest,index=False)
    validation=tmp_path/'check.json';validation.write_text(json.dumps({'passed':True}))
    config=SD15Config(checkpoint_path='unused.pt',test_manifest_path=manifest,
        output_dir=tmp_path/'output',validation_path=validation,anchors_reviewed=True,
        inference_config={'image_size':400,'num_inference_steps':34},max_pairs=1)
    calls=[];bundle={'fake':True}
    def vanilla(conf,**kwargs):
        assert kwargs=={'conditions':('vanilla',)}
        assert conf.strengths==(.30,)
        assert list(pd.read_csv(conf.test_manifest_path).sample_id)==['p0']
        calls.append('vanilla')
        return dict(manifest=pd.DataFrame({'model_name':['SD15_vanilla@0.3']}),failures=pd.DataFrame(),
            generated_roots={'SD15_vanilla@0.3':'vanilla'})
    def picking(conf,**kwargs):
        assert kwargs['bundle'] is bundle
        assert conf.resume and not conf.overwrite
        assert conf.target_age_strength_map is None
        assert conf.delta_age_strength_map=={0:.04,30:.29}
        assert conf.num_inference_steps==34
        assert list(pd.read_csv(conf.test_manifest_path).sample_id)==['p0']
        calls.append('picking')
        return dict(manifest=pd.DataFrame({'model_name':['ours_picking_no_prompt','ours_picking_prompt']}),
            failures=pd.DataFrame(),generated_roots={'ours_picking_no_prompt':'base','ours_picking_prompt':'prompt'})
    monkeypatch.setattr(mechanism,'run_sd15_before_after',vanilla)
    monkeypatch.setattr(mechanism,'run_picking_variants',picking)
    result=mechanism.run_sd15_vs_picking(config,picking_bundle=bundle,delta_age_strength_map={0:.04,30:.29})
    assert calls==['vanilla','picking']
    assert len(result['generated_roots'])==3
    assert len(pd.read_csv(result['paths']['manifest']))==3
    assert len(pd.read_csv(result['paths']['transitions']))==1


def test_invalid_strength_stops_before_assets_or_generation():
    with pytest.raises(ValueError,match='vanilla_strength'):
        mechanism.run_sd15_vs_picking(None,vanilla_strength=0)
