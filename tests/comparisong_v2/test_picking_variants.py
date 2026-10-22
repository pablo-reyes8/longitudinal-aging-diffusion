import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_smart_variants_keep_prompt_even_when_worse_and_resume(tmp_path):
    from src.comparisong.picking_variants import PickingVariantsConfig,run_picking_variants
    source=tmp_path/'source.png';Image.new('RGB',(16,16)).save(source)
    panel=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id='id_1__20_to_50',identity_id='id_1',source_path=str(source),target_path=str(source),source_age=20,target_age=50)]).to_csv(panel,index=False)
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy adapter')
    calls=[]
    def sweep(**kw):
        calls.append(kw)
        frame=pd.DataFrame([dict(strength=.27,age_error=0)])
        frame.attrs['selected_results']=[{'image':Image.new('RGB',(16,16),(255,0,0))}]
        frame.attrs['assisted_results']=[{'image':Image.new('RGB',(16,16),(0,0,255))}]
        frame.attrs['assisted']=pd.DataFrame([dict(strength=.27,age_error=9)])
        return frame
    class Module:
        def __init__(self):self.placements=[]
        def to(self,d):self.placements.append(d);return self
    module=Module();bundle=dict(unet=module,identity_encoder=object(),age_estimator=object())
    cfg=PickingVariantsConfig(checkpoint,panel,tmp_path/'out',device='cpu',target_age_strength_map={50:.27},overwrite=False)
    result=run_picking_variants(cfg,bundle=bundle,sweep_backend=sweep)
    assert len(calls)==1
    assert calls[0]['min_strength']==.04 and calls[0]['max_strength']==.45
    assert calls[0]['max_trials_per_target']==10
    assert calls[0]['use_bias_corrected_mivolo_target'] is True
    assert calls[0]['prompt_assistance_scale']==.30
    assert calls[0]['negative_prompt_assistance_scale']==.25
    assert calls[0]['prompt_assistance_config']['use_expression_rules'] is False
    assert calls[0]['target_age_strength_map']=={50:.27}
    assert set(result['manifest'].model_name)=={'ours_picking_no_prompt','ours_picking_prompt'}
    rows=result['manifest'].set_index('model_name')
    assert Image.open(rows.loc['ours_picking_no_prompt','generated_path']).getpixel((0,0))==(255,0,0)
    assert Image.open(rows.loc['ours_picking_prompt','generated_path']).getpixel((0,0))==(0,0,255)
    run_picking_variants(cfg,bundle=bundle,sweep_backend=lambda **kw:(_ for _ in ()).throw(AssertionError('cached inference repeated')))
    assert module.placements==['cpu','cpu']  # Activate once, offload once, cached run inactive.


def test_default_picking_start_depends_on_gap_not_absolute_target_and_resumes(tmp_path):
    from src.comparisong.picking_variants import PickingVariantsConfig,run_picking_variants
    source=tmp_path/'source.png';Image.new('RGB',(16,16)).save(source)
    pairs=[(38,43,.095),(38,68,.23),(18,48,.23),(68,38,.23)]
    panel=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id=f'p{i}',identity_id='id_1',source_path=str(source),target_path=str(source),
        source_age=a,target_age=b) for i,(a,b,_) in enumerate(pairs)]).to_csv(panel,index=False)
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy adapter')
    starts=[]
    def sweep(**kwargs):
        target=kwargs['target_ages'][0]
        assert set(kwargs['target_age_strength_map'])=={target}
        start=kwargs['target_age_strength_map'][target];starts.append(start)
        frame=pd.DataFrame([dict(strength=start)])
        frame.attrs['selected_results']=[{'image':Image.new('RGB',(16,16))}]
        frame.attrs['assisted_results']=[{'image':Image.new('RGB',(16,16))}]
        frame.attrs['assisted']=frame.copy()
        return frame
    config=PickingVariantsConfig(checkpoint,panel,tmp_path/'out',device='cpu',overwrite=False)
    bundle=dict(unet=object(),identity_encoder=object(),age_estimator=object())
    result=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    assert starts==pytest.approx([.095,.23,.23,.23])
    assert result['manifest'].initial_strength.tolist()==pytest.approx([v for _,_,s in pairs for v in [s,s]])
    cached=run_picking_variants(config,bundle=bundle,sweep_backend=lambda **k:pytest.fail('cached pair repeated'))
    assert cached['failures'].empty
    config.delta_age_strength_map={0:.2,60:.2}
    refreshed=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    assert refreshed['failures'].empty
    assert starts[4:]==pytest.approx([.2]*4)
    assert len(list((tmp_path/'out/stale').rglob('*.png')))==8


def test_gap_start_interpolates_and_respects_search_bounds():
    from src.comparisong.picking_variants import PickingVariantsConfig,initial_strength_for_transition
    config=PickingVariantsConfig('checkpoint','manifest','output')
    assert initial_strength_for_transition(40,42,config)==pytest.approx(.062)
    assert initial_strength_for_transition(38,68,config)==pytest.approx(.23)
    assert initial_strength_for_transition(30,45,config)==pytest.approx(.185)
    config.max_strength=.25
    assert initial_strength_for_transition(38,78,config)==.25
    config.delta_age_strength_map={-1:.2}
    with pytest.raises(ValueError,match='delta_age_strength_map'):
        initial_strength_for_transition(38,68,config)


def test_mouth_protection_reaches_actual_prompt_without_classifier_or_state_guess():
    from src.comparisong.picking_variants import protected_prompt_assistance
    from src.inference.prompt_assistance import resolve_prompt_assistance
    supplied=dict(positive_global_terms=[],negative_global_terms=[],max_support_terms_per_block=4,
        use_mouth_rules=True,use_eye_rules=True,use_expression_rules=False)
    effective=protected_prompt_assistance(supplied)
    resolved=resolve_prompt_assistance(base_target_prompt='portrait at age 60',
        prompt_assistance_config=effective,source_mouth_state='unknown',source_expression_state='unknown')
    for term in ['preserve mouth shape','preserve original mouth opening','preserve natural teeth visibility']:
        assert term in resolved['positive_terms']
    for term in ['malformed mouth','distorted lips','deformed teeth','extra teeth']:
        assert term in resolved['negative_prompt']
    assert 'natural eyes' in resolved['positive_terms']
    assert 'keep lips closed' not in resolved['target_prompt']
    assert 'open mouth' not in resolved['negative_terms']
    assert resolved['warnings']==[]
    assert supplied['positive_global_terms']==[]  # User config is not mutated.


def test_overwrite_is_default_even_with_resume_and_never_exposes_old_image_on_failure(tmp_path):
    from src.comparisong.picking_variants import PickingVariantsConfig,run_picking_variants
    source=tmp_path/'source.png';Image.new('RGB',(16,16)).save(source)
    panel=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id='p0',identity_id='id_1',source_path=str(source),target_path=str(source),
        source_age=40,target_age=42)]).to_csv(panel,index=False)
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy adapter')
    config=PickingVariantsConfig(checkpoint,panel,tmp_path/'out',device='cpu',resume=True)
    bundle=dict(unet=object(),identity_encoder=object(),age_estimator=object())
    calls=[]
    def sweep(**kwargs):
        calls.append(1)
        frame=pd.DataFrame([dict(strength=.1)])
        color=(len(calls)*20,0,0)
        frame.attrs['selected_results']=[{'image':Image.new('RGB',(16,16),color)}]
        frame.attrs['assisted_results']=[{'image':Image.new('RGB',(16,16),color)}]
        frame.attrs['assisted']=frame.copy()
        return frame
    first=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    second=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    assert len(calls)==2
    assert second['manifest'].generated_path.tolist()==first['manifest'].generated_path.tolist()
    for path in second['manifest'].generated_path:
        assert Image.open(path).getpixel((0,0))==(40,0,0)
    assert len(list((tmp_path/'out/stale').rglob('*.png')))==2
    def fail(**kwargs):raise RuntimeError('synthetic generation failure')
    failed=run_picking_variants(config,bundle=bundle,sweep_backend=fail)
    assert len(failed['failures'])==2
    assert not list((tmp_path/'out/generated').rglob('*.png'))
    assert len(list((tmp_path/'out/stale').rglob('*.png')))==4
    refreshed=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    assert refreshed['failures'].empty  # overwrite also retries earlier failed files.


def test_old_mouthless_outputs_are_replaced_and_preserved_in_stale(tmp_path):
    from dataclasses import asdict
    from src.comparisong.picking_variants import PickingVariantsConfig,run_picking_variants
    from src.comparisong.evaluation_backends_v2 import file_hash
    from src.aging_bias.features import stable_hash
    from src.quantitative_metrics.image_io import ImageResolver
    source=tmp_path/'source.png';Image.new('RGB',(16,16)).save(source)
    panel=tmp_path/'panel.csv'
    pd.DataFrame([dict(sample_id='p0',identity_id='id_1',source_path=str(source),target_path=str(source),
        source_age=40,target_age=42)]).to_csv(panel,index=False)
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy adapter')
    config=PickingVariantsConfig(checkpoint,panel,tmp_path/'out',device='cpu',overwrite=False,
        prompt_assistance_config=dict(positive_global_terms=[],negative_global_terms=[],max_support_terms_per_block=4))
    params=asdict(config)
    for key in ['checkpoint_path','test_manifest_path','output_dir','source_root','resume','retry_failed','device','overwrite']:params.pop(key)
    signature=stable_hash(dict(config=params,checkpoint_sha256=file_hash(checkpoint),version='paired-smart-prompts-v2-delta-start'))
    old_key=stable_hash(dict(run=signature,source=ImageResolver().load(source)[1],source_age=40,target_age=42))
    for variant in ['ours_picking_no_prompt','ours_picking_prompt']:
        path=tmp_path/'out/generated'/variant/'id_1/p0.png';path.parent.mkdir(parents=True)
        Image.new('RGB',(16,16),(255,0,0)).save(path)
        path.with_suffix('.json').write_text(json.dumps(dict(signature=old_key,selection=dict(strength=.04))))
    def sweep(**kw):
        from src.inference.prompt_assistance import resolve_prompt_assistance
        resolved=resolve_prompt_assistance(base_target_prompt='portrait',prompt_assistance_config=kw['prompt_assistance_config'],
            source_mouth_state=kw['source_mouth_state'],source_expression_state=kw['source_expression_state'])
        assert 'deformed teeth' in resolved['negative_terms']
        frame=pd.DataFrame([dict(strength=.1)])
        frame.attrs['selected_results']=[{'image':Image.new('RGB',(16,16),(0,255,0))}]
        frame.attrs['assisted_results']=[{'image':Image.new('RGB',(16,16),(0,0,255))}]
        frame.attrs['assisted']=frame.copy()
        return frame
    bundle=dict(unet=object(),identity_encoder=object(),age_estimator=object())
    result=run_picking_variants(config,bundle=bundle,sweep_backend=sweep)
    assert result['failures'].empty
    assert Image.open(result['manifest'].iloc[1].generated_path).getpixel((0,0))==(0,0,255)
    assert len(list((tmp_path/'out/stale').rglob('*.png')))==2
    cached=run_picking_variants(config,bundle=bundle,sweep_backend=lambda **k:pytest.fail('new results not reused'))
    assert cached['failures'].empty


def test_reports_export_all_and_aging_for_every_model(tmp_path):
    from src.aging_bias.reporting import report_paper
    rows=[];ops=[]
    for model in ['SAM','ours_picking_no_prompt','ours_picking_prompt']:
        for subset,value in [('all',.6),('aging',.2)]:
            r=dict(model_name=model,panel='common',subset=subset,scheme='bin_balanced',status='ok')
            for k in ['D','pi','drift','homog']:r.update({k:value,k+'_ci_low':value-.1,k+'_ci_high':value+.1})
            rows.append(r)
            for scheme in ['unweighted','bin_balanced']:
                ops.append(dict(model_name=model,panel='common',subset=subset,scheme=scheme,delta_bin='all',delta_id_gain=value))
    pd.DataFrame(rows).to_csv(tmp_path/'bias_indices.csv',index=False)
    pd.DataFrame(ops).to_csv(tmp_path/'operational_metrics.csv',index=False)
    pd.DataFrame(columns=['model_name','panel','subset','scheme','delta_bin']).to_csv(tmp_path/'bias_indices_by_bin.csv',index=False)
    pd.DataFrame([dict(model_name=m,metric_contaminated__dex_age_mae=False,critic_provenance_unknown=False) for m in ['SAM','ours_picking_no_prompt','ours_picking_prompt']]).to_parquet(tmp_path/'per_sample.parquet')
    pd.DataFrame([dict(model_name=m,subset=s,kid_mean=v) for m in ['SAM','ours_picking_no_prompt','ours_picking_prompt'] for s,v in [('all',.03),('aging',.01)]]).to_csv(tmp_path/'kid_metrics.csv',index=False)
    report_paper(tmp_path)
    for suffix,expected in [('all',.6),('aging',.2)]:
        f=pd.read_csv(tmp_path/f'tables/table1_bias_{suffix}.csv')
        assert len(f)==3 and np.allclose(f.pi,expected)
        assert (tmp_path/f'tables/table2_operational_{suffix}.csv').exists()
        appendix=pd.read_csv(tmp_path/f'tables/tableA1_legacy_{suffix}.csv')
        assert np.allclose(appendix.kid_mean,.03 if suffix=='all' else .01)


def test_kid_is_computed_on_all_and_aging_panels_not_copied(tmp_path):
    from test_evaluator import setup
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    panel=pd.read_csv(cfg.test_manifest_path)
    panel.loc[:9,['source_age','target_age']]=panel.loc[:9,['target_age','source_age']].to_numpy()
    panel.to_csv(cfg.test_manifest_path,index=False)
    class SizedKid:
        def compute(self,real,generated,**kwargs):return float(len(real)),0.
    bundle['kid_metric']=SizedKid()
    evaluate_paper_v2(cfg,bundle=bundle)
    kid=pd.read_csv(cfg.output_dir/'kid_metrics.csv')
    assert set(kid.subset)=={'all','aging'}
    assert kid[kid.subset=='all'].kid_mean.eq(30).all()
    assert kid[kid.subset=='aging'].kid_mean.eq(20).all()
