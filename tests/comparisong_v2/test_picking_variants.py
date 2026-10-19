import json
import numpy as np
import pandas as pd
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
    cfg=PickingVariantsConfig(checkpoint,panel,tmp_path/'out',device='cpu',target_age_strength_map={50:.27})
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
