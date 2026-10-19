import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image


class Features:
    cache_id='toy-pixel-features-v1'
    def __init__(self):self.calls=0
    def extract(self,image):
        self.calls+=1
        x=np.asarray(image).mean((0,1))/255
        return {'phi__a':x[0],'phi__b':x[1]}


class Identity:
    def embed_batch(self,images):return [np.asarray(x).mean((0,1))+.1 for x in images]


class Age:
    cache_id='test-independent-age-v1'
    def predict_batch(self,images):return [np.asarray(x)[...,0].mean()/3 for x in images]


class Kid:
    def compute(self,*args,**kwargs):return .01,.002


def setup(tmp_path):
    from src.aging_bias.references import ReferenceBundle
    from src.aging_bias.config import PaperV2Config,WeightingConfig
    from src.aging_bias.features import FeatureStore
    backend=Features();store=FeatureStore(tmp_path/'temp',feature_backend=backend)
    ref=ReferenceBundle(np.array([0.,40.,80.]),np.array([[0.,0.],[.4,.2],[.8,.4]]),
        np.array([[.1,.2],[.5,.4],[.9,.6]]),np.array([0.,40.,80.]),
        np.array([[.3,.1],[.4,.6],[.7,.9]]),np.zeros(2),np.ones(2),['a','b'],np.array([.9,.9]),store.config_hash,(0,80))
    ref.save(tmp_path/'ref')
    rows=[];roots={m:tmp_path/m for m in ['SAM','M1','M2']}
    for i in range(30):
        identity='id_'+str(i//2);sample='pair_'+str(i)
        source=tmp_path/f's{i}.png';target=tmp_path/f't{i}.png'
        Image.new('RGB',(16,16),(20+i,80+i,100)).save(source)
        Image.new('RGB',(16,16),(80+i,120+i,100)).save(target)
        for m,root in roots.items():
            p=root/identity/f'{sample}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(16,16),(60+i,110+i,100)).save(p)
        a=5+i;b=45+(i%13)
        rows.append(dict(sample_id=sample,identity_id=identity,source_path=str(source),target_path=str(target),source_age=a,target_age=b,delta_age=b-a))
    manifest=tmp_path/'transitions.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    cfg=PaperV2Config(manifest,roots,list(roots),tmp_path/'ref',tmp_path/'out',n_boot=3,weighting=WeightingConfig(min_bin_n=1))
    bundle=dict(feature_backend=backend,aligner=lambda x:x,identity_encoder=Identity(),
        independent_age_backend=Age(),age_estimator=Age(),kid_metric=Kid(),metadata={'cache_id':'toy-eval-v1'})
    return cfg,bundle,backend


def test_I1_I2_I3_I5_I6_evaluation_resume_and_missing(tmp_path):
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,features=setup(tmp_path)
    for i in [0,1,2]:(tmp_path/'M1'/f'id_{i//2}'/f'pair_{i}.png').unlink()
    result=evaluate_paper_v2(cfg,bundle=bundle)
    assert result['metadata']['n_common_pairs']==27
    assert len(result['failures'].query("stage=='generated_load'"))==3
    for file in ['per_sample.parquet','bias_indices.csv','bias_indices_by_bin.csv','paired_differences.csv',
                 'operational_metrics.csv','failures.csv','run_metadata.json','validation/real_anchors.json','validation/self_checks.json']:
        assert (tmp_path/'out'/file).is_file()
    per=result['per_sample'];assert {'source_noedit','oracle_target'}<=set(per.model_name)
    assert per.query("model_name=='source_noedit'").delta_id_gain.eq(0).all()
    flags=per.query("model_name=='SAM'")['metric_contaminated__dex_age_mae']
    assert flags.all()
    noedit=result['bias_indices'].query("model_name=='source_noedit'")
    assert noedit.status.eq('no_edit').all()
    assert 'oracle_target' not in set(result['bias_indices'].model_name)
    before=features.calls
    evaluate_paper_v2(cfg,bundle=bundle)
    assert features.calls==before
    cfg.independent_age_name='DEX'
    with pytest.raises(ValueError,match='independent|critic'):evaluate_paper_v2(cfg,bundle=bundle)


def test_independent_age_rejects_backend_disguised_as_fairface(tmp_path):
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path)
    bundle['independent_age_backend'].name='MiVOLO_v2'
    with pytest.raises(ValueError,match='independent|critic'):evaluate_paper_v2(cfg,bundle=bundle)


def test_hard_first_source_does_not_abort_valid_rest_of_panel(tmp_path):
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    (tmp_path/'s0.png').unlink()
    result=evaluate_paper_v2(cfg,bundle=bundle)
    assert result['metadata']['n_common_pairs']==29
    assert len(result['failures'].query("stage=='source_load'"))>=1
