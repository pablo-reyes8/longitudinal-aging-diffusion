"""The integration regression must recover known w through real image IO/cache."""
import numpy as np
import pandas as pd
from PIL import Image
import pytest


def test_I4_toy_prior_recovered_through_feature_extraction(tmp_path):
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.references import ReferenceBundle
    from src.aging_bias.features import FeatureStore
    from src.aging_bias.config import PaperV2Config,WeightingConfig
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    from test_evaluator import Identity,Age
    toy=simulate_toy_aging(60,2,10,.4,.7,0,0,33);p=toy['curve_parameters'];n=len(toy['g'])
    features={}
    class Backend:
        cache_id='toy-exact-phi-v1'
        def extract(self,image):
            a=np.asarray(image)[0,0];code=int(a[0])*256+int(a[1])
            return {'phi__a'+str(j):v for j,v in enumerate(features[code])}
    backend=Backend();rows=[];counter=0
    def write(v):
        nonlocal counter
        counter+=1;features[counter]=v;path=tmp_path/f'image{counter}.png'
        Image.new('RGB',(16,16),(counter//256,counter%256,123)).save(path)
        return path
    root=tmp_path/'generated'
    for i,r in toy['transitions'].iterrows():
        source=write(toy['source_phi'][i]);target=write(toy['target_phi'][i]);gen=write(toy['generated_phi'][i])
        sid='pair'+str(i);dest=root/r.identity_id/(sid+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(gen.read_bytes())
        rows.append({**r.to_dict(),'sample_id':sid,'source_path':str(source),'target_path':str(target)})
    pd.DataFrame(rows).to_csv(tmp_path/'panel.csv',index=False)
    knots=np.sort(np.unique(np.r_[toy['transitions'].source_age,toy['transitions'].target_age]))
    def curve(a,b):return knots[:,None]/30*a+(knots[:,None]/50)**2*b
    c=curve(p['ac'],p['bc']);proxy=curve(p['ap'],p['bp'])+p['gap']
    ref=ReferenceBundle(knots,c,c,knots,proxy,np.zeros(10),np.ones(10),['a'+str(j) for j in range(10)],
        np.ones(10),FeatureStore(tmp_path/'cache',feature_backend=backend).config_hash,(knots.min(),knots.max()))
    cfg=PaperV2Config(tmp_path/'panel.csv',{'ours_picking':root},['ours_picking'],tmp_path/'unused',tmp_path/'out',
        n_boot=4,include_kid=False,include_dex_appendix=False,weighting=WeightingConfig(min_bin_n=1))
    result=evaluate_paper_v2(cfg,bundle=dict(feature_backend=backend,aligner=lambda x:x,
        identity_encoder=Identity(),independent_age_backend=Age()),reference_bundle=ref)
    primary=result['bias_indices'].query("model_name=='ours_picking' and panel=='common' and subset=='aging' and scheme=='unweighted'").iloc[0]
    assert primary.status=='ok'
    assert primary.pi==pytest.approx(.7,abs=.03)
    assert primary.D==pytest.approx(.6,abs=.03)
