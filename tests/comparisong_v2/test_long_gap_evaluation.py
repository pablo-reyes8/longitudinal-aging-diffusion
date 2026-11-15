from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.aging_bias.references import ReferenceBundle
from src.comparisong.long_gap_evaluation import LongGapEvaluationConfig, evaluate_long_gap, _structural


def setup_case(tmp_path):
    reference = ReferenceBundle(np.array([18., 80.]), np.array([[0.], [6.2]]),
        np.array([[0.], [3.1]]), np.array([18., 80.]), np.array([[0.], [12.4]]),
        np.array([10.]), np.array([2.]), ['zone_upper'], np.array([.9]), 'frozen', (18.,80.),
        metadata={'reliability_policy': {'retest_min_age':18.,'attribute_weights':[.9]}})
    panel = pd.DataFrame([dict(sample_id='a',identity_id='p',source_age=20.,target_age=60.,source_path='source'),
                          dict(sample_id='b',identity_id='q',source_age=25.,target_age=70.,source_path='source2')])
    root = tmp_path/'generated'
    root.mkdir()
    for sid in panel.sample_id: (root/(sid+'.png')).touch()
    class Store:
        config_hash='frozen'
        def extract(self,path):
            value = 10. if str(path) in ['source','source2'] else 22.
            return {'phi__zone_upper':value,'image_sha256':str(path)}
    cfg=LongGapEvaluationConfig(panel,{'m':root},['m'],tmp_path/'ref',tmp_path/'out',n_boot=10)
    return cfg,reference,Store()


def test_counterfactual_signed_residual_and_noedit(tmp_path):
    cfg,reference,store=setup_case(tmp_path)
    result=evaluate_long_gap(cfg,bundle={},reference_bundle=reference,feature_store=store)
    per=result['per_measure'].set_index(['sample_id','model_name'])
    assert per.loc[('a','m'),'error']==pytest.approx(2.)
    assert per.loc[('a','m'),'ffhq_error']==pytest.approx(-2.)
    assert per.loc[('a','source_noedit'),'error']==pytest.approx(-4.)
    assert per.loc[('a','m'),'latino_cross_error']==pytest.approx(4.)
    assert per.loc[('a','m'),'ffhq_absolute_error']==pytest.approx(2.)
    assert per.loc[('a','m'),'latino_cross_absolute_error']==pytest.approx(4.)
    assert set(result['structural_bias'].status)=={'not_identified'}
    assert result['summary'].n_identities.max()==2
    assert (cfg.output_dir/'failures.csv').is_file()


def test_support_and_common_panel_preserve_missingness(tmp_path):
    cfg,reference,store=setup_case(tmp_path)
    cfg.manifest_path.loc[1,'target_age']=90.
    (cfg.generated_roots['m']/'a.png').unlink()
    result=evaluate_long_gap(cfg,bundle={},reference_bundle=reference,feature_store=store)
    per=result['per_measure']
    assert not per.in_common_panel.any()
    assert per.loc[per.sample_id.eq('b'),'error'].isna().all()
    assert per.loc[per.sample_id.eq('b'),'extrapolated'].all()
    assert len(result['failures'])==1


def test_reject_hash_and_real_target(tmp_path):
    cfg,reference,store=setup_case(tmp_path)
    store.config_hash='changed'
    with pytest.raises(ValueError,match='configuration'):
        evaluate_long_gap(cfg,bundle={},reference_bundle=reference,feature_store=store)
    store.config_hash='frozen'
    cfg.manifest_path['target_path']='real.png'
    with pytest.raises(ValueError,match='target'):
        evaluate_long_gap(cfg,bundle={},reference_bundle=reference,feature_store=store)


def test_operational_age_compares_requested_age_and_source_identity(tmp_path, monkeypatch):
    cfg,reference,store=setup_case(tmp_path)
    calls=[]
    class FakeOperational:
        def __init__(self,*args): pass
        def get(self,path):
            calls.append(str(path))
            source=str(path) in ['source','source2']
            return {'embedding':[1.,0.] if source else [0.,1.], 'independent_age':22. if source else 65.}
    monkeypatch.setattr('src.comparisong.long_gap_evaluation.OperationalStore',FakeOperational)
    bundle=dict(aligner=object(),identity_encoder=object(),independent_age_backend=SimpleNamespace(name='FairFace'))
    result=evaluate_long_gap(cfg,bundle=bundle,reference_bundle=reference,feature_store=store)
    per=result['per_sample'].set_index(['sample_id','model_name'])
    assert per.loc[('a','m'),'age_signed_indep']==5.
    assert per.loc[('b','m'),'age_signed_indep']==-5.
    assert per.loc[('a','m'),'adaface_id_source']==0.
    assert per.loc[('a','source_noedit'),'adaface_id_source']==1.
    assert per.loc[('a','m'),'age_change_indep']==43.
    assert per.loc[('a','m'),'age_change_error_indep']==3.
    assert per.loc[('a','m'),'age_source_calibration_indep']==2.
    assert per.loc[('a','source_noedit'),'age_change_error_indep']==-40.
    assert calls.count('source')==1


def test_identified_structural_fit_and_noedit_are_distinct(tmp_path):
    cfg,reference,_=setup_case(tmp_path)
    reference.attributes=['x','y','z']
    reference.metadata['reliability_policy']['attribute_weights']=[1.,1.,1.]
    rng=np.random.default_rng(4)
    values={key:rng.normal(size=(60,3)) for key in ['d_C','d_P','G_P','zeta']}
    rows=[]
    for model in ['m','source_noedit']:
        for i in range(60):
            row=dict(sample_id=str(i),identity_id=str(i//2),model_name=model)
            for j,name in enumerate(reference.attributes):
                for key,array in values.items(): row[key+'__'+name]=array[i,j]
                row['g__'+name]=.6*values['d_C'][i,j]+.4*values['d_P'][i,j] if model=='m' else 0.
                row['common_valid__'+name]=True
            rows.append(row)
    result=_structural(pd.DataFrame(rows),reference,['m','source_noedit'],cfg).set_index('model_name')
    assert result.loc['m','status']=='ok'
    assert result.loc['m','pi']==pytest.approx(.4)
    assert result.loc['source_noedit','status']=='no_edit'
    assert np.isnan(result.loc['source_noedit','pi'])


def test_quality_covariates_and_missing_measure_reasons(tmp_path):
    cfg,reference,store=setup_case(tmp_path)
    original=store.extract
    def extract(path):
        row=original(path)
        row.update(pose_quality_ok=False,expression_quality_ok=True,is_grayscale=False,yaw_deg=47.)
        if str(path)=='source': row['phi__zone_upper']=np.nan
        return row
    store.extract=extract
    cfg.manifest_path.loc[1,'target_age']=90.
    result=evaluate_long_gap(cfg,bundle={},reference_bundle=reference,feature_store=store)
    per=result['per_sample'].set_index(['sample_id','model_name'])
    assert per.loc[('a','m'),'source_quality__pose_quality_ok']==False
    assert per.loc[('a','m'),'generated_quality__yaw_deg']==47.
    assert per.loc[('a','m'),'n_finite_source_features']==0
    assert per.loc[('a','m'),'n_finite_generated_features']==1
    measure=result['per_measure'].set_index(['sample_id','model_name'])
    assert measure.loc[('a','m'),'missing_reason']=='source_measure_missing'
    assert measure.loc[('b','m'),'missing_reason']=='reference_age_out_of_support'
