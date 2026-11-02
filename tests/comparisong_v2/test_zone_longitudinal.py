from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


def fixture():
    from src.aging_bias.references import ReferenceBundle
    from src.aging_bias.config import PaperV2Config,WeightingConfig
    names=['zone_upper','zone_mid','zone_lower','wrinkle_forehead','wrinkle_nasolabial','geom_jaw_sag']
    shape=(3,len(names))
    ref=ReferenceBundle(np.array([0.,40.,80.]),np.tile(np.arange(3)[:,None],(1,len(names))),
        np.zeros(shape),np.array([0.,40.,80.]),np.ones(shape),np.zeros(len(names)),np.ones(len(names)),names,
        np.ones(len(names))*.8,'fake',(0,80),metadata={'module1_close':True,'attribute_orientations':dict.fromkeys(names,1),
        'reliability_policy':{'retest_min_age':30}})
    cfg=PaperV2Config('manifest',{},['SAM','ours@0.3','SD15_vanilla@0.3'],'ref','out',n_boot=50,
        weighting=WeightingConfig(min_bin_n=1))
    rows=[]
    for i in range(12):
        for model in cfg.models+['source_noedit','oracle_target']:
            error={'SAM':[-1.,-2.,3.,-1.,-2.,3.],'ours@0.3':[0.]*6,
                'SD15_vanilla@0.3':[-2.,-2.,4.,-2.,-2.,4.],'source_noedit':[-2.]*6,'oracle_target':[0.]*6}[model]
            row=dict(sample_id=f'p{i}',identity_id=f'id_{i//2}',model_name=model,source_age=35,target_age=65,
                delta_age=30,reliability_age_supported=True)
            for j,name in enumerate(names):
                row['phi_source__'+name]=1.;row['phi_target__'+name]=3.;row['phi_generated__'+name]=3.+error[j]
            rows.append(row)
    return pd.DataFrame(rows),cfg,ref


def test_exact_observed_error_ffhq_pattern_and_oracle():
    from src.comparisong.zone_evaluation import longitudinal_errors
    per,cfg,ref=fixture();errors=longitudinal_errors(per,ref,cfg.models)
    sam=errors[(errors.model_name=='SAM')&(errors.sample_id=='p0')].set_index('measure')
    assert sam.loc['zone_upper','error']==-1
    assert sam.loc['zone_lower','error']==3
    assert sam.loc['zone_upper','model_change']==1 and sam.loc['zone_upper','real_change']==2
    assert errors[errors.model_name.eq('oracle_target')].error.eq(0).all()
    assert errors[errors.model_name.eq('source_noedit')].model_change.eq(0).all()
    assert np.allclose(errors.error,errors.model_change-errors.real_change)


def test_train_frozen_orientation_and_missing_zone_excluded_from_common_only():
    from src.comparisong.zone_evaluation import longitudinal_errors
    per,cfg,ref=fixture();ref.metadata['attribute_orientations']['wrinkle_forehead']=-1
    per.loc[(per.model_name=='SAM')&(per.sample_id=='p0'),'phi_generated__zone_mid']=np.nan
    errors=longitudinal_errors(per,ref,cfg.models)
    assert errors.query("model_name=='SAM' and measure=='wrinkle_forehead'").error.eq(1).all()
    assert not errors.query("sample_id=='p0' and measure=='zone_mid'").in_common_panel.any()
    assert errors.query("sample_id=='p0' and measure=='zone_upper'").in_common_panel.all()
    assert errors.query("sample_id=='p0' and measure=='zone_mid' and model_name=='SAM'").error.isna().all()


def test_paired_zone_bootstrap_and_sd15_same_strength(tmp_path):
    from src.comparisong.zone_evaluation import evaluate_zone_errors
    per,cfg,ref=fixture();cfg.output_dir=tmp_path
    result=evaluate_zone_errors(per,cfg,ref)
    summary=result['zone_summary']
    main=summary.query("panel=='common' and subset=='aging' and age_support=='supported' and measure=='zone_lower'")
    assert main.set_index('model_name').loc['SAM','signed_error_mean']==3
    assert main.set_index('model_name').loc['ours@0.3','mae']==0
    pairs=result['sd15_paired']
    row=pairs.query("measure=='zone_lower' and subset=='aging' and age_support=='supported'").iloc[0]
    assert row.mae_difference_after_minus_before==-4
    assert row.mae_ci_low==pytest.approx(-4) and row.mae_ci_high==pytest.approx(-4)
    assert row.strength==.3
    assert (tmp_path/'tables/zone_errors_aging.csv').is_file()
    assert (tmp_path/'zone_errors_per_transition.csv').is_file()


def test_cluster_weights_match_existing_weighting_after_duplicated_id_draw():
    from src.comparisong.zone_evaluation import cluster_weights
    from src.aging_bias.weighting import transition_weights
    from src.aging_bias.config import WeightingConfig
    base=pd.DataFrame(dict(identity_id=['a','a','b','c'],delta_age=[5,25,5,35]))
    for balance in [True,False]:
        cfg=WeightingConfig(min_bin_n=2,identity_balance=balance)
        # Draw a,a,c: each a has two transitions, c one.
        sampled=base.iloc[[0,1,0,1,3]].copy().reset_index(drop=True)
        sampled['identity_id']=['draw0','draw0','draw1','draw1','draw2']
        expected=transition_weights(sampled,cfg).w_bin_balanced.to_numpy()
        collapsed=np.array([expected[[0,2]].sum(),expected[[1,3]].sum(),0.,expected[4]])
        actual=cluster_weights(base,np.array([2,2,0,1]),cfg)
        np.testing.assert_allclose(actual,collapsed)


def test_unreliable_age_rows_not_silently_counted_in_primary_and_long_bins_stay_long(tmp_path):
    from src.comparisong.zone_evaluation import evaluate_zone_errors
    per,cfg,ref=fixture();cfg.output_dir=tmp_path;cfg.n_boot=0
    per.loc[per.sample_id.eq('p0'),'reliability_age_supported']=False
    result=evaluate_zone_errors(per,cfg,ref)
    rows=result['zone_summary'].query("measure=='zone_upper' and model_name=='SAM' and panel=='common' and subset=='long_aging'")
    assert rows.set_index('age_support').loc['supported','n_pairs']==11
    assert rows.set_index('age_support').loc['all_ages_exploratory','n_pairs']==12
    assert rows.mae_ci_low.isna().all()


@pytest.mark.parametrize('scheme',['unweighted','bin_balanced','operational'])
def test_compressed_bootstrap_weights_match_actual_cluster_slots_across_merges(scheme):
    from src.comparisong.zone_evaluation import cluster_weights
    from src.aging_bias.weighting import transition_weights
    from src.aging_bias.config import WeightingConfig
    base=pd.DataFrame(dict(identity_id=['a','a','b','b','c','d','d'],delta_age=[2,25,15,32,22,4,45]))
    rng=np.random.default_rng(5)
    for _ in range(12):
        draw=rng.choice(['a','b','c','d'],4);indices=[];names=[]
        for slot,person in enumerate(draw):
            ix=np.flatnonzero(base.identity_id.eq(person));indices.extend(ix);names.extend([str(slot)]*len(ix))
        sampled=base.iloc[indices].copy().reset_index(drop=True);sampled.identity_id=names
        cfg=WeightingConfig(scheme_primary=scheme,min_bin_n=3,operational_target=(.25,.25,.25,.25) if scheme=='operational' else None)
        try:weights=transition_weights(sampled,cfg)['w_'+scheme].to_numpy()
        except ValueError:
            with pytest.raises(ValueError):cluster_weights(base,np.bincount(indices,minlength=len(base)),cfg)
            continue
        expected=np.zeros(len(base));np.add.at(expected,indices,weights)
        np.testing.assert_allclose(cluster_weights(base,np.bincount(indices,minlength=len(base)),cfg),expected,atol=1e-12)
