import numpy as np
import pandas as pd
import pytest


def estimate(toy, **kwargs):
    from src.aging_bias.decomposition import estimate_bias_decomposition
    return estimate_bias_decomposition(**{k: toy[k] for k in ('g','d_C','d_P','G_P','zeta')},
                                       weights=np.ones(len(toy['g'])), **kwargs)


def test_T1_T3_T5_recovery_oracle_and_pure_prior():
    from src.aging_bias.validation import simulate_toy_aging
    t=simulate_toy_aging(400, 3, 10, .4, .6, .01, 0, 2026)
    fit=estimate(t)
    assert fit.status=='ok'
    assert fit.pi==pytest.approx(.6, abs=.05)
    assert fit.D==pytest.approx(.6, abs=.05)
    assert fit.homog==pytest.approx(1, abs=.1)
    t['g']=t['d_C']
    oracle=estimate(t)
    assert oracle.D==pytest.approx(1, abs=.05)
    assert max(abs(oracle.pi),abs(oracle.drift),abs(oracle.homog))<.05
    assert estimate(simulate_toy_aging(400,3,10,.4,1,.01,0,2)).pi>.95


def test_T2_strength_invariance():
    from src.aging_bias.validation import simulate_toy_aging
    fits=[estimate(simulate_toy_aging(400,3,10,k,.6,.01,0,3)) for k in [.2,.4,.6,.8]]
    assert np.ptp([x.pi for x in fits])<.05
    assert np.all(np.diff([x.D for x in fits])<0)


def test_T4_T6_noedit_and_rank_guard():
    from src.aging_bias.validation import simulate_toy_aging
    t=simulate_toy_aging(200,2,10,.4,.6,0,0,4)
    t['g']=np.zeros_like(t['g'])
    f=estimate(t)
    assert f.status=='no_edit' and np.isnan(f.pi)
    t['d_P']=t['d_C'].copy()
    with pytest.warns(UserWarning, match='identif'):
        f=estimate(t)
    assert f.status=='not_identified' and np.isnan(f.pi)


def test_T7_T8_T9_T13_noise_placebo_and_missing():
    from src.aging_bias.validation import simulate_toy_aging
    rng=np.random.default_rng(6)
    t=simulate_toy_aging(1000,3,10,.4,0,0,0,5)
    t['g']=t['d_C'].copy()
    estimates=[]
    for _ in range(50):
        estimates.append(estimate({**t,'d_P':rng.permutation(t['d_P'])}).pi)
    assert abs(np.mean(estimates))<.05
    original=estimate(t)
    noisy=estimate({**t,'g':t['g']+rng.normal(0,.1,t['g'].shape)})
    assert abs(noisy.beta_C-original.beta_C)<.03
    # Observed source noise appears with negative sign in both g and -zeta.
    clean=rng.normal(size=t['zeta'].shape); noise=rng.normal(size=clean.shape)*.5
    f=estimate({**t,'zeta':clean+noise,'g':t['d_C']-noise})
    assert f.homog==pytest.approx(.25/1.25, abs=.05)
    t=simulate_toy_aging(600,3,10,.4,.6,.01,0,6)
    t['g'][:int(.4*len(t['g'])),0]=np.nan
    f=estimate(t)
    assert f.pi==pytest.approx(.6, abs=.07)
    assert f.n_rows==t['g'].size-int(.4*len(t['g']))


def test_W1_W7_weighting():
    from src.aging_bias.weighting import transition_weights
    from src.aging_bias.config import WeightingConfig
    df=pd.DataFrame({'identity_id':['a']*10+['b']+['c']*3,
                     'delta_age':[5]*11+[25]*3})
    w=transition_weights(df,WeightingConfig(min_bin_n=1))
    assert w.w_unweighted.to_numpy()==pytest.approx(np.ones(14)/14)
    assert w.w_bin_balanced.sum()==pytest.approx(1)
    assert (w.w_bin_balanced>=0).all()
    assert w.groupby('delta_bin').w_bin_balanced.sum().to_numpy()==pytest.approx([.5,.5])
    assert w.loc[:9,'w_bin_balanced'].sum()==pytest.approx(w.loc[10,'w_bin_balanced'])
    values=np.array([2]*11+[10]*3)
    assert np.dot(w.w_bin_balanced,values)==pytest.approx(6)
    op=transition_weights(df,WeightingConfig(min_bin_n=1, operational_target=[.2,0,.8,0]))
    assert op.groupby('delta_bin').w_operational.sum().to_numpy()==pytest.approx([.2,.8])
    merged=transition_weights(df,WeightingConfig(min_bin_n=5))
    assert merged.delta_bin.nunique()==1 and merged.attrs['merges']


def test_T10_weighted_prior():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.decomposition import estimate_bias_decomposition
    t=simulate_toy_aging(1000,3,10,.4,.2,.01,0,8)
    # Independent regressor distribution across bins: otherwise the pooled
    # heterogeneous-slope WLS estimand need not be a convex average of w.
    rng=np.random.default_rng(19)
    for key in ['d_C','d_P','G_P','zeta']:t[key]=rng.normal(size=t[key].shape)
    t['g']=.6*(.8*t['d_C']+.2*t['d_P']+.2*t['G_P']-t['zeta'])
    long=t['transitions'].delta_age.to_numpy()>=30
    t['g'][long]=.6*(.2*t['d_C'][long]+.8*t['d_P'][long]+.8*t['G_P'][long]-t['zeta'][long])
    f=estimate_bias_decomposition(**{k:t[k][long] for k in ['g','d_C','d_P','G_P','zeta']},weights=np.ones(long.sum()))
    assert f.pi==pytest.approx(.8, abs=.05)
    assert .2<estimate(t).pi<.8


def test_T12_cluster_paired_bootstrap():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    a=simulate_toy_aging(80,2,8,.4,.2,.01,0,7)
    b=simulate_toy_aging(80,2,8,.4,.7,.01,0,7)
    result=bootstrap_models(a['transitions'], {'a':a,'b':b},WeightingConfig(min_bin_n=1),n_boot=30,seed=1)
    diff=result['paired'].query("model_a=='a' and model_b=='b' and index_name=='pi' and scheme=='unweighted'").iloc[0]
    assert diff.ci_high<0
    same=bootstrap_models(a['transitions'],{'a':a,'b':a},WeightingConfig(min_bin_n=1),n_boot=10,seed=2)
    assert same['paired'].query("index_name=='pi'").ci_low.abs().max()<1e-12


def test_T9_noise_widens_cluster_intervals():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    t=simulate_toy_aging(70,3,8,.4,.6,0,0,42)
    rng=np.random.default_rng(43)
    noisy={**t,'g':t['g']+rng.normal(0,.15,t['g'].shape)}
    r=bootstrap_models(t['transitions'],{'clean':t,'noisy':noisy},WeightingConfig(min_bin_n=1),n_boot=20,seed=44)
    f=r['indices'].query("scheme=='unweighted'").set_index('model_name')
    assert f.loc['noisy','pi_ci_high']-f.loc['noisy','pi_ci_low']>f.loc['clean','pi_ci_high']-f.loc['clean','pi_ci_low']+1e-4


def test_T12_equal_prior_independent_noise_difference_covers_zero_in_majority():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    covered=0
    for seed in range(5):
        t=simulate_toy_aging(50,2,8,.4,.6,0,0,30)
        rng=np.random.default_rng(seed)
        a={**t,'g':t['g']+rng.normal(0,.03,t['g'].shape)}
        b={**t,'g':t['g']+rng.normal(0,.03,t['g'].shape)}
        r=bootstrap_models(t['transitions'],{'a':a,'b':b},WeightingConfig(min_bin_n=1),n_boot=20,seed=seed)
        row=r['paired'].query("scheme=='unweighted' and index_name=='pi'").iloc[0]
        covered+=row.ci_low<=0<=row.ci_high
    assert covered>=3


def test_self_checks(tmp_path):
    from src.aging_bias import run_self_checks
    r=run_self_checks(output_dir=tmp_path)
    assert r.passed.all() and len(r)>=7
    assert (tmp_path/'self_checks.json').exists()


def test_operational_missing_bootstrap_bin_does_not_invalidate_unweighted():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    t=simulate_toy_aging(20,1,10,.4,.6,0,0,55)
    t['transitions']['delta_age']=[5]*19+[25]
    with pytest.warns(UserWarning,match='share_ok'):
        result=bootstrap_models(t['transitions'],{'M':t},WeightingConfig(min_bin_n=1,operational_target=(.5,0,.5,0)),n_boot=30,seed=56)
    rows=result['indices'].set_index('scheme')
    assert rows.loc['unweighted','share_ok']==1
    assert rows.loc['operational','share_ok']<.9
