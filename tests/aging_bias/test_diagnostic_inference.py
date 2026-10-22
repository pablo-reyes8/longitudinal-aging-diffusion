import numpy as np
import pandas as pd
import pytest


def arrays(seed=4):
    rng=np.random.default_rng(seed)
    return {k:rng.normal(size=(80,3)) for k in ['d_C','d_P','G_P','zeta']}


def test_large_feature_change_can_have_zero_population_dose_and_high_r2():
    from src.aging_bias.decomposition import estimate_bias_decomposition
    a=arrays();g=np.tile([1.,2.,3.],(80,1))
    fit=estimate_bias_decomposition(g,**a,weights=np.ones(80))
    assert fit.status=='no_edit'
    assert fit.g_norm_mean==pytest.approx(np.sqrt(14))
    assert fit.r2==pytest.approx(1)
    assert abs(fit.age_component_rms)<1e-10


def test_dose_ci_includes_identified_draws_below_no_edit_threshold():
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    a=arrays();a['g']=-.2*a['d_C']
    panel=pd.DataFrame(dict(identity_id=np.repeat(np.arange(40),2),delta_age=20))
    result=bootstrap_models(panel,{'ours':a,'SAM':a},WeightingConfig(min_bin_n=1),n_boot=10)
    row=result['indices'].iloc[0]
    assert row.status=='no_edit' and row.share_ok==0
    assert row.D_ci_low==pytest.approx(-.2)
    assert row.D_ci_high==pytest.approx(-.2)
    assert np.isnan(row.pi_ci_low)
    assert result['paired'].comparison_status.eq('unstable_bootstrap').all()


def test_oracle_json_contains_complete_rows_and_bootstrap_failure():
    from src.aging_bias.validation import validate_real_anchors
    frame=pd.DataFrame([dict(model_name='oracle_target',scheme='bin_balanced',subset='aging',
        D=.2,pi=0.,drift=0.,homog=1.8,status='ok',D_ci_low=-.1,D_ci_high=1.,
        pi_ci_low=np.nan,pi_ci_high=np.nan,share_ok=.6)])
    report=validate_real_anchors(frame,.35)
    assert report['oracle_rows'][0]['D_ci_low']==-.1
    assert 'oracle_bootstrap_stability' in report['failed_checks']
