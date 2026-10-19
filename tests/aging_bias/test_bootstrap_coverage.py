import os
import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('AGING_BIAS_SLOW')!='1',reason='Explicit Monte Carlo: AGING_BIAS_SLOW=1')
def test_T11_identity_bootstrap_coverage_exceeds_pair_bootstrap():
    from src.aging_bias.validation import simulate_toy_aging
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    cluster_cover=pair_cover=0
    for seed in range(100):
        t=simulate_toy_aging(60,8,8,.4,.6,.01,0,seed)
        rng=np.random.default_rng(seed+500)
        # Random slopes shared by identity induce within-person dependence.
        slope=np.repeat(rng.normal(0,.18,60),8)[:,None]
        t['g']+=slope*(t['d_P']-t['d_C']+t['G_P'])
        def interval(frame):
            r=bootstrap_models(frame,{'M':t},WeightingConfig(min_bin_n=1),n_boot=150,seed=seed)
            return r['indices'].query("scheme=='unweighted'").iloc[0]
        a=interval(t['transitions']);incorrect=t['transitions'].copy()
        incorrect['identity_id']=np.arange(len(incorrect)).astype(str)
        b=interval(incorrect)
        cluster_cover+=a.pi_ci_low<=.6<=a.pi_ci_high
        pair_cover+=b.pi_ci_low<=.6<=b.pi_ci_high
    assert cluster_cover>=88 and pair_cover<cluster_cover
