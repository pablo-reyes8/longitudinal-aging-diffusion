"""Reproducible identifiable simulations and real-anchor diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .decomposition import estimate_bias_decomposition
from .weighting import transition_weights
from .config import WeightingConfig


def simulate_toy_aging(n_ids=400,pairs_per_id=3,p=10,k=.4,w=.6,noise_g=.01,noise_src=0,seed=2026):
    rng=np.random.default_rng(seed);n=n_ids*pairs_per_id
    ids=np.repeat(np.arange(n_ids),pairs_per_id)
    source=rng.uniform(10,45,n);delta=rng.uniform(1,40,n);target=source+delta
    # Different nonlinear trajectories by attribute; no artificial collinear
    # population columns. Individual deviations include a shared cluster term.
    ac=rng.normal(0,.4,p);bc=rng.normal(0,.6,p)
    ap=rng.normal(0,.4,p);bp=rng.normal(0,.6,p);gap=rng.normal(0,1,p)
    def curve(age,a,b): return age[:,None]/30*a+(age[:,None]/50)**2*b
    cs=curve(source,ac,bc);ct=curve(target,ac,bc)
    ps=curve(source,ap,bp)+gap;pt=curve(target,ap,bp)+gap
    z=rng.normal(size=(n_ids,p))[ids]+rng.normal(0,.3,(n,p))
    eps=rng.normal(0,noise_src,(n,p))
    g=(1-k)*((1-w)*(ct-cs)+w*(pt-ps)+w*(ps-cs)-z)
    g+=rng.normal(0,noise_g,(n,p))-eps
    transitions=pd.DataFrame({'identity_id':['id_'+str(i) for i in ids],
        'source_age':source,'target_age':target,'delta_age':delta})
    return dict(g=g,d_C=ct-cs,d_P=pt-ps,G_P=ps-cs,zeta=z+eps,transitions=transitions,
        source_phi=cs+z+eps,target_phi=ct+z,generated_phi=cs+z+eps+g,
        curve_parameters=dict(ac=ac,bc=bc,ap=ap,bp=bp,gap=gap))


def run_self_checks(output_dir=None):
    t=simulate_toy_aging(n_ids=180)
    def fit(toy):
        return estimate_bias_decomposition(**{k:toy[k] for k in ('g','d_C','d_P','G_P','zeta')},weights=np.ones(len(toy['g'])),warn=False)
    rows=[]
    def add(name,expected,observed,passed): rows.append(dict(check=name,expected=expected,observed=observed,passed=bool(passed)))
    f=fit(t);add('T1','pi=.6; D=.6',[f.pi,f.D],abs(f.pi-.6)<.05 and abs(f.D-.6)<.05)
    fs=[fit(simulate_toy_aging(n_ids=180,k=k)) for k in [.2,.4,.6,.8]]
    add('T2','strength-invariant pi',[x.pi for x in fs],np.ptp([x.pi for x in fs])<.05 and np.all(np.diff([x.D for x in fs])<0))
    f=fit({**t,'g':t['d_C']});add('T3','oracle D=1, ratios=0',[f.D,f.pi,f.drift,f.homog],abs(f.D-1)<.05 and max(abs(f.pi),abs(f.drift),abs(f.homog))<.05)
    f=fit({**t,'g':np.zeros_like(t['g'])});add('T4','no_edit',f.status,f.status=='no_edit' and np.isnan(f.pi))
    f=fit({**t,'d_P':t['d_C']});add('T6','not_identified',f.status,f.status=='not_identified')
    rng=np.random.default_rng(12);z=rng.normal(size=(5000,10));e=rng.normal(0,.5,z.shape)
    tt=simulate_toy_aging(1000,5,10)
    f=fit({**tt,'zeta':z+e,'g':tt['d_C']-e})
    add('T8','homog=.2',f.homog,abs(f.homog-.2)<.05)
    weights=transition_weights(t['transitions'],WeightingConfig(min_bin_n=1))
    mass=weights.groupby('delta_bin').w_bin_balanced.sum().to_numpy()
    add('W2','equal bin mass',mass.tolist(),np.ptp(mass)<1e-12)
    result=pd.DataFrame(rows)
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        (path/'self_checks.json').write_text(result.to_json(orient='records',indent=2))
    return result


def validate_real_anchors(indices, noise_share, d_bounds=(.7,1.3), ratio_tolerance=.15, homog_tolerance=.15):
    records=[]
    for _,r in indices.query("scheme=='bin_balanced' and subset=='aging'").iterrows():
        if r.model_name not in {'oracle_target','source_noedit'}:continue
        if r.model_name=='source_noedit':
            records.append(dict(check='source_noedit',observed=r.status,passed=r.status=='no_edit'))
        else:
            for name,passed in [('D',d_bounds[0]<=r.D<=d_bounds[1]),
                ('pi',abs(r.pi)<ratio_tolerance or r.pi_ci_low<=0<=r.pi_ci_high),
                ('drift',abs(r.drift)<ratio_tolerance),('homog',abs(r.homog-noise_share)<homog_tolerance)]:
                records.append(dict(check='oracle_'+name,observed=float(r[name]),passed=bool(passed)))
    return dict(passed=bool(records) and all(r['passed'] for r in records),checks=records,
        expected_noise_share=float(noise_share),
        note='If anchors fail, inspect ICC, alignment, resolution harmonization, reference support and identification before interpreting model ratios.')
