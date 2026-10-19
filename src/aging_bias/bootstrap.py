"""Cluster resampling recomputes weights and shares draws across every model."""
from dataclasses import asdict,replace
import warnings
import numpy as np
import pandas as pd
from .decomposition import estimate_bias_decomposition,BiasDecomposition
from .weighting import transition_weights

INDICES=('D','pi','drift','homog')


def cluster_draw(transitions, rng):
    ids=transitions.identity_id.astype(str).to_numpy()
    unique=np.unique(ids)
    draw=rng.choice(unique,len(unique),replace=True)
    indices=[]; renamed=[]
    for slot,identity in enumerate(draw):
        ix=np.flatnonzero(ids==identity)
        indices.extend(ix);renamed.extend([f'draw_{slot}']*len(ix))
    panel=transitions.iloc[indices].copy().reset_index(drop=True)
    panel['identity_id']=renamed
    return np.array(indices,int),panel


def bootstrap_models(transitions, model_arrays, weighting, n_boot=1000, seed=2026, **fit_options):
    if int(n_boot)<0: raise ValueError('n_boot must be nonnegative')
    model_arrays={m:{k:np.asarray(a[k]) for k in ('g','d_C','d_P','G_P','zeta')} for m,a in model_arrays.items()}
    weighted=transition_weights(transitions,weighting)
    schemes=[c[2:] for c in weighted if c.startswith('w_')]
    fits={}; draws={}
    for m,a in model_arrays.items():
        for scheme in schemes:
            fits[m,scheme]=estimate_bias_decomposition(**a,weights=weighted['w_'+scheme].to_numpy(),**fit_options)
            draws[m,scheme]=[]
    rng=np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        ix,panel=cluster_draw(transitions,rng)
        try:
            bw=transition_weights(panel,weighting)
        except ValueError as exc:
            if weighting.operational_target is None:raise
            # An operational bin can disappear in a cluster draw. It cannot
            # acquire fabricated observations; count the draw as unsupported.
            bw=transition_weights(panel,replace(weighting,operational_target=None,scheme_primary='bin_balanced'))
        for m,a in model_arrays.items():
            selected={k:v[ix] for k,v in a.items()}
            for scheme in schemes:
                if 'w_'+scheme not in bw:
                    draws[m,scheme].append(BiasDecomposition())
                else:
                    draws[m,scheme].append(estimate_bias_decomposition(**selected,weights=bw['w_'+scheme].to_numpy(),warn=False,**fit_options))
    records=[];paired=[]
    def interval(values):
        return np.percentile(values,[2.5,97.5]) if len(values) else (np.nan,np.nan)
    for (m,scheme),fit in fits.items():
        row={'model_name':m,'scheme':scheme,**asdict(fit)}
        ok=[v for v in draws[m,scheme] if v.status=='ok']
        row['share_ok']=len(ok)/n_boot if n_boot else np.nan
        if n_boot and fit.status=='ok' and row['share_ok']<.9:
            warnings.warn(f'Bootstrap {m}/{scheme}: share_ok={row["share_ok"]:.2f} < .9')
        for k in INDICES:
            row[k+'_ci_low'],row[k+'_ci_high']=interval([getattr(v,k) for v in ok])
        records.append(row)
    names=list(model_arrays)
    for ai,a in enumerate(names):
        for b in names[ai+1:]:
            for scheme in schemes:
                good=[(u,v) for u,v in zip(draws[a,scheme],draws[b,scheme]) if u.status==v.status=='ok']
                for k in INDICES:
                    lo,hi=interval([getattr(u,k)-getattr(v,k) for u,v in good])
                    paired.append(dict(model_a=a,model_b=b,scheme=scheme,index_name=k,
                        difference=getattr(fits[a,scheme],k)-getattr(fits[b,scheme],k),
                        ci_low=lo,ci_high=hi,share_ok=len(good)/n_boot if n_boot else np.nan))
    return {'indices':pd.DataFrame(records),'paired':pd.DataFrame(paired),'draws':draws,'weights':weighted}
