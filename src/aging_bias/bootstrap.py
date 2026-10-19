"""Cluster resampling recomputes weights and shares draws across every model."""
from dataclasses import asdict,replace
import warnings
import numpy as np
import pandas as pd
from .decomposition import estimate_bias_decomposition,BiasDecomposition
from .weighting import transition_weights

INDICES=('D','pi','drift','homog')


def calibrate_homogenization(indices,draws):
    """Paired excess over the observed oracle; retain raw values for validation.

    Each bootstrap difference uses the SAME identity draw for model and oracle.
    Unknown/failed oracle identification never becomes a zero correction.
    """
    result=indices.copy()
    for column in ['homog_excess','homog_excess_ci_low','homog_excess_ci_high','homog_oracle','homog_excess_share_ok']:
        result[column]=np.nan
    result['homog_excess_status']='oracle_unavailable'
    for i,row in result.iterrows():
        oracle=result[(result.model_name=='oracle_target')&(result.scheme==row.scheme)]
        if len(oracle)!=1:continue
        oracle=oracle.iloc[0]
        if row.get('status','ok')!='ok' or oracle.get('status','ok')!='ok':continue
        if not np.isfinite([row.homog,oracle.homog]).all():continue
        result.loc[i,'homog_oracle']=oracle.homog
        result.loc[i,'homog_excess']=row.homog-oracle.homog
        result.loc[i,'homog_excess_status']='ok'
        model_draws=draws.get((row.model_name,row.scheme),[])
        oracle_draws=draws.get(('oracle_target',row.scheme),[])
        good=[a.homog-b.homog for a,b in zip(model_draws,oracle_draws)
            if a.status==b.status=='ok' and np.isfinite([a.homog,b.homog]).all()]
        if model_draws:result.loc[i,'homog_excess_share_ok']=len(good)/len(model_draws)
        if good:
            result.loc[i,['homog_excess_ci_low','homog_excess_ci_high']]=np.percentile(good,[2.5,97.5])
    return result


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
    homog_mask=fit_options.pop('homogenization_mask',None)
    def fit_one(arrays,weights,warn=True):
        fitted=estimate_bias_decomposition(**arrays,weights=weights,warn=warn,**fit_options)
        if homog_mask is not None:
            mask=np.asarray(homog_mask,bool)
            if mask.shape!=(arrays['g'].shape[1],):raise ValueError('homogenization_mask shape mismatch')
            strict=fitted if mask.all() else estimate_bias_decomposition(**arrays,weights=weights,warn=warn,
                attribute_mask=np.broadcast_to(mask,arrays['g'].shape),**fit_options)
            fitted.homog=strict.homog;fitted.h=strict.h;fitted.homog_status=strict.status
        return fitted
    for m,a in model_arrays.items():
        for scheme in schemes:
            fits[m,scheme]=fit_one(a,weighted['w_'+scheme].to_numpy())
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
                    draws[m,scheme].append(fit_one(selected,bw['w_'+scheme].to_numpy(),warn=False))
    records=[];paired=[]
    def interval(values):
        values=np.asarray(values,float);values=values[np.isfinite(values)]
        return np.percentile(values,[2.5,97.5]) if len(values) else (np.nan,np.nan)
    for (m,scheme),fit in fits.items():
        row={'model_name':m,'scheme':scheme,**asdict(fit)}
        ok=[v for v in draws[m,scheme] if v.status=='ok']
        row['share_ok']=len(ok)/n_boot if n_boot else np.nan
        if n_boot and fit.status=='ok' and row['share_ok']<.9:
            warnings.warn(f'Bootstrap {m}/{scheme}: share_ok={row["share_ok"]:.2f} < .9')
        for k in INDICES:
            row[k+'_ci_low'],row[k+'_ci_high']=interval([getattr(v,k) for v in ok])
        row['homog_share_ok']=sum(np.isfinite(v.homog) for v in ok)/n_boot if n_boot else np.nan
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
