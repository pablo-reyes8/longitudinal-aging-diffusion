"""Population contrasts, cluster/image bootstrap and identification diagnostics."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from .references import fit_panel_fe,spline_basis
from .bootstrap import cluster_draw


def mean_faces(*args,**kwargs):
    """Descriptive mean faces/HF maps only. Statistical inference is in tables."""
    from .module1_reports import mean_faces as run
    return run(*args,**kwargs)


def capture_sensitivity(bundle,transitions=((20,50),(30,60),(40,70))):
    """Within-person contrast with/without capture controls; not an APC solution."""
    data=bundle.training_data
    if not data:return pd.DataFrame()
    unadjusted=data.get('f_coeff_unadjusted')
    if unadjusted is None:
        unadjusted,_=fit_panel_fe(data['c_age'],data['c_phi'],data['c_ids'],bundle.knots)
    source=np.array([a for a,b in transitions]);target=np.array([b for a,b in transitions])
    delta=spline_basis(np.clip(target,*bundle.support),bundle.knots)-spline_basis(np.clip(source,*bundle.support),bundle.knots)
    adjusted=delta@bundle.f_coeff;raw=delta@unadjusted
    rows=[]
    for i,(a,b) in enumerate(transitions):
        for j,name in enumerate(bundle.attributes):
            valid=np.isfinite([adjusted[i,j],raw[i,j]]).all()
            rows.append(dict(attribute=name,source_age=a,target_age=b,d_C_adjusted=adjusted[i,j],
                d_C_unadjusted=raw[i,j],capture_adjustment=adjusted[i,j]-raw[i,j],
                sign_changed=bool(valid and adjusted[i,j]*raw[i,j]<0),
                extrapolated=bool(bundle.extrapolated([a],[b])[0]),
                interpretation='Capture sensitivity, not proof of an age/period causal effect'))
    return pd.DataFrame(rows)


def describe_population_aging(bundle,transitions=((20,50),(30,60),(40,70)),n_boot=1000,seed=2026,output_dir=None,*,include_cross_ci=False):
    a=np.array([t[0] for t in transitions]);b=np.array([t[1] for t in transitions])
    dc=bundle.d_C(a,b);dp=bundle.d_P(a,b)
    cross=bundle._c(b,bundle.mu_c)-bundle._c(a,bundle.mu_c)
    draws=[];cross_draws=[];rng=np.random.default_rng(seed);data=bundle.training_data
    if n_boot and not data:
        warnings.warn('Reference training arrays unavailable: population bootstrap intervals cannot be computed')
    clusters=None
    if data and include_cross_ci:
        # Thousands of FairFace singleton IDs: don't scan every photo for every
        # identity in every replicate. The resampling unit remains identity.
        _,inverse=np.unique(data['c_ids'],return_inverse=True)
        order=np.argsort(inverse,kind='stable')
        clusters=np.split(order,np.cumsum(np.bincount(inverse))[:-1])
    for iteration in range(n_boot if data else 0):
        if clusters is None:
            frame=pd.DataFrame({'identity_id':data['c_ids'],'delta_age':np.ones(len(data['c_ids']))})
            ix,panel=cluster_draw(frame,rng);draw_ids=panel.identity_id.to_numpy()
        else:
            groups=[clusters[j] for j in rng.integers(0,len(clusters),len(clusters))]
            ix=np.concatenate(groups);draw_ids=np.repeat(np.arange(len(groups)),[len(g) for g in groups])
        keep=data.get('c_longitudinal',np.ones(len(data['c_age']),bool))[ix]
        coef,_=fit_panel_fe(data['c_age'][ix][keep],data['c_phi'][ix][keep],draw_ids[keep],bundle.knots,data['c_capture'][ix][keep])
        cdraw=(spline_basis(np.clip(b,*bundle.support),bundle.knots)-spline_basis(np.clip(a,*bundle.support),bundle.knots))@coef
        p_mu=[]
        for age in bundle.p_ages:
            values=data['p_phi'][data['p_age']==age]
            p_mu.append(np.nanmean(values[rng.integers(0,len(values),len(values))],axis=0))
        def interp(age):return np.column_stack([np.interp(age,bundle.p_ages,np.array(p_mu)[:,j]) for j in range(len(bundle.attributes))])
        draws.append(interp(b)-interp(a)-cdraw)
        if include_cross_ci:
            mu=np.full_like(bundle.mu_c,np.nan)
            values=data['c_phi'][ix];ages=data['c_age'][ix]
            for j in range(len(bundle.attributes)):
                valid=np.isfinite(values[:,j])&np.isfinite(data['c_capture'][ix]).all(1)
                if valid.any():mu[:,j]=np.linalg.lstsq(spline_basis(ages[valid],bundle.knots),values[valid,j],rcond=None)[0]
            cross_draws.append(interp(b)-interp(a)-(spline_basis(np.clip(b,*bundle.support),bundle.knots)-spline_basis(np.clip(a,*bundle.support),bundle.knots))@mu)
            if (iteration+1)%100==0:print(f'[module1 population bootstrap] {iteration+1}/{n_boot}',flush=True)
    rows=[]
    for i,(source,target) in enumerate(transitions):
        for j,attribute in enumerate(bundle.attributes):
            lo,hi=np.nanpercentile(np.array(draws)[:,i,j],[2.5,97.5]) if draws else (np.nan,np.nan)
            rows.append(dict(attribute=attribute,source_age=source,target_age=target,d_C=dc[i,j],d_P=dp[i,j],
                d_C_cross_sectional=cross[i,j],difference=dp[i,j]-dc[i,j],ci_low=lo,ci_high=hi,
                reliability_age_supported=min(source,target)>= (bundle.metadata.get('reliability_policy',{}).get('retest_min_age') or 0),
                extrapolated=bool(bundle.extrapolated([source],[target])[0])))
            if include_cross_ci:
                low,high=np.nanpercentile(np.array(cross_draws)[:,i,j],[2.5,97.5]) if cross_draws else (np.nan,np.nan)
                rows[-1].update(difference_long=rows[-1]['difference'],difference_cross=dp[i,j]-cross[i,j],
                    ci_low_long=lo,ci_high_long=hi,ci_low_cross=low,ci_high_cross=high)
    corr=float(np.corrcoef(dc.ravel(),dp.ravel())[0,1])
    # Use actual training source deviations where available, rather than
    # computing an artificially identifiable design with invented zeta.
    if data:
        src=data['c_age'];tgt=np.minimum(src+30,bundle.support[1]);n,p=data['c_phi'].shape
        columns=[bundle.d_C(src,tgt),bundle.d_P(src,tgt),bundle.G_P(src),-bundle.zeta(data['c_phi'],src)]
        x=np.column_stack([np.tile(np.eye(p),(n,1)),*[v.ravel() for v in columns]])
        x=x[np.isfinite(x).all(1)]
        condition=float(np.linalg.cond(x)) if len(x) else np.inf
    else:condition=np.inf
    weak=abs(corr)>.95 or condition>1e3
    if weak:warnings.warn(f'Weak population identification: correlation={corr:.3f}, condition={condition:.3g}; inspect the actual evaluation design too')
    diagnostics=dict(corr_dC_dP=corr,condition_number=condition,weak_identification=weak,
        note='P is cross-sectional; inspect d_C_cross_sectional alongside longitudinal d_C. APC confounding remains.')
    root=Path(output_dir or bundle.metadata.get('config',{}).get('output_dir','outputs/aging_bias/references'))
    root.mkdir(parents=True,exist_ok=True);table=pd.DataFrame(rows)
    sensitivity=capture_sensitivity(bundle,transitions)
    if not sensitivity.empty:
        sensitivity.to_csv(root/'capture_sensitivity.csv',index=False)
        table=table.merge(sensitivity[['attribute','source_age','target_age','d_C_unadjusted','capture_adjustment','sign_changed']],
            on=['attribute','source_age','target_age'],validate='one_to_one')
        if sensitivity.sign_changed.any():
            warnings.warn('Longitudinal contrast changes sign with capture adjustment; inspect capture_sensitivity.csv. APC confounding is not resolved.')
    table.to_csv(root/'population_contrasts.csv',index=False)
    (root/'identification.json').write_text(json.dumps(diagnostics,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(bundle.attributes),2,figsize=(10,3*len(bundle.attributes)),squeeze=False)
    age=np.linspace(*bundle.support,100)
    for j,attribute in enumerate(bundle.attributes):
        axes[j,0].plot(age,bundle._c(age,bundle.mu_c)[:,j],label='latam cross-sectional' if bundle.metadata.get('module1_close') else 'Colombia cross-sectional')
        axes[j,0].plot(age,bundle._p(age)[:,j],label='FFHQ-Aging proxy')
        axes[j,0].set_title(attribute);axes[j,0].legend()
        sub=table[table.attribute==attribute]
        axes[j,1].bar(range(len(sub)),sub.difference)
        axes[j,1].set_xticks(range(len(sub)),[f'{s}→{t}' for s,t in transitions]);axes[j,1].set_title('d_P − d_C (longitudinal)')
    fig.tight_layout();fig.savefig(root/'population_aging.png');plt.close(fig)
    table.attrs['identification']=diagnostics
    return table
