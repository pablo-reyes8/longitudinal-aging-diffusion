"""Population contrasts, cluster/image bootstrap and identification diagnostics."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from .references import fit_panel_fe,spline_basis
from .bootstrap import cluster_draw


def describe_population_aging(bundle,transitions=((20,50),(30,60),(40,70)),n_boot=1000,seed=2026,output_dir=None):
    a=np.array([t[0] for t in transitions]);b=np.array([t[1] for t in transitions])
    dc=bundle.d_C(a,b);dp=bundle.d_P(a,b)
    cross=bundle._c(b,bundle.mu_c)-bundle._c(a,bundle.mu_c)
    draws=[];rng=np.random.default_rng(seed);data=bundle.training_data
    if n_boot and not data:
        warnings.warn('Reference training arrays unavailable: population bootstrap intervals cannot be computed')
    for _ in range(n_boot if data else 0):
        frame=pd.DataFrame({'identity_id':data['c_ids'],'delta_age':np.ones(len(data['c_ids']))})
        ix,panel=cluster_draw(frame,rng)
        coef,_=fit_panel_fe(data['c_age'][ix],data['c_phi'][ix],panel.identity_id,bundle.knots,data['c_capture'][ix])
        cdraw=(spline_basis(np.clip(b,*bundle.support),bundle.knots)-spline_basis(np.clip(a,*bundle.support),bundle.knots))@coef
        p_mu=[]
        for age in bundle.p_ages:
            values=data['p_phi'][data['p_age']==age]
            p_mu.append(np.nanmean(values[rng.integers(0,len(values),len(values))],axis=0))
        def interp(age):return np.column_stack([np.interp(age,bundle.p_ages,np.array(p_mu)[:,j]) for j in range(len(bundle.attributes))])
        draws.append(interp(b)-interp(a)-cdraw)
    rows=[]
    for i,(source,target) in enumerate(transitions):
        for j,attribute in enumerate(bundle.attributes):
            lo,hi=np.nanpercentile(np.array(draws)[:,i,j],[2.5,97.5]) if draws else (np.nan,np.nan)
            rows.append(dict(attribute=attribute,source_age=source,target_age=target,d_C=dc[i,j],d_P=dp[i,j],
                d_C_cross_sectional=cross[i,j],difference=dp[i,j]-dc[i,j],ci_low=lo,ci_high=hi,
                extrapolated=bool(bundle.extrapolated([source],[target])[0])))
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
    table.to_csv(root/'population_contrasts.csv',index=False)
    (root/'identification.json').write_text(json.dumps(diagnostics,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(bundle.attributes),2,figsize=(10,3*len(bundle.attributes)),squeeze=False)
    age=np.linspace(*bundle.support,100)
    for j,attribute in enumerate(bundle.attributes):
        axes[j,0].plot(age,bundle._c(age,bundle.mu_c)[:,j],label='Colombia cross-sectional')
        axes[j,0].plot(age,bundle._p(age)[:,j],label='FFHQ-Aging proxy')
        axes[j,0].set_title(attribute);axes[j,0].legend()
        sub=table[table.attribute==attribute]
        axes[j,1].bar(range(len(sub)),sub.difference)
        axes[j,1].set_xticks(range(len(sub)),[f'{s}→{t}' for s,t in transitions]);axes[j,1].set_title('d_P − d_C (longitudinal)')
    fig.tight_layout();fig.savefig(root/'population_aging.png');plt.close(fig)
    table.attrs['identification']=diagnostics
    return table
