"""Internal FFHQ-vs-FFHQ split placebo. Never exported as paper evidence."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .module1_reports import zone_consistency


def placebo_ffhq(bundle,transitions,n_splits,n_boot,seed,output_dir):
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    data=bundle.training_data;ages=data['p_age'];phi=data['p_phi'];rng=np.random.default_rng(seed)
    groups=np.unique(ages);n_attr=len(bundle.attributes);a=np.array([x for x,y in transitions]);b=np.array([y for x,y in transitions])
    summaries=[];cells=[]
    def contrast(means):
        def interp(v):return np.column_stack([np.interp(v,groups,means[:,j]) for j in range(n_attr)])
        return interp(b)-interp(a)
    for split in range(n_splits):
        halves=[[],[]];valid=True
        for age in groups:
            ix=rng.permutation(np.flatnonzero(ages==age));mid=len(ix)//2
            if not mid:valid=False;break
            halves[0].append(phi[ix[:mid]]);halves[1].append(phi[ix[mid:]])
        if not valid:
            summaries.append(dict(split=split,n_cells=0,fraction_significant=np.nan,n_zones_consistent=0,status='insufficient_age_group_support'));continue
        point=contrast(np.array([x.mean(0) for x in halves[1]]))-contrast(np.array([x.mean(0) for x in halves[0]]))
        draws=[]
        for start in range(0,n_boot,32):
            size=min(32,n_boot-start);means=[]
            for half in halves:
                means.append(np.stack([x[rng.integers(0,len(x),(size,len(x)))].mean(1) for x in half],axis=1))
            for t in range(size):draws.append(contrast(means[1][t])-contrast(means[0][t]))
        lo,hi=np.percentile(draws,[2.5,97.5],axis=0) if draws else (np.full_like(point,np.nan),np.full_like(point,np.nan))
        current=[]
        for i,(source,target) in enumerate(transitions):
            for j,attribute in enumerate(bundle.attributes):
                current.append(dict(split=split,attribute=attribute,source_age=source,target_age=target,
                    difference_cross=point[i,j],difference_long=point[i,j],ci_low=lo[i,j],ci_high=hi[i,j],
                    significant=bool(lo[i,j]>0 or hi[i,j]<0)))
        table=pd.DataFrame(current);consistency=zone_consistency(table,bundle.attributes)
        zones=consistency[consistency.attribute.str.startswith('zone_')]
        summaries.append(dict(split=split,n_cells=len(table),fraction_significant=table.significant.mean() if n_boot else np.nan,
            n_zones_consistent=int(zones.status.eq('consistent').sum()),status='ok' if n_boot else 'no_bootstrap'))
        cells.extend(current)
    summary=pd.DataFrame(summaries,columns=['split','n_cells','fraction_significant','n_zones_consistent','status'])
    summary.to_csv(root/'placebo_ffhq.csv',index=False);pd.DataFrame(cells).to_csv(root/'placebo_cells.csv',index=False)
    fraction=float(summary.fraction_significant.mean())
    print(f'6. Placebo interno FFHQ: fracción significativa={fraction:.3f}; zonas consistentes por azar={summary.n_zones_consistent.sum()} (NO paper).',flush=True)
    if fraction>.15:warnings.warn('Internal FFHQ placebo >15% significant cells: human review required; pipeline not stopped')
    return summary
