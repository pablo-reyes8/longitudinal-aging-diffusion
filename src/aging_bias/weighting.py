"""Explicit horizon masses, with cluster balance and auditable small-bin merging."""
import numpy as np
import pandas as pd
from .config import WeightingConfig


def transition_weights(transitions, config=None):
    config = config or WeightingConfig()
    out = transitions.copy().reset_index(drop=True)
    if out.empty:
        raise ValueError('Cannot weight an empty panel')
    bins = np.asarray(config.bins, float)
    if len(bins)==0 or not np.all(np.diff(bins)>0) or config.min_bin_n<1:
        raise ValueError('Bins must be increasing and min_bin_n >= 1')
    gaps = np.abs(out['delta_age'].to_numpy(float))
    if not np.isfinite(gaps).all() or np.any(gaps<bins[0]):
        raise ValueError('Every nonzero transition must be covered by the bins')
    original = np.searchsorted(bins, gaps, side='right')-1
    assignment = original.copy()
    merges=[]
    while True:
        present, counts = np.unique(assignment, return_counts=True)
        small = [int(b) for b,n in zip(present,counts) if n<config.min_bin_n]
        if not small or len(present)==1:
            break
        b=small[0]; lower=present[present<b]
        neighbor=int(lower[-1] if len(lower) else present[present>b][0])
        assignment[assignment==b]=neighbor
        merges.append({'from':int(b),'into':neighbor})
    # Track original bins, including empty ones, through the same merge map.
    mapping={int(b):int(b) for b in range(len(bins))}
    for change in merges:
        mapping={k:(change['into'] if v==change['from'] else v) for k,v in mapping.items()}
    def label(b):
        members=[i for i,v in mapping.items() if v==b]
        lo=min(members); hi=max(members)+1
        return f'[{bins[lo]:g},{bins[hi]:g})' if hi<len(bins) else f'[{bins[lo]:g},inf)'
    out['delta_bin']= [label(int(b)) for b in assignment]
    out['w_unweighted']=1/len(out)  # Strictly simple mean, by specification W3.
    groups=np.unique(assignment)
    target=config.operational_target
    if target is not None:
        target=np.asarray(target,float)
        if len(target)!=len(bins) or not np.isfinite(target).all() or np.any(target<0) or target.sum()<=0:
            raise ValueError('operational_target must give nonnegative mass for each original bin')
        target=target/target.sum()
        absent=sum(target[i] for i in range(len(bins)) if mapping[i] not in groups)
        if absent>1e-12:
            raise ValueError('Operational target assigns mass to an unsupported bin')
    for scheme in ['bin_balanced']+(['operational'] if target is not None else []):
        weights=np.zeros(len(out))
        for b in groups:
            ix=np.flatnonzero(assignment==b)
            mass=1/len(groups) if scheme=='bin_balanced' else sum(target[i] for i in range(len(bins)) if mapping[i]==b)
            base=np.ones(len(ix))
            if config.identity_balance:
                ids=out.iloc[ix]['identity_id'].astype(str)
                base=1/ids.map(ids.value_counts()).to_numpy(float)
            weights[ix]=mass*base/base.sum()
        out['w_'+scheme]=weights/weights.sum()
    if 'w_'+config.scheme_primary not in out:
        raise ValueError('Requested primary scheme is unavailable')
    out.attrs['merges']=merges
    return out
