"""Absolute-agreement ICC(2,1); no repeat pairs means unknown, never perfect."""
import numpy as np
import pandas as pd


def icc_2_1(measurements):
    x=np.asarray(measurements,float)
    x=x[np.isfinite(x).all(axis=1)]
    if x.ndim!=2 or len(x)<2 or x.shape[1]<2:return np.nan
    n,k=x.shape;grand=x.mean();row=x.mean(1);col=x.mean(0)
    msr=k*np.sum((row-grand)**2)/(n-1)
    msc=n*np.sum((col-grand)**2)/(k-1)
    mse=np.sum((x-row[:,None]-col[None,:]+grand)**2)/((n-1)*(k-1))
    denom=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denom) if denom>0 else np.nan


def reliability_table(repeats,min_icc=.5,min_pairs=2):
    rows=[]
    for name,values in repeats.items():
        a=np.asarray(values,float).reshape(-1,2);n=int(np.isfinite(a).all(1).sum())
        icc=icc_2_1(a) if n>=min_pairs else np.nan
        rows.append(dict(attribute=name,icc=icc,noise_share=1-icc,n_retest_pairs=n,
            active=bool(np.isfinite(icc) and icc>=min_icc),
            exclusion_reason='' if np.isfinite(icc) and icc>=min_icc else ('insufficient_retests' if n<min_pairs else 'low_icc')))
    return pd.DataFrame(rows)
