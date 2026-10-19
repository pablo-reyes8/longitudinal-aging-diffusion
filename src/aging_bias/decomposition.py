"""Structural WLS. Ratios are unconstrained; nonidentification is not success."""
from dataclasses import dataclass
import warnings
import numpy as np


@dataclass
class BiasDecomposition:
    beta_C: float = np.nan
    beta_P: float = np.nan
    gamma: float = np.nan
    h: float = np.nan
    D: float = np.nan
    pi: float = np.nan
    drift: float = np.nan
    homog: float = np.nan
    status: str = 'not_identified'
    n_pairs: int = 0
    n_rows: int = 0
    cond_number: float = np.inf
    corr_dC_dP: float = np.nan
    homog_status: str = 'same_as_structural_fit'


def estimate_bias_decomposition(g, d_C, d_P, G_P, zeta, weights,
                                attribute_mask=None, d_min=.10, max_cond=1e3, max_corr=.95,
                                warn=True,attribute_weights=None):
    arrays=[np.asarray(v,float) for v in (g,d_C,d_P,G_P,zeta)]
    if arrays[0].ndim!=2 or any(a.shape!=arrays[0].shape for a in arrays):
        raise ValueError('All feature arrays must share shape (pairs, attributes)')
    n,p=arrays[0].shape
    aw=np.ones(p) if attribute_weights is None else np.asarray(attribute_weights,float)
    if aw.shape!=(p,) or not np.isfinite(aw).all() or np.any(aw<0) or not np.any(aw>0):
        raise ValueError('attribute_weights must be finite nonnegative attribute weights with positive mass')
    w=np.asarray(weights,float)
    if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('weights must be finite nonnegative pair weights')
    mask=np.logical_and.reduce([np.isfinite(a) for a in arrays]) & (w[:,None]>0) & (aw[None,:]>0)
    if attribute_mask is not None:
        if np.shape(attribute_mask)!=(n,p): raise ValueError('attribute_mask shape mismatch')
        mask &= attribute_mask
    i,j=np.where(mask)
    fit=BiasDecomposition(n_pairs=len(np.unique(i)),n_rows=len(i))
    if not len(i):
        if warn: warnings.warn('Bias decomposition not identified: no valid attributes')
        return fit
    active=np.unique(j)
    fe=(j[:,None]==active).astype(float)
    x=np.column_stack([fe,arrays[1][i,j],arrays[2][i,j],arrays[3][i,j],-arrays[4][i,j]])
    root=np.sqrt(w[i]*aw[j]/w.sum()); xw=x*root[:,None]; yw=arrays[0][i,j]*root
    coef,_,rank,sv=np.linalg.lstsq(xw,yw,rcond=None)
    fit.cond_number=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>0 else np.inf
    a,b=x[:,-4],x[:,-3]
    fit.corr_dC_dP=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan
    if rank<x.shape[1] or fit.cond_number>max_cond or (np.isfinite(fit.corr_dC_dP) and abs(fit.corr_dC_dP)>max_corr):
        if warn: warnings.warn(f'Bias decomposition not identified (rank={rank}/{x.shape[1]}, condition={fit.cond_number:.2g}, correlation={fit.corr_dC_dP:.3g})')
        return fit
    fit.beta_C,fit.beta_P,fit.gamma,fit.h=map(float,coef[-4:])
    fit.D=fit.beta_C+fit.beta_P
    fit.status='no_edit' if fit.D<d_min else 'ok'
    if fit.status=='ok':
        fit.pi=fit.beta_P/fit.D;fit.drift=fit.gamma/fit.D;fit.homog=fit.h/fit.D
    return fit
