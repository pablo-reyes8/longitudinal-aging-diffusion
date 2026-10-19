"""Absolute-agreement ICC(2,1); no repeat pairs means unknown, never perfect."""
import numpy as np
import pandas as pd
from itertools import combinations


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


def reliability_table(repeats,min_icc=.5,min_pairs=2,*,min_homog_icc=None):
    if not 0 <= min_icc <= 1 or (min_homog_icc is not None and not 0 <= min_homog_icc <= 1):
        raise ValueError('ICC thresholds must be in [0, 1]')
    if min_pairs < 2:raise ValueError('min_pairs must be >= 2')
    rows=[]
    for name,values in repeats.items():
        a=np.asarray(values,float).reshape(-1,2);n=int(np.isfinite(a).all(1).sum())
        icc=icc_2_1(a) if n>=min_pairs else np.nan
        active=bool(np.isfinite(icc) and icc>=min_icc)
        rows.append(dict(attribute=name,icc=icc,noise_share=1-icc,n_retest_pairs=n,
            active=active,active_pi=active,
            active_homog=active and icc >= (min_icc if min_homog_icc is None else min_homog_icc),
            attribute_weight=float(np.clip(icc,0,1)) if active else 0.,
            exclusion_reason='' if np.isfinite(icc) and icc>=min_icc else ('insufficient_retests' if n<min_pairs else 'low_icc')))
    return pd.DataFrame(rows)


def reference_reliability(frame, attributes, config):
    """Adult-only reliability with an all-age sensitivity report, no fallback.

    Repeated rows/content are deduplicated within identity. Pair counts are
    not independent sample sizes: identity counts are reported alongside them.
    Adult ICC does not establish reliability for children.
    """
    frame=frame.drop_duplicates(['identity_id','image_sha256']).copy()
    def collect(min_age):
        repeats={a:[] for a in attributes};identities={a:set() for a in attributes}
        ages=[]
        eligible=frame if min_age is None else frame[frame.age>=min_age]
        for identity,group in eligible.groupby('identity_id'):
            for a,b in combinations(group.to_dict('records'),2):
                if abs(float(a['age'])-float(b['age']))>1:continue
                limit=config.retest_max_expression_delta
                if limit is not None:
                    keys=[k for k in ['smile_score','jaw_open_score'] if k in frame]
                    if not keys:continue  # Missing expression is not evidence of matching.
                    if any(not np.isfinite([a[k],b[k]]).all() or abs(a[k]-b[k])>limit for k in keys):continue
                ages.extend([float(a['age']),float(b['age'])])
                for name in attributes:
                    values=[a['phi__'+name],b['phi__'+name]]
                    repeats[name].append(values)
                    if np.isfinite(values).all():identities[name].add(identity)
        report=reliability_table(repeats,config.min_icc,config.min_retest_pairs,min_homog_icc=config.min_homog_icc)
        report['n_retest_identities']=[len(identities[a]) for a in report.attribute]
        report['retest_observed_age_min']=min(ages) if ages else np.nan
        report['retest_observed_age_max']=max(ages) if ages else np.nan
        return report
    report=collect(config.retest_min_age)
    all_ages=collect(None).set_index('attribute')
    for column in ['icc','n_retest_pairs','n_retest_identities']:
        report[column+'_all_ages']=report.attribute.map(all_ages[column])
    report['retest_min_age']=config.retest_min_age
    return report
