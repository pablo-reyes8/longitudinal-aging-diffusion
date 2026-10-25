"""Train-frozen facial zone indices; no proxy or test-dependent orientation."""
import numpy as np
import pandas as pd
from .references import fit_panel_fe,spline_basis,CAPTURE

ZONE_DEFINITIONS={
    'zone_upper':['wrinkle_forehead','geom_brow_eye'],
    'zone_mid':['wrinkle_nasolabial','geom_malar_descent','geom_midface_height','skin_hf_texture','wrinkle_periocular'],
    'zone_lower':['geom_jaw_sag','geom_lower_third','wrinkle_perioral'],
}


def longitudinal_mask(frame):
    if 'population_source' in frame:return ~frame.population_source.eq('fairface_latino_hispanic').to_numpy()
    return ~frame.identity_id.astype(str).str.startswith('fairface::').to_numpy()


def fit_zone_transform(train,reliability,knots,definitions=None):
    definitions=ZONE_DEFINITIONS if definitions is None else definitions
    active=reliability[reliability.active].set_index('attribute');out={}
    panel=train.loc[longitudinal_mask(train)]
    for zone,attributes in definitions.items():
        columns=[];means=[];scales=[];signs=[];weights=[];excluded=[]
        for name in attributes:
            key='phi__'+name
            if name not in active.index or key not in train:
                excluded.append(dict(attribute=name,reason='inactive_or_missing'));continue
            mean=float(train[key].mean());scale=float(train[key].std())
            if not np.isfinite(scale) or scale<=1e-12:
                excluded.append(dict(attribute=name,reason='nonvariable'));continue
            q=panel.reindex(columns=CAPTURE,fill_value=0).to_numpy(float)
            coef,_=fit_panel_fe(panel.age,(panel[[key]].to_numpy()-mean)/scale,panel.identity_id,knots,q)
            change=float(((spline_basis([70],knots)-spline_basis([30],knots))@coef)[0,0])
            if not np.isfinite(change) or abs(change)<1e-12:
                excluded.append(dict(attribute=name,reason='unidentified_orientation'));continue
            columns.append(name);means.append(mean);scales.append(scale);signs.append(float(np.sign(change)))
            weights.append(float(active.loc[name,'icc']))
        out[zone]=dict(attributes=columns,means=means,scales=scales,signs=signs,weights=weights,
            excluded=excluded,status='ok' if columns else 'no_active_attributes',orientation='Colombian longitudinal 30->70 only')
    return out


def zone_values(row,transform):
    result={}
    for zone,spec in transform.items():
        values=np.array([row.get('phi__'+a,np.nan) for a in spec['attributes']],float)
        valid=len(values)>0 and np.isfinite(values).all()
        # Never reweight the available subset: every image has the same index.
        result['phi__'+zone]=float(np.average((values-spec['means'])/spec['scales']*spec['signs'],
            weights=spec['weights'])) if valid else np.nan
    return result


def apply_zone_transform(frame,transform):
    result=frame.copy()
    values=[zone_values(row,transform) for row in frame.to_dict('records')]
    for zone in transform:result['phi__'+zone]=[row['phi__'+zone] for row in values]
    return result
