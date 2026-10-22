"""Train-only longitudinal and cross-sectional references.

Within identity, age and capture period are confounded. Capture covariates
mitigate but do not solve age-period-cohort identification. Sharing d_C across
models does not make conclusions immune to reference misspecification.
"""
from dataclasses import dataclass,field,asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from .features import FeatureStore
from .reliability import reference_reliability

CAPTURE=('is_grayscale','log_min_side','blur_laplacian_var')


def spline_basis(ages,knots):
    knots=np.asarray(knots,float);age=np.asarray(ages,float).reshape(-1)
    if len(knots)<2 or not np.all(np.diff(knots)>0):raise ValueError('Need increasing spline knots')
    age=np.clip(age,knots[0],knots[-1]);j=np.clip(np.searchsorted(knots,age,side='right')-1,0,len(knots)-2)
    t=(age-knots[j])/(knots[j+1]-knots[j]);x=np.zeros((len(age),len(knots)))
    x[np.arange(len(age)),j]=1-t;x[np.arange(len(age)),j+1]=t
    return x


def fit_panel_fe(ages,values,identities,knots,capture=None):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    ids=np.asarray(identities);basis=spline_basis(ages,knots)
    q=np.zeros((len(ids),0)) if capture is None else np.asarray(capture,float)
    coef=np.full((len(knots),values.shape[1]),np.nan);capture_coef=np.full((q.shape[1],values.shape[1]),np.nan)
    for j in range(values.shape[1]):
        valid=np.isfinite(values[:,j])&np.isfinite(q).all(1)
        if valid.sum()<3:continue
        x=np.column_stack([basis[valid,1:],q[valid]]);y=values[valid,j].copy();groups=ids[valid]
        # Demean columns AND outcomes within each person.
        _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
        sums=np.zeros((len(counts),x.shape[1]+1))
        np.add.at(sums,inverse,np.column_stack([x,y]))
        means=sums/counts[:,None]
        x-=means[inverse,:-1];y-=means[inverse,-1]
        fitted=np.linalg.lstsq(x,y,rcond=None)[0]
        coef[0,j]=0;coef[1:,j]=fitted[:len(knots)-1];capture_coef[:,j]=fitted[len(knots)-1:]
    return coef,capture_coef


@dataclass
class ReferenceBundle:
    knots: np.ndarray
    f_coeff: np.ndarray
    mu_c: np.ndarray
    p_ages: np.ndarray
    mu_p: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    attributes: list
    icc: np.ndarray
    feature_config_hash: str
    support: tuple
    metadata: dict=field(default_factory=dict)
    training_data: dict=field(default_factory=dict,repr=False)
    strata: dict=field(default_factory=dict,repr=False)

    def standardize(self,phi): return (np.asarray(phi,float)-self.mean)/self.sd
    def _c(self,age,curve):return spline_basis(np.clip(np.asarray(age,float),*self.support),self.knots)@curve
    def _p(self,age):
        a=np.asarray(age,float).reshape(-1)
        return np.column_stack([np.interp(a,self.p_ages,self.mu_p[:,j]) for j in range(len(self.attributes))])
    def _stratified(self,name,args,sex):
        result=getattr(self,name)(*args)
        labels=np.broadcast_to(np.asarray(sex).reshape(-1),(len(result),))
        for label,ref in self.strata.items():
            ix=labels==label
            if ix.any():result[ix]=getattr(ref,name)(*[np.broadcast_to(np.asarray(v), (len(result),)+np.shape(v)[1:])[ix] for v in args])
        return result
    def d_C(self,a_s,a_t,sex=None):
        if sex is not None and self.strata:return self._stratified('d_C',(a_s,a_t),sex)
        return self._c(a_t,self.f_coeff)-self._c(a_s,self.f_coeff)
    def d_P(self,a_s,a_t,sex=None):
        if sex is not None and self.strata:return self._stratified('d_P',(a_s,a_t),sex)
        return self._p(a_t)-self._p(a_s)
    def G_P(self,a_s,sex=None):
        if sex is not None and self.strata:return self._stratified('G_P',(a_s,),sex)
        return self._p(a_s)-self._c(a_s,self.mu_c)
    def zeta(self,phi_s,a_s,sex=None):
        """phi_s must already be in the stored train-standardized coordinates."""
        if sex is not None and self.strata:return self._stratified('zeta',(phi_s,a_s),sex)
        return np.asarray(phi_s)-self._c(a_s,self.mu_c)
    def extrapolated(self,a_s,a_t,sex=None):
        if sex is not None and self.strata:return self._stratified('extrapolated',(a_s,a_t),sex)
        lo=max(self.support[0],self.p_ages[0],self.knots[0]);hi=min(self.support[1],self.p_ages[-1],self.knots[-1])
        a,b=np.broadcast_arrays(np.atleast_1d(a_s),np.atleast_1d(a_t))
        return (a<lo)|(b<lo)|(a>hi)|(b>hi)

    def save(self,destination):
        root=Path(destination);root.mkdir(parents=True,exist_ok=True)
        arrays={k:getattr(self,k) for k in ['knots','f_coeff','mu_c','p_ages','mu_p','mean','sd','icc']}
        for key,value in self.training_data.items():
            value=np.asarray(value)
            if key=='c_ids':
                if not all(isinstance(v,(str,np.str_)) for v in value.ravel()):
                    raise ValueError('Training identity IDs must be strings')
                value=value.astype(str)
            if value.dtype.hasobject:raise ValueError(f'Training array {key} cannot contain Python objects')
            arrays['training__'+key]=value
        np.savez_compressed(root/'references.npz',**arrays)
        meta=dict(version='aging-bias-reference-v1',created_at=datetime.now(timezone.utc).isoformat(),
            attributes=self.attributes,feature_config_hash=self.feature_config_hash,support=list(self.support),metadata=self.metadata,strata=list(self.strata))
        (root/'references.json').write_text(json.dumps(meta,indent=2,default=str))
        for label,bundle in self.strata.items():bundle.save(root/('stratum_'+label))
        return root

    @classmethod
    def load(cls,source,*,load_training_data=True):
        """Safe numeric/string loading; legacy IDs are recovered from features.

        Evaluation only needs the fitted curves, so callers may omit training
        arrays explicitly. Object arrays are NEVER deserialized with pickle.
        """
        root=Path(source);meta=json.loads((root/'references.json').read_text())
        if meta['version']!='aging-bias-reference-v1':raise ValueError('Unsupported reference version')
        with np.load(root/'references.npz',allow_pickle=False) as a:
            arrays={k:a[k].copy() for k in ['knots','f_coeff','mu_c','p_ages','mu_p','mean','sd','icc']}
            training={};recover_ids=False
            if load_training_data:
                for key in a.files:
                    if not key.startswith('training__'):continue
                    try:training[key[10:]]=a[key].copy()
                    except ValueError as exc:
                        if key!='training__c_ids' or 'Object arrays' not in str(exc):raise
                        recover_ids=True
        if recover_ids:
            sidecar=root/'train_features.parquet'
            sex=meta.get('metadata',{}).get('sex')
            if not sidecar.is_file() and sex:sidecar=root.parent/'train_features.parquet'
            if not sidecar.is_file():
                raise ValueError('Legacy object identity IDs require train_features.parquet. '
                    'For module 2 only, use ReferenceBundle.load(path, load_training_data=False); no refit is needed.')
            frame=pd.read_parquet(sidecar)
            if sex:
                if 'gender' not in frame:raise ValueError('Training sidecar does not match stratum: missing gender')
                frame=frame[frame.gender==sex]
            columns=['phi__'+name for name in meta['attributes']]
            required=['identity_id','age']+columns
            if not set(required)<=set(frame) or frame.identity_id.isna().any():
                raise ValueError('Training sidecar does not match the reference columns')
            values=(frame[columns].to_numpy(float)-arrays['mean'])/arrays['sd']
            ages=frame.age.to_numpy(float)
            if ('c_age' not in training or 'c_phi' not in training or
                ages.shape!=training['c_age'].shape or values.shape!=training['c_phi'].shape or
                not np.allclose(ages,training['c_age'],equal_nan=True) or
                not np.allclose(values,training['c_phi'],equal_nan=True)):
                raise ValueError('Training sidecar rows do not match the saved reference arrays; refusing to guess identity order')
            training['c_ids']=np.asarray(frame.identity_id.astype(str).tolist(),dtype=str)
        return cls(**arrays,attributes=meta['attributes'],feature_config_hash=meta['feature_config_hash'],
            support=tuple(meta['support']),metadata=meta['metadata'],training_data=training,
            strata={label:cls.load(root/('stratum_'+label),load_training_data=load_training_data) for label in meta.get('strata',[])})


def load_colombia_manifest(config):
    if config.train_manifest_path:
        df=pd.read_csv(config.train_manifest_path)
        df=df.rename(columns={'person_id':'identity_id','relative_path':'image_path'})
    else:
        from data.indexing import build_image_manifest
        records,_=build_image_manifest(config.colombia_root,min_age=0,max_age=120,normalize_over_100=False)
        df=pd.DataFrame([dict(identity_id=r.person_id,age=r.age,image_path=str(r.path(config.colombia_root))) for r in records])
    if df.empty:raise ValueError('Empty Colombian training manifest')
    if config.split_path:
        split=pd.read_csv(config.split_path).rename(columns={'person_id':'identity_id'})
        df=df.merge(split[['identity_id','split']],on='identity_id',how='left',validate='many_to_one')
        if df.split.isna().any():raise ValueError('Missing split assignments')
        df=df[df.split=='train'].copy()
    elif 'split' in df:
        if not df['split'].eq('train').all():raise ValueError('Reference manifest contains non-train/test data')
    elif not config.train_only:
        raise ValueError('Provide split_path or explicitly acknowledge train_only=True')
    excluded=set(map(str,config.excluded_test_ids))
    if config.test_manifest_path:
        test=pd.read_csv(config.test_manifest_path)
        excluded.update(test['identity_id'].astype(str))
    overlap=excluded.intersection(df.identity_id.astype(str))
    if overlap:raise ValueError(f'Test identity leakage in reference train: {sorted(overlap)[:10]}')
    if df.empty:raise ValueError('No training images after split filtering')
    df['image_path']=[str(Path(p) if Path(p).is_absolute() else Path(config.colombia_root)/p) for p in df.image_path]
    return df.reset_index(drop=True)


def load_ffhq_manifest(root,labels_path=None,min_conf=.6,max_per_group=None,seed=2026):
    """Official groups: 0-2,3-6,7-9,10-14,15-19,20-29,30-39,40-49,50-69,70-120.

    Midpoints are literal arithmetic means; last midpoint is 95, not a
    made-up chronological annotation. Source: royorel/FFHQ-Aging-Dataset,
    ffhq_aging_labels.csv. FFHQ is a proxy, not a nationality classifier.
    """
    root=Path(root);labels=Path(labels_path) if labels_path else root/'ffhq_aging_labels.csv'
    df=pd.read_csv(labels);required={'image_number','age_group','age_group_confidence'}
    if not required.issubset(df):raise ValueError('Expected official FFHQ-Aging label columns')
    df=df[df.age_group_confidence>=min_conf].copy()
    allowed={'0-2','3-6','7-9','10-14','15-19','20-29','30-39','40-49','50-69','70-120'}
    if not set(df.age_group.astype(str))<=allowed:raise ValueError('Unknown FFHQ age group convention')
    df['age']=[np.mean(list(map(float,x.split('-')))) for x in df.age_group]
    if max_per_group:
        df=pd.concat([g.sample(min(len(g),max_per_group),random_state=seed) for _,g in df.groupby('age_group')])
    photos=root/'images' if (root/'images').is_dir() else root
    images={}
    for path in photos.rglob('*'):
        if path.suffix.lower() not in {'.png','.jpg','.jpeg'}:continue
        relative=path.relative_to(photos)
        if any(any(token in part.lower() for token in ['mask','parsing','segmentation','semantic']) for part in relative.parts[:-1]):continue
        if path.stem in images:raise ValueError(f'Ambiguous FFHQ image stem {path.stem}; point ffhq_aging_root to one photo resolution and supply ffhq_labels_path')
        images[path.stem]=path
    df['image_path']=[str(images.get(f'{int(i):05d}',images.get(str(int(i)),''))) for i in df.image_number]
    if df.image_path.eq('').any():
        raise FileNotFoundError('FFHQ labels reference missing images; provide complete images or a labels CSV for your explicit subset')
    return df.reset_index(drop=True)


def fit_reference_frames(colombia,proxy,attributes,knots,mean,sd,icc,feature_config_hash,metadata=None):
    columns=['phi__'+a for a in attributes]
    cv=(colombia[columns].to_numpy(float)-mean)/sd
    pv=(proxy[columns].to_numpy(float)-mean)/sd
    q=colombia.reindex(columns=CAPTURE,fill_value=0).to_numpy(float)
    age=colombia.age.to_numpy(float);ids=colombia.identity_id.to_numpy(dtype=str)
    coef,capture=fit_panel_fe(age,cv,ids,knots,q)
    unadjusted,_=fit_panel_fe(age,cv,ids,knots)
    mu=np.full_like(coef,np.nan)
    for j in range(len(attributes)):
        valid=np.isfinite(cv[:,j])&np.isfinite(q).all(1)
        if valid.any():mu[:,j]=np.linalg.lstsq(spline_basis(age[valid],knots),cv[valid,j],rcond=None)[0]
    p_age=np.sort(proxy.age.unique().astype(float))
    p_mu=np.array([np.nanmean(pv[proxy.age.to_numpy()==a],axis=0) for a in p_age])
    if len(p_age)<2 or not np.isfinite(p_mu).all():raise ValueError('Proxy needs >=2 age groups and finite active attributes in each group')
    return ReferenceBundle(np.asarray(knots,float),coef,mu,p_age,p_mu,np.asarray(mean),np.asarray(sd),
        list(attributes),np.asarray(icc),feature_config_hash,(max(float(age.min()),float(knots[0])),min(float(age.max()),float(knots[-1]))),
            metadata=metadata or {},training_data=dict(c_age=age,c_phi=cv,c_ids=ids,c_capture=q,
            capture_coef=capture,f_coeff_unadjusted=unadjusted,
            p_age=proxy.age.to_numpy(float),p_phi=pv))


def build_reference_bundle(config,*,feature_backend=None,landmark_backend=None):
    """Fit references on declared train only; no automatic model/download side effects.

    Missing/low ICC excludes features, not a fallback to perfect reliability.
    No test images may influence means, scales, PCA, ICC, or population curves.
    """
    colombia=load_colombia_manifest(config)  # Leakage check precedes every backend.
    if config.feature_backend!='handcrafted' and feature_backend is None:raise ValueError('Supply a fitted robustness feature_backend explicitly')
    root=Path(config.output_dir);root.mkdir(parents=True,exist_ok=True)
    proxy=load_ffhq_manifest(config.ffhq_aging_root,config.ffhq_labels_path,config.ffhq_min_conf,config.ffhq_max_per_group,config.seed)
    store=FeatureStore(root/'feature_cache',config.features,landmark_backend,feature_backend,config.resume)
    def extract(df):
        rows=[];failures=[]
        for i,r in enumerate(df.to_dict('records')):
            try:rows.append({**r,**store.extract(r['image_path'])})
            except Exception as exc:failures.append({**r,'error':f'{type(exc).__name__}: {exc}'})
            if (i+1)%100==0:print(f'[references] {i+1}/{len(df)} images, {len(failures)} failures',flush=True)
        return pd.DataFrame(rows),failures
    c,cf=extract(colombia);p,pf=extract(proxy)
    pd.DataFrame(cf+pf,columns=['identity_id','image_path','age','error']).to_csv(root/'reference_failures.csv',index=False)
    return fit_reference_feature_frames(config,c,p,store.config_hash)


def fit_reference_feature_frames(config,c,p,feature_config_hash):
    """Fit only statistical references from already extracted train/proxy rows."""
    root=Path(config.output_dir);root.mkdir(parents=True,exist_ok=True)
    if c.empty or p.empty:raise ValueError('No reference features extracted; inspect reference_failures.csv and landmark backend')
    attrs=[k[5:] for k in c if k.startswith('phi__')]
    quality=[]
    for population,frame in [('colombia_train',c),('ffhq_proxy',p)]:
        phi=frame[['phi__'+a for a in attrs]]
        quality.append(dict(population=population,n_extracted=len(frame),
            n_pose_excluded=int(frame.get('pose_quality_ok',pd.Series(True,index=frame.index)).eq(False).sum()),
            n_expression_flagged=int(frame.get('expression_quality_ok',pd.Series(True,index=frame.index)).eq(False).sum()),
            n_with_any_attributes=int(phi.notna().any(axis=1).sum()),
            n_with_all_attributes=int(phi.notna().all(axis=1).sum()),
            n_grayscale=int(frame.get('is_grayscale',pd.Series(False,index=frame.index)).sum()),
            n_adult_images=int(frame.age.ge(config.retest_min_age or 0).sum())))
    pd.DataFrame(quality).to_csv(root/'reference_quality.csv',index=False)
    reliability=reference_reliability(c,attrs,config)
    sd=c[['phi__'+k for k in attrs]].std(ddof=1).to_numpy(float)
    invalid=~np.isfinite(sd)|(sd<=1e-12)
    reliability.loc[invalid,['active','active_pi','active_homog']]=False
    reliability.loc[invalid,'attribute_weight']=0.
    reliability.loc[invalid,'exclusion_reason']='nonvariable_attribute'
    # A reliable train descriptor may have no measurements in a proxy age bin
    # (e.g. only grayscale photos). Exclude explicitly rather than crash later
    # or fill unknown color attributes with zero.
    proxy_means=p.groupby('age')[['phi__'+a for a in attrs]].mean()
    unavailable=~np.isfinite(proxy_means.to_numpy()).all(axis=0)
    reliability.loc[unavailable,['active','active_pi','active_homog']]=False
    reliability.loc[unavailable,'attribute_weight']=0.
    reliability.loc[unavailable,'exclusion_reason']='missing_proxy_age_group'
    reliability.to_csv(root/'attribute_reliability.csv',index=False)
    active=reliability[reliability.active]
    if active.empty:raise ValueError('No active reference attributes: inspect attribute_reliability.csv and reference_quality.csv for adult retests, ICC, variation and proxy support. No threshold or age fallback was applied.')
    attrs=active.attribute.tolist();columns=['phi__'+k for k in attrs]
    bundle=fit_reference_frames(c,p,attrs,config.knots,c[columns].mean().to_numpy(),c[columns].std().to_numpy(),
        active.icc.to_numpy(),feature_config_hash,dict(config=asdict(config),train_identities=sorted(c.identity_id.astype(str).unique()),
            proxy='FFHQ-Aging (not a demographic nationality label)',age_period_cohort_limitation=True,
            reliability_policy=dict(weight_by_icc=config.weight_attributes_by_icc,
                min_icc_pi=config.min_icc,min_icc_homog=config.min_homog_icc,
                retest_min_age=config.retest_min_age,
                active_homog=active.active_homog.tolist(),
                attribute_weights=active.attribute_weight.tolist(),
                homogenization='excess_over_oracle' if config.min_homog_icc is not None else 'raw'),
            measurement_caveats=['Adult reliability is not evidence of reliability in children.',
                'Local illumination correction does not identify global exposure or white balance.',
                'ICC weighting does not remove measurement error in regressors or capture confounding.']))
    if config.use_sex:
        if 'gender' not in c or 'gender' not in p:
            warnings.warn('Sex labels unavailable in train/proxy: using pooled references, no inferred labels')
        else:
            for sex in ['male','female']:
                cs=c[c.gender==sex];ps=p[p.gender==sex]
                if cs.identity_id.nunique()<config.min_sex_identities or ps.age.nunique()<2:
                    warnings.warn(f'Insufficient labeled reference support for {sex}; pooled fallback recorded')
                    continue
                bundle.strata[sex]=fit_reference_frames(cs,ps,attrs,config.knots,bundle.mean,bundle.sd,
                    bundle.icc,feature_config_hash,dict(sex=sex,standardization='pooled Colombian train'))
            bundle.metadata['sex_strata']=list(bundle.strata)
    bundle.save(root);c.to_parquet(root/'train_features.parquet',index=False);p.to_parquet(root/'proxy_features.parquet',index=False)
    coefficients=bundle.training_data['capture_coef']
    pd.DataFrame([dict(attribute=a,covariate=key,coefficient=coefficients[i,j])
        for j,a in enumerate(attrs) for i,key in enumerate(CAPTURE)]).to_csv(root/'capture_coefficients.csv',index=False)
    return bundle
