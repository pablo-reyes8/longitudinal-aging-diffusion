"""Manifest-first aging-bias evaluation; never generates or retrains a model."""
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import platform
import warnings
import numpy as np
import pandas as pd
from src.aging_bias.features import FeatureStore,COLOR_ATTRIBUTES,stable_hash
from src.aging_bias.references import ReferenceBundle
from src.aging_bias.weighting import transition_weights
from src.aging_bias.bootstrap import bootstrap_models
from src.aging_bias.validation import run_self_checks,validate_real_anchors
from src.quantitative_metrics.image_io import ImageResolver
from src.quantitative_metrics.evaluator import _cosine,_git_commit,_safe_version
from .critic_provenance import assert_independent_age,contamination_flags,CRITIC_PROVENANCE


def load_test_panel(manifest_path,source_root=None):
    df=pd.read_csv(manifest_path,dtype={'sample_id':str,'identity_id':str}) if not isinstance(manifest_path,pd.DataFrame) else manifest_path.copy()
    required=['sample_id','identity_id','source_age','target_age','source_path','target_path']
    if not set(required)<=set(df):raise ValueError(f'Test manifest requires {required}')
    if df[required].isna().any().any():raise ValueError('Test manifest contains missing IDs, ages or image paths')
    for key in ['sample_id','identity_id']:df[key]=df[key].astype(str)
    panel=df[required+(['gender'] if 'gender' in df else [])].drop_duplicates()
    if panel.duplicated('sample_id').any():raise ValueError('Conflicting source/target metadata for a sample_id')
    for k in ['source_age','target_age']:
        panel[k]=pd.to_numeric(panel[k],errors='raise')
        if not panel[k].between(0,120).all():raise ValueError('Test ages must be in [0,120]')
    panel['delta_age']=panel.target_age-panel.source_age
    if panel.empty or panel.delta_age.eq(0).any():raise ValueError('Need nonempty, nonzero longitudinal transitions')
    if source_root:
        for role in ['source','target']:
            panel[role+'_path']=[str(Path(source_root)/str(r.identity_id)/Path(getattr(r,role+'_path')).name) for r in panel.itertuples()]
    return panel.reset_index(drop=True)


def _generated_path(config,model,row,original):
    root=config.generated_roots.get(model)
    if root is not None:
        root=Path(root)
        candidates=[root/row['identity_id']/(row['sample_id']+'.png'),root/(row['sample_id']+'.png'),
            root/'generated'/model/row['identity_id']/(row['sample_id']+'.png')]
        return next((p for p in candidates if p.is_file()),candidates[0])
    match=original[(original.sample_id==row['sample_id'])&(original.model_name==model)] if 'model_name' in original else pd.DataFrame()
    if len(match) and pd.notna(match.iloc[0].get('generated_path')):return Path(match.iloc[0].generated_path)
    return None


class OperationalStore:
    """Content+backend keyed embeddings and age expectations, not just CSV reuse."""
    def __init__(self,bundle,root,resume=True):
        self.bundle=bundle;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.resume=resume;self.resolver=ImageResolver()
        keys=['identity_encoder','independent_age_backend','age_estimator','aligner']
        self.signature=stable_hash({'metadata':bundle.get('metadata',{}),
            'backends':{k:getattr(bundle.get(k),'cache_id',type(bundle.get(k)).__qualname__) for k in keys}})

    def get(self,path):
        image,digest=self.resolver.load(path);self.resolver._cache.clear()
        key=self.root/(digest+'_'+self.signature+'.json')
        if self.resume and key.exists():return json.loads(key.read_text())
        aligned=self.bundle['aligner'](image)
        if aligned is None:raise ValueError('No face detected for operational metrics')
        embedding=np.asarray(self.bundle['identity_encoder'].embed_batch([aligned])[0],float)
        age=float(self.bundle['independent_age_backend'].predict_batch([aligned])[0])
        dex=float(self.bundle['age_estimator'].predict_batch([aligned])[0]) if self.bundle.get('age_estimator') else np.nan
        if not np.isfinite(embedding).all() or not np.isfinite(age):raise ValueError('Nonfinite identity/independent-age output')
        row=dict(embedding=embedding.tolist(),independent_age=age,dex_age=dex,image_sha256=digest)
        temp=key.with_suffix('.tmp');temp.write_text(json.dumps(row,allow_nan=True));temp.replace(key)
        return row

    def prefetch(self,paths,batch_size=16):
        """Bounded batches; failed items are retried individually by get()."""
        if batch_size<1:raise ValueError('batch_size must be >= 1')
        pending=[];seen=set()
        def flush():
            if not pending:return
            images=[x[0] for x in pending]
            try:
                embeddings=self.bundle['identity_encoder'].embed_batch(images)
                ages=self.bundle['independent_age_backend'].predict_batch(images)
                dex=self.bundle['age_estimator'].predict_batch(images) if self.bundle.get('age_estimator') else [np.nan]*len(images)
                if any(len(v)!=len(images) for v in [embeddings,ages,dex]):raise ValueError('Backend batch length mismatch')
                for (_,digest,path),emb,age,d in zip(pending,embeddings,ages,dex):
                    if not np.isfinite(emb).all() or not np.isfinite(age):continue
                    row=dict(embedding=np.asarray(emb,float).tolist(),independent_age=float(age),dex_age=float(d),image_sha256=digest)
                    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,allow_nan=True));temp.replace(path)
            except Exception:
                pass  # per-image retry records exact exceptions, never fabricates values
            pending.clear()
        for path in paths:
            try:
                image,digest=self.resolver.load(path);self.resolver._cache.clear()
                if digest in seen:continue
                seen.add(digest);key=self.root/(digest+'_'+self.signature+'.json')
                if self.resume and key.exists():continue
                aligned=self.bundle['aligner'](image)
                if aligned is not None:pending.append((aligned,digest,key))
                if len(pending)>=batch_size:flush()
            except (OSError,ValueError,TypeError):continue
        flush()


def _arrays(frame,attributes):
    return {k:frame[[k+'__'+a for a in attributes]].to_numpy(float) for k in ['g','d_C','d_P','G_P','zeta']}


def _operational_summary(frame,weights,model,subset,panel,delta_bin='all'):
    rows=[]
    columns=['adaface_id_source','adaface_id_gt','delta_id_gain','age_rel_err_indep','age_rel_signed_indep','dex_age_mae']
    for name in [c for c in weights if c.startswith('w_')]:
        row=dict(model_name=model,scheme=name[2:],subset=subset,panel=panel,delta_bin=delta_bin,
                 n_pairs=len(frame),n_identities=frame.identity_id.nunique())
        for k in columns:
            value=frame[k].to_numpy(float);w=weights[name].to_numpy(float);valid=np.isfinite(value)
            row[k]=float(np.average(value[valid],weights=w[valid])) if valid.any() and w[valid].sum()>0 else np.nan
            row['n_valid__'+k]=int(valid.sum())
        rows.append(row)
    return rows


def evaluate_paper_v2(config,*,bundle,reference_bundle=None):
    """Recalculate metrics from saved images. Models stay CPU/resident in bundle.

    Main panel is the common feature-valid intersection; full per-model panels
    are reported separately. Missing attributes remain NaN. Bootstrap intervals
    condition on the fitted references (population-curve uncertainty is reported
    separately by describe_population_aging, not silently claimed as joint CI).
    """
    assert_independent_age(config.independent_age_name,config.models)
    required=['aligner','identity_encoder','independent_age_backend']
    if config.include_dex_appendix:required+=['age_estimator']
    if config.include_kid:required+=['kid_metric']
    missing=[k for k in required if bundle.get(k) is None]
    if missing:raise ValueError('Evaluation bundle missing: '+', '.join(missing))
    backend=bundle['independent_age_backend']
    assert_independent_age(getattr(backend,'name',config.independent_age_name),config.models)
    assert_independent_age(type(backend).__name__,config.models)
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    checks=run_self_checks(output/'validation')
    if not checks.passed.all():raise RuntimeError('Synthetic self-checks failed; evaluation stopped before any network execution')
    ref=reference_bundle or ReferenceBundle.load(config.reference_bundle_path)
    panel=load_test_panel(config.test_manifest_path,config.source_root)
    for direction in ['all','aging','rejuvenation']:
        subset=panel if direction=='all' else panel[panel.delta_age>0] if direction=='aging' else panel[panel.delta_age<0]
        if len(subset):transition_weights(subset,config.weighting)
    overlap=set(panel.identity_id.astype(str))&set(map(str,ref.metadata.get('train_identities',[])))
    if overlap:raise ValueError(f'Test/reference training identity overlap: {sorted(overlap)[:10]}')
    original=pd.read_csv(config.test_manifest_path,dtype={'sample_id':str,'identity_id':str})
    models=list(dict.fromkeys(config.models))
    if not models or {'source_noedit','oracle_target'}&set(models):raise ValueError('Supply actual models; anchors are added automatically')
    store=FeatureStore(output/'cache'/'features',config.features,bundle.get('landmark_backend'),bundle.get('feature_backend'),config.resume)
    if store.config_hash!=ref.feature_config_hash:raise ValueError('Feature/preprocessing configuration differs from reference bundle; use the same FeatureConfig and backend')
    operational=OperationalStore(bundle,output/'cache'/'operational',config.resume)
    failures=[];rows=[];paths=[]
    print(f'[paper v2] {len(panel)} transitions; cached descriptors and CPU metrics, no generation',flush=True)
    def feature(path,sample,model,role):
        try:return store.extract(path)
        except Exception as exc:
            failures.append(dict(sample_id=sample,model_name=model,stage=role+'_load',path=str(path),error=f'{type(exc).__name__}: {exc}'))
            return None
    def measure(path,sample,model,role):
        try:return operational.get(path)
        except Exception as exc:
            failures.append(dict(sample_id=sample,model_name=model,stage=role+'_operational',path=str(path),error=f'{type(exc).__name__}: {exc}'))
            return None
    # Smoke uses real inputs before bulk processing; generated-image failures
    # are recorded per model, never take down a completed long run.
    smoke_paths=[]
    for sample in panel.head(8).to_dict('records'):
        for role in ['source','target']:
            path=sample[role+'_path']
            if path in smoke_paths:continue
            if feature(path,sample['sample_id'],'preflight',role) is not None and measure(path,sample['sample_id'],'preflight',role) is not None:
                smoke_paths.append(path)
            if len(smoke_paths)>=2:break
        if len(smoke_paths)>=2:break
    if not smoke_paths:
        pd.DataFrame(failures).to_csv(output/'failures.csv',index=False)
        raise RuntimeError('Paper v2 backend smoke failed; inspect failures.csv before processing the panel')
    if config.include_kid:
        try:
            smoke=[store.resolver.load(p)[0] for p in (smoke_paths+[smoke_paths[0]])[:2]]
            bundle['kid_metric'].compute(smoke,smoke,seed=config.seed,subsets=1,subset_size=2)
            store.resolver._cache.clear()
        except Exception as exc:raise RuntimeError(f'Paper v2 KID smoke failed before bulk evaluation: {exc}') from exc
    refs=[]
    for sample in panel.to_dict('records'):
        refs.extend([sample['source_path'],sample['target_path']])
        refs.extend(_generated_path(config,m,sample,original) for m in models)
    operational.prefetch(refs,config.batch_size)
    for i,sample in enumerate(panel.to_dict('records')):
        sid=sample['sample_id'];fs=feature(sample['source_path'],sid,'shared','source');ft=feature(sample['target_path'],sid,'shared','target')
        os=measure(sample['source_path'],sid,'shared','source');ot=measure(sample['target_path'],sid,'shared','target')
        for model in models+['source_noedit','oracle_target']:
            path=sample['source_path'] if model=='source_noedit' else sample['target_path'] if model=='oracle_target' else _generated_path(config,model,sample,original)
            fg=fs if model=='source_noedit' else ft if model=='oracle_target' else feature(path,sid,model,'generated')
            og=os if model=='source_noedit' else ot if model=='oracle_target' else measure(path,sid,model,'generated') if fg is not None else None
            row={**sample,'model_name':model,'generated_path':str(path) if path is not None else None,
                **contamination_flags(model),'extrapolated':bool(ref.extrapolated([sample['source_age']],[sample['target_age']],
                    sex=[sample['gender']] if sample.get('gender') else None)[0]),
                'reference_sex_fallback':bool(ref.strata and sample.get('gender') not in ref.strata)}
            row.update(source_sha256=fs.get('image_sha256') if fs else None,
                target_sha256=ft.get('image_sha256') if ft else None,generated_sha256=fg.get('image_sha256') if fg else None)
            for k in ['adaface_id_source','adaface_id_gt','delta_id_gain','age_rel_err_indep','age_rel_signed_indep','dex_age_mae']:row[k]=np.nan
            if os is not None and ot is not None and og is not None:
                row.update(independent_age_source=os['independent_age'],independent_age_target=ot['independent_age'],
                    independent_age_generated=og['independent_age'])
                row['adaface_id_source']=_cosine(os['embedding'],og['embedding'])
                row['adaface_id_gt']=_cosine(og['embedding'],ot['embedding'])
                row['delta_id_gain']=row['adaface_id_gt']-_cosine(os['embedding'],ot['embedding'])
                row['age_rel_signed_indep']=og['independent_age']-ot['independent_age']
                row['age_rel_err_indep']=abs(row['age_rel_signed_indep'])
                row['dex_age_mae']=abs(og['dex_age']-sample['target_age'])
            sex=[sample['gender']] if sample.get('gender') else None
            dc=ref.d_C([sample['source_age']],[sample['target_age']],sex=sex)[0]
            dp=ref.d_P([sample['source_age']],[sample['target_age']],sex=sex)[0];gp=ref.G_P([sample['source_age']],sex=sex)[0]
            def vector(f):return np.array([f.get('phi__'+a,np.nan) if f else np.nan for a in ref.attributes])
            src=ref.standardize(vector(fs));tgt=ref.standardize(vector(ft));gen=ref.standardize(vector(fg))
            for j,a in enumerate(ref.attributes):
                color_ok=a not in COLOR_ATTRIBUTES or (fs is not None and ft is not None and not fs.get('is_grayscale',False) and not ft.get('is_grayscale',False))
                for key,v in [('phi_source',src[j]),('phi_target',tgt[j]),('phi_generated',gen[j]),
                    ('g',gen[j]-src[j]),('d_C',dc[j]),('d_P',dp[j]),('G_P',gp[j]),('zeta',ref.zeta(src[None,:],[sample['source_age']],sex=sex)[0,j])]:
                    row[key+'__'+a]=float(v) if color_ok else np.nan
            row['feature_valid']=bool(np.logical_and.reduce([np.isfinite(np.array([row[k+'__'+a] for a in ref.attributes])) for k in ['g','d_C','d_P','G_P','zeta']]).any())
            row['operational_valid']=bool(np.isfinite(row['delta_id_gain']) and np.isfinite(row['age_rel_err_indep']))
            rows.append(row);paths.append(dict(sample,model_name=model,generated_path=row['generated_path']))
        if (i+1)%20==0:print(f'[paper v2] descriptors/metrics {i+1}/{len(panel)}',flush=True)
    per=pd.DataFrame(rows)
    valid_sets=[set(per.loc[(per.model_name==m)&per.feature_valid&per.operational_valid,'sample_id']) for m in models]
    common=set.intersection(*valid_sets)
    per['in_common_panel']=per.sample_id.isin(common)
    per['direction']=np.where(per.delta_age>0,'aging','rejuvenation')
    all_indices=[];all_bybin=[];paired=[];ops=[];anchor_indices=[]
    panels=[('common',per[per.in_common_panel])]+[(f'full:{m}',per[(per.model_name==m)&per.feature_valid]) for m in models]
    fit_opts=dict(d_min=config.d_min,max_cond=config.max_cond,max_corr=config.max_corr)
    for panel_name,frame in panels:
        for subset in ['all','aging','rejuvenation']:
            sub=frame if subset=='all' else frame[frame.direction==subset]
            names=list(dict.fromkeys(sub.model_name))
            if sub.empty:continue
            print(f'[paper v2] {panel_name}/{subset}: structural fit + {config.n_boot if panel_name=="common" else 0} identity bootstrap draws',flush=True)
            base=sub[sub.model_name==names[0]].reset_index(drop=True)
            weights=transition_weights(base,config.weighting)
            arrays={m:_arrays(sub[sub.model_name==m].set_index('sample_id').loc[base.sample_id].reset_index(),ref.attributes) for m in names}
            result=bootstrap_models(base,arrays,config.weighting,n_boot=config.n_boot if panel_name=='common' else 0,seed=config.seed,**fit_opts)
            idx=result['indices'].assign(panel=panel_name,subset=subset,
                weighting_merges=json.dumps(result['weights'].attrs['merges']))
            anchor_indices.append(idx[idx.model_name.isin(['source_noedit','oracle_target'])])
            all_indices.append(idx[idx.model_name!='oracle_target'])
            pdiff=result['paired']
            if not pdiff.empty:
                pdiff=pdiff[~pdiff.model_a.isin(['oracle_target','source_noedit'])&~pdiff.model_b.isin(['oracle_target','source_noedit'])].copy()
                # Orient every contrast as ours - baseline, not alphabetically.
                for j,r in pdiff.iterrows():
                    if str(r.model_b).startswith('ours') and not str(r.model_a).startswith('ours'):
                        pdiff.loc[j,['model_a','model_b']]=[r.model_b,r.model_a]
                        pdiff.loc[j,['difference','ci_low','ci_high']]=[-r.difference,-r.ci_high,-r.ci_low]
                pdiff=pdiff[pdiff.model_a.str.startswith('ours')]
                paired.append(pdiff.assign(panel=panel_name,subset=subset))
            for m in names:
                aligned=sub[sub.model_name==m].set_index('sample_id').loc[base.sample_id].reset_index()
                ops+=_operational_summary(aligned,weights,m,subset,panel_name)
            for label,group in weights.groupby('delta_bin',sort=False):
                ix=group.index.to_numpy();binconfig=replace(config.weighting,operational_target=None,scheme_primary='bin_balanced')
                br=bootstrap_models(base.iloc[ix].reset_index(drop=True),{m:{k:v[ix] for k,v in a.items()} for m,a in arrays.items()},
                    binconfig,n_boot=config.n_boot if panel_name=='common' else 0,seed=config.seed,**fit_opts)
                all_bybin.append(br['indices'].query("model_name!='oracle_target'").assign(panel=panel_name,subset=subset,delta_bin=label,
                    n_identities=group.identity_id.nunique(),n_bin_pairs=len(group)))
                bw=transition_weights(base.iloc[ix].reset_index(drop=True),binconfig)
                for m in names:
                    aligned=sub[sub.model_name==m].set_index('sample_id').loc[base.iloc[ix].sample_id].reset_index()
                    ops+=_operational_summary(aligned,bw,m,subset,panel_name,label)
            if panel_name=='common' and subset=='all':
                for col in [c for c in weights if c.startswith('w_')]+['delta_bin']:
                    mapping=dict(zip(base.sample_id,weights[col]));per[col]=per.sample_id.map(mapping)
    if not common:warnings.warn('No common valid panel: primary comparisons and real anchors are unavailable; inspect failures.csv')
    index_columns=['model_name','scheme','panel','subset','status','D','pi','drift','homog']
    indices=pd.concat(all_indices,ignore_index=True) if all_indices else pd.DataFrame(columns=index_columns)
    bybin=pd.concat(all_bybin,ignore_index=True) if all_bybin else pd.DataFrame(columns=index_columns+['delta_bin'])
    contrasts=pd.concat(paired,ignore_index=True) if paired else pd.DataFrame(columns=['model_a','model_b','index_name','difference','ci_low','ci_high','share_ok'])
    anchor=pd.concat(anchor_indices,ignore_index=True) if anchor_indices else pd.DataFrame(columns=['scheme','subset'])
    anchors=validate_real_anchors(anchor,1-float(np.mean(ref.icc)),config.oracle_d_bounds,config.oracle_ratio_tolerance,config.oracle_homog_tolerance)
    if not anchors['passed']:warnings.warn('REAL ANCHORS FAILED: do not interpret bias ratios or start SD1.5 counterfactual; inspect validation/real_anchors.json')
    (output/'validation'/'real_anchors.json').write_text(json.dumps(anchors,indent=2,allow_nan=True))
    anchor.to_csv(output/'validation'/'oracle_indices.csv',index=False)
    kid_rows=[]
    if config.include_kid:
        resolver=ImageResolver()
        for m in models+['source_noedit']:
            f=per[(per.model_name==m)&per.in_common_panel]
            mean=std=np.nan;status='insufficient_samples'
            if len(f)>=2:
                try:
                    real=[resolver.load(p)[0] for p in f.target_path];gen=[resolver.load(p)[0] for p in f.generated_path]
                    mean,std=bundle['kid_metric'].compute(real,gen,seed=config.seed,subset_size=min(50,len(f)))
                    status='ok'
                except Exception as exc:
                    status='backend_error';failures.append(dict(model_name=m,stage='kid',error=str(exc)))
            kid_rows.append(dict(model_name=m,kid_mean=mean,kid_std=std,kid_status=status,n_kid=len(f),panel='common'))
            resolver._cache.clear()
    failures=pd.DataFrame(failures,columns=['sample_id','model_name','stage','path','error'])
    operations=pd.DataFrame(ops) if ops else pd.DataFrame(columns=['model_name','scheme','subset','panel','delta_bin'])
    operations[operations.model_name.eq('oracle_target')].to_csv(output/'validation'/'oracle_operational_metrics.csv',index=False)
    operations=operations[operations.model_name.ne('oracle_target')].reset_index(drop=True)
    # Content signature: moving outputs does not invalidate descriptors.
    refhash=stable_hash(dict(feature=ref.feature_config_hash,knots=ref.knots.tolist(),f=ref.f_coeff.tolist(),
        c=ref.mu_c.tolist(),p=ref.mu_p.tolist(),mean=ref.mean.tolist(),sd=ref.sd.tolist(),icc=ref.icc.tolist()))
    metadata=dict(config=asdict(config),n_requested_pairs=len(panel),n_common_pairs=len(common),n_common_identities=panel[panel.sample_id.isin(common)].identity_id.nunique(),
        feature_config_hash=store.config_hash,reference_hash=refhash,operational_hash=operational.signature,
        critic_provenance=CRITIC_PROVENANCE,git_commit=_git_commit(),python=platform.python_version(),
        versions={k:_safe_version(k) for k in ['numpy','pandas','scipy','mediapipe','torch']},
        weighting_merges=transition_weights(panel,config.weighting).attrs['merges'],seed=config.seed,
        ci_scope='identity bootstrap conditional on fitted references; not joint reference uncertainty',anchors_passed=anchors['passed'])
    per.to_parquet(output/'per_sample.parquet',index=False)
    per.to_csv(output/'per_sample.csv',index=False)
    for name,df in [('bias_indices',indices),('bias_indices_by_bin',bybin),('paired_differences',contrasts),('operational_metrics',operations),('failures',failures),('kid_metrics',pd.DataFrame(kid_rows))]:df.to_csv(output/(name+'.csv'),index=False)
    panel.to_csv(output/'transitions.csv',index=False);pd.DataFrame(paths).to_csv(output/'prediction_manifest.csv',index=False)
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str,allow_nan=True))
    if config.robustness_backend is not None:
        if config.robustness_backend!='embedding':raise ValueError('Unknown robustness backend')
        robust=bundle.get('robustness_bundle');robust_ref=bundle.get('robustness_reference')
        if robust is None or robust_ref is None:raise ValueError('Embedding robustness requires independently train-fitted robustness_bundle and robustness_reference')
        evaluate_paper_v2(replace(config,output_dir=output/'robustness',robustness_backend=None),bundle=robust,reference_bundle=robust_ref)
    from src.aging_bias.reporting import report_paper
    report_paper(output)
    return dict(per_sample=per,bias_indices=indices,bias_indices_by_bin=bybin,paired_differences=contrasts,
        operational_metrics=operations,summary=indices,failures=failures,metadata=metadata,
        paths={p.name:str(p) for p in output.iterdir() if p.is_file()})
