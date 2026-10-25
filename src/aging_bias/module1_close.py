"""Versioned, opt-in closure of module 1. Never trains/generates diffusion."""
from dataclasses import dataclass,field,replace,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .config import ReferenceConfig,FeatureConfig,FFHQSamplingConfig
from .features import FeatureStore,HandcraftedFeatureBackend,NEW_GEOMETRY,ATTRIBUTE_IDS
from .references import load_colombia_manifest,load_ffhq_manifest,fit_reference_feature_frames,fit_reference_frames,ReferenceBundle
from .reliability import reference_reliability
from .zones import fit_zone_transform,apply_zone_transform,ZONE_DEFINITIONS,longitudinal_mask


@dataclass
class Module1CloseConfig:
    reference_dir_out: str | Path
    latam_sources: dict = field(default_factory=dict)
    ffhq_aging_root: str | Path | None = None
    previous_reference_dir: str | Path | None = None
    reference_config: ReferenceConfig | None = None
    feature_backend: str = 'handcrafted'
    new_attributes: tuple = NEW_GEOMETRY
    resolution_matching: bool = True
    flip_tta: bool = True
    zone_indices: bool = True
    zone_definitions: dict = field(default_factory=lambda:dict(ZONE_DEFINITIONS))
    ffhq_sampling: FFHQSamplingConfig = field(default_factory=FFHQSamplingConfig)
    mean_faces: bool = True
    face_zone_figure: bool = True
    internal_placebo: bool = True
    min_n_mean_face: int = 50
    min_icc_pi: float = .30
    min_icc_homog: float = .50
    n_boot: int = 1000
    seed: int = 2026
    placebo_splits: int = 20
    transitions: tuple = ((30,60),(40,70),(50,75))
    allow_download: bool = False
    assets_root: str | Path = '/content/evaluation_models_v2'
    estimate_ffhq_50_69: bool = True
    ffhq_age_checkpoint_path: str | Path | None = None

    @classmethod
    def from_reference(cls,config):
        sampling=config.ffhq_sampling or FFHQSamplingConfig()
        if isinstance(sampling,dict):sampling=FFHQSamplingConfig(**sampling)
        return cls(reference_dir_out=config.output_dir,previous_reference_dir=config.previous_reference_dir,
            latam_sources=dict(colombia_train=config.colombia_root,fairface_latino=config.fairface_root),
            ffhq_aging_root=config.ffhq_aging_root,reference_config=config,
            feature_backend=config.feature_backend,ffhq_sampling=sampling,seed=config.seed,
            allow_download=config.module1_allow_download,assets_root=config.module1_assets_root)


def validate_destination(root,previous=None):
    root=Path(root).resolve()
    if previous:
        old=Path(previous).resolve()
        if root==old or old in root.parents:raise ValueError('Previous reference output is read-only; choose a separate new directory')
    if (root/'references.json').is_file():
        meta=json.loads((root/'references.json').read_text()).get('metadata',{})
        if not meta.get('module1_close'):raise ValueError('Legacy reference bundle exists; refusing to overwrite it. Choose references_v2.')


class CachedLandmarks:
    """Reusable original-image detections; caches belong only to NEW output."""
    def __init__(self,backend,root,fallback_id='injected-landmarks'):
        self.backend=backend;self.cache_id=getattr(backend,'cache_id',fallback_id)
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def __call__(self,image):
        signature=hashlib.sha256(self.cache_id.encode()+str(image.size).encode()+image.convert('RGB').tobytes()).hexdigest()
        path=self.root/(signature+'.json')
        if path.is_file():
            return {k:v if k.startswith('_') else np.asarray(v,float) for k,v in json.loads(path.read_text()).items()}
        values={k:v for k,v in self.backend(image).items() if k!='_mesh'}
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(values,default=lambda a:np.asarray(a).tolist()));temp.replace(path)
        return values


def _resolution_target(frame,landmarks,root,percentile):
    from src.quantitative_metrics.image_io import ImageResolver
    from .module1_reports import save_landmark_check
    resolver=ImageResolver();rows=[];control=False
    for i,row in enumerate(frame.to_dict('records'),1):
        try:
            image,_=resolver.load(row['image_path']);resolver._cache.clear();points=landmarks(image)
            iod=float(np.linalg.norm(points['left_eye'].mean(0)-points['right_eye'].mean(0)))
            rows.append(dict(image_path=row['image_path'],original_iod_px=iod,status='ok'))
            if not control and 'malar' in points:
                save_landmark_check(image,points,Path(root)/'figures/malar_landmarks_check.png');control=True
        except Exception as exc:rows.append(dict(image_path=row['image_path'],original_iod_px=np.nan,status=str(exc)))
        if i%100==0:print(f'[module1 IOD calibration] {i}/{len(frame)}',flush=True)
    table=pd.DataFrame(rows);table.to_csv(Path(root)/'diagnostics/resolution_calibration_latam.csv',index=False)
    valid=table.original_iod_px.to_numpy(float);valid=valid[np.isfinite(valid)&(valid>0)]
    if not len(valid):raise ValueError('No valid latam interocular distances; inspect resolution_calibration_latam.csv')
    return float(np.percentile(valid,percentile))


def prepare_closure_features(config,landmark_backend):
    """Called before DINO PCA too: exactly the same train-only target/crops."""
    validate_destination(config.output_dir,config.previous_reference_dir)
    root=Path(config.output_dir)
    for name in ['diagnostics','figures','tables','internal']:(root/name).mkdir(parents=True,exist_ok=True)
    frame=load_colombia_manifest(config)
    landmarks=CachedLandmarks(landmark_backend,root/'landmark_cache',config.features.landmark_backend_id)
    target=_resolution_target(frame,landmarks,root,config.features.resolution_target_percentile)
    features=replace(config.features,version='module1-close-v2',
        new_attributes=NEW_GEOMETRY if config.feature_backend=='handcrafted' else (),
        resolution_matching=True,target_iod_px=target,flip_tta=True,zone_transform={})
    return replace(config,features=features),landmarks


def _configuration(config):
    base=config.reference_config
    if base is None and config.previous_reference_dir:
        prior=ReferenceBundle.load(config.previous_reference_dir,load_training_data=False)
        raw=dict(prior.metadata['config']);raw['features']=FeatureConfig(**raw['features']);base=ReferenceConfig(**raw)
    if base is None:
        base=ReferenceConfig(config.latam_sources['colombia_train'],config.ffhq_aging_root,config.reference_dir_out,train_only=True)
    return replace(base,colombia_root=config.latam_sources.get('colombia_train') or base.colombia_root,
        fairface_root=config.latam_sources.get('fairface_latino',base.fairface_root),
        ffhq_aging_root=config.ffhq_aging_root or base.ffhq_aging_root,output_dir=Path(config.reference_dir_out),
        module1_close=False,min_icc=config.min_icc_pi,min_homog_icc=config.min_icc_homog,
        weight_attributes_by_icc=True,ffhq_sampling=config.ffhq_sampling,seed=config.seed)


def _extract(frame,store,root,label):
    rows=[];failures=[]
    for i,row in enumerate(frame.to_dict('records'),1):
        try:rows.append({**row,**store.extract(row['image_path'])})
        except Exception as exc:failures.append({**row,'error':f'{type(exc).__name__}: {exc}'})
        if i%100==0:print(f'[module1 {label}] {i}/{len(frame)} | {len(failures)} failures',flush=True)
    pd.DataFrame(failures).to_csv(Path(root)/'diagnostics'/f'{label}_failures.csv',index=False)
    if not rows:raise ValueError(f'No features for {label}; inspect diagnostics/{label}_failures.csv')
    return pd.DataFrame(rows)


def _coverage(latam,eligible,selected,c,p,root):
    rows=[]
    labels=['0-2','3-6','7-9','10-14','15-19','20-29','30-39','40-49','50-69','70-120']
    def groups(frame):
        return frame.age_group if 'age_group' in frame else frame.age.map(lambda age:next((g for g in labels if int(g.split('-')[0])<=age<=int(g.split('-')[1])+1-1e-9),'unknown'))
    for pop,before,after,measured in [('latam',latam,latam,c),('ffhq',eligible,selected,p)]:
        for group in labels:
            sub=measured[groups(measured).eq(group)]
            pose=sub.get('pose_quality_ok',pd.Series(True,index=sub.index)).fillna(False)
            expr=sub.get('expression_quality_ok',pd.Series(True,index=sub.index)).fillna(False)
            rows.append(dict(population=pop,age_group=group,n_eligible=int(groups(before).eq(group).sum()),
                n_selected=int(groups(after).eq(group).sum()),n_extracted=len(sub),n_pose_pass=int(pose.sum()),
                n_expression_pass=int(expr.sum()),n_both_pass=int((pose&expr).sum())))
    table=pd.DataFrame(rows);table.to_csv(root/'diagnostics/ffhq_age_coverage.csv',index=False)
    resolution=[]
    for pop,frame in [('latam',c),('ffhq',p)]:
        for role in ['original_iod_px','effective_iod_px']:
            v=frame[role].dropna()
            resolution.append(dict(population=pop,measurement=role,n=len(v),min=v.min(),p10=v.quantile(.1),median=v.median(),p90=v.quantile(.9),max=v.max(),
                n_below_target=int(frame.below_target_resolution.sum())))
    res=pd.DataFrame(resolution);res.to_csv(root/'diagnostics/resolution_matching.csv',index=False)
    return table,res


def close_module1(config,*,feature_backend=None,landmark_backend=None,age_backend=None):
    """Execute closure on explicit local data; returns tables, bundle and paths.

    All new artifacts go to reference_dir_out. previous_reference_dir is
    read-only configuration input, never a destination or implicit replacement.
    """
    from .population_aging import describe_population_aging
    from .module1_reports import zone_consistency,plot_face_zones,mean_faces
    validate_destination(config.reference_dir_out,config.previous_reference_dir)
    cfg=_configuration(config);root=Path(cfg.output_dir)
    if config.n_boot<0 or config.min_n_mean_face<1:raise ValueError('Invalid bootstrap/mean cell size')
    if landmark_backend is None:
        from .preprocessing import MediaPipeLandmarks
        landmark_backend=MediaPipeLandmarks(cfg.features.landmark_model_path)
    cfg,landmarks=prepare_closure_features(cfg,landmark_backend)
    cfg.features=replace(cfg.features,new_attributes=config.new_attributes if cfg.feature_backend=='handcrafted' else (),
        resolution_matching=config.resolution_matching,flip_tta=config.flip_tta)
    if isinstance(feature_backend,HandcraftedFeatureBackend) or feature_backend is None and cfg.feature_backend=='handcrafted':
        feature_backend=HandcraftedFeatureBackend(cfg.features.new_attributes)
    if feature_backend is None:raise ValueError('Embedding closure requires the train/proxy fitted DINO backend')
    state=dict(complete=False,landmarks_reviewed=False,anatomical_indices='candidate 116/345, 61/291, 234/454')
    status=root/'diagnostics/closure_status.json';status.write_text(json.dumps(state,indent=2))
    owned_age=False
    try:
        latam=load_colombia_manifest(cfg)
        eligible=load_ffhq_manifest(cfg.ffhq_aging_root,cfg.ffhq_labels_path,cfg.ffhq_min_conf,None,cfg.seed)
        selected=load_ffhq_manifest(cfg.ffhq_aging_root,cfg.ffhq_labels_path,cfg.ffhq_min_conf,cfg.ffhq_max_per_group,cfg.seed,sampling=cfg.ffhq_sampling)
        if config.mean_faces and config.estimate_ffhq_50_69 and selected.age_group.eq('50-69').any() and age_backend is None:
            from .ffhq_age_bands import prepare_band_age_backend
            # Validate/load the one small age backend BEFORE long extraction/bootstrap.
            age_backend=prepare_band_age_backend(config.assets_root,checkpoint_path=config.ffhq_age_checkpoint_path,allow_download=config.allow_download)
            owned_age=True
        store=FeatureStore(root/'feature_cache',cfg.features,landmarks,feature_backend,cfg.resume)
        c=_extract(latam,store,root,'latam');p=_extract(selected,store,root,'ffhq')
        component_cfg=replace(cfg,output_dir=root/'diagnostics/components')
        components=fit_reference_feature_frames(component_cfg,c,p,store.config_hash)
        reliability=pd.read_csv(component_cfg.output_dir/'attribute_reliability.csv')
        transform=fit_zone_transform(c,reliability,cfg.knots,config.zone_definitions) if config.zone_indices else {}
        c=apply_zone_transform(c,transform);p=apply_zone_transform(p,transform)
        cfg.features=replace(cfg.features,zone_transform=transform)
        final_store=FeatureStore(root/'feature_cache',cfg.features,landmarks,feature_backend,cfg.resume)
        c['feature_config_hash']=final_store.config_hash;p['feature_config_hash']=final_store.config_hash
        ref=fit_reference_feature_frames(cfg,c,p,final_store.config_hash)
        ref.metadata.update(module1_close=True,population='latam',zone_transform=transform,previous_reference_dir=str(config.previous_reference_dir) if config.previous_reference_dir else None,
            closure_config=asdict(config),landmark_anatomy_review='required; inspect figures/malar_landmarks_check.png')
        ref.save(root)
        original=c.copy()
        for key in [k for k in c if k.startswith('no_tta__')]:original['phi__'+key[8:]]=c[key]
        original=apply_zone_transform(original,transform)
        r0=reference_reliability(original.loc[longitudinal_mask(original)],[k[5:] for k in c if k.startswith('phi__')],cfg)
        r1=pd.read_csv(root/'attribute_reliability.csv')
        tta=r1.merge(r0[['attribute','icc','n_retest_pairs']],on='attribute',suffixes=('_tta','_no_tta'))
        tta.to_csv(root/'diagnostics/icc_tta_comparison.csv',index=False)
        print('\n1. ICC sin/con TTA (sin pares FairFace):\n'+tta.to_string(index=False),flush=True)
        coverage,resolution=_coverage(latam,eligible,selected,c,p,root)
        print('\n2. Cobertura de edades:\n'+coverage.to_string(index=False)+'\nResolución:\n'+resolution.to_string(index=False),flush=True)
        table=describe_population_aging(ref,config.transitions,config.n_boot,config.seed,root,include_cross_ci=True)
        # Hold zone signs/weights/scales fixed when isolating resolution effects.
        rawcfg=replace(cfg,output_dir=root/'sensitivity_no_resolution',features=replace(cfg.features,resolution_matching=False))
        rawstore=FeatureStore(root/'feature_cache',rawcfg.features,landmarks,feature_backend,cfg.resume)
        rc=_extract(latam,rawstore,root,'latam_no_resolution');rp=_extract(selected,rawstore,root,'ffhq_no_resolution')
        rawref=fit_reference_feature_frames(rawcfg,rc,rp,rawstore.config_hash)
        main_scale={a:(ref.mean[i],ref.sd[i]) for i,a in enumerate(ref.attributes)}
        fixed_mean=np.array([main_scale.get(a,(rawref.mean[i],rawref.sd[i]))[0] for i,a in enumerate(rawref.attributes)])
        fixed_sd=np.array([main_scale.get(a,(rawref.mean[i],rawref.sd[i]))[1] for i,a in enumerate(rawref.attributes)])
        rawref=fit_reference_frames(rc,rp,rawref.attributes,rawcfg.knots,fixed_mean,fixed_sd,rawref.icc,rawstore.config_hash,rawref.metadata)
        rawref.metadata['sensitivity_standardization']='Matched train scale for common attributes; frozen matched zone transform'
        rawref.metadata.update(module1_close=True,zone_transform=transform);rawref.save(rawcfg.output_dir)
        rawtable=describe_population_aging(rawref,config.transitions,config.n_boot,config.seed,rawcfg.output_dir,include_cross_ci=True)
        comparison=table.merge(rawtable,on=['attribute','source_age','target_age'],how='outer',suffixes=('_matched','_unmatched'))
        comparison.to_csv(root/'diagnostics/resolution_sensitivity.csv',index=False)
        print('\n3. Contrastes (igualado / sin igualar):\n'+comparison.to_string(index=False),flush=True)
        orientations={a:float(np.sign(ref.d_C([30],[70])[0,j])) or 1. for j,a in enumerate(ref.attributes)}
        for zone in transform:orientations[zone]=1.
        attributes=list(dict.fromkeys([*getattr(feature_backend,'attribute_ids',ATTRIBUTE_IDS),*transform]))
        consistency=zone_consistency(table,attributes,orientations);consistency.to_csv(root/'tables/zone_consistency.csv',index=False)
        ref.metadata['attribute_orientations']=orientations;ref.save(root)
        print('\n4. Consistencia:\n'+consistency.to_string(index=False),flush=True)
        if config.face_zone_figure:plot_face_zones(consistency,root/'figures')
        if config.mean_faces:
            if config.estimate_ffhq_50_69 and selected.age_group.eq('50-69').any():
                from .ffhq_age_bands import refine_ffhq_mean_ages
                selected=refine_ffhq_mean_ages(selected,cfg.features,landmarks,age_backend,root/'age_band_cache')
                selected.to_csv(root/'diagnostics/ffhq_50_69_age_predictions.csv',index=False)
                print('[FFHQ 50–69 únicamente] Edades para figuras:\n'+selected.loc[selected.age_group.eq('50-69'),'mean_face_age_status'].value_counts().to_string(),flush=True)
            mean_faces(cfg.features,latam,selected,landmarks,root/'figures',min_n=config.min_n_mean_face)
        paths={name:str(root/'figures'/name) for name in ['mean_faces_grid.png','aging_difference_maps.png','face_zones.png','malar_landmarks_check.png']}
        print('\n5. Figuras: '+json.dumps(paths,indent=2),flush=True)
        placebo=pd.DataFrame()
        if config.internal_placebo:
            from .module1_placebo import placebo_ffhq
            placebo=placebo_ffhq(ref,config.transitions,config.placebo_splits,config.n_boot,config.seed,root/'internal')
        state.update(complete=True,feature_config_hash=ref.feature_config_hash,output_dir=str(root))
        status.write_text(json.dumps(state,indent=2))
        return dict(references=ref,icc=tta,coverage=coverage,resolution=resolution,contrasts=table,
            resolution_sensitivity=comparison,zone_consistency=consistency,placebo=placebo,paths=paths)
    except BaseException as exc:
        state.update(complete=False,error=f'{type(exc).__name__}: {exc}');status.write_text(json.dumps(state,indent=2));raise
    finally:
        if owned_age and hasattr(age_backend,'model'):
            age_backend.model.to('cpu')
