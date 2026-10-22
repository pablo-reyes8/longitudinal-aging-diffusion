"""Refit reliability sensitivities without image extraction or diffusion generation."""
from dataclasses import replace
from pathlib import Path
import pandas as pd
from .config import FeatureConfig,ReferenceConfig
from .references import ReferenceBundle,fit_reference_feature_frames


def load_saved_reference_frames(source,excluded_test_ids=()):
    root=Path(source)
    ref=ReferenceBundle.load(root,load_training_data=False)
    frames=[pd.read_parquet(root/name) for name in ['train_features.parquet','proxy_features.parquet']]
    for frame in frames:
        if frame.empty or 'feature_config_hash' not in frame or not frame.feature_config_hash.eq(ref.feature_config_hash).all():
            raise ValueError('Saved feature rows do not match their reference hash')
    c,p=frames
    declared=set(map(str,ref.metadata.get('train_identities',[])))
    if set(c.identity_id.astype(str))!=declared:raise ValueError('Saved train identities do not match reference provenance')
    if declared&set(map(str,excluded_test_ids)):raise ValueError('Test identity leakage in cached reference frames')
    raw=dict(ref.metadata['config']);raw['features']=FeatureConfig(**raw['features'])
    return ref,ReferenceConfig(**raw),c,p


def refit_cached_references(source,output_dir,*,min_icc=.30,min_homog_icc=.50,excluded_test_ids=()):
    ref,cfg,c,p=load_saved_reference_frames(source,excluded_test_ids)
    if Path(source).resolve()==Path(output_dir).resolve():
        raise ValueError('Sensitivity references must not overwrite the original module 1 bundle')
    cfg=replace(cfg,output_dir=Path(output_dir),min_icc=min_icc,min_homog_icc=min_homog_icc,weight_attributes_by_icc=True)
    result=fit_reference_feature_frames(cfg,c,p,ref.feature_config_hash)
    result.metadata['sensitivity']=dict(source=str(source),min_icc_pi=min_icc,min_icc_homog=min_homog_icc,
        note='Exploratory reliability sensitivity; not a threshold selected to pass test anchors')
    result.save(output_dir)
    return result


def preflight_dino_inputs(source,excluded_test_ids=()):
    _,_,c,p=load_saved_reference_frames(source,excluded_test_ids)
    missing=[str(path) for frame in [c,p] for path in frame.image_path if not Path(path).is_file()]
    if missing:raise FileNotFoundError('DINO requires the module 1 original train/FFHQ images. '
        f'Restore those paths (no picking rerun needed). First missing: {missing[:3]}')
    return c,p


def build_cached_embedding_references(source,output_dir,*,root,landmark_backend,
                                      allow_download=False,excluded_test_ids=(),encoder=None,device='auto'):
    """Mandatory robustness setup: GPU-preferred DINO and cached PCA, never test-fit.

    Uses original images listed by module 1. Feature rows alone cannot recover
    DINO vectors; missing originals fail with instructions before model setup.
    """
    from .features import FeatureStore
    from src.comparisong.evaluation_backends_v2 import prepare_embedding_reference_backend
    _,cfg,c,p=load_saved_reference_frames(source,excluded_test_ids)
    cfg=replace(cfg,output_dir=Path(output_dir),feature_backend='embedding',min_icc=.30,
        min_homog_icc=.50,weight_attributes_by_icc=True)
    preflight_dino_inputs(source,excluded_test_ids)
    backend=prepare_embedding_reference_backend(cfg,root=root,landmark_backend=landmark_backend,
        allow_download=allow_download,encoder=encoder,reference_frames_path=source,excluded_test_ids=excluded_test_ids,device=device)
    try:
        store=FeatureStore(cfg.output_dir/'feature_cache',cfg.features,backend['landmark_backend'],backend['feature_backend'],cfg.resume)
        extracted=[];failures=[]
        for label,frame in [('train',c),('proxy',p)]:
            rows=[]
            for i,row in enumerate(frame.to_dict('records')):
                metadata={k:v for k,v in row.items() if not k.startswith('phi__')}
                try:rows.append({**metadata,**store.extract(row['image_path'])})
                except Exception as exc:failures.append(dict(split=label,image_path=row['image_path'],error=f'{type(exc).__name__}: {exc}'))
                if (i+1)%100==0:print(f'[DINO references] {label}: {i+1}/{len(frame)}',flush=True)
            extracted.append(pd.DataFrame(rows))
        pd.DataFrame(failures,columns=['split','image_path','error']).to_csv(cfg.output_dir/'reference_failures.csv',index=False)
        ref=fit_reference_feature_frames(cfg,*extracted,store.config_hash)
        ref.metadata['embedding_device']=backend['embedding_device'];ref.save(cfg.output_dir)
        return ref,backend
    except BaseException:
        active_encoder=getattr(backend['feature_backend'],'encoder',None)
        if hasattr(active_encoder,'to'):active_encoder.to('cpu')
        raise
