"""Model-estimated ages ONLY for splitting FFHQ's broad 50–69 mean-face band."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .features import stable_hash
from .preprocessing import preprocess_image
from src.quantitative_metrics.image_io import ImageResolver


def prepare_band_age_backend(root,*,checkpoint_path=None,allow_download=False,device='auto'):
    from src.comparisong.evaluation_backends_v2 import FairFaceAgeBackend,_download
    if device=='auto':
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    path=Path(checkpoint_path or Path(root)/'fairface_resnet34.pt')
    _download('https://drive.google.com/file/d/11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB/view',path,allow_download)
    return FairFaceAgeBackend(path,device=device)


def refine_ffhq_mean_ages(frame,features,landmarks,backend,cache_dir,*,batch_size=16):
    """Preserve official ages. Raw predictions outside [50,70) are reported,
    not clipped into an invented valid age. They stay excluded from fine means.
    """
    if batch_size<1:raise ValueError('age batch_size must be positive')
    result=frame.copy();result['mean_face_age']=result.age
    result['mean_face_age_source']='official_group';result['mean_face_age_status']='not_refined'
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    resolver=ImageResolver();pending=[]
    config=asdict(features);config.pop('zone_transform',None)
    signature=stable_hash(dict(features=config,backend=backend.cache_id,version='ffhq50-69-age-v1'))
    def assign(ix,row):
        age=row.get('age_pred',np.nan)
        result.loc[ix,'mean_face_age']=age if row.get('status')=='ok' and 50<=age<70 else np.nan
        result.loc[ix,'mean_face_age_source']='model_prediction'
        result.loc[ix,'mean_face_age_status']=row.get('status','backend_error') if not np.isfinite(age) or 50<=age<70 else 'outside_official_group'
        result.loc[ix,'age_pred_raw']=age
    def flush():
        if not pending:return
        try:
            values=np.asarray(backend.predict_batch([image for _,image,_ in pending]),float)
            if len(values)!=len(pending):raise ValueError('Age backend length mismatch')
            predictions=[dict(age_pred=float(age),status='ok' if np.isfinite(age) else 'nonfinite') for age in values]
        except Exception as exc:predictions=[dict(age_pred=np.nan,status='backend_error',error=str(exc)) for _ in pending]
        for (ix,_,path),row in zip(pending,predictions):
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row));temp.replace(path);assign(ix,row)
        pending.clear()
    selected=result[result.age_group.eq('50-69')]
    for number,(ix,row) in enumerate(selected.iterrows(),1):
        try:
            image,digest=resolver.load(row.image_path);resolver._cache.clear();path=root/(digest+'_'+signature+'.json')
            if path.is_file():assign(ix,json.loads(path.read_text()));continue
            prep=preprocess_image(image,landmarks,features)
            if not prep.covariates['pose_quality_ok'] or not prep.covariates['expression_quality_ok']:
                assign(ix,dict(age_pred=np.nan,status='quality_excluded'));continue
            pending.append((ix,prep.image,path))
            if len(pending)>=batch_size:flush()
        except Exception as exc:assign(ix,dict(age_pred=np.nan,status='image_error',error=str(exc)))
        if number%100==0:print(f'[FFHQ 50–69 age estimates] {number}/{len(selected)}',flush=True)
    flush();return result
