"""Classical facial descriptors. Landmarks are learned, descriptor values are not.

Heuristic regions are not semantic segmentations: hair/skin measurements must
be validated with ICC and real anchors before interpreting population effects.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from src.quantitative_metrics.image_io import ImageResolver
from .preprocessing import preprocess_image, is_grayscale
from .config import FeatureConfig

ATTRIBUTE_IDS=('wrinkle_forehead','wrinkle_periocular','wrinkle_nasolabial','wrinkle_perioral',
    'skin_hf_texture','skin_ita','hair_gray','geom_jaw_sag','geom_lip_thickness',
    'geom_eye_aperture','geom_brow_eye','geom_lower_third')
COLOR_ATTRIBUTES={'skin_ita','hair_gray'}
ATTRIBUTE_REGISTRY={k:dict(id=k,requires_color=k in COLOR_ATTRIBUTES,
    region=k.split('_',1)[-1]) for k in ATTRIBUTE_IDS}


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,allow_nan=True).encode()).hexdigest()


def geometry_attributes(l):
    out={k:np.nan for k in ATTRIBUTE_IDS if k.startswith('geom_')}
    required=['left_eye','right_eye','mouth','jaw','left_brow','right_brow','chin']
    if any(k not in l or not np.isfinite(l[k]).all() for k in required):return out
    dist=lambda a,b:float(np.linalg.norm(np.asarray(a)-b))
    le,re=l['left_eye'].mean(0),l['right_eye'].mean(0);inter=dist(le,re)
    mouth=l['mouth'];jaw=l['jaw'];width=dist(mouth[0],mouth[4])
    if min(inter,width)<1e-9:return out
    out['geom_jaw_sag']=np.mean([np.min(np.linalg.norm(jaw-c,axis=1)) for c in mouth[[0,4]]])/inter
    out['geom_lip_thickness']=dist(mouth[2],mouth[6])/width
    def ear(e):return (dist(e[1],e[5])+dist(e[2],e[4]))/(2*dist(e[0],e[3]))
    out['geom_eye_aperture']=np.mean([ear(l['left_eye']),ear(l['right_eye'])])
    out['geom_brow_eye']=(dist(l['left_brow'].mean(0),le)+dist(l['right_brow'].mean(0),re))/(2*inter)
    out['geom_lower_third']=dist(mouth.mean(0),l['chin'].mean(0))/dist(jaw[0],jaw[-1])
    return out


def rgb_to_lab(rgb):
    v=np.asarray(rgb,float)/255
    v=np.where(v>.04045,((v+.055)/1.055)**2.4,v/12.92)
    xyz=v@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.072175],[.0193339,.119192,.9503041]]).T
    xyz/=np.array([.95047,1.,1.08883])
    f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)


def ita_from_lab(lab):
    lab=np.asarray(lab,float)
    with np.errstate(divide='ignore',invalid='ignore'):
        v=np.degrees(np.arctan((lab[...,0]-50)/lab[...,2]))
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def region_masks(shape,l):
    h,w=shape;out={}
    if not all(k in l for k in ['left_eye','right_eye','mouth']):return out
    le=l['left_eye'].mean(0);re=l['right_eye'].mean(0);m=l['mouth'].mean(0)
    eye=(le+re)/2;inter=np.linalg.norm(re-le)
    if inter<=0:return out
    def box(x,y,rx,ry):
        yy,xx=np.ogrid[:h,:w]
        return (abs(xx-x)<=rx)&(abs(yy-y)<=ry)
    out['forehead']=box(eye[0],eye[1]-.38*inter,.65*inter,.21*inter)
    out['periocular']=box(le[0],le[1]+.09*inter,.36*inter,.19*inter)|box(re[0],re[1]+.09*inter,.36*inter,.19*inter)
    out['nasolabial']=box(m[0]-.28*inter,m[1]-.18*inter,.13*inter,.26*inter)|box(m[0]+.28*inter,m[1]-.18*inter,.13*inter,.26*inter)
    out['perioral']=box(m[0],m[1],.46*inter,.25*inter)&~box(m[0],m[1],.34*inter,.12*inter)
    out['cheeks']=box(le[0],(eye[1]+m[1])/2,.22*inter,.20*inter)|box(re[0],(eye[1]+m[1])/2,.22*inter,.20*inter)
    out['hair']=box(eye[0],eye[1]-.8*inter,.72*inter,.16*inter)
    return out


class HandcraftedFeatureBackend:
    cache_id='handcrafted-v1'
    attribute_ids=ATTRIBUTE_IDS
    def extract_aligned(self,image,landmarks,allow_color=True):
        rgb=np.asarray(image.convert('RGB'),float);gray=rgb@np.array([.2126,.7152,.0722])/255
        local_mean=gaussian_filter(gray,5)
        local_sd=np.sqrt(np.maximum(gaussian_filter((gray-local_mean)**2,5),0))
        contrast=(gray-local_mean)/(local_sd+.08)
        # Multidirectional ridge curvature (squared Hessian energy).
        xx=gaussian_filter(contrast,1.2,order=(0,2));yy=gaussian_filter(contrast,1.2,order=(2,0))
        xy=gaussian_filter(contrast,1.2,order=(1,1));ridge=xx*xx+yy*yy+2*xy*xy
        masks=region_masks(gray.shape,landmarks)
        out={'phi__'+k:np.nan for k in ATTRIBUTE_IDS}
        for name in ['forehead','periocular','nasolabial','perioral']:
            mask=masks.get(name)
            if mask is not None and mask.sum()>=8:out['phi__wrinkle_'+name]=float(ridge[mask].mean())
        cheeks=masks.get('cheeks')
        if cheeks is not None and cheeks.sum()>=8:
            out['phi__skin_hf_texture']=float(np.var((contrast-gaussian_filter(contrast,1))[cheeks]))
        if allow_color and not is_grayscale(image):
            if cheeks is not None and cheeks.any():out['phi__skin_ita']=ita_from_lab(rgb_to_lab(rgb)[cheeks])
            hair=masks.get('hair')
            if hair is not None and hair.any():
                val=rgb.max(-1)/255;sat=(rgb.max(-1)-rgb.min(-1))/np.maximum(rgb.max(-1),1)
                out['phi__hair_gray']=float(np.mean((sat[hair]<.15)&(val[hair]>.55)))
        out.update({'phi__'+k:v for k,v in geometry_attributes(landmarks).items()})
        return out


class FeatureStore:
    def __init__(self,cache_dir,config=None,landmark_backend=None,feature_backend=None,resume=True):
        self.config=config or FeatureConfig();self.backend=feature_backend or HandcraftedFeatureBackend()
        self.landmarks=landmark_backend
        self.root=Path(cache_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.resolver=ImageResolver();self.resume=resume
        configuration=asdict(self.config)
        task=configuration.pop('landmark_model_path',None)
        if task and Path(task).is_file():configuration['landmark_model_sha256']=hashlib.sha256(Path(task).read_bytes()).hexdigest()
        else:configuration['landmark_model_sha256']=None
        self.config_hash=stable_hash(dict(config=configuration,backend=self.backend.cache_id,
            landmarks=getattr(landmark_backend,'cache_id',self.config.landmark_backend_id)))

    def extract(self,path):
        image,digest=self.resolver.load(path)
        self.resolver._cache.clear()
        file=self.root/(digest+'_'+self.config_hash+'.json')
        if self.resume and file.exists():return json.loads(file.read_text())
        if hasattr(self.backend,'extract'):
            row=self.backend.extract(image)
        else:
            if self.landmarks is None:
                from .preprocessing import MediaPipeLandmarks
                self.landmarks=MediaPipeLandmarks(self.config.landmark_model_path)
            prepared=preprocess_image(image,self.landmarks,self.config)
            row={**self.backend.extract_aligned(prepared.image,prepared.landmarks),**prepared.covariates}
        row={**row,'image_sha256':digest,'feature_config_hash':self.config_hash}
        temp=file.with_suffix('.tmp');temp.write_text(json.dumps(row,default=float,allow_nan=True));temp.replace(file)
        return row

    def frame(self,paths):
        return pd.DataFrame([self.extract(p) for p in paths]).set_index('image_sha256',drop=False)
