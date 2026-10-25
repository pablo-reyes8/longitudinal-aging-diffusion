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
from .preprocessing import preprocess_image, is_grayscale, normalize_local_illumination,mirror_landmarks
from .config import FeatureConfig

ATTRIBUTE_IDS=('wrinkle_forehead','wrinkle_periocular','wrinkle_nasolabial','wrinkle_perioral',
    'skin_hf_texture','skin_ita','hair_gray','geom_jaw_sag','geom_lip_thickness',
    'geom_eye_aperture','geom_brow_eye','geom_lower_third')
COLOR_ATTRIBUTES={'skin_ita','hair_gray'}
EXPRESSION_ATTRIBUTES={'wrinkle_periocular','wrinkle_nasolabial','wrinkle_perioral',
    'geom_eye_aperture','geom_lip_thickness','geom_brow_eye','geom_malar_descent','geom_midface_height'}
ATTRIBUTE_REGISTRY={k:dict(id=k,requires_color=k in COLOR_ATTRIBUTES,
    region=k.split('_',1)[-1]) for k in ATTRIBUTE_IDS}
ATTRIBUTE_REGISTRY.update({k:dict(id=k,requires_color=False,region='midface')
    for k in ['geom_malar_descent','geom_midface_height']})


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


NEW_GEOMETRY=('geom_malar_descent','geom_midface_height')


def additional_geometry_attributes(l):
    """Signed malar height toward mouth; geometry in an eye-oriented frame."""
    out={k:np.nan for k in (*NEW_GEOMETRY,'geom_malar_descent_rel')}
    required=['left_eye','right_eye','malar','mouth_corners','zygomatic']
    if any(k not in l or not np.isfinite(l[k]).all() for k in required):return out
    le,re=l['left_eye'].mean(0),l['right_eye'].mean(0)
    eye=(le+re)/2;mouth=np.mean(l['mouth_corners'],axis=0)
    axis=re-le;iod=np.linalg.norm(axis);width=np.linalg.norm(l['zygomatic'][1]-l['zygomatic'][0])
    if min(iod,width)<1e-9:return out
    normal=np.array([-axis[1],axis[0]])/iod
    if (mouth-eye)@normal<0:normal=-normal
    depth=float(np.mean((l['malar']-eye)@normal));gap=float((mouth-eye)@normal)
    out.update(geom_malar_descent=depth/iod,geom_midface_height=float(np.linalg.norm(mouth-eye)/width),
        geom_malar_descent_rel=depth/gap if gap>1e-9 else np.nan)
    return out


def average_tta(original,flipped):
    values={};single=False
    for key in original.keys()|flipped.keys():
        a,b=original.get(key,np.nan),flipped.get(key,np.nan)
        valid=np.isfinite([a,b]);single|=bool(valid.sum()==1)
        values[key]=float(np.mean(np.array([a,b])[valid])) if valid.any() else np.nan
    return values,single


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
    def __init__(self,new_attributes=()):
        if set(new_attributes)-set(NEW_GEOMETRY):raise ValueError('Unknown additional geometric attributes')
        self.new_attributes=tuple(new_attributes)
        self.attribute_ids=ATTRIBUTE_IDS+self.new_attributes
        if self.new_attributes:self.cache_id='handcrafted-cheek-v2'
    def extract_aligned(self,image,landmarks,allow_color=True,*,normalize_color=False):
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
            color=normalize_local_illumination(rgb) if normalize_color else rgb
            if cheeks is not None and cheeks.any():out['phi__skin_ita']=ita_from_lab(rgb_to_lab(color)[cheeks])
            hair=masks.get('hair')
            if hair is not None and hair.any():
                val=color.max(-1)/255;sat=(color.max(-1)-color.min(-1))/np.maximum(color.max(-1),1)
                out['phi__hair_gray']=float(np.mean((sat[hair]<.15)&(val[hair]>.55)))
        out.update({'phi__'+k:v for k,v in geometry_attributes(landmarks).items()})
        if self.new_attributes:
            new=additional_geometry_attributes(landmarks)
            out.update({'phi__'+k:new[k] for k in self.new_attributes})
            out['diagnostic__geom_malar_descent_rel']=new['geom_malar_descent_rel']
        return out


class FeatureStore:
    def __init__(self,cache_dir,config=None,landmark_backend=None,feature_backend=None,resume=True):
        self.config=config or FeatureConfig();self.backend=feature_backend or HandcraftedFeatureBackend(self.config.new_attributes)
        if isinstance(self.backend,HandcraftedFeatureBackend) and self.backend.new_attributes!=tuple(self.config.new_attributes):
            self.backend=HandcraftedFeatureBackend(self.config.new_attributes)
        self.landmarks=landmark_backend
        self.root=Path(cache_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.resolver=ImageResolver();self.resume=resume
        configuration=asdict(self.config)
        # Preserve the exact legacy hash when new opt-in fields are disabled.
        for key,default in dict(new_attributes=(),resolution_matching=False,resolution_target_percentile=10.,
                target_iod_px=None,flip_tta=False,zone_transform={}).items():
            if configuration[key]==default or (key=='new_attributes' and not configuration[key]):configuration.pop(key)
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
            kwargs={'normalize_color':self.config.normalize_color_illumination} if isinstance(self.backend,HandcraftedFeatureBackend) else {}
            ids=getattr(self.backend,'attribute_ids',())
            reject=not prepared.covariates['pose_quality_ok'] or (not prepared.covariates['expression_quality_ok'] and
                any(a.startswith('embedding_') for a in ids))
            values={'phi__'+a:np.nan for a in ids} if reject and ids else self.backend.extract_aligned(prepared.image,prepared.landmarks,**kwargs)
            if self.config.flip_tta:
                base=dict(values)
                flipped={'phi__'+a:np.nan for a in ids} if reject and ids else self.backend.extract_aligned(
                    prepared.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),mirror_landmarks(prepared.landmarks,prepared.image.width),**kwargs)
                values,single=average_tta(base,flipped)
                values.update({'no_tta__'+k[5:]:v for k,v in base.items() if k.startswith('phi__')})
                values['tta_single']=single
            row={**values,**prepared.covariates}
        if not row.get('pose_quality_ok',True):
            row.update({k:np.nan for k in row if k.startswith('phi__')})
        elif not row.get('expression_quality_ok',True):
            # For embeddings expression cannot be isolated to selected coordinates.
            row.update({k:np.nan for k in row if k.startswith('phi__') and
                (k[5:] in EXPRESSION_ATTRIBUTES or k[5:].startswith('embedding_'))})
        for key in list(row):
            if key.startswith('no_tta__') and not np.isfinite(row.get('phi__'+key[8:],np.nan)):row[key]=np.nan
        if self.config.zone_transform:
            from .zones import zone_values
            row.update(zone_values(row,self.config.zone_transform))
        row={**row,'image_sha256':digest,'feature_config_hash':self.config_hash}
        temp=file.with_suffix('.tmp');temp.write_text(json.dumps(row,default=float,allow_nan=True));temp.replace(file)
        return row

    def frame(self,paths):
        return pd.DataFrame([self.extract(p) for p in paths]).set_index('image_sha256',drop=False)
