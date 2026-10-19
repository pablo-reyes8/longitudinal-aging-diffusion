"""Common similarity alignment and capture harmonization, never a silent resize fallback."""
from dataclasses import dataclass
from typing import Protocol
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace
from .config import FeatureConfig


class LandmarkBackend(Protocol):
    def __call__(self, image: Image.Image) -> dict[str,np.ndarray]: ...


def is_grayscale(image, gray_tol=2.):
    a=np.asarray(image.convert('RGB') if isinstance(image,Image.Image) else image,float)
    return bool(np.mean(np.std(a,axis=-1))<gray_tol)


class MediaPipeLandmarks:
    """CPU task API (not deprecated mp.solutions); download is explicit in prepare_paper_bundle."""
    cache_id='mediapipe-face-landmarker-quality-v2'
    def __init__(self,model_path):
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError('FaceLandmarker task missing; call prepare_paper_bundle or set FeatureConfig.landmark_model_path')
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp=mp
        self.detector=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),num_faces=1,
            output_face_blendshapes=True,output_facial_transformation_matrixes=True))

    def __call__(self,image):
        a=np.asarray(image.convert('RGB'))
        result=self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=a))
        if not result.face_landmarks: raise ValueError('No face landmarks detected')
        points=np.array([[v.x*image.width,v.y*image.height] for v in result.face_landmarks[0]])
        groups={'left_eye':[33,160,158,133,153,144], 'right_eye':[362,385,387,263,373,380],
                'mouth':[61,40,0,270,291,321,17,91],
                'jaw':[234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454],
                'left_brow':[70,63,105,66,107], 'right_brow':[336,296,334,293,300],
                'nose':[1], 'chin':[152]}
        scores={c.category_name:float(c.score) for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
        yaw=np.nan
        if result.facial_transformation_matrixes:
            matrix=np.asarray(result.facial_transformation_matrixes[0],float)[:3,:3]
            u,_,vt=np.linalg.svd(matrix);rotation=u@vt
            yaw=float(np.degrees(np.arctan2(-rotation[2,0],np.hypot(rotation[0,0],rotation[1,0]))))
        return {**{k:points[v] for k,v in groups.items()},'_quality':dict(yaw_deg=yaw,
            smile_score=max(scores.get('mouthSmileLeft',np.nan),scores.get('mouthSmileRight',np.nan)),
            jaw_open_score=scores.get('jawOpen',np.nan))}

    def close(self): self.detector.close()


@dataclass
class PreparedImage:
    image: Image.Image
    landmarks: dict
    covariates: dict


def normalize_local_illumination(rgb):
    """Remove smooth multiplicative shading, retaining image median/color ratios.

    This is NOT white balancing or mapping all skin to a common tone. Global
    exposure/white-balance differences cannot be identified without calibration;
    ITA remains exploratory even after this local correction.
    """
    rgb=np.asarray(rgb,float)
    luminance=rgb@np.array([.2126,.7152,.0722])
    field=gaussian_filter(luminance,max(1,min(luminance.shape)*.08))
    positive=field[field>1]
    if not len(positive):return rgb.copy()
    gain=np.clip(np.median(positive)/np.maximum(field,1),.5,2)
    return np.clip(rgb*gain[...,None],0,255)


def preprocess_image(image, landmark_backend, config=None):
    config=config or FeatureConfig()
    if config.crop_size<32 or config.harmonize_sigma<0: raise ValueError('Invalid crop size or blur')
    image=image.convert('RGB');a=np.asarray(image,float)/255
    gray=a.mean(axis=2)
    cov=dict(is_grayscale=is_grayscale(image,config.gray_tol),
             log_min_side=float(np.log(min(image.size))),blur_laplacian_var=float(laplace(gray).var()))
    landmarks=landmark_backend(image)
    cov.update(landmarks.get('_quality',{}))
    for name in ['yaw_deg','smile_score','jaw_open_score']:cov.setdefault(name,np.nan)
    cov['pose_quality_ok']=bool(config.max_abs_yaw is None or
        (np.isfinite(cov['yaw_deg']) and abs(cov['yaw_deg'])<config.max_abs_yaw))
    cov['expression_quality_ok']=all(limit is None or (np.isfinite(cov[name]) and cov[name]<=limit)
        for name,limit in [('smile_score',config.max_smile_score),('jaw_open_score',config.max_jaw_open_score)])
    src=np.array([landmarks['left_eye'].mean(0),landmarks['right_eye'].mean(0),landmarks['mouth'].mean(0)])
    dst=np.array([[.32,.40],[.68,.40],[.5,.68]])*config.crop_size
    # Isotropic Procrustes: y = scale * x @ R + offset, no shear.
    x=src-src.mean(0);y=dst-dst.mean(0)
    u,s,vt=np.linalg.svd(x.T@y);r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt;s[-1]*=-1
    denom=(x*x).sum()
    if denom<1e-12:raise ValueError('Degenerate landmarks')
    scale=s.sum()/denom;offset=dst.mean(0)-scale*src.mean(0)@r
    inv=np.linalg.inv(scale*r);translation=-offset@inv
    coeff=(inv[0,0],inv[1,0],translation[0],inv[0,1],inv[1,1],translation[1])
    aligned=image.transform((config.crop_size,config.crop_size),Image.Transform.AFFINE,coeff,
                            resample=Image.Resampling.BICUBIC)
    pixels=np.asarray(aligned,float)
    if config.harmonize_sigma:
        pixels=gaussian_filter(pixels,(config.harmonize_sigma,config.harmonize_sigma,0))
    aligned=Image.fromarray(np.uint8(np.clip(pixels,0,255)))
    transformed={k:scale*np.asarray(v)@r+offset for k,v in landmarks.items() if not k.startswith('_')}
    return PreparedImage(aligned,transformed,cov)
