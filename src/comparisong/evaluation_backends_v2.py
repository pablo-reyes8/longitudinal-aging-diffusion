"""Explicit CPU evaluation setup. No imports/downloads until called by a notebook."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np


class FairFaceAgeBackend:
    """Official ResNet34 18-logit head; only age logits 9:18 are consumed.

    70+ has no midpoint: 85 is an explicit configurable convention, not an
    observed age. Relative errors reduce, but do NOT guarantee cancellation
    of estimator bias (pose/domain effects may remain).
    """
    name='FairFace'
    def __init__(self,checkpoint_path,device='cpu',age_midpoints=(1,6,14.5,24.5,34.5,44.5,54.5,64.5,85)):
        import torch
        from torchvision.models import resnet34
        from torchvision import transforms
        self.device=torch.device(device);self.model=resnet34(weights=None)
        self.model.fc=torch.nn.Linear(self.model.fc.in_features,18)
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True);self.model.eval().requires_grad_(False).to(self.device)
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.midpoints=torch.tensor(age_midpoints,device=self.device)
        self.cache_id='fairface:'+file_hash(checkpoint_path)+':'+str(tuple(age_midpoints))

    def predict_batch(self,images):
        import torch
        with torch.inference_mode():
            inputs=torch.stack([self.transform(x.convert('RGB')) for x in images]).to(self.device)
            probs=self.model(inputs)[:,9:18].softmax(-1)
            return (probs*self.midpoints).sum(-1).cpu().numpy()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _download(url,path,allow_download):
    path=Path(path)
    if path.is_file() and path.stat().st_size>1024:return path
    if not allow_download:raise FileNotFoundError(f'Missing evaluation asset {path}; enable allow_download or supply a local file')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part')
    if 'drive.google.com' in url:
        import gdown
        result=gdown.download(url,str(temp),fuzzy=True,quiet=False)
        if result is None:raise RuntimeError('FairFace download failed; supply fairface_checkpoint_path or a verified checkpoint URL')
    else:
        with urllib.request.urlopen(url,timeout=120) as source,temp.open('wb') as target:
            import shutil
            shutil.copyfileobj(source,target)
    if temp.stat().st_size<=1024:raise ValueError(f'Invalid downloaded asset: {url}')
    temp.replace(path);return path


def prepare_reference_backend(root='/content/evaluation_models_v2',*,feature_config=None,allow_download=False):
    """Module 1 needs only landmarks, not AdaFace, DEX, Inception or FairFace."""
    from src.aging_bias.config import FeatureConfig
    from src.aging_bias.features import HandcraftedFeatureBackend
    from src.aging_bias.preprocessing import MediaPipeLandmarks
    features=feature_config or FeatureConfig();task=Path(features.landmark_model_path or Path(root)/'face_landmarker.task')
    _download('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',task,allow_download)
    features.landmark_model_path=str(task)
    return dict(feature_config=features,feature_backend=HandcraftedFeatureBackend(),landmark_backend=MediaPipeLandmarks(task))


def prepare_embedding_reference_backend(reference_config,*,root='/content/evaluation_models_v2',
                                        encoder=None,landmark_backend=None,allow_download=False,
                                        reference_frames_path=None,excluded_test_ids=(),device='auto'):
    """Explicit DINOv2 setup and train/proxy PCA, safe to resume.

    Leakage validation happens BEFORE constructing/downloading any model.
    Call build_reference_bundle with the returned backends, as for handcrafted.
    """
    from src.aging_bias.references import load_colombia_manifest,load_ffhq_manifest
    from src.aging_bias.embedding_features import DinoV2Encoder,fit_reference_embedding
    config=reference_config
    if config.module1_close and config.ffhq_sampling is None:
        from dataclasses import replace
        from src.aging_bias.config import FFHQSamplingConfig
        config=replace(config,ffhq_sampling=FFHQSamplingConfig())
    if reference_frames_path is None:
        colombia=load_colombia_manifest(config)
        proxy=load_ffhq_manifest(config.ffhq_aging_root,config.ffhq_labels_path,
            config.ffhq_min_conf,config.ffhq_max_per_group,config.seed,
            **({'sampling':config.ffhq_sampling} if config.ffhq_sampling is not None else {}))
    else:
        from src.aging_bias.cached_references import load_saved_reference_frames
        _,_,colombia,proxy=load_saved_reference_frames(reference_frames_path,excluded_test_ids)
    if landmark_backend is None:
        prepared=prepare_reference_backend(root,feature_config=config.features,allow_download=allow_download)
        landmark_backend=prepared['landmark_backend']
    if config.module1_close:
        from src.aging_bias.module1_close import prepare_closure_features
        config,landmark_backend=prepare_closure_features(config,landmark_backend)
    if encoder is None:
        if device=='auto':
            import torch
            device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[DINOv2] Encoder resident on {device}; bounded batches, PCA on CPU.',flush=True)
        encoder=DinoV2Encoder(local_files_only=not allow_download,device=device,cache_dir=str(Path(root)/'huggingface'))
    try:
        fitted=fit_reference_embedding(config,colombia,proxy,encoder,landmark_backend)
    except BaseException:
        if hasattr(encoder,'to'):encoder.to('cpu')
        raise
    return dict(feature_config=config.features,feature_backend=fitted,landmark_backend=landmark_backend,
        embedding_device=str(getattr(encoder,'device',device)))


def prepare_paper_bundle(root='/content/evaluation_models_v2',*,feature_config=None,
                         fairface_checkpoint_path=None,fairface_url=None,
                         allow_download=False,include_dex=True,include_kid=True,legacy_metrics_config=None):
    """Prepare/load once on CPU. Called explicitly, never from legacy functions.

    Existing AdaFace/DEX/KID paths can be reused via legacy_metrics_config.
    No training dependencies (py-feat, MiVOLO, timm) are needed.
    """
    from src.aging_bias.config import FeatureConfig
    from src.aging_bias.features import HandcraftedFeatureBackend
    from src.aging_bias.preprocessing import MediaPipeLandmarks
    from src.quantitative_metrics.backends import AdaFaceAligner,AdaFaceEncoder,DexAgeEstimator,TorchMetricsKid
    from src.quantitative_metrics.api import build_default_metrics_config
    from src.quantitative_metrics.assets import DOWNLOAD_SPECS,_ensure_download,_clone_adaface_repository,_valid_adaface_repository
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    reference_backend=prepare_reference_backend(root,feature_config=feature_config,allow_download=allow_download)
    features=reference_backend['feature_config'];task=Path(features.landmark_model_path)
    fairface=Path(fairface_checkpoint_path or root/'fairface_resnet34.pt')
    _download(fairface_url or 'https://drive.google.com/file/d/11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB/view',fairface,allow_download)
    config=build_default_metrics_config(root)
    config.update(legacy_metrics_config or {})
    repo=Path(config['adaface_repo_path'])
    if not _valid_adaface_repository(repo):
        if not allow_download:raise FileNotFoundError(f'AdaFace inference repository missing: {repo}')
        _clone_adaface_repository(repo)
    selected=['adaface_checkpoint_path']
    if include_dex:selected+=['dex_prototxt_path','dex_checkpoint_path']
    if include_kid:selected+=['kid_inception_weights_path']
    for key in selected:
        if allow_download:_ensure_download(Path(config[key]),DOWNLOAD_SPECS[key])
        elif not Path(config[key]).is_file():raise FileNotFoundError(config[key])
    bundle=dict(**reference_backend,
        aligner=AdaFaceAligner(repo,device='cpu'),
        identity_encoder=AdaFaceEncoder(repo,config['adaface_checkpoint_path'],device='cpu'),
        independent_age_backend=FairFaceAgeBackend(fairface,device='cpu'))
    if include_dex:bundle['age_estimator']=DexAgeEstimator(config['dex_prototxt_path'],config['dex_checkpoint_path'],device='cpu')
    if include_kid:bundle['kid_metric']=TorchMetricsKid(device='cpu',inception_weights_path=config['kid_inception_weights_path'])
    bundle['metadata']=dict(cache_id='paper-evaluation-v2',fairface_sha256=file_hash(fairface),
        landmark_sha256=file_hash(task),adaface_sha256=file_hash(config['adaface_checkpoint_path']),
        independent_age='FairFace',fairface_age_midpoints=[1,6,14.5,24.5,34.5,44.5,54.5,64.5,85],
        independent_age_alignment='AdaFace aligned crop, resized224; differs from original dlib crop, validate on real anchors')
    for key in selected:bundle['metadata'][key+'_sha256']=file_hash(config[key])
    return bundle

