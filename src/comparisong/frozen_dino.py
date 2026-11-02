"""Read-only DINO/PCA loading for already closed module-1 references."""
from pathlib import Path
import numpy as np
from src.aging_bias.references import ReferenceBundle
from src.aging_bias.config import FeatureConfig


def validate_frozen_dino_files(reference_dir):
    directory=Path(reference_dir)/'dinov2'
    required=[directory/name for name in ['references.json','references.npz','embedding_pca.npz']]
    missing=[str(path) for path in required if not path.is_file()]
    if missing:raise FileNotFoundError('Frozen module-1 DINO references missing; finish its DINO step first. No refit performed: '+str(missing))
    return directory


def load_frozen_dino_backend(reference_dir,*,root,landmark_backend,cache_dir,allow_download=False,device='auto'):
    directory=validate_frozen_dino_files(reference_dir)
    ref=ReferenceBundle.load(directory,load_training_data=False)
    if not ref.metadata.get('module1_close'):
        raise ValueError('DINO bundle is not a closed module-1 reference; finish that step without refitting here')
    cfg=ref.metadata['config'];features=FeatureConfig(**cfg['features'])
    with np.load(directory/'embedding_pca.npz',allow_pickle=False) as data:
        encoder_id=str(data['encoder_id']);components=int(data['components'].shape[0])
    from src.aging_bias.embedding_features import DinoV2Encoder,CachedEncoder,EmbeddingFeatureBackend
    from src.aging_bias.features import FeatureStore
    if device=='auto':
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    encoder=DinoV2Encoder(model_id=encoder_id.split(':',1)[0],device=device,
        local_files_only=not allow_download,cache_dir=str(Path(root)/'huggingface'))
    try:
        cached=CachedEncoder(encoder,Path(cache_dir)/'dino_vectors')
        backend=EmbeddingFeatureBackend(cached,components).load(directory/'embedding_pca.npz')
        probe=FeatureStore(Path(cache_dir)/'features',features,landmark_backend,backend)
        if probe.config_hash!=ref.feature_config_hash:raise ValueError('Frozen DINO feature hash differs; restore the module-1 landmark asset/config. Never refit on test.')
    except BaseException:
        encoder.to('cpu');raise
    print(f'[DINO] Frozen PCA/reference loaded read-only; encoder resident on {device}. No train/FFHQ scan.',flush=True)
    return ref,dict(feature_config=features,feature_backend=backend,landmark_backend=landmark_backend,embedding_device=str(device))
