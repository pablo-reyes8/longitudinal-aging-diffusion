import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tests.aging_bias.test_features import landmarks


def test_embedding_setup_fits_train_proxy_in_batches_reuses_pca_and_rejects_test(tmp_path):
    from src.aging_bias.config import ReferenceConfig, FeatureConfig
    from src.comparisong.evaluation_backends_v2 import prepare_embedding_reference_backend
    class Encoder:
        cache_id='synthetic-encoder'
        def encode_batch(self,images):
            assert len(images)<=3
            if self.disabled:raise AssertionError('Cached setup must not execute encoder again')
            values=np.array([np.asarray(im).mean((0,1)) for im in images])
            return np.c_[values,values[:,0]**2/255]
        disabled=False
    train=tmp_path/'train';proxy=tmp_path/'proxy';proxy.mkdir()
    for i in range(8):
        folder=train/f'id_{i}';folder.mkdir(parents=True)
        for age in [30,50]:
            Image.new('RGB',(256,256),(40+i*12,age*2,100+i*4)).save(folder/f'{age}.png')
    for i in range(8):Image.new('RGB',(256,256),(40+i*10,140,70+i*7)).save(proxy/f'{i:05d}.png')
    pd.DataFrame(dict(image_number=range(8),age_group=['30-39']*4+['50-69']*4,
                      age_group_confidence=[1.]*8)).to_csv(proxy/'ffhq_aging_labels.csv',index=False)
    config=ReferenceConfig(train,proxy,tmp_path/'refs',train_only=True,feature_backend='embedding',
        features=FeatureConfig(),embedding_components=2,embedding_fit_max_images=12,embedding_batch_size=3)
    encoder=Encoder()
    result=prepare_embedding_reference_backend(config,encoder=encoder,landmark_backend=lambda image:landmarks())
    assert result['feature_backend'].components.shape==(2,4)
    assert (tmp_path/'refs/embedding_pca.npz').is_file()
    encoder.disabled=True
    cached=prepare_embedding_reference_backend(config,encoder=encoder,landmark_backend=lambda image:landmarks())
    assert cached['feature_backend'].cache_id==result['feature_backend'].cache_id
    config.excluded_test_ids=('id_0',)
    with pytest.raises(ValueError,match='leakage'):
        prepare_embedding_reference_backend(config,encoder=encoder,landmark_backend=lambda image:landmarks())


def test_embedding_pca_rejects_mismatched_encoder(tmp_path):
    from src.aging_bias.embedding_features import EmbeddingFeatureBackend
    class Encoder:
        cache_id='one'
        def encode_batch(self,images):return np.asarray(images)
    backend=EmbeddingFeatureBackend(Encoder(),2).fit(np.random.default_rng(2).normal(size=(10,4)).tolist(),splits=['train']*10)
    backend.save(tmp_path/'pca.npz')
    encoder=Encoder();encoder.cache_id='different-weights'
    with pytest.raises(ValueError,match='encoder'):
        EmbeddingFeatureBackend(encoder,2).load(tmp_path/'pca.npz')


def test_dino_setup_prefers_gpu_and_offloads_on_fit_error(tmp_path,monkeypatch):
    from src.aging_bias.config import ReferenceConfig
    import src.aging_bias.references as refs
    import src.aging_bias.embedding_features as embeddings
    from src.comparisong.evaluation_backends_v2 import prepare_embedding_reference_backend
    import torch
    devices=[]
    class Encoder:
        def __init__(self,**kwargs):
            self.device=kwargs['device'];devices.append(self.device)
        def to(self,device):devices.append(device);self.device=device
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(refs,'load_colombia_manifest',lambda *a:pd.DataFrame())
    monkeypatch.setattr(refs,'load_ffhq_manifest',lambda *a:pd.DataFrame())
    monkeypatch.setattr(embeddings,'DinoV2Encoder',Encoder)
    def fail(*a):raise ValueError('synthetic PCA failure')
    monkeypatch.setattr(embeddings,'fit_reference_embedding',fail)
    cfg=ReferenceConfig(tmp_path,tmp_path,tmp_path/'refs',train_only=True)
    with pytest.raises(ValueError,match='synthetic PCA'):
        prepare_embedding_reference_backend(cfg,landmark_backend=object())
    assert devices==['cuda','cpu']
