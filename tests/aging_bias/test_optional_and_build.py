import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_real_reference_builder_with_stub_features_and_retests(tmp_path):
    from src.aging_bias import ReferenceConfig,build_reference_bundle
    class Backend:
        cache_id='train-only-toy-v1'
        def extract(self,image):
            a=np.asarray(image)[0,0].astype(float)
            return {'phi__signal':a[0]+a[1]*.01,'is_grayscale':False,'log_min_side':3.,'blur_laplacian_var':0.}
    train=tmp_path/'train';proxy=tmp_path/'proxy';proxy.mkdir()
    for i in range(15):
        folder=train/f'id_{i}';folder.mkdir(parents=True)
        for age in [20,40,60]:
            for repeat in [0,1]:Image.new('RGB',(16,16),(age+3*i,repeat*2,30)).save(folder/f'{age}_{repeat}.png')
    for i in range(4):Image.new('RGB',(16,16),(20+30*i,100,40)).save(proxy/f'{i:05}.png')
    pd.DataFrame({'image_number':range(4),'age_group':['15-19','30-39','50-69','70-120'],
        'age_group_confidence':[1.]*4}).to_csv(proxy/'ffhq_aging_labels.csv',index=False)
    cfg=ReferenceConfig(train,proxy,tmp_path/'out',train_only=True,knots=(0,20,40,60,80),min_retest_pairs=10)
    ref=build_reference_bundle(cfg,feature_backend=Backend())
    assert ref.attributes==['signal'] and ref.icc[0]>.95
    assert ref.mean[0]==pytest.approx(61.01)
    assert len(ref.metadata['train_identities'])==15
    assert (tmp_path/'out/train_features.parquet').exists()


def test_embedding_pca_never_fit_on_test_and_roundtrip(tmp_path):
    from src.aging_bias.embedding_features import EmbeddingFeatureBackend
    class Encoder:
        cache_id='toy-encoder'
        def encode_batch(self,images):return np.array(images)
    rng=np.random.default_rng(1);vectors=rng.normal(size=(30,8)).tolist()
    backend=EmbeddingFeatureBackend(Encoder(),n_components=3)
    with pytest.raises(ValueError,match='test'):backend.fit(vectors,splits=['test']*30)
    backend.fit(vectors,splits=['train']*20+['proxy']*10)
    backend.save(tmp_path/'pca.npz')
    restored=EmbeddingFeatureBackend(Encoder(),n_components=3).load(tmp_path/'pca.npz')
    assert backend.cache_id==restored.cache_id
    assert backend.extract_aligned(vectors[0],{})==restored.extract_aligned(vectors[0],{})
