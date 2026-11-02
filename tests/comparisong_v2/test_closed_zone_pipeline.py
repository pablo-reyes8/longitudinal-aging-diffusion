from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest


def closed_setup(tmp_path):
    from test_evaluator import setup
    from src.aging_bias.references import ReferenceBundle
    from src.aging_bias.features import FeatureStore
    cfg,bundle,backend=setup(tmp_path);cfg.n_boot=2
    # Rename the real toy descriptors at the neural boundary, not the evaluator.
    class Zones:
        cache_id='synthetic-zone-features'
        def extract(self,image):
            values=backend.extract(image)
            return {'phi__zone_upper':values['phi__a'],'phi__zone_lower':values['phi__b']}
    bundle['feature_backend']=Zones()
    ref=ReferenceBundle.load(cfg.reference_bundle_path,load_training_data=False)
    ref.attributes=['zone_upper','zone_lower']
    ref.feature_config_hash=FeatureStore(tmp_path/'temp',feature_backend=bundle['feature_backend']).config_hash
    ref.metadata={'module1_close':True,'config':{'features':asdict(cfg.features)},
        'zone_transform':{},'attribute_orientations':{'zone_upper':1,'zone_lower':1}}
    ref.save(cfg.reference_bundle_path)
    return cfg,bundle,ref


def test_closed_evaluator_uses_observed_errors_and_never_regenerates(tmp_path):
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,ref=closed_setup(tmp_path)
    before={p:p.read_bytes() for p in cfg.reference_bundle_path.iterdir()}
    result=evaluate_paper_v2(cfg,bundle=bundle)
    assert result['metadata']['protocol']=='observed-longitudinal-zone-error-v1'
    assert result['zone_validation']['passed']
    assert result['bias_indices'].empty
    assert len(result['summary'])>0
    assert result['zone_errors'].query("model_name=='oracle_target'").error.eq(0).all()
    for p,data in before.items():assert p.read_bytes()==data
    assert (cfg.output_dir/'tables/zone_errors_aging.csv').is_file()


def test_closed_analysis_requires_frozen_dino_no_train_images_or_refits(tmp_path,monkeypatch):
    import src.comparisong.paper_analysis as api
    cfg,bundle,ref=closed_setup(tmp_path)
    before={p:p.read_bytes() for p in cfg.reference_bundle_path.iterdir()}
    def forbidden(*a,**k):raise AssertionError('Module 1 must not be rebuilt')
    monkeypatch.setattr(api,'preflight_dino_inputs',forbidden)
    monkeypatch.setattr(api,'refit_cached_references',forbidden)
    monkeypatch.setattr(api,'build_cached_embedding_references',forbidden)
    def frozen(*a,**k):return ref,dict(feature_backend=bundle['feature_backend'],feature_config=cfg.features)
    monkeypatch.setattr(api,'load_frozen_dino_backend',frozen)
    result=api.run_paper_analysis(cfg,bundle=bundle)
    assert result['analysis_status']['complete'] and result['analysis_status']['passed']
    assert result['analysis_status']['protocol']=='observed-longitudinal-zone-error-v1'
    assert set(result['analyses'])=={'handcrafted_primary','dinov2'}
    for p,data in before.items():assert p.read_bytes()==data


def test_closed_analysis_missing_dino_fails_before_expensive_evaluation(tmp_path,monkeypatch):
    import src.comparisong.paper_analysis as api
    cfg,bundle,ref=closed_setup(tmp_path)
    def should_not_run(*a,**k):raise AssertionError('Fail frozen-reference preflight first')
    monkeypatch.setattr(api,'evaluate_paper_v2',should_not_run)
    with pytest.raises(FileNotFoundError,match='DINO|dinov2'):
        api.run_paper_analysis(cfg,bundle=bundle,allow_download=False)


def test_sd_metric_preflight_extracts_real_zones_without_operational_models(tmp_path):
    from src.comparisong.zone_paper import preflight_zone_protocol
    cfg,bundle,ref=closed_setup(tmp_path)
    ref.save(cfg.reference_bundle_path/'dinov2')
    np.savez(cfg.reference_bundle_path/'dinov2/embedding_pca.npz',mean=np.zeros(2),components=np.eye(2),encoder_id=np.array('synthetic'))
    path=preflight_zone_protocol(cfg,bundle={'feature_backend':bundle['feature_backend']})
    state=json.loads(path.read_text())
    assert state['passed'] and state['checked'][0]['valid_zones']


def test_more_long_pairs_are_inventory_only_within_current_test_identities(tmp_path):
    from PIL import Image
    from src.comparisong.zone_paper import report_long_pair_support
    from src.aging_bias.config import PaperV2Config,WeightingConfig
    root=tmp_path/'test';folder=root/'id_1';folder.mkdir(parents=True)
    for age in [20,25,60]:Image.new('RGB',(8,8)).save(folder/f'{age}.png')
    other=root/'id_not_in_test_manifest';other.mkdir();Image.new('RGB',(8,8)).save(other/'70.png')
    panel=pd.DataFrame([dict(sample_id='p',identity_id='id_1',source_age=20,target_age=25,delta_age=5,
        source_path=str(folder/'20.png'),target_path=str(folder/'25.png'))])
    cfg=PaperV2Config('manifest',{},['SAM'],'ref',tmp_path/'out',source_root=root,weighting=WeightingConfig(min_bin_n=1))
    report_long_pair_support(panel,cfg)
    candidates=pd.read_csv(cfg.output_dir/'diagnostics/long_pair_candidates.csv')
    assert len(candidates)==2 and candidates.identity_id.eq('id_1').all()
    assert not candidates.in_current_manifest.any()
    assert panel.sample_id.tolist()==['p']


def test_frozen_dino_loader_uses_saved_pca_and_does_not_write_references(tmp_path,monkeypatch):
    from src.comparisong.frozen_dino import load_frozen_dino_backend
    from src.aging_bias.embedding_features import EmbeddingFeatureBackend
    import src.aging_bias.embedding_features as embedding
    from src.aging_bias.features import FeatureStore
    cfg,bundle,ref=closed_setup(tmp_path)
    class Encoder:
        cache_id='toy/checkpoint:fixed'
        def __init__(self,**kwargs):self.device=kwargs.get('device','cpu')
        def to(self,device):self.device=device
    encoder=Encoder();backend=EmbeddingFeatureBackend(encoder,2).fit_vectors(
        np.random.default_rng(7).normal(size=(12,3)),splits=['train']*6+['proxy']*6)
    directory=cfg.reference_bundle_path/'dinov2';directory.mkdir()
    backend.save(directory/'embedding_pca.npz')
    ref.attributes=['embedding_0','embedding_1']
    ref.feature_config_hash=FeatureStore(tmp_path/'probe',cfg.features,None,backend).config_hash
    ref.save(directory)
    before={p:p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(embedding,'DinoV2Encoder',Encoder)
    loaded,prepared=load_frozen_dino_backend(cfg.reference_bundle_path,root=tmp_path/'assets',
        landmark_backend=None,cache_dir=tmp_path/'evalcache',allow_download=False,device='cuda')
    assert prepared['feature_backend'].cache_id==backend.cache_id
    assert prepared['embedding_device']=='cuda'
    assert loaded.attributes==['embedding_0','embedding_1']
    for p,data in before.items():assert p.read_bytes()==data
