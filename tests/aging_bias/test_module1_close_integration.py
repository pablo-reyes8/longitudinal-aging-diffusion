import json
import numpy as np
import pandas as pd
from PIL import Image


def test_closure_keeps_previous_immutable_and_saves_loadable_zone_bundle(tmp_path):
    from src.aging_bias import ReferenceConfig
    from src.aging_bias.module1_close import close_module1,Module1CloseConfig
    from src.aging_bias.references import ReferenceBundle
    from src.aging_bias.features import FeatureStore
    from src.aging_bias.config import FeatureConfig
    from tests.aging_bias.test_module1_close_features import face
    train=tmp_path/'train';proxy=tmp_path/'proxy';proxy.mkdir()
    for i in range(10):
        folder=train/f'id_{i}';folder.mkdir(parents=True)
        for age in [30,31,40,41,60,61,70,71]:Image.new('RGB',(256,256),(age+4*i,100,80)).save(folder/f'{age}.png')
    for i in range(12):Image.new('RGB',(256,256),(30+5*i,100,80)).save(proxy/f'{i:05d}.png')
    pd.DataFrame(dict(image_number=range(12),age_group=['30-39']*3+['40-49']*3+['50-69']*3+['70-120']*3,
        age_group_confidence=[1.]*12)).to_csv(proxy/'ffhq_aging_labels.csv',index=False)
    old=tmp_path/'old';old.mkdir();(old/'keep.txt').write_text('unchanged')
    base=ReferenceConfig(train,proxy,tmp_path/'new',train_only=True,min_retest_pairs=5,knots=(0,30,40,60,70,80))
    class Backend:
        cache_id='synthetic-zone-signal'
        attribute_ids=('wrinkle_forehead','geom_brow_eye')
        def extract_aligned(self,image,landmarks):
            value=float(np.asarray(image)[128,128,0])
            return dict(phi__wrinkle_forehead=value,phi__geom_brow_eye=value*2)
    class AgeBackend:
        cache_id='synthetic-age-only'
        def predict_batch(self,images):return np.array([55.,64.,66.])[:len(images)]
    cfg=Module1CloseConfig(base.output_dir,reference_config=base,previous_reference_dir=old,
        n_boot=3,placebo_splits=2,mean_faces=True,min_n_mean_face=1)
    result=close_module1(cfg,feature_backend=Backend(),landmark_backend=lambda _:face(),age_backend=AgeBackend())
    assert list(old.iterdir())==[old/'keep.txt'] and (old/'keep.txt').read_text()=='unchanged'
    restored=ReferenceBundle.load(base.output_dir)
    assert restored.metadata['module1_close'] and 'zone_upper' in restored.attributes
    features=FeatureConfig(**restored.metadata['config']['features'])
    store=FeatureStore(tmp_path/'evaluation_cache',features,lambda _:face(),Backend())
    assert store.config_hash==restored.feature_config_hash
    values=store.extract(train/'id_0/40.png')
    assert np.isfinite(values['phi__zone_upper'])
    for path in ['diagnostics/icc_tta_comparison.csv','diagnostics/resolution_matching.csv',
        'diagnostics/ffhq_age_coverage.csv','tables/zone_consistency.csv','figures/face_zones.png',
        'internal/placebo_ffhq.csv','diagnostics/resolution_sensitivity.csv',
        'figures/mean_faces_grid.png','figures/aging_difference_maps.png','diagnostics/ffhq_50_69_age_predictions.csv']:
        assert (base.output_dir/path).is_file()
    assert json.loads((base.output_dir/'diagnostics/closure_status.json').read_text())['complete']
    assert set(result['zone_consistency'].status)<= {'consistent','inconsistent','not_measured'}
