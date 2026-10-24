from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.aging_bias.config import ReferenceConfig
from src.aging_bias.references import load_colombia_manifest, fit_panel_fe
from src.aging_bias.reliability import reference_reliability


def setup_train(tmp_path):
    train=tmp_path/'colombia'/'id_1';train.mkdir(parents=True)
    for age in [30,31,32]:Image.new('RGB',(16,16)).save(train/f'{age}.jpg')
    return ReferenceConfig(train.parent,tmp_path/'proxy',tmp_path/'out',train_only=True)


def test_optional_fairface_adds_only_available_frontal_labeled_photos(tmp_path):
    cfg=setup_train(tmp_path)
    original=load_colombia_manifest(cfg)
    extra=tmp_path/'fairface';extra.mkdir()
    names=['train_000001.jpg','train_000002.jpg','train_000003.jpg','train_000004.jpg']
    for name in names:Image.new('RGB',(16,16)).save(extra/name)
    pd.DataFrame(dict(filename=names+['missing.jpg'],age_pred=[40.94,30.,32.,150.,50.])).to_csv(extra/'age_predictions.csv',index=False)
    pd.DataFrame(dict(filename=names[0:1]+names[2:]+['missing.jpg'],
        gender=['Female']*4,race=['Latino_Hispanic','White','Latino_Hispanic','Latino_Hispanic'],
        path=['/obsolete/path.jpg']*4)).to_csv(extra/'metadata_frontal.csv',index=False)
    enriched=load_colombia_manifest(replace(cfg,fairface_root=extra))
    assert len(enriched)==4
    pd.testing.assert_frame_equal(enriched.iloc[:3][original.columns].reset_index(drop=True),original,check_dtype=False)
    row=enriched.iloc[-1]
    assert row.age==40.94 and row.gender=='female'
    assert row.image_path==str(extra/names[0])
    assert row.population_source=='fairface_latino_hispanic'
    assert row.age_source=='predicted_age'
    assert row.identity_id not in original.identity_id.tolist()


def test_singleton_fairface_cannot_create_icc_retests_or_longitudinal_changes(tmp_path):
    cfg=setup_train(tmp_path);extra=tmp_path/'fairface';extra.mkdir()
    for name in ['a.jpg','b.jpg']:Image.new('RGB',(16,16)).save(extra/name)
    pd.DataFrame(dict(filename=['a.jpg','b.jpg'],age_pred=[30.,31.])).to_csv(extra/'age_predictions.csv',index=False)
    base=load_colombia_manifest(cfg);full=load_colombia_manifest(replace(cfg,fairface_root=extra))
    base['phi__a']=[1.,2.,3.];base['image_sha256']=['x','y','v']
    full['phi__a']=[1.,2.,3.,100.,-100.];full['image_sha256']=['x','y','v','z','w']
    a,_=fit_panel_fe(base.age,base[['phi__a']],base.identity_id,[0,30,60])
    b,_=fit_panel_fe(full.age,full[['phi__a']],full.identity_id,[0,30,60])
    np.testing.assert_allclose(a,b,atol=1e-10)
    pd.testing.assert_frame_equal(reference_reliability(base,['a'],cfg),reference_reliability(full,['a'],cfg))


def test_fairface_conflicting_age_duplicates_and_empty_subset_fail_early(tmp_path):
    cfg=setup_train(tmp_path);extra=tmp_path/'fairface';extra.mkdir()
    Image.new('RGB',(16,16)).save(extra/'a.jpg')
    pd.DataFrame(dict(filename=['a.jpg','a.jpg'],age_pred=[30.,45.])).to_csv(extra/'age_predictions.csv',index=False)
    cfg=replace(cfg,fairface_root=extra)
    with pytest.raises(ValueError,match='[Cc]onflicting'):load_colombia_manifest(cfg)
    pd.DataFrame(dict(filename=['absent.jpg'],age_pred=[30.])).to_csv(extra/'age_predictions.csv',index=False)
    with pytest.raises(ValueError,match='[Nn]o.*FairFace'):load_colombia_manifest(cfg)


def test_existing_train_leakage_check_still_precedes_supplement(tmp_path):
    cfg=setup_train(tmp_path)
    cfg=replace(cfg,fairface_root=tmp_path/'missing',excluded_test_ids=('id_1',))
    with pytest.raises(ValueError,match='leakage'):load_colombia_manifest(cfg)
