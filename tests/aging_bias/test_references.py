import numpy as np
import pandas as pd
import pytest


def test_P1_within_panel_recovers_age_not_cohort():
    from src.aging_bias.references import fit_panel_fe, spline_basis
    rng=np.random.default_rng(12);n=500
    ids=np.repeat(np.arange(n),3);baseline=np.repeat(rng.uniform(10,50,n),3)
    ages=baseline+np.tile([0,10,20],n)
    cohort=np.repeat(rng.normal(size=n),3)-baseline*.4
    y=.08*ages+cohort+rng.normal(0,.005,len(ages))
    knots=np.array([0,10,20,30,40,50,60,70,80.])
    coef,_=fit_panel_fe(ages,y[:,None],ids,knots)
    difference=(spline_basis([50],knots)-spline_basis([20],knots))@coef
    assert difference[0,0]==pytest.approx(2.4,abs=.05)
    cross=np.linalg.lstsq(np.c_[np.ones(len(ages)),ages],y,rcond=None)[0][1]*30
    assert abs(cross-2.4)>3*abs(difference[0,0]-2.4)


def reference():
    from src.aging_bias.references import ReferenceBundle
    return ReferenceBundle(knots=np.array([10.,30.,50.]),f_coeff=np.array([[0.,0.],[2.,4.],[4.,8.]]),
        mu_c=np.array([[1.,2.],[3.,6.],[5.,10.]]),p_ages=np.array([10.,30.,50.]),
        mu_p=np.array([[4.,6.],[6.,10.],[8.,14.]]),mean=np.array([0.,0.]),sd=np.array([1.,1.]),
        attributes=['a','b'],icc=np.array([.8,.9]),feature_config_hash='test',support=(10.,50.))


def test_P2_P3_P5_interpolation_signs_serialization(tmp_path):
    from src.aging_bias.references import ReferenceBundle
    b=reference()
    np.testing.assert_allclose(b.d_C([10],[40]),[[3,6]])
    np.testing.assert_allclose(b.d_P([10],[40]),b.d_C([10],[40]))
    np.testing.assert_allclose(b.G_P([20,40]),[[3,4],[3,4]])
    b.save(tmp_path)
    c=ReferenceBundle.load(tmp_path)
    for name,args in [('d_C',([20],[40])),('d_P',([20],[40])),('G_P',([20],)),('zeta',(np.array([[1.,2.]]),[20]))]:
        np.testing.assert_array_equal(getattr(b,name)(*args),getattr(c,name)(*args))
    assert b.extrapolated([0],[40]).tolist()==[True]
    np.testing.assert_allclose(b.d_C([0],[90]),b.d_C([10],[50]))


def test_P4_test_identity_leakage_rejected_before_models(tmp_path):
    from src.aging_bias import ReferenceConfig,build_reference_bundle
    from PIL import Image
    p=tmp_path/'id_test';p.mkdir();Image.new('RGB',(32,32)).save(p/'20.png')
    cfg=ReferenceConfig(tmp_path,tmp_path,tmp_path/'out',train_only=True,excluded_test_ids=('id_test',))
    with pytest.raises(ValueError,match='[Ll]eak|test'):
        build_reference_bundle(cfg)


def test_ffhq_label_convention_and_confidence(tmp_path):
    from src.aging_bias.references import load_ffhq_manifest
    from PIL import Image
    for i in range(3):Image.new('RGB',(16,16)).save(tmp_path/f'{i:05}.png')
    pd.DataFrame({'image_number':[0,1,2],'age_group':['3-6','70-120','20-29'],
                  'age_group_confidence':[1.,.9,.2]}).to_csv(tmp_path/'ffhq_aging_labels.csv',index=False)
    df=load_ffhq_manifest(tmp_path,min_conf=.6)
    assert df.age.tolist()==[4.5,95.]


def test_reference_extrapolation_includes_knots_and_sex_support():
    b=reference();b.support=(10,95);b.p_ages=np.array([10.,30.,95.])
    assert b.extrapolated([60],[90])[0]
    subgroup=reference();subgroup.support=(20,40)
    b.strata['female']=subgroup
    assert b.extrapolated([15],[35],sex=['female'])[0]


def test_ffhq_photo_selection_never_substitutes_segmentation_maps(tmp_path):
    from src.aging_bias.references import load_ffhq_manifest
    from PIL import Image
    for folder,color in [('images',(100,80,70)),('zz_masks',(0,0,0))]:
        p=tmp_path/folder;p.mkdir();Image.new('RGB',(16,16),color).save(p/'00000.png')
    pd.DataFrame({'image_number':[0],'age_group':['20-29'],'age_group_confidence':[1.]}).to_csv(tmp_path/'ffhq_aging_labels.csv',index=False)
    result=load_ffhq_manifest(tmp_path)
    assert '/images/' in result.iloc[0].image_path
