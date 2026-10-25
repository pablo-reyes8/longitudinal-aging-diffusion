from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from tests.aging_bias.test_features import landmarks


def face():
    return {**landmarks(), 'malar':np.array([[83.,130.],[173.,130.]]),
        'zygomatic':np.array([[46.,133.],[210.,133.]]),
        'mouth_corners':np.array([[97.,174.],[159.,174.]])}


def test_M1_M2_M3_M4_cheek_geometry():
    from src.aging_bias.features import additional_geometry_attributes
    from src.aging_bias.preprocessing import mirror_landmarks
    l=face();a=additional_geometry_attributes(l)
    theta=.7;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    b=additional_geometry_attributes({k:v@r.T*1.7+[20,-9] for k,v in l.items()})
    for key in a:assert b[key]==pytest.approx(a[key],abs=1e-6)
    mirrored=additional_geometry_attributes(mirror_landmarks(l,256))
    for key in a:assert mirrored[key]==pytest.approx(a[key],abs=1e-6)
    descent=[];height=[]
    for dy in [0,3,6]:
        moved={**l,'malar':l['malar']+[0,dy],'mouth_corners':l['mouth_corners']+[0,dy]}
        values=additional_geometry_attributes(moved)
        descent.append(values['geom_malar_descent']);height.append(values['geom_midface_height'])
    assert np.all(np.diff(descent)>0) and np.all(np.diff(height)>0)
    assert all(np.isnan(v) for v in additional_geometry_attributes({}).values())


def test_R2_R3_low_resolution_unchanged_and_deterministic():
    from src.aging_bias.preprocessing import match_effective_resolution
    image=Image.fromarray(np.random.default_rng(2).integers(0,255,(128,128,3),dtype=np.uint8))
    low=match_effective_resolution(image,original_iod=15,aligned_iod=50,target_iod=20)
    np.testing.assert_array_equal(low,image)
    a=match_effective_resolution(image,original_iod=80,aligned_iod=50,target_iod=20)
    b=match_effective_resolution(image,original_iod=80,aligned_iod=50,target_iod=20)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,image)


def test_R1_resolution_matched_texture_within_five_percent():
    from src.aging_bias.preprocessing import preprocess_image
    from src.aging_bias.config import FeatureConfig
    from src.aging_bias.features import HandcraftedFeatureBackend
    yy,xx=np.mgrid[:256,:256]
    pixels=np.uint8(130+40*np.sin(xx/5)*np.cos(yy/7))
    im=Image.fromarray(pixels).convert('RGB');values=[]
    for size in [256,512,1024]:
        prep=preprocess_image(im.resize((size,size),Image.Resampling.BICUBIC),lambda _:landmarks(size),
            FeatureConfig(resolution_matching=True,target_iod_px=30))
        values.append(HandcraftedFeatureBackend().extract_aligned(prep.image,prep.landmarks)['phi__skin_hf_texture'])
    assert np.ptp(values)/np.mean(values)<.05


def test_TTA_permutation_involution_and_noise_reduction():
    from src.aging_bias.preprocessing import mirror_landmarks
    from src.aging_bias.features import average_tta
    l=face();twice=mirror_landmarks(mirror_landmarks(l,256),256)
    for key in l:np.testing.assert_allclose(twice[key],l[key])
    rng=np.random.default_rng(4);a=rng.normal(size=20000);b=rng.normal(size=20000)
    combined=np.array([average_tta({'x':x},{'x':y})[0]['x'] for x,y in zip(a,b)])
    assert np.var(combined)/np.var(a)==pytest.approx(.5,rel=.1)
    values,single=average_tta({'a':np.nan,'b':2.},{'a':4.,'b':2.})
    assert values=={'a':4.,'b':2.} and single


def test_TTA1_symmetric_image_and_legacy_hash_are_preserved(tmp_path):
    from src.aging_bias.features import FeatureStore,stable_hash
    from src.aging_bias.config import FeatureConfig
    photo=tmp_path/'symmetric.png';Image.new('RGB',(256,256),(130,130,130)).save(photo)
    plain=FeatureStore(tmp_path/'a',FeatureConfig(),lambda _:landmarks())
    mirror=FeatureStore(tmp_path/'b',FeatureConfig(flip_tta=True),lambda _:landmarks())
    a=plain.extract(photo);b=mirror.extract(photo)
    keys=sorted(k for k in a if k.startswith('phi__'))
    np.testing.assert_allclose([a[k] for k in keys],[b[k] for k in keys],atol=1e-8,equal_nan=True)
    # Literal pre-closure schema: new default-false fields must not alter it.
    legacy=dict(crop_size=256,harmonize_sigma=1.,gray_tol=2.,version='handcrafted-v1',
        landmark_backend_id='mediapipe-face-landmarker-v1',max_abs_yaw=None,
        max_smile_score=None,max_jaw_open_score=None,normalize_color_illumination=False,landmark_model_sha256=None)
    assert plain.config_hash==stable_hash(dict(config=legacy,backend='handcrafted-v1',landmarks='mediapipe-face-landmarker-v1'))
    assert mirror.config_hash!=plain.config_hash


def test_Z1_Z2_Z3_Z4_train_zones():
    from src.aging_bias.zones import fit_zone_transform,apply_zone_transform
    from src.aging_bias.reliability import icc_2_1
    rng=np.random.default_rng(8);n=3000
    signal=rng.normal(size=n);age=rng.uniform(30,70,n)
    # Strong same-age person variance and independent repeat noise.
    x=np.repeat(signal,2)+np.tile([0,.01],n)+rng.normal(0,1,2*n)
    y=np.repeat(signal,2)+np.tile([0,.01],n)+rng.normal(0,1,2*n)
    f=pd.DataFrame(dict(identity_id=np.repeat([str(i) for i in range(n)],2),age=np.repeat(age,2)+np.tile([0,.1],n),phi__a=x,phi__b=y,phi__bad=1e6))
    rel=pd.DataFrame(dict(attribute=['a','b','bad'],active=[True,True,False],icc=[.5,.5,.01]))
    transform=fit_zone_transform(f,rel,[0,30,70,80],{'z':['a','b','bad'],'empty':['bad']})
    result=apply_zone_transform(f,transform)
    value=icc_2_1(result.phi__z.to_numpy().reshape(n,2))
    assert value==pytest.approx(2*.5/(1+.5),abs=.05)
    assert value>max(icc_2_1(x.reshape(n,2)),icc_2_1(y.reshape(n,2)))
    assert transform['empty']['status']=='no_active_attributes' and result.phi__empty.isna().all()
    f['phi__bad']=-1e9
    np.testing.assert_allclose(result.phi__z,apply_zone_transform(f,transform).phi__z)
    f['phi__a']=-f.phi__a
    reverse=fit_zone_transform(f,rel,[0,30,70,80],{'z':['a','b']})
    assert reverse['z']['signs'][0]==-transform['z']['signs'][0]
    np.testing.assert_allclose(result.phi__z,apply_zone_transform(f,reverse).phi__z)


def test_S1_more_older_ffhq_only_available_labeled_images(tmp_path):
    from src.aging_bias.references import load_ffhq_manifest
    from src.aging_bias.config import FFHQSamplingConfig
    for i in range(12):Image.new('RGB',(8,8)).save(tmp_path/f'{i:05d}.png')
    pd.DataFrame(dict(image_number=range(12),age_group=['20-29']*4+['40-49']*4+['70-120']*4,
        age_group_confidence=[1.]*12)).to_csv(tmp_path/'ffhq_aging_labels.csv',index=False)
    f=load_ffhq_manifest(tmp_path,max_per_group=2,sampling=FFHQSamplingConfig())
    assert f.groupby('age_group').size().to_dict()=={'20-29':2,'40-49':4,'70-120':4}
