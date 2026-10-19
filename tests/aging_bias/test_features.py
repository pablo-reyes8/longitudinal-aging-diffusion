import numpy as np
import pytest
from PIL import Image


def landmarks(size=256):
    # Six points per eye in EAR order, eight outer mouth points.
    return {k:np.array(v,float)*size for k,v in {
        'left_eye':[(.28,.4),(.30,.38),(.34,.38),(.36,.4),(.34,.42),(.30,.42)],
        'right_eye':[(.64,.4),(.66,.38),(.70,.38),(.72,.4),(.70,.42),(.66,.42)],
        'mouth':[(.38,.68),(.43,.65),(.5,.64),(.57,.65),(.62,.68),(.57,.71),(.5,.72),(.43,.71)],
        'jaw':[(.18,.52),(.22,.65),(.3,.78),(.4,.84),(.5,.86),(.6,.84),(.7,.78),(.78,.65),(.82,.52)],
        'left_brow':[(.27,.32),(.32,.30),(.37,.32)],
        'right_brow':[(.63,.32),(.68,.30),(.73,.32)],
        'nose':[(.5,.55)],'chin':[(.5,.86)]}.items()}


def test_F1_wrinkle_monotonicity():
    from src.aging_bias.features import HandcraftedFeatureBackend
    b=HandcraftedFeatureBackend()
    vals=[]
    for k in [0,2,4,8]:
        a=np.full((256,256,3),150,np.uint8)
        xx=np.arange(70,186)
        for yy in np.linspace(42,73,k):
            y=np.round(yy+1.5*np.sin(xx/14)).astype(int)
            a[y,xx]=55
        vals.append(b.extract_aligned(Image.fromarray(a),landmarks())['phi__wrinkle_forehead'])
    assert np.all(np.diff(vals)>0),vals


def test_F2_resolution_harmonization():
    from src.aging_bias.preprocessing import preprocess_image
    from src.aging_bias.features import HandcraftedFeatureBackend
    from src.aging_bias.config import FeatureConfig
    rng=np.random.default_rng(2);a=np.uint8(rng.uniform(60,190,(256,256,3)))
    image=Image.fromarray(a);b=HandcraftedFeatureBackend();values=[];raw=[]
    for size in [256,512,1024]:
        im=image.resize((size,size),Image.Resampling.BICUBIC)
        p=preprocess_image(im,lambda _:landmarks(size),FeatureConfig())
        values.append(b.extract_aligned(p.image,p.landmarks)['phi__skin_hf_texture'])
        raw.append(b.extract_aligned(im,landmarks(size))['phi__skin_hf_texture'])
    assert np.ptp(values)/np.mean(values)<.1
    assert np.ptp(raw)/np.mean(raw)>.1


def test_F3_ita_analytic():
    from src.aging_bias.features import ita_from_lab,HandcraftedFeatureBackend
    assert ita_from_lab(np.array([[[70.,0.,20.]]]))==pytest.approx(45,abs=.5)
    red=Image.new('RGB',(256,256),(255,0,0))
    # Independent sRGB D65 reference: L*=53.2408, b*=67.2032.
    assert HandcraftedFeatureBackend().extract_aligned(red,landmarks())['phi__skin_ita']==pytest.approx(2.761,abs=.5)


def test_F4_F6_gray_and_missing_never_zero():
    from src.aging_bias.preprocessing import is_grayscale
    from src.aging_bias.features import HandcraftedFeatureBackend
    gray=Image.new('RGB',(256,256),(100,100,100));color=Image.new('RGB',(256,256),(160,100,70))
    assert is_grayscale(gray) and not is_grayscale(color)
    f=HandcraftedFeatureBackend().extract_aligned(gray,landmarks())
    assert np.isnan(f['phi__skin_ita']) and np.isnan(f['phi__hair_gray'])
    f=HandcraftedFeatureBackend().extract_aligned(color,{})
    assert np.isnan(f['phi__geom_jaw_sag'])


def test_F5_geometry_similarity_invariance():
    from src.aging_bias.features import geometry_attributes
    l=landmarks();theta=.71;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    moved={k:v@rotation.T*1.7+[33,-18] for k,v in l.items()}
    a=geometry_attributes(l);b=geometry_attributes(moved)
    np.testing.assert_allclose(list(a.values()),list(b.values()),atol=1e-6)


def test_R1_icc_and_exclusion():
    from src.aging_bias.reliability import icc_2_1, reliability_table
    rng=np.random.default_rng(1);signal=rng.normal(0,np.sqrt(.3),2000)
    repeated=signal[:,None]+rng.normal(0,np.sqrt(.7),(2000,2))
    assert icc_2_1(repeated)==pytest.approx(.3,abs=.05)
    report=reliability_table({'a':repeated},min_icc=.5)
    assert not report.iloc[0].active
