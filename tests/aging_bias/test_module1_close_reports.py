import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_consistency_fixed_majorities_and_FZ1_color_mapping(tmp_path):
    from src.aging_bias.module1_reports import zone_consistency,face_zone_styles,plot_face_zones
    rows=[]
    for name,cross,long in [('zone_upper',[1,2,-.1],[2,1,1]),('zone_mid',[-1,-2,-3],[-2,-1,1]),('zone_lower',[1,1,1],[-1,-1,-1])]:
        for i,(a,b) in enumerate(zip(cross,long)):
            rows.append(dict(attribute=name,source_age=[30,40,50][i],target_age=[60,70,75][i],difference_cross=a,difference_long=b))
    table=zone_consistency(pd.DataFrame(rows),['zone_upper','zone_mid','zone_lower','missing'])
    assert table.set_index('attribute').direction.to_dict()=={'zone_upper':'ffhq_more','zone_mid':'ffhq_less','zone_lower':'inconsistent','missing':'not_measured'}
    styles=face_zone_styles(table)
    assert styles['zone_upper']['color']!=styles['zone_mid']['color']
    assert styles['zone_lower']['color']=='#aaaaaa'
    plot_face_zones(table,tmp_path)
    assert (tmp_path/'face_zones.png').is_file() and (tmp_path/'face_zones_es.png').is_file()


def test_MF1_MF2_MF3_maps_and_small_cells(tmp_path):
    from src.aging_bias.module1_reports import aggregate_mean_faces,aging_maps
    yy,xx=np.mgrid[:64,:64];base=100+20*np.sin(xx/7)+15*np.cos(yy/8)
    forehead=(yy>8)&(yy<20)&(xx>18)&(xx<46)
    records=[]
    for population in ['latam','ffhq']:
        for age in [35,65]:
            a=base.copy()
            if population=='ffhq' and age==65:a[forehead]+=40
            for i in range(2):records.append((population,age,Image.fromarray(np.uint8(a)).convert('RGB')))
    means,coverage=aggregate_mean_faces(records,min_n=2,normalize_histogram=False)
    maps=aging_maps(means)
    pos=np.unravel_index(np.nanargmax(maps['difference']),maps['difference'].shape)
    assert forehead[pos]
    same=records[:4]+[('ffhq',age,image) for _,age,image in records[:4]]
    means,_=aggregate_mean_faces(same,min_n=2)
    assert abs(aging_maps(means)['difference']).max()<1e-6
    means,coverage=aggregate_mean_faces(records,min_n=3)
    assert not means and coverage.status.eq('insufficient_n').all()


def test_ffhq_broad_labels_never_invent_exact_sixties_mean():
    from src.aging_bias.module1_reports import age_band
    assert age_band(59.5,'50-69') is None
    assert age_band(34.5,'30-39')=='30-39'
    assert age_band(65,None)=='60-69'


def test_ffhq_age_refinement_only_broad_group_preserves_official_ages_and_resumes(tmp_path):
    from src.aging_bias.ffhq_age_bands import refine_ffhq_mean_ages
    from src.aging_bias.config import FeatureConfig
    from tests.aging_bias.test_features import landmarks
    for i in range(3):Image.new('RGB',(256,256),(100+i,80,70)).save(tmp_path/f'{i}.png')
    frame=pd.DataFrame(dict(image_path=[str(tmp_path/f'{i}.png') for i in range(3)],
        age_group=['30-39','50-69','50-69'],age=[34.5,59.5,59.5]))
    class AgeBackend:
        cache_id='synthetic-age'
        def predict_batch(self,images):
            assert len(images)==2
            if self.disabled:raise AssertionError('Resume must reuse age predictions')
            return [55.,65.]
        disabled=False
    backend=AgeBackend()
    refined=refine_ffhq_mean_ages(frame,FeatureConfig(),lambda _:landmarks(),backend,tmp_path/'cache')
    assert refined.age.tolist()==[34.5,59.5,59.5]
    assert refined.mean_face_age.tolist()==[34.5,55.,65.]
    backend.disabled=True
    again=refine_ffhq_mean_ages(frame,FeatureConfig(),lambda _:landmarks(),backend,tmp_path/'cache')
    pd.testing.assert_frame_equal(refined,again)


def test_closure_rejects_legacy_destination_before_any_model(tmp_path):
    from src.aging_bias.module1_close import Module1CloseConfig,close_module1
    old=tmp_path/'old';old.mkdir();(old/'references.json').write_text('{}')
    before=(old/'references.json').read_bytes()
    cfg=Module1CloseConfig(reference_dir_out=old,latam_sources={'colombia_train':'missing'},ffhq_aging_root='missing')
    with pytest.raises(ValueError,match='[Oo]verwrite|[Ll]egacy|[Pp]revious'):close_module1(cfg)
    assert (old/'references.json').read_bytes()==before


def test_population_bootstrap_cross_ci_and_no_fairface_within(tmp_path):
    from src.aging_bias.references import fit_reference_frames
    from src.aging_bias.population_aging import describe_population_aging
    c=pd.DataFrame([dict(identity_id=str(i),age=a,phi__x=i+a*.1) for i in range(15) for a in [30.,50.,70.]])
    p=pd.DataFrame([dict(age=a,phi__x=i+a*.2) for i in range(10) for a in [30.,50.,70.]])
    ref=fit_reference_frames(c,p,['x'],[0,30,50,70,80],np.array([0.]),np.array([1.]),np.array([.8]),'test')
    table=describe_population_aging(ref,transitions=[(30,60),(40,70),(50,70)],n_boot=8,output_dir=tmp_path,include_cross_ci=True)
    assert np.isfinite(table[['ci_low_cross','ci_high_cross','difference_cross','difference_long']]).all().all()
    assert (table.ci_low_cross<=table.ci_high_cross).all()
