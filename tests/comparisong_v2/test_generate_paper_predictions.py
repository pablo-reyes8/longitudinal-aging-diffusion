from types import SimpleNamespace
import pandas as pd
from PIL import Image
import pytest
from src.comparisong.generate_paper_predictions import generate_paper_predictions


def configuration(tmp_path,models):
    image=tmp_path/'20.png';Image.new('RGB',(8,8),'red').save(image)
    manifest=tmp_path/'transitions.csv'
    pd.DataFrame([dict(sample_id=f'p{i}',identity_id='id_new',source_age=20,target_age=age,
        source_path=str(image),target_path=str(image)) for i,age in enumerate([40,60])]).to_csv(manifest,index=False)
    return SimpleNamespace(test_manifest_path=manifest,source_root=None,models=models,
        generated_roots={m:tmp_path/'generated'/m for m in models},output_dir=tmp_path/'metrics',seed=2026,resume=True)


def test_exact_manifest_failure_persistence_and_resume(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    import src.comparisong.load_models as loader
    from src.comparisong.resident_models import ACTIVE_MODEL_SESSION
    config=configuration(tmp_path,['SAM']);calls=[]
    import src.comparisong.evaluation_backends_v2 as backends
    load=backends.file_hash;reads=[]
    def counted_load(path):
        reads.append(str(path));return load(path)
    monkeypatch.setattr(backends,'file_hash',counted_load)
    def generate(bundle,**kw):
        assert ACTIVE_MODEL_SESSION.get().name=='SAM'
        assert not kw['metrics']
        calls.append(kw['target_ages'][0])
        if kw['target_ages']==[60]:raise RuntimeError('no face')
        return {'SAM':{40:Image.new('RGB',(8,8))}}
    monkeypatch.setattr(generator,'age_image',generate)
    monkeypatch.setattr(loader,'load_aging_models',lambda **kw:{'SAM':{'model':None}})
    result=generate_paper_predictions(config)
    assert list(result['manifest'].inference_status)==['ok','failed']
    assert calls==[40,60]
    assert len(reads)==1  # The shared source is read once, not per transition.
    generate_paper_predictions(config)
    assert calls==[40,60]
    assert len(reads)==1  # Persisted fingerprint reused while size/mtime match.
    assert pd.read_csv(config.test_manifest_path).shape[0]==2
    assert ACTIVE_MODEL_SESSION.get() is None


def test_missing_gender_fails_before_model_load(tmp_path,monkeypatch):
    import src.comparisong.load_models as loader
    monkeypatch.setattr(loader,'load_aging_models',lambda **kw:pytest.fail('must not load'))
    with pytest.raises(ValueError,match='Missing gender'):
        generate_paper_predictions(configuration(tmp_path,['FADING']))


def test_picking_outputs_are_copied_to_exact_evaluation_roots(tmp_path,monkeypatch):
    import src.comparisong.picking_variants as picking
    config=configuration(tmp_path,['ours_picking_no_prompt','ours_picking_prompt'])
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy')
    bundle={'fake':True}
    def generate(pc,**kw):
        assert kw['bundle'] is bundle
        assert pc.overwrite is False and pc.resume
        rows=[]
        for sample in pd.read_csv(pc.test_manifest_path).to_dict('records'):
            for name in config.models:
                image=tmp_path/f'{name}_{sample["sample_id"]}.png';Image.new('RGB',(8,8)).save(image)
                rows.append({**sample,'model_name':name,'generated_path':str(image),'inference_status':'ok'})
        return {'manifest':pd.DataFrame(rows)}
    monkeypatch.setattr(picking,'run_picking_variants',generate)
    result=generate_paper_predictions(config,checkpoint_path=checkpoint,picking_bundle=bundle)
    assert len(result['manifest'])==4
    for row in result['manifest'].itertuples():
        expected=config.generated_roots[row.model_name]/row.identity_id/(row.sample_id+'.png')
        assert expected.is_file() and row.generated_path==str(expected)


def test_fading_groups_same_source_and_csv_gender_overrides_legacy(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    config=configuration(tmp_path,['FADING'])
    csv=tmp_path/'gender.csv'
    csv.write_text('id,age,gender,confidence,image_path\nid_new,20,female,0.99,unused.jpg\n')
    calls=[]
    def generate(bundle,**kw):
        calls.append(kw['target_ages'])
        assert kw['gender']=='female'
        return {'FADING':{age:Image.new('RGB',(8,8)) for age in kw['target_ages']}}
    monkeypatch.setattr(generator,'age_image',generate)
    result=generate_paper_predictions(config,comparison_bundle={'FADING':{'script':'unused'}},gender_csv_path=csv)
    assert calls==[[40,60]]
    assert list(result['manifest'].inference_status)==['ok','ok']
    generate_paper_predictions(config,comparison_bundle={'FADING':{'script':'unused'}},gender_csv_path=csv)
    assert calls==[[40,60]]  # Resume neither repeats inversion nor editing.


def test_fading_group_failure_falls_back_and_explicit_gender_has_priority(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    config=configuration(tmp_path,['FADING']);csv=tmp_path/'gender.csv'
    csv.write_text('identity_id,gender\nid_new,female\n')
    calls=[]
    def generate(bundle,**kw):
        calls.append(kw['target_ages']);assert kw['gender']=='male'
        if len(kw['target_ages'])>1:raise RuntimeError('group problem')
        return {'FADING':{kw['target_ages'][0]:Image.new('RGB',(8,8))}}
    monkeypatch.setattr(generator,'age_image',generate)
    result=generate_paper_predictions(config,comparison_bundle={'FADING':{}},
        gender_csv_path=csv,gender_by_id={'id_new':'H'})
    assert calls==[[40,60],[40],[60]]
    assert result['manifest'].inference_status.eq('ok').all()


def test_gender_csv_validates_repeated_labels_and_rejects_conflicts(tmp_path):
    from src.comparisong.identity_metadata import load_gender_csv
    csv=tmp_path/'gender.csv';csv.write_text('id,gender\nid_1,H\nid_1,male\nid_2,M\n')
    assert load_gender_csv(csv)=={'id_1':'male','id_2':'female'}
    csv.write_text('id,gender\nid_1,H\nid_1,M\n')
    with pytest.raises(ValueError,match='Conflicting'):
        load_gender_csv(csv)


def test_aging_resume_preserves_previous_sam_and_removes_rejuvenation(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    config=configuration(tmp_path,['SAM'])
    panel=pd.read_csv(config.test_manifest_path)
    reverse={**panel.iloc[0].to_dict(),'sample_id':'reverse','source_age':40,'target_age':20}
    pd.concat([panel,pd.DataFrame([reverse])],ignore_index=True).to_csv(config.test_manifest_path,index=False)
    calls=[]
    def generate(bundle,**kw):
        calls.append(kw['target_ages'][0])
        return {'SAM':{kw['target_ages'][0]:Image.new('RGB',(8,8))}}
    monkeypatch.setattr(generator,'age_image',generate)
    # Create a compatible previous mixed-direction run, then switch defaults.
    config.aging_only=False
    generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    assert calls==[40,60,20]
    reverse_path=config.generated_roots['SAM']/'id_new'/'reverse.png'
    assert reverse_path.is_file()
    config.aging_only=True
    result=generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    assert calls==[40,60,20]  # Existing aging signatures remained compatible.
    assert result['manifest'].target_age.gt(result['manifest'].source_age).all()
    assert not reverse_path.exists() and not reverse_path.with_suffix('.json').exists()
    assert len(pd.read_csv(config.output_dir/'generation'/'transitions.csv'))==2


def test_fingerprint_cache_survives_restart_and_invalidates_changed_file(tmp_path):
    from src.comparisong.generation_cache import FingerprintCache
    source=tmp_path/'source';source.write_bytes(b'first');calls=[]
    def compute(path):calls.append(path);return str(len(source.read_bytes()))
    path=tmp_path/'cache.json';cache=FingerprintCache(path)
    assert cache.get(source,compute)=='5';cache.save()
    cache=FingerprintCache(path)
    assert cache.get(source,compute)=='5' and len(calls)==1
    source.write_bytes(b'changed content')
    assert cache.get(source,compute)=='15' and len(calls)==2


def test_sam_batches_only_pending_preserving_legacy_singleton_signatures(tmp_path,monkeypatch,capsys):
    import src.comparisong.generate_images as generator
    import torch
    config=configuration(tmp_path,['SAM']);legacy=[]
    def single(bundle,**kw):
        legacy.append(kw['target_ages']);print('AGING SWEEP COMPLETE')
        return {'SAM':{kw['target_ages'][0]:Image.new('RGB',(8,8),'red')}}
    monkeypatch.setattr(generator,'age_image',single)
    generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    old=config.generated_roots['SAM']/'id_new'/'p0.png';previous=old.read_bytes()
    panel=pd.read_csv(config.test_manifest_path)
    added={**panel.iloc[0].to_dict(),'sample_id':'p2','target_age':70}
    pd.concat([panel,pd.DataFrame([added])],ignore_index=True).to_csv(config.test_manifest_path,index=False)
    batches=[]
    class Model(torch.nn.Module):
        def forward(self,x,**kw):batches.append(len(x));return x[:,:3]
    entry=dict(model=Model(),align_face=lambda **kw:Image.new('RGB',(8,8)),predictor=None,
        AgeTransformer=lambda **kw:lambda x:x,tensor2im=lambda x:Image.new('RGB',(8,8),'blue'))
    result=generate_paper_predictions(config,comparison_bundle={'SAM':entry},device='cpu')
    assert batches==[1] and len(legacy)==2 and len(result['manifest'])==3, result['manifest'].to_dict('records')
    assert old.read_bytes()==previous
    assert 'AGING SWEEP COMPLETE' not in capsys.readouterr().out


def test_completed_phase_resumes_from_one_progress_file_without_scanning(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    import src.comparisong.load_models as loader
    import src.comparisong.prediction_resume as resume
    from src.comparisong.generation_cache import FingerprintCache
    config=configuration(tmp_path,['SAM'])
    monkeypatch.setattr(generator,'age_image',lambda bundle,**kw:{'SAM':{kw['target_ages'][0]:Image.new('RGB',(8,8))}})
    generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    def forbidden(*args,**kwargs):raise AssertionError('completed phase must not scan/load/reopen')
    monkeypatch.setattr(Image,'open',forbidden)
    monkeypatch.setattr(resume.os,'scandir',forbidden)
    monkeypatch.setattr(FingerprintCache,'get',forbidden)
    monkeypatch.setattr(loader,'load_aging_models',forbidden)
    result=generate_paper_predictions(config)
    assert len(result['manifest'])==2 and result['manifest'].inference_status.eq('ok').all()


def test_legacy_fast_import_never_opens_completed_images_or_markers(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    from src.comparisong.generation_cache import FingerprintCache
    from pathlib import Path
    config=configuration(tmp_path,['SAM'])
    monkeypatch.setattr(generator,'age_image',lambda bundle,**kw:{'SAM':{kw['target_ages'][0]:Image.new('RGB',(8,8))}})
    generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    (config.output_dir/'generation'/'resume_state.json').unlink()
    original=Path.read_text
    def read(self,*args,**kwargs):
        assert self.parent.name!='id_new','legacy completed marker must not be opened'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    def forbidden(*args,**kwargs):raise AssertionError('legacy import must not reopen PNGs or fingerprint sources')
    monkeypatch.setattr(Image,'open',forbidden)
    monkeypatch.setattr(FingerprintCache,'get',forbidden)
    assert len(generate_paper_predictions(config)['manifest'])==2


def test_fast_resume_does_not_adopt_files_rejected_by_interrupted_strict_scan(tmp_path,monkeypatch):
    import src.comparisong.generate_images as generator
    import src.comparisong.load_models as loader
    config=configuration(tmp_path,['SAM']);calls=[]
    def generate(bundle,**kw):
        age=kw['target_ages'][0];calls.append(age)
        return {'SAM':{age:Image.new('RGB',(8,8))}}
    monkeypatch.setattr(generator,'age_image',generate)
    generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    panel=pd.read_csv(config.test_manifest_path);panel.loc[0,'target_age']=70
    panel.to_csv(config.test_manifest_path,index=False)
    def fail(**kw):raise RuntimeError('setup interrupted')
    monkeypatch.setattr(loader,'load_aging_models',fail)
    with pytest.raises(RuntimeError,match='setup interrupted'):
        generate_paper_predictions(config,resume_validation='strict')
    result=generate_paper_predictions(config,comparison_bundle={'SAM':{}})
    assert calls==[40,60,70]
    assert result['manifest'].inference_status.eq('ok').all()
