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
    generate_paper_predictions(config)
    assert calls==[40,60]
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
