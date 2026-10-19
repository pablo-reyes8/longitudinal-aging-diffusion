import json
import numpy as np
import pandas as pd
from PIL import Image
import pytest


def test_sd15_paired_seed_once_loaded_restore_and_resume(tmp_path):
    from src.comparisong.sd15_counterfactual import run_sd15_before_after
    from src.aging_bias.config import SD15Config
    from test_sd15 import toy
    b=toy();conv=b['unet'].conv_in
    b['_sd15_original_conv_in']={k:v.clone() for k,v in conv.state_dict().items()}
    source=tmp_path/'source.png';target=tmp_path/'target.png'
    Image.new('RGB',(16,16),(100,110,120)).save(source);Image.new('RGB',(16,16),(120,110,100)).save(target)
    frame=pd.DataFrame([dict(sample_id='p',identity_id='id_1',source_path=str(source),target_path=str(target),source_age=20,target_age=50)])
    frame.to_csv(tmp_path/'panel.csv',index=False)
    (tmp_path/'anchors.json').write_text(json.dumps({'passed':True}))
    checkpoint=tmp_path/'adapter.pt';checkpoint.write_bytes(b'toy checkpoint only')
    config=SD15Config(checkpoint,tmp_path/'panel.csv',tmp_path/'out',tmp_path/'anchors.json',
        anchors_reviewed=True,device='cpu',strengths=(.1,.3))
    calls=[]
    def infer(**kwargs):
        calls.append((kwargs['seed'],kwargs['strength'],b['use_age_delta_conditioning']))
        return {'image':Image.open(source)}
    before={k:v.clone() for k,v in conv.state_dict().items()}
    result=run_sd15_before_after(config,bundle=b,infer_backend=infer)
    assert calls==[(2026,.1,False),(2026,.3,False),(2026,.1,True),(2026,.3,True)]
    assert len(result['manifest'])==4
    # A resumed run must not even construct diffusion weights.
    result=run_sd15_before_after(config,infer_backend=lambda **k:pytest.fail('inference repeated'))
    assert len(result['manifest'])==4
    for k,v in conv.state_dict().items():assert (v==before[k]).all()
    config.anchors_reviewed=False
    with pytest.raises(ValueError,match='review'):run_sd15_before_after(config,bundle=b,infer_backend=infer)


def test_missing_all_generated_produces_failure_report_not_crash(tmp_path):
    from test_evaluator import setup
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    for path in tmp_path.glob('M*/*/*.png'):path.unlink()
    for path in tmp_path.glob('SAM/*/*.png'):path.unlink()
    with pytest.warns(UserWarning):result=evaluate_paper_v2(cfg,bundle=bundle)
    assert result['metadata']['n_common_pairs']==0
    assert len(result['failures'])>=90


def test_download_wrapper_never_connects_when_asset_present_or_download_disabled(tmp_path,monkeypatch):
    from src.comparisong.evaluation_backends_v2 import _download
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('unexpected network access'))
    path=tmp_path/'asset';path.write_bytes(b'x'*2048)
    assert _download('https://example.test/asset',path,False)==path
    with pytest.raises(FileNotFoundError):_download('https://example.test/asset',tmp_path/'missing',False)


def test_counterfactual_loader_normalizes_serialized_dtype_and_snapshots_before_load(tmp_path,monkeypatch):
    import functools
    import torch
    import src.model.load_diffusion_models as model_api
    import src.inference as inference
    from test_sd15 import toy
    from src.comparisong.sd15_counterfactual import load_counterfactual_bundle
    calls=[]
    @functools.wraps(model_api.build_face_aging_diffusion_bundle)
    def build(**kwargs):
        assert kwargs['trainable_dtype'] is torch.float32
        assert kwargs['load_auxiliary_models'] is False
        calls.append('build');return toy()
    def apply(bundle,*args,**kwargs):
        before=bundle['unet'].conv_in.weight.clone()
        assert torch.equal(bundle['_sd15_original_conv_in']['weight'],before)
        calls.append('apply')
    monkeypatch.setattr(model_api,'build_face_aging_diffusion_bundle',build)
    monkeypatch.setattr(inference,'load_face_aging_adapter_for_inference',apply)
    path=tmp_path/'checkpoint.pt';torch.save({'config':{'model_id':'synthetic/base','trainable_dtype':'torch.float32'}},path)
    load_counterfactual_bundle(path)
    assert calls==['build','apply']


def test_changed_sd_protocol_failed_refresh_cannot_expose_stale_images(tmp_path):
    from src.comparisong.sd15_counterfactual import run_sd15_before_after
    from src.aging_bias.config import SD15Config
    from test_sd15 import toy
    b=toy();b['_sd15_original_conv_in']={k:v.clone() for k,v in b['unet'].conv_in.state_dict().items()}
    source=tmp_path/'source.png';Image.new('RGB',(16,16),(100,100,100)).save(source)
    pd.DataFrame([dict(sample_id='p',identity_id='id_1',source_path=str(source),target_path=str(source),source_age=20,target_age=50)]).to_csv(tmp_path/'panel.csv',index=False)
    (tmp_path/'anchors.json').write_text('{"passed": true}')
    (tmp_path/'ckpt.pt').write_bytes(b'toy checkpoint')
    c=SD15Config(tmp_path/'ckpt.pt',tmp_path/'panel.csv',tmp_path/'out',tmp_path/'anchors.json',
        anchors_reviewed=True,device='cpu',strengths=(.2,))
    run_sd15_before_after(c,bundle=b,infer_backend=lambda **kw:{'image':Image.open(source)})
    c.seed=33
    def fail(**kw):raise ValueError('synthetic failed regeneration')
    result=run_sd15_before_after(c,bundle=b,infer_backend=fail)
    assert result['manifest'].inference_status.eq('failed').all()
    assert not list((tmp_path/'out/generated').rglob('*.png'))
    assert len(list((tmp_path/'out/stale').rglob('*.png')))==2
    # Failed same-signature transitions are reused as failures, not reexecuted.
    run_sd15_before_after(c,infer_backend=lambda **kw:pytest.fail('failed transition retried'))


def test_sd_missing_source_persists_failure_without_loading_model(tmp_path):
    from src.comparisong.sd15_counterfactual import run_sd15_before_after
    from src.aging_bias.config import SD15Config
    (tmp_path/'anchors.json').write_text('{"passed": true}');(tmp_path/'ckpt.pt').write_bytes(b'toy')
    pd.DataFrame([dict(sample_id='p',identity_id='id_1',source_path=str(tmp_path/'missing.png'),target_path='unused',source_age=20,target_age=50)]).to_csv(tmp_path/'panel.csv',index=False)
    c=SD15Config(tmp_path/'ckpt.pt',tmp_path/'panel.csv',tmp_path/'out',tmp_path/'anchors.json',anchors_reviewed=True,strengths=(.2,))
    result=run_sd15_before_after(c)
    assert len(result['failures'])==2
    assert result['manifest'].inference_status.eq('failed').all()
    assert (tmp_path/'out/inference_failures.csv').exists()


def test_numeric_manifest_ids_normalized_without_losing_leading_zeros(tmp_path):
    from src.comparisong.evaluate_paper_v2 import load_test_panel
    p=tmp_path/'panel.csv'
    p.write_text('sample_id,identity_id,source_age,target_age,source_path,target_path\n001,0002,20,40,s.png,t.png\n')
    panel=load_test_panel(p)
    assert panel.iloc[0].sample_id=='001' and panel.iloc[0].identity_id=='0002'
