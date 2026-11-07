import json
from pathlib import Path
import sys
from PIL import Image
from src.comparisong.resident_models import ModelSession


def fake_hrfae_script(tmp_path):
    # Original CLI shape with tiny tensor boundaries; no real weights/GPU.
    script=tmp_path/'test.py'
    script.write_text('''import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
parser=argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--target_age',type=int)
parser.add_argument('--img_path',default='./test/input/')
parser.add_argument('--out_path',default='./test/output/')
opts=parser.parse_args()
device='cpu'
config={}
class Trainer:
    def __init__(self,config):
        with open('loads.txt','a') as f:f.write('load\\n')
    def load_checkpoint(self,path):pass
    def to(self,device):return self
    def test_eval(self,images,ages,target_age=0,hist_trans=True):
        with open('batches.txt','a') as f:f.write(str((len(images),target_age))+'\\n')
        assert ages.eq(target_age).all() and hist_trans
        return images+target_age/255
trainer=Trainer(config)
trainer.load_checkpoint('fake')
trainer.to(device)
def preprocess(name):
    image=Image.open(opts.img_path+name).convert('RGB')
    return torch.tensor(np.array(image).copy()).permute(2,0,1).float()/255
def clip_img(tensor):return [tensor[0].clamp(0,1)]
class Utils:
    @staticmethod
    def save_image(images,path):
        image=(images[0].permute(1,2,0)*255+.5).clamp(0,255).byte().numpy()
        Image.fromarray(image).save(path)
utils=Utils()
with torch.no_grad():
    for name in []:
        trainer.test_eval(None,None)
''')
    return script


def test_hrfae_batch_worker_loads_once_groups_age_and_matches_singleton_codec(tmp_path):
    script=fake_hrfae_script(tmp_path)
    source=tmp_path/'source.png';Image.new('RGB',(8,8),'gray').save(source)
    output=tmp_path/'output';output.mkdir()
    entries=[dict(sample_id=str(i),source_path=str(source),source_age=20,target_age=age,
        output_path=str(output/f'{i}.jpg')) for i,age in enumerate([40,60,40])]
    with ModelSession({'HRFAE':{'script':str(script)}},'HRFAE') as session:
        result=session.run_hrfae_batch([sys.executable,str(script),'--config','fake'],
            str(tmp_path),entries,batch_size=32)
        assert all(row['status']=='ok' for row in result['results'])
        again=session.run_hrfae_batch([sys.executable,str(script),'--config','fake'],
            str(tmp_path),[{**entries[0],'output_path':str(output/'single.jpg')}],batch_size=1)
        assert again['results'][0]['status']=='ok'
    assert (tmp_path/'loads.txt').read_text().splitlines()==['load']
    assert (tmp_path/'batches.txt').read_text().splitlines()==['(2, 40)','(1, 60)','(1, 40)']
    assert (output/'0.jpg').read_bytes()==(output/'single.jpg').read_bytes()


def test_hrfae_pipeline_batches_pending_and_preserves_compatible_signatures(tmp_path,capsys):
    from types import SimpleNamespace
    import hashlib
    import pandas as pd
    from src.aging_bias.features import stable_hash
    from src.comparisong.generate_paper_predictions import generate_paper_predictions
    script=fake_hrfae_script(tmp_path)
    source=tmp_path/'source.png';Image.new('RGB',(8,8),'gray').save(source)
    panel=pd.DataFrame([dict(sample_id=str(i),identity_id='id_1',source_path=str(source),
        target_path=str(source),source_age=20,target_age=age) for i,age in enumerate([40,40,40,60])])
    manifest=tmp_path/'manifest.csv';panel.to_csv(manifest,index=False)
    root=tmp_path/'generated'/'HRFAE';root.joinpath('id_1').mkdir(parents=True)
    old=root/'id_1'/'0.png';Image.new('RGB',(8,8),'red').save(old);previous=old.read_bytes()
    signature=stable_hash(dict(protocol='hrfae-batched-eval-v1',model='HRFAE',source=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_age=20,target_age=40,seed=2026,gender=None,inference=None,checkpoint=None,loader=None))
    old.with_suffix('.json').write_text(json.dumps(dict(signature=signature,status='ok')))
    config=SimpleNamespace(test_manifest_path=manifest,source_root=None,models=['HRFAE'],generated_roots={'HRFAE':root},
        output_dir=tmp_path/'metrics',resume=True,seed=2026)
    bundle={'HRFAE':dict(script=str(script),root=str(tmp_path),config='fake')}
    result=generate_paper_predictions(config,comparison_bundle=bundle)
    assert len(result['manifest'])==4 and result['manifest'].inference_status.eq('ok').all()
    assert old.read_bytes()==previous
    assert (tmp_path/'batches.txt').read_text().splitlines()==['(2, 40)','(1, 60)']
    generate_paper_predictions(config,comparison_bundle=bundle)
    assert (tmp_path/'loads.txt').read_text().splitlines()==['load']
    output=capsys.readouterr().out
    assert 'AGING SWEEP COMPLETE' not in output and '[HRFAE] 4/4 | remaining=0' in output


def test_hrfae_oom_releases_retained_features_and_freezes_spectral_norm(tmp_path):
    script=fake_hrfae_script(tmp_path)
    text=script.read_text().replace('class Trainer:', 'class Trainer(torch.nn.Module):')
    text=text.replace('    def __init__(self,config):\n',
        '    def __init__(self,config):\n        super().__init__()\n'
        '        torch.manual_seed(2026)\n'
        '        self.layer=torch.nn.utils.spectral_norm(torch.nn.Conv2d(3,3,1))\n')
    text=text.replace('        return images+target_age/255',
        "        assert not self.training\n"
        "        assert getattr(self,'content_code_a',None) is None\n"
        "        before=self.layer.weight_u.clone()\n"
        "        self.content_code_a=self.layer(images)\n"
        "        assert torch.equal(before,self.layer.weight_u)\n"
        "        if len(images)>2:raise torch.cuda.OutOfMemoryError('toy OOM')\n"
        "        return self.content_code_a+target_age/255")
    script.write_text(text)
    source=tmp_path/'source.png';Image.new('RGB',(8,8),'gray').save(source)
    output=tmp_path/'output';output.mkdir()
    samples=[dict(sample_id=str(i),source_path=str(source),source_age=20,target_age=40,
        output_path=str(output/f'{i}.jpg')) for i in range(4)]
    with ModelSession({'HRFAE':{'script':str(script)}},'HRFAE') as session:
        result=session.run_hrfae_batch([sys.executable,str(script),'--config','fake'],str(tmp_path),samples,batch_size=32)
        assert result['batch_size']==2 and all(row['status']=='ok' for row in result['results'])
        result=session.run_hrfae_batch([sys.executable,str(script),'--config','fake'],str(tmp_path),
            [{**samples[0],'output_path':str(output/'single.jpg')}],batch_size=1)
        assert result['results'][0]['status']=='ok'
    assert (output/'0.jpg').read_bytes()==(output/'single.jpg').read_bytes()
