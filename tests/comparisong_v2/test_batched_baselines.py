from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

from src.comparisong.batched_baselines import BaselineBatchRunner,split_fran_process


def toy_fran_windows(tensor,window_size,stride,model):
    return model(tensor)


def toy_fran_process(model,image,video,source_age,target_age=0,window_size=512,stride=256):
    # Same two-stage interface as the real upstream function, with tiny arrays.
    input_tensor=torch.tensor([[[[source_age]],[[target_age]]]],dtype=torch.float32)
    if video:raise ValueError('not used')
    aged_cropped_image=toy_fran_windows(input_tensor,window_size,stride,model)
    return Image.new('RGB',(2,2),(int(aged_cropped_image[0,0,0,0]),0,0))


def test_sam_real_batch_keeps_order_and_aligns_shared_source_once(tmp_path):
    calls=[];aligned=[]
    class Model(torch.nn.Module):
        def forward(self,x,**kw):
            calls.append(len(x));assert kw==dict(randomize_noise=False,resize=False)
            return x[:,:3]
    def align(**kw):aligned.append(kw['filepath']);return Image.new('RGB',(12,12),(30,50,70))
    def add_age(x):return torch.cat([x,torch.zeros_like(x[:1])])
    def to_image(x):return Image.fromarray(((x.permute(1,2,0).numpy()+1)*127.5).round().astype('uint8'))
    entry=dict(model=Model(),align_face=align,predictor=None,AgeTransformer=lambda **kw:add_age,tensor2im=to_image)
    source=tmp_path/'same.png';Image.new('RGB',(12,12)).save(source)
    samples=[dict(source_path=str(source),source_age=20,target_age=a) for a in [30,40,50]]
    runner=BaselineBatchRunner('SAM',entry,device='cpu',cache_dir=tmp_path,batch_size=32)
    result=runner.infer(samples)
    assert calls==[3] and len(aligned)==1
    assert len(result)==3 and all(isinstance(x,Image.Image) for x in result)
    assert all(np.array_equal(np.array(result[0]),np.array(x)) for x in result)


def test_fran_split_preserves_postprocessing_and_batches_network(tmp_path):
    calls=[]
    class Model(torch.nn.Module):
        def forward(self,x):calls.append(len(x));return x[:,1:2]-x[:,:1]
    entry=dict(model=Model(),process_image=toy_fran_process,sliding_window_tensor=toy_fran_windows,
        window_size=2,stride=1)
    samples=[dict(source_path='unused',source_age=20,target_age=a) for a in [30,40,50]]
    entry['load_image']=lambda path:Image.new('RGB',(2,2))
    runner=BaselineBatchRunner('FRAN',entry,device='cpu',cache_dir=tmp_path,batch_size=32)
    result=runner.infer(samples)
    assert calls==[3]
    assert [im.getpixel((0,0))[0] for im in result]==[10,20,30]
    singles=[toy_fran_process(entry['model'],None,False,20,a) for a in [30,40,50]]
    assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(result,singles))


def test_batch_oom_shrinks_without_reordering_or_losing_samples(tmp_path):
    calls=[]
    class Model(torch.nn.Module):
        def forward(self,x):
            calls.append(len(x))
            if len(x)>2:raise torch.cuda.OutOfMemoryError('toy OOM')
            return x[:,1:2]-x[:,:1]
    entry=dict(model=Model(),process_image=toy_fran_process,sliding_window_tensor=toy_fran_windows,
        window_size=2,stride=1,load_image=lambda path:Image.new('RGB',(2,2)))
    runner=BaselineBatchRunner('FRAN',entry,device='cpu',cache_dir=tmp_path,batch_size=32)
    samples=[dict(source_path='unused',source_age=20,target_age=a) for a in [30,40,50,60]]
    result=runner.infer(samples)
    assert calls==[4,2,2] and runner.batch_size==2
    assert [im.getpixel((0,0))[0] for im in result]==[10,20,30,40]


def test_cusp_native_conditioning_and_random_blur_are_batch_independent(tmp_path):
    from src.comparisong.cusp_batch import load_cusp_batch_entry
    calls=[]
    class Decoder(torch.nn.Module):
        def forward(self,x,noise_mode='random'):
            if noise_mode=='const':return x+.01
            return x+torch.rand_like(x)*.10
    class Model(torch.nn.Module):
        img_resolution=8
        def __init__(self):
            super().__init__();self.content_enc=torch.nn.Identity();self.attr_map=SimpleNamespace(z_dim=10)
            self.image_dec=Decoder()
        def _batch_blur(self,x,blur_val=None):return x+float(np.random.uniform(0,.05))
        def forward(self,img,cls,style_mixing=False):
            assert torch.is_grad_enabled()  # Real CUSP guided masks need it.
            calls.append((len(img),cls.argmax(1).tolist()))
            assert cls.shape==(len(img),10) and cls.sum(1).eq(1).all()
            return self.image_dec(self._batch_blur(torch.zeros_like(img)+cls.argmax(1).reshape(-1,1,1,1)/10))
    source=tmp_path/'source.png';Image.new('RGB',(8,8)).save(source)
    samples=[dict(source_path=str(source),source_age=20,target_age=a) for a in [25,45,75]]
    entry=load_cusp_batch_entry({'model':Model(),'variant':'ffhq_ls'},2026)
    runner=BaselineBatchRunner('CUSP',entry,device='cpu',cache_dir=tmp_path,batch_size=32)
    state=np.random.get_state()
    batched=runner.infer(samples)
    assert all(isinstance(x,Image.Image) for x in batched)
    assert calls==[(3,[5.,7.,9.])]
    assert np.array_equal(np.random.get_state()[1],state[1])
    singles=[runner.infer([sample])[0] for sample in samples]
    assert all(np.array_equal(np.array(a),np.array(b)) for a,b in zip(batched,singles))
    assert 'forward' not in vars(entry['model'].image_dec)


def test_one_bad_preprocessing_does_not_drop_other_batch_items(tmp_path):
    calls=[]
    class Model(torch.nn.Module):
        def forward(self,x,**kw):calls.append(len(x));return x[:,:3]
    source=tmp_path/'ok.png';Image.new('RGB',(4,4)).save(source)
    entry=dict(model=Model(),align_face=lambda **kw:Image.new('RGB',(4,4)),predictor=None,
        AgeTransformer=lambda **kw:lambda x:x,tensor2im=lambda x:Image.new('RGB',(4,4)))
    runner=BaselineBatchRunner('SAM',entry,device='cpu',cache_dir=tmp_path,batch_size=32)
    result=runner.infer([dict(source_path=str(source),target_age=40),dict(source_path='missing.png',target_age=50)])
    assert isinstance(result[0],Image.Image) and isinstance(result[1],Exception)
    assert calls==[1]


def test_generation_session_skips_legacy_figures_and_duplicate_pngs(tmp_path):
    from src.comparisong.generate_images import _save_model_sweep,_save_standard_result
    from src.comparisong.resident_models import ModelSession
    output=tmp_path/'unused_figures'
    with ModelSession({},'FADING',generation_only=True):
        assert _save_model_sweep(None,20,'FADING',{},[40],str(output)) is None
        assert _save_standard_result(None,str(output),'FADING',40) is None
    assert not output.exists()
