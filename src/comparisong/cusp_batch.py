"""Native CUSP Image2Image batching, replacing the non-executable CLI stub."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import MethodType
import numpy as np
from PIL import Image
import torch
from src.aging_bias.features import stable_hash


def load_cusp_batch_entry(entry,seed):
    entry=dict(entry)
    if entry.get('model') is None:
        root=Path(entry['root']);checkpoint=Path(entry['checkpoint'])
        if not checkpoint.is_file():raise FileNotFoundError(f'CUSP requires a local network .pkl: {checkpoint}')
        # Use the repository's persistence-aware reader, never a generic GAN
        # projector: CUSP is an image-to-image network with an attribute map.
        old_path=list(sys.path);sys.path.insert(0,str(root))
        try:
            spec=importlib.util.spec_from_file_location('_paper_cusp_legacy',root/'legacy.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with checkpoint.open('rb') as stream:payload=module.load_network_pkl(stream)
            entry['model']=payload['G_ema'].eval().to('cpu')
        finally:sys.path[:]=old_path
    model=entry['model']
    if not hasattr(model,'content_enc') or not hasattr(model,'attr_map'):
        raise ValueError('CUSP checkpoint must contain the official Image2Image G_ema')
    variant=entry.get('variant','ffhq_ls')
    if variant not in {'ffhq_ls','ffhq_rr'}:raise ValueError(f'Unsupported CUSP variant: {variant}')
    label_dim=10 if variant=='ffhq_ls' else 101
    if getattr(model.attr_map,'z_dim',None)!=label_dim:
        raise ValueError(f'CUSP {variant} generator requires {label_dim}-way age labels; checkpoint conditioning differs')
    resolution=int(model.img_resolution) if hasattr(model,'img_resolution') else (256 if variant=='ffhq_ls' else 224)
    for parameter in model.parameters():parameter.requires_grad_(False)
    def prepare(sample):
        with Image.open(sample['source_path']) as image:
            image=image.convert('RGB')
            # Match Resize(short edge), then center to the checkpoint's square
            # domain; no generated face alignment or identity substitution.
            width,height=image.size;scale=resolution/min(width,height)
            image=image.resize((int(width*scale),int(height*scale)),Image.Resampling.BILINEAR)
            left=(image.width-resolution)//2;top=(image.height-resolution)//2
            image=image.crop((left,top,left+resolution,top+resolution))
            tensor=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().div(127.5).sub(1)
        age=float(sample['target_age'])
        condition=sum(age>=edge for edge in [3,7,10,15,20,30,40,50,70]) if variant=='ffhq_ls' else age
        if variant=='ffhq_rr' and not 0<=age<=100:raise ValueError('CUSP FFHQ-RR supports target ages 0..100')
        label=torch.full_like(tensor[:1],condition)
        def finish(out):
            array=(out[0].permute(1,2,0)*127.5+128).clamp(0,255).to(torch.uint8).numpy()
            return Image.fromarray(array,'RGB')
        return torch.cat([tensor,label]).unsqueeze(0),finish
    def forward(network,tensor):
        # Guided-backprop masks in the released CUSP model require gradients
        # w.r.t. the input even during inference; no parameters are optimized.
        with torch.inference_mode(False),torch.enable_grad(),constant_cusp_noise(network):
            labels=torch.nn.functional.one_hot(tensor[:,3,0,0].long(),num_classes=label_dim).float()
            return network(tensor[:,:3].clone(),labels,style_mixing=False)
    entry.update(prepare_batch_sample=prepare,forward_batch=forward,seed=seed)
    return entry


@contextmanager
def constant_cusp_noise(model):
    """Use checkpoint noise maps, independent of batch partition or resume."""
    decoder=getattr(model,'image_dec',None)
    if decoder is None:
        yield;return
    original=decoder.forward;had_instance='forward' in vars(decoder);previous=vars(decoder).get('forward')
    def forward(this,*args,**kwargs):
        return original(*args,**{**kwargs,'noise_mode':'const'})
    decoder.forward=MethodType(forward,decoder)
    try:yield
    finally:
        if had_instance:decoder.forward=previous
        else:delattr(decoder,'forward')


@contextmanager
def reproducible_cusp_blur(model,samples,seed):
    """Keep per-image random structural blur stable across batch/resume splits."""
    original=getattr(model,'_batch_blur',None)
    if original is None:
        yield;return
    had_instance='_batch_blur' in vars(model);previous=vars(model).get('_batch_blur');counter=0
    def blur(this,tensor,blur_val=None):
        nonlocal counter
        call=counter;counter+=1
        if blur_val is not None:return original(tensor,blur_val=blur_val)
        outputs=[];state=np.random.get_state()
        try:
            for index,sample in enumerate(samples):
                key=stable_hash(dict(seed=seed,source=sample.get('_source_digest',sample['source_path']),
                    source_age=sample['source_age'],target_age=sample['target_age'],call=call))
                np.random.seed(int(key[:8],16))
                outputs.append(original(tensor[index:index+1],blur_val=None))
            return torch.cat(outputs)
        finally:np.random.set_state(state)
    model._batch_blur=MethodType(blur,model)
    try:yield
    finally:
        if had_instance:model._batch_blur=previous
        else:delattr(model,'_batch_blur')
