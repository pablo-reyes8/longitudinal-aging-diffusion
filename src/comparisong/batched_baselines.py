"""Tensor batches for paper generation, preserving baseline preprocessing.

FRAN's original process_image is suspended at its sliding-window call; its
original preprocessing and compositing run unchanged. Only that tensor call
is shared across the batch. No neural weights are imported or fetched here.
"""
import ast
from collections import OrderedDict
import inspect
from pathlib import Path
import textwrap
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from src.aging_bias.features import stable_hash
from .resident_models import ACTIVE_MODEL_SESSION


def split_fran_process(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    class Split(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='aged_cropped_image' for t in node.targets):
                if not isinstance(node.value,ast.Call) or not node.value.args:
                    raise ValueError('Unsupported FRAN sliding-window assignment')
                self.count+=1
                node.value=ast.Yield(value=node.value.args[0])
            return self.generic_visit(node)
    split=Split();tree=split.visit(tree)
    if split.count!=1:raise ValueError('Unsupported FRAN runner: expected one image sliding-window call')
    tree.body[0].decorator_list=[]
    namespace=dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(function) or '<FRAN>', 'exec'),namespace)
    return namespace[function.__name__]


class BaselineBatchRunner:
    def __init__(self,name,entry,*,device,cache_dir,batch_size=32):
        self.name=name;self.entry=entry;self.device=device;self.batch_size=batch_size
        self.cache_dir=Path(cache_dir);self.memory=OrderedDict()
        session=ACTIVE_MODEL_SESSION.get()
        model=entry['model']
        self.model=session.place(model,device) if session else model.to(device).eval()
        entry['model']=self.model
        if name=='FRAN':
            self.process=split_fran_process(entry['process_image'])
            self.windows=entry.get('sliding_window_tensor') or entry['process_image'].__globals__['sliding_window_tensor']
            detector=self.process.__globals__.get('face_recognition')
            if detector is not None:
                def face_locations(image,*args,**kwargs):
                    key=('fran_face',self.current_source,repr(args),repr(sorted(kwargs.items())))
                    return self._remember(key,lambda:detector.face_locations(image,*args,**kwargs))
                self.process.__globals__['face_recognition']=SimpleNamespace(face_locations=face_locations)
        elif name not in {'SAM','CUSP'}:raise ValueError(f'No tensor batch adapter for {name}')

    def _remember(self,key,factory):
        if key not in self.memory:self.memory[key]=factory()
        self.memory.move_to_end(key)
        while len(self.memory)>64:self.memory.popitem(last=False)
        return self.memory[key]

    def _sam_base(self,sample):
        key=stable_hash(dict(source=sample.get('_source_digest',sample['source_path']),protocol='sam-align-jpeg95-256-v1'))
        def prepare():
            path=self.cache_dir/'sam_alignment'/(key+'.pt')
            if path.is_file():
                try:
                    tensor=torch.load(path,map_location='cpu',weights_only=True)
                    if tensor.shape==(3,256,256) and torch.isfinite(tensor).all():return tensor
                except (OSError,RuntimeError,ValueError):pass
            # Legacy age_image recompresses the RGB source as JPEG quality=95
            # before SAM alignment. Preserve that preprocessing for resume.
            with tempfile.TemporaryDirectory(prefix='sam_batch_') as folder:
                source=Path(folder)/'source.jpg'
                with Image.open(sample['source_path']) as image:image.convert('RGB').save(source,quality=95)
                aligned=self.entry['align_face'](filepath=str(source),predictor=self.entry['predictor'])
            image=aligned.resize((256,256),Image.Resampling.BILINEAR)
            tensor=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().div(255).sub(.5).div(.5)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp.pt');torch.save(tensor,temp);temp.replace(path)
            return tensor
        return self._remember(key,prepare)

    def _prepare(self,sample):
        if self.name=='SAM':
            tensor=self.entry['AgeTransformer'](target_age=sample['target_age'])(self._sam_base(sample).clone())
            return tensor.unsqueeze(0),lambda out:self.entry['tensor2im'](out[0]).convert('RGB')
        if self.name=='CUSP':
            return self.entry['prepare_batch_sample'](sample)
        def load():
            if self.entry.get('load_image'):return self.entry['load_image'](sample['source_path'])
            with Image.open(sample['source_path']) as image:return image.convert('RGB')
        image=self._remember(sample['source_path'],load)
        self.current_source=sample.get('_source_digest',sample['source_path'])
        generator=self.process(self.model,image.copy(),video=False,source_age=float(sample['source_age']),
            target_age=float(sample['target_age']),window_size=self.entry['window_size'],stride=self.entry['stride'])
        tensor=next(generator)
        def finish(out):
            try:generator.send(out)
            except StopIteration as done:return done.value
            raise RuntimeError('FRAN yielded more than one inference request')
        return tensor,finish

    def _forward(self,tensors,samples):
        tensor=torch.cat(tensors).to(self.device).float()
        if self.name=='SAM':
            result=self.model(tensor,randomize_noise=False,resize=False)
            return (result[0] if isinstance(result,(tuple,list)) else result).detach().cpu()
        if self.name=='CUSP':
            from .cusp_batch import reproducible_cusp_blur
            with reproducible_cusp_blur(self.model,samples,self.entry.get('seed',2026)):
                return self.entry['forward_batch'](self.model,tensor).detach().cpu()
        return self.windows(tensor,self.entry['window_size'],self.entry['stride'],self.model)

    def infer(self,samples):
        result=[None]*len(samples);ready=[]
        for index,sample in enumerate(samples):
            try:ready.append((index,*self._prepare(sample)))
            except Exception as exc:result[index]=exc
        def run(items):
            if not items:return
            error=None
            try:
                with torch.inference_mode():prediction=self._forward([item[1] for item in items],[samples[item[0]] for item in items])
                if len(prediction)!=len(items):raise ValueError('Baseline returned the wrong batch size')
            except Exception as exc:
                error=exc
            if error is not None:
                if len(items)>1:
                    middle=max(1,len(items)//2)
                    if isinstance(error,torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
                        self.batch_size=min(self.batch_size,middle)
                        print(f'[{self.name}] VRAM limit: reducing batch to {self.batch_size}',flush=True)
                        if torch.cuda.is_available():torch.cuda.empty_cache()
                    # Tracebacks can hold failed CUDA activations. Drop them
                    # before retrying smaller, independently saved batches.
                    error=None
                    run(items[:middle]);run(items[middle:]);return
                result[items[0][0]]=error;return
            for row,item in zip(prediction,items):
                try:
                    image=item[2](row.unsqueeze(0))
                    if not isinstance(image,Image.Image):raise ValueError('Baseline postprocessing returned no image')
                    result[item[0]]=image
                except Exception as exc:result[item[0]]=exc
        start=0
        while start<len(ready):
            chunk=ready[start:start+self.batch_size];run(chunk);start+=len(chunk)
        return result
