"""HRFAE tensor batches inside its existing isolated resident process.

Uses the actual checkout's setup, preprocess, test_eval, histogram correction,
clip_img and JPEG writer. This module never imports the host project packages.
"""
import ast
from collections import OrderedDict
import os
from pathlib import Path
import sys
import tempfile


def _setup(runtime,request):
    script=Path(request['cmd'][1]).resolve()
    tree=ast.parse(script.read_text(),filename=str(script));removed=0;body=[]
    for node in tree.body:
        # Keep all original setup and preprocessing, remove only image execution.
        is_execution=isinstance(node,(ast.With,ast.For)) and any(
            isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and call.func.attr=='test_eval' for call in ast.walk(node))
        if is_execution:removed+=1
        else:body.append(node)
    if removed!=1:raise ValueError('Unsupported HRFAE test.py: expected one test_eval image loop')
    tree.body=body
    work=tempfile.TemporaryDirectory(prefix='hrfae_batch_')
    namespace={'__name__':'__main__','__file__':str(script),'__package__':None}
    sys.argv=[*map(str,request['cmd'][1:]),'--target_age',str(int(request['samples'][0]['target_age'])),
        '--img_path',work.name+'/', '--out_path',work.name+'/']
    exec(compile(ast.fix_missing_locations(tree),str(script),'exec'),namespace)
    for name in ['trainer','preprocess','clip_img','utils','opts']:
        if name not in namespace:raise ValueError(f'HRFAE test.py does not expose {name}')
    if hasattr(namespace['trainer'],'eval'):namespace['trainer'].eval()
    runtime.model=namespace['trainer']
    return dict(namespace=namespace,work=work,cache=OrderedDict(),batch_size=int(request['batch_size']),
        key=(str(script),tuple(request['cmd'][2:])))


def run_batch(runtime,request):
    import torch
    from PIL import Image
    if runtime.name!='HRFAE':raise ValueError('Batch request sent to another baseline')
    if not request['samples']:return dict(results=[])
    old_cwd,old_argv,old_path=os.getcwd(),sys.argv,list(sys.path)
    rows=[]
    try:
        os.chdir(request['cwd']);sys.path.insert(0,str(Path(request['cmd'][1]).resolve().parent))
        state=getattr(runtime,'hrfae_batch_state',None)
        key=(str(Path(request['cmd'][1]).resolve()),tuple(request['cmd'][2:]))
        if state is None or state['key']!=key:
            state=_setup(runtime,request);runtime.hrfae_batch_state=state
        namespace=state['namespace'];trainer=namespace['trainer']
        device=namespace.get('device','cuda' if torch.cuda.is_available() else 'cpu')
        groups={}
        for sample in request['samples']:
            try:
                source=Path(sample['source_path']);stat=source.stat()
                cache_key=(str(source.resolve()),stat.st_size,stat.st_mtime_ns,sample.get('_source_digest'))
                if cache_key not in state['cache']:
                    # Preserve the old wrapper's RGB PNG conversion before the
                    # upstream preprocessing (including for grayscale/RGBA).
                    path=Path(state['work'].name)/'source.png'
                    with Image.open(source) as image:image.convert('RGB').save(path)
                    namespace['opts'].img_path=state['work'].name+'/'
                    state['cache'][cache_key]=namespace['preprocess'](path.name)
                state['cache'].move_to_end(cache_key)
                tensor=state['cache'][cache_key]
                while len(state['cache'])>64:state['cache'].popitem(last=False)
                groups.setdefault(int(sample['target_age']),[]).append((sample,tensor))
            except Exception as exc:
                rows.append(dict(sample_id=sample['sample_id'],status='failed',error=f'{type(exc).__name__}: {exc}'))
        def execute(items,age):
            error=None
            try:
                with torch.no_grad():
                    images=torch.stack([tensor for _,tensor in items]).to(device)
                    ages=torch.full((len(items),),age,dtype=torch.long,device=device)
                    prediction=trainer.test_eval(images,ages,target_age=age,hist_trans=True)
                if len(prediction)!=len(items):raise ValueError('HRFAE returned an incorrect batch size')
            except Exception as exc:error=exc
            if error is not None:
                if len(items)>1:
                    middle=max(1,len(items)//2)
                    if isinstance(error,torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
                        state['batch_size']=min(state['batch_size'],middle)
                    error=None
                    # Free temporary inputs before retrying; failed traceback
                    # is released above. OOM affects only this resident phase.
                    images=None;ages=None
                    if hasattr(trainer,'content_code_a'):trainer.content_code_a=None
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    execute(items[:middle],age);execute(items[middle:],age);return
                rows.append(dict(sample_id=items[0][0]['sample_id'],status='failed',error=f'{type(error).__name__}: {error}'))
                if hasattr(trainer,'content_code_a'):trainer.content_code_a=None
                return
            for index,(sample,_) in enumerate(items):
                try:
                    destination=Path(sample['output_path']);destination.parent.mkdir(parents=True,exist_ok=True)
                    # clip_img deliberately reads [0]; slice every image rather
                    # than writing one grid or discarding the rest of the batch.
                    namespace['utils'].save_image(namespace['clip_img'](prediction[index:index+1]),str(destination))
                    rows.append(dict(sample_id=sample['sample_id'],status='ok',output_path=str(destination)))
                except Exception as exc:
                    rows.append(dict(sample_id=sample['sample_id'],status='failed',error=f'{type(exc).__name__}: {exc}'))
            if hasattr(trainer,'content_code_a'):trainer.content_code_a=None
        for age,items in groups.items():
            start=0
            while start<len(items):
                size=min(int(request['batch_size']),state['batch_size'])
                chunk=items[start:start+size];execute(chunk,age);start+=len(chunk)
        return dict(results=rows,batch_size=state['batch_size'])
    finally:
        sys.argv,sys.path=old_argv,old_path;os.chdir(old_cwd)
