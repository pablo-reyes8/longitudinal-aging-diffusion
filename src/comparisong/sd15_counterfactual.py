"""Paired SD1.5 counterfactual, without modifying the public inference module."""
from contextlib import contextmanager,nullcontext
from dataclasses import asdict
from pathlib import Path
import inspect
import json
import shutil
import uuid
import numpy as np
import pandas as pd
import torch
from src.aging_bias.features import stable_hash
from src.quantitative_metrics.image_io import ImageResolver
from .evaluate_paper_v2 import load_test_panel


@contextmanager
def adapters_disabled(bundle):
    """Bypass manual DoRA/LoRA; zero source channels and age contribution.

    Real counterfactual runs MUST supply _sd15_original_conv_in captured BEFORE
    checkpoint application: conv_in's original four channels and bias are also
    trainable. Keeping their fine-tuned values would not be vanilla SD1.5.
    Toy/custom callers without that snapshot preserve their current first four.
    Every temporary tensor/hook/flag is restored even on exceptions.
    """
    from src.model.DoRa import DoRALinear
    from src.model.LoRa import LoRALinear
    unet=bundle['unet'];conv=unet.conv_in
    if conv.in_channels!=8:raise ValueError('Counterfactual expects an expanded 8-channel conv_in')
    weight=conv.weight.detach().clone();bias=conv.bias.detach().clone() if conv.bias is not None else None
    hooks=[];conditioners=[];scales=[]
    present='use_age_delta_conditioning' in bundle;flag=bundle.get('use_age_delta_conditioning')
    try:
        for module in unet.modules():
            if isinstance(module,(DoRALinear,LoRALinear)):
                hooks.append(module.register_forward_hook(lambda m,args,out:m.base_layer(args[0])))
        for key in ['age_conditioner','age_delta_conditioner']:
            module=bundle.get(key)
            if module is None or id(module) in conditioners:continue
            conditioners.append(id(module))
            hooks.append(module.register_forward_hook(lambda m,args,out:torch.zeros_like(out)))
            gate=getattr(module,'age_scale',None)
            if gate is not None:scales.append((gate,gate.detach().clone()))
        with torch.no_grad():
            original=bundle.get('_sd15_original_conv_in')
            if original is not None:
                conv.weight[:,:4].copy_(original['weight'][:,:4].to(conv.weight))
                if conv.bias is not None and original.get('bias') is not None:conv.bias.copy_(original['bias'].to(conv.bias))
            conv.weight[:,4:].zero_()
            for gate,_ in scales:gate.zero_()
        bundle['use_age_delta_conditioning']=False
        yield bundle
    finally:
        with torch.no_grad():
            conv.weight.copy_(weight)
            if bias is not None:conv.bias.copy_(bias)
            for gate,value in scales:gate.copy_(value)
        for hook in hooks:hook.remove()
        if present:bundle['use_age_delta_conditioning']=flag
        else:bundle.pop('use_age_delta_conditioning',None)


def load_counterfactual_bundle(checkpoint_path,*,device='cpu',local_files_only=True):
    """Build once, snapshot pristine conv_in, then apply the trained checkpoint once."""
    from src.model.load_diffusion_models import build_face_aging_diffusion_bundle
    from src.inference import load_face_aging_adapter_for_inference
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=dict(payload.get('config') or payload.get('bundle_config') or {})
    parameters=inspect.signature(build_face_aging_diffusion_bundle).parameters
    options={k:v for k,v in config.items() if k in parameters}
    dtype_names={'torch.float32':torch.float32,'float32':torch.float32,
        'torch.float16':torch.float16,'float16':torch.float16,
        'torch.bfloat16':torch.bfloat16,'bfloat16':torch.bfloat16}
    for key in ['trainable_dtype','dtype','auxiliary_dtype']:
        if isinstance(options.get(key),str):
            if options[key] not in dtype_names:raise ValueError(f'Unsupported checkpoint dtype: {options[key]}')
            options[key]=dtype_names[options[key]]
    if 'model_id' not in options:raise ValueError('Checkpoint must record the base model_id')
    options.update(device=device,local_files_only=local_files_only,load_auxiliary_models=False)
    bundle=build_face_aging_diffusion_bundle(**options)
    conv=bundle['unet'].conv_in
    bundle['_sd15_original_conv_in']={k:v.detach().cpu().clone() for k,v in conv.state_dict().items()}
    load_face_aging_adapter_for_inference(bundle,checkpoint_path)
    return bundle


def run_sd15_before_after(config,*,bundle=None,infer_backend=None):
    """Generate only when explicitly called; failures persisted per transition.

    This is SD1.5 backbone under the SAME project direct-img2img/CFG protocol,
    not the stock Diffusers pipeline with different guidance semantics.
    """
    if not config.anchors_reviewed:raise ValueError('Inspect real_anchors.json and set anchors_reviewed=True before the counterfactual')
    anchors=json.loads(Path(config.validation_path).read_text())
    if not anchors.get('passed'):raise ValueError('Real anchors failed/unreviewed; fix references and inspect validation before SD1.5 generation')
    if not config.strengths or any(not 0<float(s)<=1 for s in config.strengths):raise ValueError('strengths must be in (0,1]')
    from .evaluation_backends_v2 import file_hash
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    panel=load_test_panel(config.test_manifest_path,config.source_root)
    panel=panel[panel.delta_age>0].sort_values(['delta_age','sample_id'],ascending=[False,True])
    if config.max_pairs is not None:panel=panel.head(config.max_pairs)
    if panel.empty:raise ValueError('No aging transitions for SD1.5 counterfactual')
    protected={'bundle','image','source_age','target_age','strength','seed','mode','generator','use_inverse_diffusion','use_delta_dependent_strength'}
    if protected&set(config.inference_config):raise ValueError('inference_config overrides paired protocol: '+str(protected&set(config.inference_config)))
    signature=stable_hash(dict(checkpoint_sha256=file_hash(config.checkpoint_path),strengths=config.strengths,
        inference=config.inference_config,seed=config.seed,version='sd15-paired-v1'))
    rows=[];failures=[];roots={};resolver=ImageResolver();device_active=False
    def quarantine(path,marker):
        if not path.exists():return
        stale=output/'stale'/path.relative_to(output/'generated')
        stale=stale.with_name(stale.stem+'_'+uuid.uuid4().hex+stale.suffix)
        stale.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),str(stale))
        if marker.exists():shutil.copy2(marker,stale.with_suffix('.json'))
    # Lazy: a fully resumed run does not load any diffusion weights.
    live=bundle
    try:
        for condition in ['vanilla','finetuned']:
            for strength in config.strengths:
                name=('SD15_vanilla' if condition=='vanilla' else 'ours')+f'@{float(strength):g}'
                roots[name]=str(output/'generated'/name)
                pending=[]
                for sample in panel.to_dict('records'):
                    path=output/'generated'/name/sample['identity_id']/(sample['sample_id']+'.png')
                    marker=path.with_suffix('.json')
                    try:
                        source_digest=resolver.load(sample['source_path'])[1]
                    except Exception as exc:
                        quarantine(path,marker)
                        row={**sample,'model_name':name,'generated_path':None,'strength':strength,'condition':condition,
                            'seed':config.seed,'inference_status':'failed','inference_error':f'source_load: {type(exc).__name__}: {exc}'}
                        rows.append(row);failures.append(row.copy())
                        pd.DataFrame(failures).to_csv(output/'inference_failures.csv',index=False)
                        continue
                    key=stable_hash(dict(run=signature,source=source_digest,
                        source_age=sample['source_age'],target_age=sample['target_age'],condition=condition,strength=float(strength)))
                    resolver._cache.clear()
                    cached=None
                    if config.resume and marker.exists():
                        try:cached=json.loads(marker.read_text())
                        except (OSError,ValueError):pass
                    valid=False
                    if cached and cached.get('signature')==key and cached.get('status')!='failed' and path.exists():
                        from PIL import Image
                        try:
                            with Image.open(path) as im:im.verify()
                            valid=True
                        except OSError:pass
                    if not valid:quarantine(path,marker)
                    row={**sample,'model_name':name,'generated_path':str(path),'strength':strength,
                        'condition':condition,'seed':config.seed,'inference_status':'ok'}
                    if cached and cached.get('signature')==key and cached.get('status')=='failed' and not config.retry_failed:
                        row.update(inference_status='failed',generated_path=None,inference_error=cached.get('error','cached failure'))
                        rows.append(row);failures.append(row.copy())
                    elif valid:rows.append(row)
                    else:pending.append((sample,path,marker,key,row))
                if not pending:continue
                if live is None:live=load_counterfactual_bundle(config.checkpoint_path,device=config.device,local_files_only=config.local_files_only)
                if '_sd15_original_conv_in' not in live:raise ValueError('Use load_counterfactual_bundle: pristine conv_in is required for genuine SD1.5')
                if infer_backend is None:
                    from src.inference import infer_face_aging_direct
                    infer_backend=infer_face_aging_direct
                if not device_active:
                    for key in ['unet','vae','text_encoder','age_delta_conditioner']:
                        if live.get(key) is not None:live[key].to(config.device)
                    device_active=True
                context=adapters_disabled(live) if condition=='vanilla' else nullcontext(live)
                with context:
                    for sample,path,marker,key,row in pending:
                        try:
                            options=dict(config.inference_config)
                            options.update(compute_diagnostics=False,output_type='pil',return_dict=True)
                            result=infer_backend(bundle=live,image=sample['source_path'],source_age=int(sample['source_age']),
                                target_age=int(sample['target_age']),strength=float(strength),seed=config.seed,**options)
                            image=result['image'] if isinstance(result,dict) else result
                            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.png');image.save(temp);temp.replace(path)
                            marker.write_text(json.dumps(dict(signature=key,condition=condition)))
                        except Exception as exc:
                            row.update(inference_status='failed',generated_path=None,inference_error=f'{type(exc).__name__}: {exc}')
                            failures.append(row.copy())
                            marker.parent.mkdir(parents=True,exist_ok=True)
                            marker.write_text(json.dumps(dict(signature=key,status='failed',error=row['inference_error'])))
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(output/'prediction_manifest.csv',index=False)
                        pd.DataFrame(failures).to_csv(output/'inference_failures.csv',index=False)
    finally:
        if live is not None:
            for key in ['unet','vae','text_encoder','age_delta_conditioner']:
                if live.get(key) is not None:live[key].to('cpu')
    frame=pd.DataFrame(rows);frame.to_csv(output/'prediction_manifest.csv',index=False)
    pd.DataFrame(failures,columns=[*panel.columns,'model_name','inference_status','inference_error']).to_csv(output/'inference_failures.csv',index=False)
    panel.to_csv(output/'transitions.csv',index=False)
    (output/'run_config.json').write_text(json.dumps(dict(signature=signature,config=asdict(config),generated_roots=roots),indent=2,default=str))
    return dict(manifest=frame,generated_roots=roots,failures=pd.DataFrame(failures),paths={'manifest':str(output/'prediction_manifest.csv')})
