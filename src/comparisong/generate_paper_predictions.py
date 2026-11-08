"""Generate predictions on an existing paper manifest, without legacy metrics."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import os
import sys
import tempfile
import json
import pandas as pd
from PIL import Image
from src.aging_bias.features import stable_hash
from .evaluate_paper_v2 import load_test_panel
from .identity_metadata import TEST_ID_GENDERS,load_gender_csv
from .resident_models import ModelSession


def generate_paper_predictions(config, *, checkpoint_path=None, comparison_bundle=None,
        model_loader_config=None, gender_by_id=None, inference_config=None,
        picking_options=None, picking_bundle=None, device='cuda',
        local_files_only=False, retry_failed=False, gender_csv_path=None,
        fading_group_targets=True, batch_size=32, progress_every=100,
        resume_validation='fast', resume_workers=16):
    """Generate exactly config's panel and roots; save progress after every pair.

    No evaluation, test sampling, or module-1 writes. Missing gender is checked
    before generation for gender-dependent models; no inferred default labels.
    Fast resume trusts saved progress for unchanged inputs/parameters. Legacy
    runs import PNG+JSON filenames once, without opening every image/marker.
    resume_validation='strict' validates signatures and changed artifacts.
    Gender precedence: explicit dict > CSV > manifest > legacy labels. FADING
    groups pending targets by source to reuse its original CLI inversion. A
    failed group falls back to individual requests without dropping transitions.
    """
    print('[generate paper] Reading manifest and validating configuration...',flush=True)
    for key,value in [('batch_size',batch_size),('progress_every',progress_every),('resume_workers',resume_workers)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'{key} must be a positive integer')
    if resume_validation not in {'fast','strict'}:raise ValueError('resume_validation must be fast or strict')
    fast=config.resume and resume_validation=='fast'
    from .generate_images import age_image
    from .load_models import load_aging_models
    from .evaluation_backends_v2 import file_hash
    panel=load_test_panel(config.test_manifest_path,config.source_root)
    excluded=panel[panel.delta_age<0].copy()
    aging_only=getattr(config,'aging_only',True)
    if aging_only:
        panel=panel[panel.delta_age>0].copy()
        if panel.empty:raise ValueError('No aging transitions to generate')
    print(f'[generate paper] {len(panel)} transitions / {panel.source_path.nunique()} unique source images',flush=True)
    names=list(dict.fromkeys(config.models))
    external=[n for n in names if not n.startswith('ours_')]
    supported={'SAM','FRAN','FADING','HRFAE','CUSP','Cradle2Cane'}
    if set(external)-supported:raise ValueError(f'Unsupported models: {set(external)-supported}')
    picking=[n for n in names if n in {'ours_picking_no_prompt','ours_picking_prompt'}]
    direct=[n for n in names if n.startswith('ours_') and n not in picking]
    if any(not n.startswith('ours_strength_') for n in direct):raise ValueError('Direct models must be named ours_strength_<number>')
    for n in direct:
        if not 0<float(n.removeprefix('ours_strength_'))<=1:raise ValueError(f'Invalid direct strength: {n}')
    if (direct or picking) and (checkpoint_path is None or not Path(checkpoint_path).is_file()):
        raise FileNotFoundError('Supply checkpoint_path for our model')
    roots={n:Path(config.generated_roots[n]) for n in names}
    genders=dict(TEST_ID_GENDERS)
    if 'gender' in panel:
        for row in panel.dropna(subset=['gender']).itertuples():genders[row.identity_id]=row.gender
    if gender_csv_path is not None:genders.update(load_gender_csv(gender_csv_path))
    genders.update(gender_by_id or {})
    aliases={'h':'male','m':'female','male':'male','female':'female'}
    genders={str(k):aliases.get(str(v).lower().strip()) for k,v in genders.items()}
    if {'FADING','Cradle2Cane'}&set(external):
        missing=[i for i in panel.identity_id.unique() if not genders.get(i)]
        if missing:raise ValueError(f'Missing gender labels for {len(missing)} IDs; pass gender_csv_path or gender_by_id. First IDs: {missing[:20]}')
    out=Path(config.output_dir)/'generation';out.mkdir(parents=True,exist_ok=True)
    try:previous_config=json.loads((out/'generation_config.json').read_text())
    except (ValueError,OSError):previous_config={}
    already_clean=(previous_config.get('aging_only') is True
        and previous_config.get('generated_roots')=={n:str(r) for n,r in roots.items()}
        and previous_config.get('test_manifest_path')==str(config.test_manifest_path))
    if aging_only and len(excluded) and not (fast and already_clean):
        import shutil
        removed=0
        for sample in excluded.to_dict('records'):
            for name in names:
                for base in [roots[name],out/'picking'/'generated'/name]:
                    path=base/sample['identity_id']/(sample['sample_id']+'.png')
                    for artifact in [path,path.with_suffix('.json'),path.with_suffix('.tmp.png')]:
                        if artifact.is_file():artifact.unlink();removed+=1
                work=out/'literature'/name/sample['sample_id']
                if work.is_dir():shutil.rmtree(work);removed+=1
            work=out/'picking'/'picking'/sample['sample_id']
            if work.is_dir():shutil.rmtree(work);removed+=1
        # Remove excluded rows from progress files left by an interrupted run.
        for path in [out/'prediction_manifest.csv',out/'inference_failures.csv',
                     out/'picking'/'prediction_manifest.csv',out/'picking'/'inference_failures.csv']:
            if path.is_file():
                try:
                    frame=pd.read_csv(path)
                    if {'source_age','target_age'}<=set(frame):
                        frame=frame[pd.to_numeric(frame.target_age)>pd.to_numeric(frame.source_age)]
                        frame.to_csv(path,index=False)
                except pd.errors.EmptyDataError:pass
        print(f'[generate paper] Aging only: excluded {len(excluded)} rejuvenations; removed {removed} generated artifacts. Aging outputs retained.',flush=True)
    elif aging_only and len(excluded):
        print('[generate paper] Aging only: previous cleanup reused; no rejuvenation scan.',flush=True)
    panel.to_csv(out/'transitions.csv',index=False)
    from .generation_cache import FingerprintCache
    fingerprint_cache=FingerprintCache(out/'fingerprints.json')
    pd.DataFrame([dict(identity_id=i,gender=genders.get(i)) for i in panel.identity_id.unique()]).to_csv(out/'gender_by_id.csv',index=False)
    (out/'generation_config.json').write_text(json.dumps(dict(
        test_manifest_path=str(config.test_manifest_path),generated_roots={n:str(r) for n,r in roots.items()},
        gender_csv_path=str(gender_csv_path) if gender_csv_path is not None else None,
        gender_csv_sha256=file_hash(gender_csv_path) if gender_csv_path is not None else None,
        fading_group_targets=fading_group_targets,aging_only=aging_only,
        batch_size=batch_size,progress_every=progress_every,
        resume_validation=resume_validation,
        hrfae_mode='batched_eval_histogram_jpeg' if 'HRFAE' in names else None,
        cusp_decoder_noise='checkpoint_constant' if 'CUSP' in names else None),indent=2))
    options=dict(image_size=400,num_inference_steps=34,seed=config.seed,
        text_reference_mode='source_age',age_guidance_scale=3.,text_guidance_scale=7.,image_guidance_scale=1.5)
    options.update(inference_config or {})
    if {'bundle','image','source_age','target_age','strength','device','compute_diagnostics','return_dict','output_type'}&set(options):
        raise ValueError('inference_config contains reserved generation arguments')
    rows=[];own=None;active=False
    from .prediction_resume import ProgressResume,ResumeIndex
    progress=ProgressResume(out/'resume_state.json');contexts={};indexes={};blocked={}
    expected_ids=set(panel.sample_id)
    # Fast resume trusts frozen inputs; strict mode checks size/mtime before
    # reusing a fingerprint. Never decode source images merely to hash bytes.
    source_digests={}
    checkpoint_hash=None
    def persist():
        if rows:
            pd.DataFrame(rows).to_csv(out/'prediction_manifest.csv',index=False)
            pd.DataFrame([r for r in rows if r['inference_status']=='failed']).to_csv(out/'inference_failures.csv',index=False)
        progress.save(rows,contexts,expected_ids,blocked)
        for index in indexes.values():index.save()
    def external_call(*args,**kwargs):
        # This generation-only route needs PNGs, not per-pair plots/banners.
        # Tensor adapters below bypass age_image completely for SAM/FRAN.
        with open(os.devnull,'w') as quiet,redirect_stdout(quiet),redirect_stderr(quiet):
            return age_image(*args,**kwargs)
    try:
        for name in external+direct:
            loader_keys={'SAM':[],'FRAN':[],'FADING':['fading_specialized_path'],
                'HRFAE':['hrfae_checkpoint_path'],'CUSP':['cusp_variant','cusp_checkpoint_path'],
                'Cradle2Cane':[]}.get(name,[])
            loader_context={'root':str((model_loader_config or {}).get('root','/content/aging_benchmarks')),
                **{key:(model_loader_config or {}).get(key) for key in loader_keys}}
            context=stable_hash(dict(model=name,root=str(roots[name]),seed=config.seed,
                loader=loader_context if name in external else None,inference=options if name in direct else None,
                checkpoint_path=str(checkpoint_path) if name in direct else None,
                genders={identity:genders.get(identity) for identity in panel.identity_id.unique()} if name in {'FADING','Cradle2Cane'} else None,
                protocol='hrfae-batched-eval-v1' if name=='HRFAE' else 'cusp-native-batch-v1' if name=='CUSP' else 'paper-generation-v1'))
            saved=progress.load(name,context) if fast else None
            contexts[name]=context
            blocked[name]=set((saved or {}).get('pending',[]))
            saved_rows={r['sample_id']:r for r in (saved or {}).get('rows',[])
                if r['sample_id'] in expected_ids and not (retry_failed and r['inference_status']=='failed')}
            if fast and saved_rows:
                if not roots[name].is_dir():
                    raise FileNotFoundError(f'{name}: saved progress exists but its prediction folder is missing: {roots[name]}. Restore it before resuming.')
                for sample in panel.to_dict('records'):
                    previous=saved_rows.get(sample['sample_id'])
                    if previous and any(previous[k]!=sample[k] for k in ['identity_id','source_age','target_age','source_path','target_path']):
                        raise ValueError(f'Transition metadata changed for {sample["sample_id"]}; use resume_validation="strict".')
            if fast and len(saved_rows)==len(panel) and roots[name].is_dir():
                rows.extend(saved_rows.values())
                print(f'[{name}] complete: {len(panel)}/{len(panel)} recorded — skipped without scanning files.',flush=True)
                continue
            unknown=panel[~panel.sample_id.isin(saved_rows)] if fast else panel
            print(f'[resume] {name}: {len(saved_rows)} recorded; listing {unknown.identity_id.nunique()} ID folders '
                f'({resume_workers} I/O workers). No model loading.',flush=True)
            index=ResumeIndex(out/'resume'/f'{name}.json',roots[name],unknown.identity_id,workers=resume_workers)
            indexes[name]=index
            pending=[]
            print(f'[generate paper] {name}: checking inputs and saved outputs...',flush=True)
            for number,sample in enumerate(panel.to_dict('records'),1):
                if number==1 or number%100==0 or number==len(panel):
                    print(f'[resume {"filenames" if fast else "scan"}] {name}: {number}/{len(panel)}',flush=True)
                path=roots[name]/sample['identity_id']/(sample['sample_id']+'.png');marker=path.with_suffix('.json')
                row={**sample,'model_name':name,'generated_path':str(path),'inference_status':'ok','inference_error':None}
                try:
                    if fast:
                        cached_row=saved_rows.get(sample['sample_id'])
                        if cached_row is not None:
                            rows.append(cached_row);continue
                        cached,valid,_=index.fast_inspect(path) if sample['sample_id'] not in blocked[name] else (None,False,None)
                        if valid:
                            if cached.get('status')=='failed':
                                if retry_failed:valid=False
                                else:row.update(generated_path=None,inference_status='failed',inference_error=cached.get('error','Cached failure'))
                            if valid:rows.append(row);continue
                    if name in direct and checkpoint_hash is None:
                        print('[generate paper] Checking adapter fingerprint for our direct model...',flush=True)
                        checkpoint_hash=(fingerprint_cache.peek(checkpoint_path) if fast else None) or fingerprint_cache.get(checkpoint_path,file_hash)
                    source=sample['source_path']
                    if source not in source_digests:
                        source_digests[source]=(fingerprint_cache.peek(source) if fast else None) or fingerprint_cache.get(source,file_hash)
                    digest=source_digests[source]
                    protocol='hrfae-batched-eval-v1' if name=='HRFAE' else 'cusp-native-batch-v1' if name=='CUSP' else 'paper-generation-v1'
                    signature=stable_hash(dict(protocol=protocol,model=name,source=digest,
                        source_age=sample['source_age'],target_age=sample['target_age'],seed=config.seed,
                        gender=genders.get(sample['identity_id']) if name in {'FADING','Cradle2Cane'} else None,
                        inference=options if name in direct else None,checkpoint=checkpoint_hash if name in direct else None,
                        loader=model_loader_config if name in external else None))
                    cached,valid,reason=index.inspect(path,signature) if config.resume and (not fast or sample['sample_id'] in blocked[name]) else (None,False,None)
                    if valid:
                        if cached.get('status')=='failed' and not retry_failed:
                            rows.append({**row,'generated_path':None,'inference_status':'failed','inference_error':cached.get('error')});continue
                        if cached.get('status','ok')=='ok':rows.append(row);continue
                    pending.append((sample,path,marker,signature,row))
                    if not config.resume or reason in {'signature_mismatch','invalid_marker','corrupt_png'}:
                        blocked[name].add(sample['sample_id'])
                except Exception as exc:
                    rows.append({**row,'generated_path':None,'inference_status':'failed','inference_error':str(exc)});persist()
            print(f'[generate paper] {name}: {len(pending)} pending / {len(panel)}',flush=True)
            fingerprint_cache.save()
            persist()
            if not pending:persist();continue
            if name in external:
                if comparison_bundle is None or comparison_bundle.get(name) is None:
                    comparison_bundle={**(comparison_bundle or {}),**load_aging_models(**{**(model_loader_config or {}),'models':[name],'load_metrics':False})}
                if comparison_bundle.get(name) is None:raise RuntimeError(f'{name} failed to load; inspect setup log')
                if name=='CUSP' and not comparison_bundle[name].get('prepare_batch_sample'):
                    from .cusp_batch import load_cusp_batch_entry
                    comparison_bundle[name]=load_cusp_batch_entry(comparison_bundle[name],config.seed)
            elif own is None:
                from src.inference import load_face_aging_inference_bundle
                own=load_face_aging_inference_bundle(checkpoint_path,device='cpu',local_files_only=local_files_only,load_auxiliary_models=False)
                active=True
                for key in ['unet','vae','text_encoder','age_delta_conditioner','age_conditioner']:
                    if hasattr(own.get(key),'to'):own[key].to(device)
                own['_reuse_loaded_checkpoint']=True
            runner=None
            def jobs():
                if name=='HRFAE':
                    from .resident_models import ACTIVE_MODEL_SESSION
                    session=ACTIVE_MODEL_SESSION.get();entry=comparison_bundle[name]
                    groups={}
                    for item in pending:groups.setdefault(int(item[0]['target_age']),[]).append(item)
                    size=batch_size
                    with tempfile.TemporaryDirectory(prefix='hrfae_outputs_') as folder:
                        for age,items in groups.items():
                            start=0
                            while start<len(items):
                                chunk=items[start:start+size]
                                request=[{**item[0],
                                    '_source_digest':source_digests[item[0]['source_path']],
                                    'output_path':str(Path(folder)/(item[0]['sample_id']+'.jpg'))} for item in chunk]
                                try:
                                    with open(os.devnull,'w') as quiet,redirect_stdout(quiet),redirect_stderr(quiet):
                                        response=session.run_hrfae_batch([sys.executable,entry['script'],'--config',str(entry['config'])],
                                            entry['root'],request,batch_size=size)
                                    reduced=min(size,response.get('batch_size',size))
                                    if reduced<size:print(f'[HRFAE] VRAM limit: reducing batch to {reduced}',flush=True)
                                    size=reduced
                                    results={r['sample_id']:r for r in response['results']}
                                except Exception as exc:
                                    raise RuntimeError('HRFAE batch worker setup/protocol failed; completed predictions are saved. '
                                        'Inspect the worker error before resuming.') from exc
                                for item in chunk:
                                    result=results.get(item[0]['sample_id'],dict(status='failed',error='No HRFAE batch result'))
                                    if result['status']=='ok':
                                        try:
                                            with Image.open(result['output_path']) as image:prepared=image.convert('RGB')
                                            Path(result['output_path']).unlink(missing_ok=True)
                                        except Exception as exc:prepared=exc
                                    else:prepared=RuntimeError(result['error'])
                                    yield item,prepared
                                start+=len(chunk)
                    return
                if runner is not None:
                    start=0
                    while start<len(pending):
                        chunk=pending[start:start+runner.batch_size]
                        samples=[{**item[0],'_source_digest':source_digests[item[0]['source_path']]} for item in chunk]
                        predictions=runner.infer(samples)
                        for item,image in zip(chunk,predictions):yield item,image
                        start+=len(chunk)
                    return
                if name!='FADING' or not fading_group_targets:
                    for item in pending:yield item,None
                    return
                groups={}
                for item in pending:
                    sample=item[0]
                    key=(sample['source_path'],sample['source_age'],genders[sample['identity_id']])
                    groups.setdefault(key,[]).append(item)
                print(f'[FADING] {len(pending)} pending transitions / {len(groups)} source inversions (grouped targets)',flush=True)
                for (source,age,gender),items in groups.items():
                    if len(items)==1:
                        yield items[0],None
                        continue
                    targets=list(dict.fromkeys(item[0]['target_age'] for item in items))
                    results={}
                    try:
                        result=external_call(comparison_bundle,image=source,source_age=age,target_ages=targets,
                            models=['FADING'],gender=gender,seed=config.seed,show=False,strict=True,metrics=False,
                            output_dir=str(out/'literature'/'FADING'/items[0][0]['sample_id']))
                        results=result.get('FADING',{})
                    except Exception as exc:
                        print(f'[FADING] Group failed; retrying its transitions individually: {exc}',flush=True)
                    for item in items:yield item,results.get(item[0]['target_age'])
            with ModelSession(comparison_bundle,name,generation_only=True):
                if name in {'SAM','FRAN','CUSP'} and comparison_bundle[name].get('model') is not None:
                    from .batched_baselines import BaselineBatchRunner
                    runner=BaselineBatchRunner(name,comparison_bundle[name],device=device,
                        cache_dir=out/'preprocessing',batch_size=batch_size)
                print(f'[{name}] {len(panel)-len(pending)}/{len(panel)} already recorded; {len(pending)} pending'
                    +(f' | batch={runner.batch_size}' if runner else f' | batch={batch_size} (same target age)' if name=='HRFAE' else ''),flush=True)
                for number,((sample,path,marker,signature,row),prepared_image) in enumerate(jobs(),1):
                    try:
                        if isinstance(prepared_image,Exception):raise prepared_image
                        if isinstance(prepared_image,Image.Image):image=prepared_image
                        elif name in external:
                            result=external_call(comparison_bundle,image=sample['source_path'],source_age=sample['source_age'],
                                target_ages=[sample['target_age']],models=[name],gender=genders.get(sample['identity_id']) or 'male',
                                seed=config.seed,show=False,strict=True,metrics=False,
                                output_dir=str(out/'literature'/name/sample['sample_id']))
                            image=result.get(name,{}).get(sample['target_age'])
                        else:
                            from src.inference import infer_face_aging_direct
                            result=infer_face_aging_direct(bundle=own,image=sample['source_path'],source_age=sample['source_age'],
                                target_age=sample['target_age'],strength=float(name.removeprefix('ours_strength_')),
                                device=device,compute_diagnostics=False,return_dict=True,output_type='pil',**options)
                            image=result['image']
                        if not isinstance(image,Image.Image):raise RuntimeError('Model returned no image')
                        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.png');image.save(temp);temp.replace(path)
                        completed_marker=dict(signature=signature,status='ok')
                        marker.write_text(json.dumps(completed_marker))
                        if not fast:index.record(path,completed_marker)
                    except Exception as exc:
                        row.update(generated_path=None,inference_status='failed',inference_error=f'{type(exc).__name__}: {exc}')
                        path.unlink(missing_ok=True);marker.parent.mkdir(parents=True,exist_ok=True)
                        marker.write_text(json.dumps(dict(signature=signature,status='failed',error=row['inference_error'])))
                        if not fast:index.record(path,dict(signature=signature,status='failed',error=row['inference_error']))
                    rows.append(row)
                    if number%progress_every==0 or number==len(pending):
                        persist()
                        completed=len(panel)-len(pending)+number
                        failed=sum(r['model_name']==name and r['inference_status']=='failed' for r in rows)
                        print(f'[{name}] {completed}/{len(panel)} | remaining={len(pending)-number} | failed={failed}',flush=True)
    finally:
        fingerprint_cache.save()
        persist()
        if active:
            for key in ['unet','vae','text_encoder','age_delta_conditioner','age_conditioner']:
                if hasattr(own.get(key),'to'):own[key].to('cpu')
    if picking:
        from .picking_variants import PickingVariantsConfig,run_picking_variants
        own=None
        if picking_bundle is None:
            from src.inference import load_face_aging_inference_bundle
            picking_bundle=load_face_aging_inference_bundle(checkpoint_path,device='cpu',local_files_only=local_files_only,
                load_auxiliary_models=True,auxiliary_trust_remote_code=True)
        # Avoid retaining a second diffusion bundle during the picking phase.
        params={k:v for k,v in options.items() if k!='seed'}
        params.update(picking_options or {});params.setdefault('overwrite',False)
        pc=PickingVariantsConfig(checkpoint_path=checkpoint_path,test_manifest_path=out/'transitions.csv',
            output_dir=out/'picking',source_root=config.source_root,seed=config.seed,device=device,
            resume=config.resume,retry_failed=retry_failed,**params)
        selected=run_picking_variants(pc,bundle=picking_bundle)
        for row in selected['manifest'].to_dict('records'):
            name=row['model_name']
            if name not in picking:continue
            if row['inference_status']=='ok':
                import shutil
                destination=roots[name]/row['identity_id']/(row['sample_id']+'.png')
                destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.png')
                shutil.copyfile(row['generated_path'],temp);temp.replace(destination);row['generated_path']=str(destination)
            else:(roots[name]/row['identity_id']/(row['sample_id']+'.png')).unlink(missing_ok=True)
            rows.append(row)
        persist()
    panel.to_csv(out/'transitions.csv',index=False)
    return dict(manifest=pd.DataFrame(rows),generated_roots={n:str(r) for n,r in roots.items()},
        paths={'manifest':str(out/'prediction_manifest.csv')})
