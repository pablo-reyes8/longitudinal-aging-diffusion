"""Generate predictions on an existing paper manifest, without legacy metrics."""
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from src.aging_bias.features import stable_hash
from src.quantitative_metrics.image_io import ImageResolver
from .evaluate_paper_v2 import load_test_panel
from .identity_metadata import TEST_ID_GENDERS,load_gender_csv
from .resident_models import ModelSession


def generate_paper_predictions(config, *, checkpoint_path=None, comparison_bundle=None,
        model_loader_config=None, gender_by_id=None, inference_config=None,
        picking_options=None, picking_bundle=None, device='cuda',
        local_files_only=False, retry_failed=False, gender_csv_path=None,
        fading_group_targets=True):
    """Generate exactly config's panel and roots; save progress after every pair.

    No evaluation, test sampling, or module-1 writes. Missing gender is checked
    before generation for gender-dependent models; no inferred default labels.
    Successful files require a compatible signature and readable PNG to resume.
    Gender precedence: explicit dict > CSV > manifest > legacy labels. FADING
    groups pending targets by source to reuse its original CLI inversion. A
    failed group falls back to individual requests without dropping transitions.
    """
    print('[generate paper] Reading manifest and validating configuration...',flush=True)
    from .generate_images import age_image
    from .load_models import load_aging_models
    from .evaluation_backends_v2 import file_hash
    panel=load_test_panel(config.test_manifest_path,config.source_root)
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
    pd.DataFrame([dict(identity_id=i,gender=genders.get(i)) for i in panel.identity_id.unique()]).to_csv(out/'gender_by_id.csv',index=False)
    (out/'generation_config.json').write_text(json.dumps(dict(
        test_manifest_path=str(config.test_manifest_path),generated_roots={n:str(r) for n,r in roots.items()},
        gender_csv_path=str(gender_csv_path) if gender_csv_path is not None else None,
        gender_csv_sha256=file_hash(gender_csv_path) if gender_csv_path is not None else None,
        fading_group_targets=fading_group_targets),indent=2))
    options=dict(image_size=400,num_inference_steps=34,seed=config.seed,
        text_reference_mode='source_age',age_guidance_scale=3.,text_guidance_scale=7.,image_guidance_scale=1.5)
    options.update(inference_config or {})
    if {'bundle','image','source_age','target_age','strength','device','compute_diagnostics','return_dict','output_type'}&set(options):
        raise ValueError('inference_config contains reserved generation arguments')
    rows=[];resolver=ImageResolver();own=None;active=False
    # Files are immutable inputs during a run. Keep only their small digests,
    # not decoded images, across transitions/models. Re-read on the next run.
    source_digests={}
    if direct:print('[generate paper] Hashing adapter checkpoint for safe resume...',flush=True)
    checkpoint_hash=file_hash(checkpoint_path) if direct else None
    def persist():
        pd.DataFrame(rows).to_csv(out/'prediction_manifest.csv',index=False)
        pd.DataFrame([r for r in rows if r['inference_status']=='failed']).to_csv(out/'inference_failures.csv',index=False)
    try:
        for name in external+direct:
            pending=[]
            print(f'[generate paper] {name}: checking inputs and saved outputs...',flush=True)
            for number,sample in enumerate(panel.to_dict('records'),1):
                if number==1 or number%100==0 or number==len(panel):
                    print(f'[resume scan] {name}: {number}/{len(panel)} | {len(source_digests)} source fingerprints cached',flush=True)
                path=roots[name]/sample['identity_id']/(sample['sample_id']+'.png');marker=path.with_suffix('.json')
                row={**sample,'model_name':name,'generated_path':str(path),'inference_status':'ok','inference_error':None}
                try:
                    source=sample['source_path']
                    if source not in source_digests:
                        source_digests[source]=resolver.load(source)[1];resolver._cache.clear()
                    digest=source_digests[source]
                    signature=stable_hash(dict(protocol='paper-generation-v1',model=name,source=digest,
                        source_age=sample['source_age'],target_age=sample['target_age'],seed=config.seed,
                        gender=genders.get(sample['identity_id']) if name in {'FADING','Cradle2Cane'} else None,
                        inference=options if name in direct else None,checkpoint=checkpoint_hash if name in direct else None,
                        loader=model_loader_config if name in external else None))
                    try:cached=json.loads(marker.read_text()) if config.resume and marker.exists() else {}
                    except (ValueError,OSError):cached={}
                    if cached.get('signature')==signature:
                        if cached.get('status')=='failed' and not retry_failed:
                            rows.append({**row,'generated_path':None,'inference_status':'failed','inference_error':cached.get('error')});continue
                        if cached.get('status')=='ok' and path.is_file():
                            try:
                                with Image.open(path) as image:image.verify()
                                rows.append(row);continue
                            except OSError:pass
                    pending.append((sample,path,marker,signature,row))
                except Exception as exc:
                    rows.append({**row,'generated_path':None,'inference_status':'failed','inference_error':str(exc)});persist()
            print(f'[generate paper] {name}: {len(pending)} pending / {len(panel)}',flush=True)
            if not pending:persist();continue
            if name in external:
                if comparison_bundle is None or comparison_bundle.get(name) is None:
                    comparison_bundle={**(comparison_bundle or {}),**load_aging_models(**{**(model_loader_config or {}),'models':[name],'load_metrics':False})}
                if comparison_bundle.get(name) is None:raise RuntimeError(f'{name} failed to load; inspect setup log')
            elif own is None:
                from src.inference import load_face_aging_inference_bundle
                own=load_face_aging_inference_bundle(checkpoint_path,device='cpu',local_files_only=local_files_only,load_auxiliary_models=False)
                active=True
                for key in ['unet','vae','text_encoder','age_delta_conditioner','age_conditioner']:
                    if hasattr(own.get(key),'to'):own[key].to(device)
                own['_reuse_loaded_checkpoint']=True
            def jobs():
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
                        result=age_image(comparison_bundle,image=source,source_age=age,target_ages=targets,
                            models=['FADING'],gender=gender,seed=config.seed,show=False,strict=True,metrics=False,
                            output_dir=str(out/'literature'/'FADING'/items[0][0]['sample_id']))
                        results=result.get('FADING',{})
                    except Exception as exc:
                        print(f'[FADING] Group failed; retrying its transitions individually: {exc}',flush=True)
                    for item in items:yield item,results.get(item[0]['target_age'])
            with ModelSession(comparison_bundle,name):
                for number,((sample,path,marker,signature,row),prepared_image) in enumerate(jobs(),1):
                    try:
                        if isinstance(prepared_image,Image.Image):image=prepared_image
                        elif name in external:
                            result=age_image(comparison_bundle,image=sample['source_path'],source_age=sample['source_age'],
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
                        marker.write_text(json.dumps(dict(signature=signature,status='ok')))
                    except Exception as exc:
                        row.update(generated_path=None,inference_status='failed',inference_error=f'{type(exc).__name__}: {exc}')
                        path.unlink(missing_ok=True);marker.parent.mkdir(parents=True,exist_ok=True)
                        marker.write_text(json.dumps(dict(signature=signature,status='failed',error=row['inference_error'])))
                    rows.append(row);persist()
                    if number==1 or number%10==0 or number==len(pending):print(f'[{name}] {number}/{len(pending)}: {row["inference_status"]}',flush=True)
    finally:
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
        pc=PickingVariantsConfig(checkpoint_path=checkpoint_path,test_manifest_path=config.test_manifest_path,
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
