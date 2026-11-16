"""One-call Colab workflow for the long-gap experiment."""
from pathlib import Path
import json
from dataclasses import replace
import gc
import importlib.util
import subprocess
import sys

import pandas as pd

from data.indexing import build_image_manifest
from src.aging_bias.config import FeatureConfig, PaperV2Config
from src.aging_bias.references import ReferenceBundle
from .identity_metadata import TEST_ID_GENDERS, load_gender_csv
from .long_gap_panel import LongGapPanelConfig, select_long_gap_panel
from .long_gap_evaluation import LongGapEvaluationConfig, evaluate_long_gap
from .generate_paper_predictions import generate_paper_predictions
from .evaluation_backends_v2 import prepare_paper_bundle


def run_long_gap_experiment(*, train_root, reference_dir, checkpoint_path, output_dir,
        models, n_transitions=350, source_ages=(18,35), target_ages=(60,65,70),
        exclusion_margin=10, inventory_roots=None, gender_csv_path=None,
        gender_by_id=None, batch_size=32, n_boot=1000, seed=2026,
        model_loader_config=None, inference_config=None, picking_options=None,
        assets_root='/content/evaluation_models_v2', allow_download=True,run_dino=True):
    """Select, generate/resume and evaluate; all persistent output has one root.

    CSVs and audits live directly in output_dir; images are in generated/ and
    resumable working files/cache are kept together in .state/.
    """
    train_root, reference_dir, output = map(Path, (train_root, reference_dir, output_dir))
    if output.resolve()==reference_dir.resolve() or reference_dir.resolve() in output.resolve().parents:
        raise ValueError('Module 1 is read-only; choose a separate output directory')
    if not train_root.is_dir():raise FileNotFoundError(train_root)
    reference=ReferenceBundle.load(reference_dir,load_training_data=False)
    if run_dino:
        from .frozen_dino import validate_frozen_dino_files
        validate_frozen_dino_files(reference_dir)
    output.mkdir(parents=True,exist_ok=True)
    state=output/'.state';state.mkdir(exist_ok=True)
    genders=dict(TEST_ID_GENDERS)
    if gender_csv_path is not None and Path(gender_csv_path).is_file():
        genders.update(load_gender_csv(gender_csv_path))
    genders.update(gender_by_id or {})
    roots=[train_root,*(inventory_roots if inventory_roots is not None else [
        train_root.parent/'val',train_root.parent/'test',
        train_root.parent.parent/'dataset_unificado',
        train_root.parent.parent/'daytaset_unificado_test'])]
    audits, unknown, supplemental, train = {}, set(), [], None
    seen=set()
    for root in map(Path,roots):
        key=str(root.resolve())
        if key in seen:continue
        seen.add(key)
        if not root.is_dir():
            audits[key]={'missing_root':True};continue
        records,audit=build_image_manifest(root,use_cached_manifest=False,min_age=0,
            max_age=120,normalize_over_100=False,gender_by_person=genders)
        audits[key]=audit
        for item in audit['skipped_files']+audit['filtered_ages']:
            relative=Path(item['relative_path'])
            if relative.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}:
                unknown.add(relative.parts[0])
        for identity in root.iterdir():
            if identity.is_dir() and any(p.is_dir() for p in identity.iterdir()):
                unknown.add(identity.name)
        frame=pd.DataFrame([dict(identity_id=r.person_id,age=r.age,path=str(r.path(root)),
            gender=r.gender or '') for r in records],columns=['identity_id','age','path','gender'])
        if root.resolve()==train_root.resolve():train=frame
        elif len(frame):supplemental.append(frame)
    train_manifest=state/'train_inventory.csv';train.to_csv(train_manifest,index=False)
    inventories=[]
    if supplemental:
        path=state/'additional_inventory.csv'
        pd.concat(supplemental,ignore_index=True).to_csv(path,index=False);inventories.append(path)
    (output/'inventory_audit.json').write_text(json.dumps(dict(roots=audits,
        unknown_age_identities=sorted(unknown)),indent=2,default=str),encoding='utf-8')
    selected=select_long_gap_panel(LongGapPanelConfig(train_manifest,output,
        inventory_manifest_paths=tuple(inventories),n_transitions=n_transitions,
        source_age_min=source_ages[0],source_age_max=source_ages[1],target_ages=tuple(target_ages),
        exclusion_margin=exclusion_margin,seed=seed,reference_bundle_path=reference_dir,
        excluded_identity_ids=tuple(sorted(unknown))))
    panel=selected['panel'];manifest=selected['paths']['panel']
    print(f'[long gap] {len(panel)} transitions / {panel.identity_id.nunique()} identities; '
        f'minimum observed-target distance={panel.nearest_observed_target_age_distance.min():g}',flush=True)
    supported=~reference.extrapolated(panel.source_age,panel.target_age)
    supported &= panel.source_age.ge(reference.metadata.get('reliability_policy',{}).get('retest_min_age') or 0).to_numpy()
    if not supported.any():raise ValueError('No requested transition has frozen reference support')
    if any(name in {'FADING','Cradle2Cane'} for name in models):
        missing=sorted(set(panel.identity_id)-set(genders))
        if missing:
            pd.DataFrame(dict(identity_id=missing,gender=['']*len(missing))).to_csv(output/'missing_gender.csv',index=False)
            raise ValueError(f'Missing gender for {len(missing)} identities; complete {output / "missing_gender.csv"}')
    if any(name.startswith('ours_picking_') for name in models):
        if importlib.util.find_spec('feat') is None:
            subprocess.check_call([sys.executable,'-m','pip','install','-e','.[auxiliary]'],cwd=Path(__file__).resolve().parents[2])
        if importlib.util.find_spec('mivolo') is None:
            subprocess.check_call([sys.executable,'-m','pip','install','setuptools==80.9.0','wheel','timm>=1.0.11,<2'])
            subprocess.check_call([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps',
                'git+https://github.com/WildChlamydia/MiVOLO.git@37475e3f8818b5f22448003feec3e64b01bfb188'])
    roots={name:output/'generated'/name for name in models}
    raw=dict(reference.metadata['config']['features'])
    raw['new_attributes']=tuple(raw.get('new_attributes',()))
    features=FeatureConfig(**raw)
    generation_config=PaperV2Config(manifest,roots,list(models),reference_dir,state,
        features=features,resume=True,seed=seed,include_dex_appendix=False,include_kid=False)
    inference=dict(image_size=400,num_inference_steps=37,text_reference_mode='source_age',
        age_guidance_scale=2.,text_guidance_scale=7.,image_guidance_scale=1.5)
    inference.update(inference_config or {})
    picking=dict(delta_age_strength_map={0:.04,10:.15,20:.22,30:.23,40:.29,60:.34},
        max_trials_per_target=10,min_strength=.04,max_strength=.45,overwrite=False)
    picking.update(picking_options or {})
    generated=generate_paper_predictions(generation_config,checkpoint_path=checkpoint_path,
        model_loader_config=model_loader_config or {'root':'/content/aging_benchmarks'},
        gender_by_id=genders,inference_config=inference,picking_options=picking,
        batch_size=batch_size,progress_every=100,resume_validation='fast',local_files_only=not allow_download)
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()
    features.landmark_model_path=str(Path(assets_root)/'face_landmarker.task')
    bundle=prepare_paper_bundle(assets_root,feature_config=features,allow_download=allow_download,
        include_dex=False,include_kid=False)
    evaluation_config=LongGapEvaluationConfig(manifest,roots,list(models),reference_dir,output,
        cache_dir=state/'cache',n_boot=n_boot,seed=seed,batch_size=batch_size)
    result=evaluate_long_gap(evaluation_config,bundle=bundle,reference_bundle=reference)
    dino_result=None
    if run_dino:
        from .frozen_dino import load_frozen_dino_backend
        dino_reference,dino_bundle=load_frozen_dino_backend(reference_dir,root=assets_root,
            landmark_backend=bundle['landmark_backend'],cache_dir=state/'cache',
            allow_download=allow_download,device='auto',landmark_model_path=features.landmark_model_path)
        try:
            dino_config=replace(evaluation_config,reference_bundle_path=reference_dir/'dinov2',
                output_dir=state/'dinov2',cache_dir=state/'cache_dinov2')
            dino_result=evaluate_long_gap(dino_config,bundle={**bundle,**dino_bundle},reference_bundle=dino_reference)
            for name,value in dino_result.items():
                if isinstance(value,pd.DataFrame):value.to_csv(output/('dinov2_'+name+'.csv'),index=False)
            for name in ['run_config.json','completeness.json']:
                (output/('dinov2_'+name)).write_text((dino_config.output_dir/name).read_text(),encoding='utf-8')
        finally:
            encoder=getattr(dino_bundle.get('feature_backend'),'encoder',None)
            if hasattr(encoder,'to'):encoder.to('cpu')
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    record=dict(protocol='long-gap-counterfactual-frozen-module1-v1',models=list(models),
        selection=selected['diagnostics'],evaluation=result['completeness'],dino_requested=run_dino,
        dino_complete=dino_result is not None,no_real_target=True)
    (output/'experiment_summary.json').write_text(json.dumps(record,indent=2,default=str),encoding='utf-8')
    return {**result,'selection':selected,'generation':generated,'dinov2':dino_result,'reference_dir':reference_dir}


def display_long_gap_analysis(result,*,n_images=6):
    """One display call for every diagnostic, plot and visual comparison."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    output=Path(result['output_dir'])
    print('SELECCION Y SOLAPAMIENTO CON MODULO 1')
    print(json.dumps(result['selection']['diagnostics'],indent=2))
    print(json.dumps(result['completeness'],indent=2))
    for filename in ['reference_quality.csv','population_contrasts.csv']:
        path=Path(result['reference_dir'])/filename
        if path.is_file():print(filename);display(pd.read_csv(path))
    print('COBERTURA Y SOPORTE');display(result['coverage'])
    print('GENERACION');display(pd.crosstab(result['generation']['manifest'].model_name,
        result['generation']['manifest'].inference_status))
    summary=result['summary']
    main=summary[summary.group.eq('all')]
    print('RESULTADOS GLOBALES (medias por identidad)')
    display(main.pivot(index=['model_name','measure'],columns='metric',values='value'))
    print('INTERVALOS POR IDENTIDAD');display(main)
    proximity=main[main.measure.str.startswith('zone_') & main.metric.isin([
        'absolute_error','ffhq_absolute_error','latino_cross_absolute_error'])].pivot(
            index=['model_name','measure'],columns='metric',values='value')
    proximity.to_csv(output/'reference_proximity_descriptive.csv')
    print('PROXIMIDAD A CADA REFERENCIA');display(proximity)
    print('RESULTADOS POR EDAD OBJETIVO');display(summary[summary.group.eq('target')])
    print('IDENTIFICACION DE PI');display(result['structural_bias'])
    print('DIFERENCIAS PAREADAS');display(result['paired'].query("group == 'all'"))
    measures=result['per_measure']
    signs=measures[measures.aging_oriented & measures.valid].groupby(
        ['model_name','measure','interpretation']).size().unstack(fill_value=0)
    signs.to_csv(output/'sign_counts.csv');print('SIGNOS POR FOTO Y ZONA');display(signs)
    sign_map=measures.pivot(index=['sample_id','identity_id','source_age','target_age','model_name'],columns='measure',values='error')
    sign_map.to_csv(output/'signed_residuals_by_photo.csv');display(sign_map)
    print('CALIDAD Y EXCLUSIONES');display(measures.groupby(['model_name','missing_reason']).size().unstack(fill_value=0))
    print('FALLOS');display(result['failures'])
    regional=main[main.measure.str.startswith('zone_') & main.metric.eq('error')]
    zones=list(regional.measure.unique())
    if zones:
        fig,axes=plt.subplots(1,len(zones),figsize=(6*len(zones),5),squeeze=False)
        for ax,zone in zip(axes[0],zones):
            frame=regional[regional.measure.eq(zone)]
            ax.scatter(np.arange(len(frame)),frame.value)
            for i,row in enumerate(frame.itertuples()):
                if np.isfinite([row.ci_low,row.ci_high]).all():ax.plot([i,i],[row.ci_low,row.ci_high],color='tab:blue')
            ax.axhline(0,color='black');ax.set_title(zone)
            ax.set_xticks(np.arange(len(frame)),frame.model_name,rotation=65,ha='right')
            ax.set_ylabel('Cambio generado - esperado latino')
        fig.tight_layout();fig.savefig(output/'signed_regional_residuals.png',dpi=180,bbox_inches='tight');plt.show()
    age=main[main.measure.eq('operational') & main.metric.eq('age_mae_indep')]
    if not age.empty:
        fig,ax=plt.subplots(figsize=(9,4));ax.bar(age.model_name,age.value)
        ax.set_ylabel('MAE edad independiente (anos)');ax.tick_params(axis='x',rotation=60)
        fig.tight_layout();fig.savefig(output/'independent_age_mae.png',dpi=180,bbox_inches='tight');plt.show()
    panel=result['selection']['panel']
    manifest=result['generation']['manifest']
    models=list(manifest.model_name.unique())
    for sample in panel.sample(n=min(n_images,len(panel)),random_state=2026).to_dict('records'):
        names=['source_noedit',*models]
        fig,axes=plt.subplots(1,len(names),figsize=(3*len(names),3.8),squeeze=False)
        for ax,name in zip(axes[0],names):
            row=manifest[manifest.sample_id.eq(sample['sample_id']) & manifest.model_name.eq(name)]
            path=sample['source_path'] if name=='source_noedit' else row.iloc[0].generated_path if len(row) else None
            if path is not None and pd.notna(path) and Path(path).is_file():
                with Image.open(path) as image:ax.imshow(image.convert('RGB'))
            ax.set_title(name,fontsize=9);ax.axis('off')
        fig.suptitle(f'{sample["identity_id"]}: {sample["source_age"]:g} -> {sample["target_age"]:g} | sin objetivo real')
        fig.tight_layout();fig.savefig(output/(sample['sample_id']+'.png'),dpi=130,bbox_inches='tight');plt.show()
        display(measures[measures.sample_id.eq(sample['sample_id']) & measures.measure_kind.eq('zone')][
            ['model_name','measure','model_change','expected_change','error','interpretation','valid']])
    dino=result.get('dinov2')
    if dino is not None:
        print('ROBUSTEZ DINO CONGELADA');display(dino['structural_bias'])
        display(dino['summary'].query("group == 'all'"))
    print('Todos los resultados:',output)
