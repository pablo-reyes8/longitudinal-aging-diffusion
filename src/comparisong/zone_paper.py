"""Module-2 report plumbing. Module-1 references are read-only inputs."""
from pathlib import Path
from itertools import combinations
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from src.aging_bias.features import FeatureStore,stable_hash
from src.aging_bias.references import ReferenceBundle
from src.aging_bias.weighting import transition_weights
from .zone_evaluation import evaluate_zone_errors


def report_long_pair_support(panel,config):
    root=Path(config.output_dir)/'diagnostics';root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for direction,frame in [('aging',panel[panel.delta_age>0]),('rejuvenation',panel[panel.delta_age<0])]:
        for lo,hi in [(1,10),(10,20),(20,30),(30,np.inf)]:
            sub=frame[frame.delta_age.abs().ge(lo)&frame.delta_age.abs().lt(hi)]
            rows.append(dict(direction=direction,delta_bin=f'[{lo},{hi})',n_pairs=len(sub),n_identities=sub.identity_id.nunique(),
                long_transition=lo>=20,status='sparse' if len(sub)<config.weighting.min_bin_n else 'ok'))
    support=pd.DataFrame(rows);support.to_csv(root/'long_pair_support.csv',index=False)
    candidates=[];status='source_root_not_provided'
    if config.source_root and Path(config.source_root).is_dir():
        from data.indexing import build_image_manifest
        records,_=build_image_manifest(config.source_root,min_age=0,max_age=120,normalize_over_100=False)
        test_ids=set(panel.identity_id.astype(str))
        known={(str(r.identity_id),Path(r.source_path).name,Path(r.target_path).name) for r in panel.itertuples()}
        groups={}
        for row in records:
            if row.person_id in test_ids:groups.setdefault(row.person_id,[]).append(row)
        for identity,group in groups.items():
            for a,b in combinations(sorted(group,key=lambda r:(r.age,r.filename)),2):
                if b.age-a.age<20:continue
                key=(identity,a.filename,b.filename)
                candidates.append(dict(identity_id=identity,source_age=a.age,target_age=b.age,delta_age=b.age-a.age,
                    source_path=str(a.path(config.source_root)),target_path=str(b.path(config.source_root)),
                    in_current_manifest=key in known,action='already_evaluated_or_requested' if key in known else 'requires_generation_all_models'))
        status='inventory_only_no_panel_change'
    columns=['identity_id','source_age','target_age','delta_age','source_path','target_path','in_current_manifest','action']
    proposed=pd.DataFrame(candidates,columns=columns)
    if not proposed.empty:proposed=proposed.sort_values(['in_current_manifest','delta_age','identity_id'],ascending=[True,False,True])
    proposed.to_csv(root/'long_pair_candidates.csv',index=False)
    (root/'long_pair_inventory.json').write_text(json.dumps(dict(status=status,panel_unchanged=True,n_candidates=len(proposed),
        n_additional=int((~proposed.in_current_manifest).sum()) if len(proposed) else 0),indent=2))
    return support


def finalize_zone_paper(per,config,ref,store,operational,failures,panel,paths):
    from .evaluate_paper_v2 import _operational_summary
    root=Path(config.output_dir)
    result=evaluate_zone_errors(per,config,ref)
    errors=result['zone_errors'];measures=ref.attributes
    common_by_measure={m:set(errors.loc[errors.measure.eq(m)&errors.in_common_panel,'sample_id']) for m in measures}
    common=set().union(*common_by_measure.values()) if common_by_measure else set()
    per=per.copy();per['in_common_panel']=per.sample_id.isin(common)
    for name in measures:per['common_valid__'+name]=per.sample_id.isin(common_by_measure[name])
    per['direction']=np.where(per.delta_age>0,'aging','rejuvenation')
    names=list(config.models)+['source_noedit','oracle_target']
    valid=[set(per.loc[per.model_name.eq(m)&per.operational_valid,'sample_id']) for m in names]
    operational_common=set.intersection(*valid) if valid else set()
    operations=[]
    for subset in ['all','aging','rejuvenation','long_aging']:
        selection=per[per.sample_id.isin(operational_common)]
        if subset=='aging':selection=selection[selection.delta_age>0]
        elif subset=='rejuvenation':selection=selection[selection.delta_age<0]
        elif subset=='long_aging':selection=selection[selection.delta_age>=20]
        for model in names:
            f=selection[selection.model_name.eq(model)].sort_values('sample_id')
            if f.empty:continue
            operations+=_operational_summary(f,transition_weights(f,config.weighting),model,subset,'common')
    ops=pd.DataFrame(operations,columns=['model_name','scheme','subset','panel','delta_bin','n_pairs','n_identities',
        *[value for k in ['adaface_id_source','adaface_id_gt','delta_id_gain','age_rel_err_indep','age_rel_signed_indep','dex_age_mae'] for value in [k,'n_valid__'+k]]])
    ops.to_csv(root/'operational_metrics.csv',index=False)
    for subset in ['all','aging','long_aging']:
        ops[ops.subset.eq(subset)].to_csv(root/'tables'/f'table2_operational_{subset}.csv',index=False)
    support=report_long_pair_support(panel,config)
    kid_rows=[]
    if config.include_kid:
        from src.quantitative_metrics.image_io import ImageResolver
        resolver=ImageResolver()
        for model in list(config.models)+['source_noedit']:
            for subset in ['all','aging','long_aging']:
                frame=per[per.model_name.eq(model)&per.sample_id.isin(common)]
                if subset=='aging':frame=frame[frame.delta_age>0]
                elif subset=='long_aging':frame=frame[frame.delta_age>=20]
                mean=std=np.nan;status='insufficient_samples'
                if len(frame)>=2:
                    try:
                        real=[resolver.load(p)[0] for p in frame.target_path]
                        generated=[resolver.load(p)[0] for p in frame.generated_path]
                        mean,std=operational.bundle['kid_metric'].compute(real,generated,seed=config.seed,subset_size=min(50,len(frame)))
                        status='ok'
                    except Exception as exc:
                        status='backend_error';failures.append(dict(model_name=model,stage='kid',error=f'{subset}: {exc}'))
                    finally:resolver._cache.clear()
                kid_rows.append(dict(model_name=model,subset=subset,kid_mean=mean,kid_std=std,kid_status=status,n_kid=len(frame)))
    kid=pd.DataFrame(kid_rows,columns=['model_name','subset','kid_mean','kid_std','kid_status','n_kid'])
    kid.to_csv(root/'kid_metrics.csv',index=False)
    validation=result['zone_validation']
    (root/'validation/real_anchors.json').write_text(json.dumps({**validation,'structural_ratios_validated':False},indent=2))
    result['feature_error_summary'].query("model_name=='oracle_target'").to_csv(root/'validation/oracle_observed_errors.csv',index=False)
    metadata=dict(protocol='observed-longitudinal-zone-error-v1',config=asdict(config),output_dir=str(root),n_requested_pairs=len(panel),n_common_pairs=len(common),
        n_common_identities=panel[panel.sample_id.isin(common)].identity_id.nunique(),
        common_panel_policy='separate common transition panel per frozen measure; missing data never imputed',
        reference_dir=str(config.reference_bundle_path),feature_config_hash=store.config_hash,
        reference_hash=stable_hash(dict(feature=ref.feature_config_hash,attributes=ref.attributes,mean=ref.mean.tolist(),sd=ref.sd.tolist(),
            knots=ref.knots.tolist(),mu_c=ref.mu_c.tolist(),mu_p=ref.mu_p.tolist(),p_ages=ref.p_ages.tolist(),icc=ref.icc.tolist(),
            orientations=ref.metadata.get('attribute_orientations',{}),reliability=ref.metadata.get('reliability_policy',{}))),operational_hash=operational.signature,
        validation_passed=validation['passed'],structural_ratios_validated=False,
        ci_scope='identity bootstrap conditional on module-1 frozen measures, masks, normalizations and signs',seed=config.seed)
    per.to_csv(root/'per_sample.csv',index=False);per.to_parquet(root/'per_sample.parquet',index=False)
    pd.DataFrame(failures,columns=['sample_id','model_name','stage','path','error']).to_csv(root/'failures.csv',index=False)
    panel.to_csv(root/'transitions.csv',index=False);pd.DataFrame(paths).to_csv(root/'prediction_manifest.csv',index=False)
    (root/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    # Compatible result keys; old mixture ratios are intentionally NOT estimated
    # or fabricated when the protocol is observed longitudinal error.
    legacy=pd.DataFrame(columns=['model_name','scheme','panel','subset','status','D','pi','drift','homog'])
    return dict(**result,per_sample=per,bias_indices=legacy,bias_indices_by_bin=legacy.copy(),
        paired_differences=result['zone_paired'],operational_metrics=ops,summary=result['zone_summary'],
        kid_metrics=kid,
        failures=pd.DataFrame(failures),metadata=metadata,long_pair_support=support,
        paths={p.name:str(p) for p in root.iterdir() if p.is_file()})


def preflight_zone_protocol(config,*,bundle,assets_root=None,allow_download=False):
    """Short real-data check before SD generation: no diffusion, no train refit."""
    from .evaluate_paper_v2 import load_test_panel
    ref=ReferenceBundle.load(config.reference_bundle_path,load_training_data=False)
    if not ref.metadata.get('module1_close'):raise ValueError('Zone protocol requires the closed module-1 bundle')
    active=[a for a in ref.attributes if a.startswith('zone_')]
    if not active:raise ValueError('No active frozen zone indices in the module-1 bundle')
    from .frozen_dino import validate_frozen_dino_files
    validate_frozen_dino_files(config.reference_bundle_path)
    root=Path(config.output_dir);refroot=Path(config.reference_bundle_path).resolve()
    if root.resolve()==refroot or refroot in root.resolve().parents:raise ValueError('Module-1 references are read-only; choose a separate output')
    out=root/'validation';out.mkdir(parents=True,exist_ok=True)
    path=out/'zone_preflight.json'
    path.write_text(json.dumps(dict(passed=False,protocol='observed-longitudinal-zone-error-v1')))
    if assets_root is not None:
        from .frozen_dino import load_frozen_dino_backend
        _,dino=load_frozen_dino_backend(config.reference_bundle_path,root=assets_root,
            landmark_backend=bundle.get('landmark_backend'),cache_dir=config.cache_dir or root/'cache',allow_download=allow_download)
        encoder=getattr(dino['feature_backend'],'encoder',None)
        if hasattr(encoder,'to'):encoder.to('cpu')
    store=FeatureStore(Path(config.cache_dir or root/'cache')/'features',config.features,bundle.get('landmark_backend'),bundle.get('feature_backend'),config.resume)
    if store.config_hash!=ref.feature_config_hash:raise ValueError('Frozen module-1 feature/preprocessing hash mismatch')
    panel=load_test_panel(config.test_manifest_path,config.source_root,aging_only=getattr(config,'aging_only',True))
    from .identity_overlap import check_identity_overlap
    identity_check=check_identity_overlap(config,panel,ref)
    checked=[];errors=[]
    for row in panel.head(8).to_dict('records'):
        try:
            s,t=[store.extract(row[role+'_path']) for role in ['source','target']]
            valid=[a for a in active if np.isfinite([s.get('phi__'+a,np.nan),t.get('phi__'+a,np.nan)]).all()]
            checked.append(dict(sample_id=row['sample_id'],valid_zones=valid))
            if valid:break
        except Exception as exc:errors.append(dict(sample_id=row['sample_id'],error=str(exc)))
    passed=any(row['valid_zones'] for row in checked)
    report=dict(passed=passed,protocol='observed-longitudinal-zone-error-v1',feature_config_hash=store.config_hash,
        checked=checked,failures=errors,dino_model_checked=assets_root is not None,identity_overlap_check=identity_check,
        note='Checks real source/target zone extraction, not diffusion checkpoints or future generated images.')
    path.write_text(json.dumps(report,indent=2))
    if not passed:raise RuntimeError(f'No usable real source/target zone in smoke; inspect {path}')
    print('[zone preflight] Frozen transforms and real source/target zone extraction OK.',flush=True)
    return path


def display_zone_results(result):
    """Compact notebook presentation; full tables remain on disk."""
    from IPython.display import display,Image
    root=Path(result['metadata']['output_dir'])
    for subset,title in [('aging','Envejecimiento'),('long_aging','Envejecimiento ≥20 años')]:
        print('\n'+title+' — error firmado y MAE frente a la persona real')
        path=root/'tables'/f'zone_errors_{subset}.csv'
        frame=pd.read_csv(path)
        cols=[c for c in ['model_name','measure','n_pairs','n_identities','signed_error_mean','mae','mae_ci_low','mae_ci_high','status'] if c in frame]
        display(frame[cols])
    print('\nHipótesis frente/surco/mandíbula (descriptiva, no causal):')
    display(pd.read_csv(root/'tables/ffhq_pattern_aging.csv'))
    sd=result.get('sd15_paired')
    if sd is not None and not sd.empty:
        print('\nSD1.5: MAE después − antes, misma semilla/strength; negativo = mejora')
        frame=sd[(sd.subset=='aging')&(sd.age_support=='supported')&(sd.measure_kind=='zone')]
        display(frame[['strength','measure','n_pairs','n_identities','mae_difference_after_minus_before','mae_ci_low','mae_ci_high']])
    print('\nSoporte de pares largos (no se fusionan con cortos):')
    display(pd.read_csv(root/'diagnostics/long_pair_support.csv'))
    for name in ['zone_error_aging.png','zone_error_long_aging.png','sd15_before_after_zones.png']:
        path=root/'figures'/name
        if path.is_file():display(Image(filename=str(path)))
    print('Validación de medición:',result.get('analysis_status',{}).get('passed',result['zone_validation']['passed']))
    print('Todos los CSV, cobertura y figuras:',root)
