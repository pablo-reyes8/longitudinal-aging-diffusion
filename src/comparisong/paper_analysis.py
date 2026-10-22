"""Re-evaluation only: cached descriptors, ICC sensitivity and mandatory DINO.

This module never imports/calls a diffusion generator or run_picking_variants.
Low dose, high feature-change norm and R² are diagnostics, not causal proof of
missing attribute coverage or demographic bias. DINO can also fail validation.
"""
from dataclasses import replace,asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.aging_bias.references import ReferenceBundle
from src.aging_bias.config import PaperV2Config,FeatureConfig,WeightingConfig
from src.aging_bias.cached_references import refit_cached_references,build_cached_embedding_references,preflight_dino_inputs
from src.aging_bias.weighting import transition_weights
from .evaluate_paper_v2 import evaluate_paper_v2,load_test_panel,_generated_path


def write_analysis_diagnostics(result,config):
    root=Path(config.output_dir);out=root/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    oracle=pd.read_csv(root/'validation/oracle_indices.csv')
    checks=json.loads((root/'validation/real_anchors.json').read_text())
    oracle.to_csv(out/'oracle_full.csv',index=False)
    indices=result['bias_indices']
    direction=pd.concat([indices,oracle[oracle.model_name.eq('oracle_target')]],ignore_index=True)
    columns=[c for c in ['model_name','scheme','panel','subset','D','D_ci_low','D_ci_high','status','share_ok',
        'pi','pi_ci_low','pi_ci_high','pi_ci_status','drift','homog','homog_excess','n_pairs','n_rows',
        'g_norm_mean','g_rms_per_coordinate','mean_valid_attributes','r2','r2_age_partial',
        'age_component_rms','residual_rms','n_boot_no_edit','n_boot_not_identified'] if c in direction]
    direction[columns].to_csv(out/'direction_diagnostics.csv',index=False)
    per=result['per_sample'];support=[]
    requested=per[per.model_name.eq('source_noedit')].copy()
    for panel_name,frame in [('requested',requested),('common',requested[requested.in_common_panel])]:
        for subset in ['all','aging','rejuvenation']:
            selected=frame if subset=='all' else frame[frame.delta_age>0] if subset=='aging' else frame[frame.delta_age<0]
            if selected.empty:continue
            weighted=transition_weights(selected,config.weighting)
            gaps=np.abs(selected.delta_age.to_numpy(float));edges=np.asarray(config.weighting.bins,float)
            for i,lo in enumerate(edges):
                hi=edges[i+1] if i+1<len(edges) else np.inf
                mask=(gaps>=lo)&(gaps<hi);group=weighted[mask]
                support.append(dict(panel=panel_name,subset=subset,original_bin=f'[{lo:g},{hi:g})',
                    n_pairs=int(mask.sum()),n_identities=selected[mask].identity_id.nunique(),
                    merged_bins=';'.join(group.delta_bin.unique()),
                    balanced_mass=float(group.w_bin_balanced.sum()),
                    fewer_than_min_bin_n=int(mask.sum())<config.weighting.min_bin_n,
                    merges=json.dumps(weighted.attrs['merges'])))
    bins=pd.DataFrame(support);bins.to_csv(out/'bin_support.csv',index=False)
    synthetic=json.loads((root/'validation/self_checks.json').read_text())
    audit=pd.DataFrame([dict(kind='synthetic',**r) for r in synthetic]+
        [dict(kind='real_anchor',**r) for r in checks['checks']])
    audit.to_csv(out/'checks_report.csv',index=False)
    paired=result['paired_differences'].copy()
    if not paired.empty:
        eligible=(paired.index_name.eq('pi')&paired.comparison_status.eq('ok')&
            paired.share_ok_model_a.gt(.9)&paired.share_ok_model_b.gt(.9)&paired.share_ok.gt(.9))
        stable=paired[eligible].copy()
        stable['anchors_passed']=checks['passed']
        if not checks['passed']:stable=stable.iloc[:0]
    else:stable=paired
    stable.to_csv(out/'stable_paired_pi.csv',index=False)
    # Report only same-scheme same-subset comparisons; never choose favorable bins.
    ops=result['operational_metrics']
    operative=ops[(ops.panel=='common')&(ops.subset=='aging')&(ops.scheme=='bin_balanced')&(ops.delta_bin=='all')]
    operative=operative.sort_values('age_rel_err_indep')
    operative.to_csv(out/'operational_aging.csv',index=False)
    print('\nOPERATIVE — aging, bin-balanced (descriptive, not a significance ranking)')
    print(operative[[c for c in ['model_name','n_pairs','age_rel_err_indep','age_rel_signed_indep','adaface_id_gt'] if c in operative]].to_string(index=False))
    print('\nREAL ORACLE — full rows and intervals: diagnostics/oracle_full.csv')
    print(oracle[[c for c in ['scheme','subset','D','D_ci_low','D_ci_high','pi','pi_ci_low','pi_ci_high','homog','share_ok'] if c in oracle]].to_string(index=False))
    print('\nDIRECTION — g norm/R² describe changes, not necessarily biological age')
    primary=direction[(direction.panel=='common')&(direction.subset=='aging')&(direction.scheme=='bin_balanced')]
    print(primary[[c for c in ['model_name','D','g_norm_mean','r2','r2_age_partial','age_component_rms','share_ok','status'] if c in primary]].to_string(index=False))
    print('\nBIN SUPPORT — actual counts/merges, not assumptions about long transitions')
    print(bins[(bins.panel=='common')&(bins.subset=='aging')].to_string(index=False))
    print('\nCHECKS (synthetic and real-anchor failures are distinguished)')
    print(audit[['kind','check','observed','passed']].to_string(index=False))
    print('\nSTABLE PAIRED pi (both models and paired draws >0.9; anchors must pass)')
    print(stable.to_string(index=False) if len(stable) else 'No eligible comparisons. No reliable bias ranking can be claimed.')
    return dict(anchors_passed=checks['passed'],failed_checks=checks.get('failed_checks',[]),
        n_common_pairs=result['metadata']['n_common_pairs'],output_dir=str(root))


def run_paper_analysis(config,*,bundle,picking_config=None,assets_root='/content/evaluation_models_v2',allow_download=False,dino_device='auto'):
    """One notebook call; reuses saved generated images and metric caches.

    Primary thresholds remain those saved in module 1. ICC .30 is a separately
    labeled sensitivity; homogenization keeps .50. DINO is mandatory, never
    silently skipped. If it fails, earlier results persist but complete/passed
    remain false. Only evaluating descriptors/metrics can download DINO;
    the picking checkpoint is neither loaded nor run. DINO alone prefers GPU;
    its encoder is resident for its full stage, then offloaded once.
    """
    # Also accept an existing notebook dataclass instance made before a module
    # reload; populate new optional defaults without touching the loaded bundle.
    raw=asdict(config)
    raw['features']=FeatureConfig(**raw['features'])
    raw['weighting']=WeightingConfig(**raw['weighting'])
    config=PaperV2Config(**raw)
    root=Path(config.output_dir);out=root/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    state=dict(complete=False,passed=False,required_backends=['handcrafted_primary','handcrafted_icc030','dinov2'],stages={})
    def persist():
        (out/'analysis_status.json').write_text(json.dumps(state,indent=2,default=str,allow_nan=True))
    persist();dino_backend=None
    try:
        roots=dict(config.generated_roots)
        if picking_config is not None:
            for name in ['ours_picking_no_prompt','ours_picking_prompt']:
                if name in config.models:roots[name]=Path(picking_config.output_dir)/'generated'/name
        config=replace(config,generated_roots=roots,cache_dir=config.cache_dir or root/'cache',robustness_backend=None)
        panel=load_test_panel(config.test_manifest_path,config.source_root)
        original=pd.read_csv(config.test_manifest_path,dtype={'sample_id':str,'identity_id':str})
        for model in config.models:
            paths=(_generated_path(config,model,row,original) for row in panel.to_dict('records'))
            if not any(p is not None and p.is_file() for p in paths):
                raise FileNotFoundError(f'No saved images for {model}; correct generated_roots. This analysis does not rerun picking.')
        ref=ReferenceBundle.load(config.reference_bundle_path,load_training_data=False)
        print('[analysis preflight] Checking saved reference provenance and original train/FFHQ paths for mandatory DINO.',flush=True)
        preflight_dino_inputs(config.reference_bundle_path,panel.identity_id.unique())
        results={}
        print('\n=== 1/3: HANDCRAFTED — original prespecified ICC policy ===',flush=True)
        results['handcrafted_primary']=evaluate_paper_v2(config,bundle=bundle,reference_bundle=ref)
        state['stages']['handcrafted_primary']=write_analysis_diagnostics(results['handcrafted_primary'],config);persist()
        print('\n=== 2/3: HANDCRAFTED — exploratory ICC .30, homog .50 ===',flush=True)
        sensitivity=refit_cached_references(config.reference_bundle_path,root/'references_icc030',
            min_icc=.30,min_homog_icc=.50,excluded_test_ids=panel.identity_id.unique())
        sensitivity_cfg=replace(config,output_dir=root/'sensitivity_icc030',reference_bundle_path=root/'references_icc030')
        print('Active sensitivity coordinates:',', '.join(sensitivity.attributes),flush=True)
        results['handcrafted_icc030']=evaluate_paper_v2(sensitivity_cfg,bundle=bundle,reference_bundle=sensitivity)
        state['stages']['handcrafted_icc030']=write_analysis_diagnostics(results['handcrafted_icc030'],sensitivity_cfg);persist()
        print('\n=== 3/3: MANDATORY DINOv2 — GPU if available, train/proxy PCA only ===',flush=True)
        dino_ref,dino_backend=build_cached_embedding_references(config.reference_bundle_path,
            Path(config.reference_bundle_path)/'dinov2',root=assets_root,
            landmark_backend=bundle.get('landmark_backend'),allow_download=allow_download,
            excluded_test_ids=panel.identity_id.unique(),device=dino_device)
        dino_bundle={**bundle,**dino_backend}
        dino_cfg=replace(config,output_dir=root/'dinov2',reference_bundle_path=Path(config.reference_bundle_path)/'dinov2',
            features=dino_backend['feature_config'])
        results['dinov2']=evaluate_paper_v2(dino_cfg,bundle=dino_bundle,reference_bundle=dino_ref)
        state['stages']['dinov2']=write_analysis_diagnostics(results['dinov2'],dino_cfg)
        sets={name:set(result['per_sample'].loc[result['per_sample'].in_common_panel,'sample_id']) for name,result in results.items()}
        overlap=pd.DataFrame([dict(backend=name,n_common_pairs=len(ids),n_shared_all_backends=len(set.intersection(*sets.values()))) for name,ids in sets.items()])
        overlap.to_csv(out/'backend_panel_overlap.csv',index=False)
        state['complete']=True
        state['passed']=all(state['stages'][name]['anchors_passed'] for name in ['handcrafted_primary','dinov2'])
        persist()
        print('\nANALYSIS COMPLETE. Bias interpretation allowed:',state['passed'],flush=True)
        print('Backend panels may differ; inspect diagnostics/backend_panel_overlap.csv before comparing their estimates.')
        return dict(**results['handcrafted_primary'],analyses=results,analysis_status=state,
            analysis_status_path=str(out/'analysis_status.json'))
    except Exception as exc:
        state.update(complete=False,passed=False,error=f'{type(exc).__name__}: {exc}')
        persist()
        print('Analysis INCOMPLETE. Completed report stages are saved. DINO cannot be omitted or replaced by handcrafted.',flush=True)
        raise
    finally:
        if dino_backend is not None:
            encoder=getattr(dino_backend.get('feature_backend'),'encoder',None)
            if hasattr(encoder,'to'):
                encoder.to('cpu')
                if str(dino_backend.get('embedding_device','')).startswith('cuda'):
                    import torch
                    torch.cuda.empty_cache()
                print('[DINOv2] Stage finished: encoder offloaded to CPU.',flush=True)
