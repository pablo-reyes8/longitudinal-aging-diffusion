"""Second mechanism: fixed vanilla versus the complete smart-picking pipeline."""
from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .evaluate_paper_v2 import load_test_panel
from .picking_variants import PickingVariantsConfig, run_picking_variants
from .sd15_counterfactual import run_sd15_before_after


def run_sd15_vs_picking(config, *, vanilla_strength=.30,
                        delta_age_strength_map=None, picking_options=None,
                        picking_bundle=None):
    """Return the same manifest/generated_roots interface as run_sd15_before_after.

    config is SD15Config. Only aging pairs are used, with its optional max_pairs.
    Vanilla has one fixed strength; the two picking variants share the canonical
    smart search. Assisted prompts are applied at the selected base strength.
    Resume is enabled without overwriting compatible picking outputs by default.
    Neural assets are loaded only when this function is explicitly executed.
    """
    if not 0<float(vanilla_strength)<=1:
        raise ValueError('vanilla_strength must be in (0,1]')
    if not config.anchors_reviewed or not json.loads(Path(config.validation_path).read_text()).get('passed'):
        raise ValueError('Run preflight_zone_protocol before this mechanism')
    root=Path(config.output_dir)
    panel=load_test_panel(config.test_manifest_path,config.source_root)
    panel=panel[panel.delta_age>0].sort_values(['delta_age','sample_id'],ascending=[False,True])
    if config.max_pairs is not None:panel=panel.head(config.max_pairs)
    if panel.empty:raise ValueError('No aging transitions for vanilla versus picking')
    options=dict(config.inference_config)
    shared={k:options[k] for k in ['image_size','num_inference_steps','text_reference_mode',
        'age_guidance_scale','text_guidance_scale','image_guidance_scale','negative_prompt'] if k in options}
    shared.update(picking_options or {})
    if delta_age_strength_map is not None:shared['delta_age_strength_map']=delta_age_strength_map
    protected={'checkpoint_path','test_manifest_path','output_dir','source_root','device','seed','resume','retry_failed'}
    if protected & set(shared):raise ValueError(f'Use SD15Config for: {sorted(protected & set(shared))}')
    if shared.get('target_age_strength_map') is not None:
        raise ValueError('Use delta_age_strength_map for this mechanism, not target ages')
    shared.setdefault('overwrite',False)
    # Validate configuration before any neural execution.
    picking=PickingVariantsConfig(checkpoint_path=config.checkpoint_path,
        test_manifest_path=root/'transitions.csv',output_dir=root/'picking_variants',
        source_root=config.source_root,device=config.device,seed=config.seed,
        resume=config.resume,retry_failed=config.retry_failed,**shared)
    from .picking_variants import initial_strength_for_transition, protected_prompt_assistance
    initial_strength_for_transition(panel.iloc[0].source_age,panel.iloc[0].target_age,picking)
    protected_prompt_assistance(picking.prompt_assistance_config)
    root.mkdir(parents=True,exist_ok=True)
    panel.to_csv(root/'transitions.csv',index=False)
    vanilla_config=replace(config,test_manifest_path=root/'transitions.csv',
        strengths=(float(vanilla_strength),),output_dir=root/'vanilla',max_pairs=None)
    vanilla=run_sd15_before_after(vanilla_config,conditions=('vanilla',))
    if picking_bundle is None:
        from src.inference import load_face_aging_inference_bundle
        picking_bundle=load_face_aging_inference_bundle(config.checkpoint_path,
            device='cpu',local_files_only=config.local_files_only,
            load_auxiliary_models=True,auxiliary_trust_remote_code=True)
    selected=run_picking_variants(picking,bundle=picking_bundle)
    manifest=pd.concat([vanilla['manifest'],selected['manifest']],ignore_index=True)
    failures=pd.concat([vanilla['failures'],selected['failures']],ignore_index=True)
    roots={**vanilla['generated_roots'],**selected['generated_roots']}
    manifest.to_csv(root/'prediction_manifest.csv',index=False)
    failures.to_csv(root/'inference_failures.csv',index=False)
    (root/'mechanism.json').write_text(json.dumps(dict(vanilla_strength=float(vanilla_strength),
        generated_roots=roots,comparison='fixed vanilla vs smart picking; not a matched-strength training effect'),indent=2))
    return dict(manifest=manifest,failures=failures,generated_roots=roots,
        paths={'manifest':str(root/'prediction_manifest.csv'),'transitions':str(root/'transitions.csv')})
