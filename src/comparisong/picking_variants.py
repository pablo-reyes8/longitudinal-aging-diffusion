"""Separate base/assisted smart-sweep outputs for paper v2; legacy APIs unchanged."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import json
import shutil
import uuid
import math
import pandas as pd
from PIL import Image
from src.aging_bias.features import stable_hash
from src.quantitative_metrics.image_io import ImageResolver
from .evaluate_paper_v2 import load_test_panel
from .evaluation_backends_v2 import file_hash


def default_prompt_assistance():
    return dict(positive_global_terms=[],negative_global_terms=[],use_mouth_rules=True,
        use_eye_rules=True,use_expression_rules=False,
        closed_mouth_positive_terms=['keep lips closed','preserve mouth shape'],
        closed_mouth_negative_terms=['open mouth','visible teeth'],
        visible_teeth_positive_terms=['preserve natural teeth visibility','preserve natural smile'],
        visible_teeth_negative_terms=['deformed teeth','extra teeth'],
        eye_positive_terms=['preserve eye shape','preserve gaze direction','natural eyes'],
        eye_negative_terms=['deformed eyes','asymmetric eyes','dark hollow eyes'],
        max_support_terms_per_block=4)


def default_delta_age_strength_map():
    """Prespecified starting heuristic, not a calibrated optimum or final strength."""
    return {0:.15,10:.15,20:.22,30:.29,40:.33,60:.40}


@dataclass
class PickingVariantsConfig:
    checkpoint_path: str | Path
    test_manifest_path: str | Path
    output_dir: str | Path
    target_age_strength_map: dict | None = None
    prompt_assistance_config: dict = field(default_factory=default_prompt_assistance)
    max_trials_per_target: int = 10
    min_strength: float = .04
    max_strength: float = .45
    confidence_aging: float = 3.
    confidence_rejuvenecer: float = 2.
    strength_step_coarse: float = .05
    strength_step_medium: float = .03
    strength_step_fine: float = .02
    prompt_assistance_scale: float = .30
    negative_prompt_assistance_scale: float = .25
    use_bias_corrected_mivolo_target: bool = True
    enable_guidance_micro_search: bool = False
    identity_tiebreak: bool = True
    identity_margin_for_tiebreak: float = 1.
    save_all_trials: bool = False
    seed: int = 2026
    num_inference_steps: int = 40
    image_size: int = 400
    age_guidance_scale: float = 3.
    text_guidance_scale: float = 7.
    image_guidance_scale: float = 1.5
    text_reference_mode: str = 'source_age'
    negative_prompt: str = ''
    device: str = 'cuda'
    source_root: str | Path | None = None
    resume: bool = True
    retry_failed: bool = False
    # Appended to preserve positional compatibility; explicit old target maps
    # remain supported, but the benchmark notebook uses this gap-based policy.
    delta_age_strength_map: dict = field(default_factory=default_delta_age_strength_map)


def initial_strength_for_transition(source_age,target_age,config):
    """Interpolate the start by absolute age gap, then clip to search bounds.

    Aging/rejuvenation of the same magnitude get the same start; the existing
    smart search still applies its directional tolerances and selection logic.
    An explicitly supplied legacy target map overrides the default gap policy.
    """
    if config.target_age_strength_map is not None:
        from src.inference.smart_age_sweep import _initial_strength
        start=_initial_strength(float(target_age),config.target_age_strength_map)
    else:
        mapping=config.delta_age_strength_map
        if not isinstance(mapping,dict) or not mapping:
            raise ValueError('delta_age_strength_map must be a non-empty mapping')
        anchors=sorted((float(gap),float(strength)) for gap,strength in mapping.items())
        if any(not math.isfinite(g) or g<0 or not math.isfinite(s) or not 0<s<=1 for g,s in anchors):
            raise ValueError('delta_age_strength_map requires finite gaps >=0 and strengths in (0,1]')
        gap=abs(float(target_age)-float(source_age))
        start=anchors[-1][1]
        if gap<=anchors[0][0]:start=anchors[0][1]
        else:
            for (left,a),(right,b) in zip(anchors,anchors[1:]):
                if gap<=right:
                    start=a+(b-a)*(gap-left)/(right-left)
                    break
    return min(config.max_strength,max(config.min_strength,start))


def run_picking_variants(config,*,bundle,sweep_backend=None):
    """One canonical smart search per pair, plus its assisted prompt variant.

    Matches diagnose_checkpoint_smart_age_sweep exactly: the assisted image
    reuses the selected base strength/CFG; it is NOT a second strength search.
    Always saves each variant separately, even when the assisted result is
    worse. MiVOLO/ArcFace are selection auxiliaries, NOT the v2 evaluators.
    Bundle must be the already prepared inference bundle, not FairFace metrics.
    No model download/loading happens here. Residency spans all transitions.
    """
    if not 0<config.min_strength<=config.max_strength<=1:raise ValueError('Invalid picking strength bounds')
    if isinstance(config.max_trials_per_target,bool) or not isinstance(config.max_trials_per_target,int) or not 1<=config.max_trials_per_target<=20:
        raise ValueError('max_trials_per_target must be an integer in [1,20]')
    if bundle.get('unet') is None or bundle.get('identity_encoder') is None or bundle.get('age_estimator') is None:
        raise ValueError('Use the inference bundle with ArcFace/MiVOLO, not the evaluation bundle')
    panel=load_test_panel(config.test_manifest_path,config.source_root)
    # Validate the policy before any inference or GPU placement.
    initial_strength_for_transition(panel.iloc[0].source_age,panel.iloc[0].target_age,config)
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    names={'ours_picking_no_prompt':('selected_results',None),'ours_picking_prompt':('assisted_results','assisted')}
    roots={name:str(output/'generated'/name) for name in names}
    params=asdict(config)
    for key in ['checkpoint_path','test_manifest_path','output_dir','source_root','resume','retry_failed','device']:params.pop(key)
    signature=stable_hash(dict(config=params,checkpoint_sha256=file_hash(config.checkpoint_path),version='paired-smart-prompts-v2-delta-start'))
    params.pop('delta_age_strength_map')  # Internal benchmark policy, not a sweep argument.
    resolver=ImageResolver();rows=[];failures=[];active=False
    previous_reuse=bundle.get('_reuse_loaded_checkpoint',False)
    def persist():
        pd.DataFrame(rows).to_csv(output/'prediction_manifest.csv',index=False)
        pd.DataFrame(failures,columns=[*panel.columns,'model_name','generated_path','inference_status','inference_error']).to_csv(output/'inference_failures.csv',index=False)
    def quarantine(path,marker):
        if path.exists():
            stale=output/'stale'/path.relative_to(output/'generated')
            stale=stale.with_name(stale.stem+'_'+uuid.uuid4().hex+stale.suffix);stale.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),str(stale))
            if marker.exists():shutil.copy2(marker,stale.with_suffix('.json'))
    try:
        for sample in panel.to_dict('records'):
            start=initial_strength_for_transition(sample['source_age'],sample['target_age'],config)
            sample={**sample,'initial_strength':start,
                'initial_strength_policy':'age_gap' if config.target_age_strength_map is None else 'explicit_target_map'}
            pending=[]
            try:
                digest=resolver.load(sample['source_path'])[1];resolver._cache.clear()
                key=stable_hash(dict(run=signature,source=digest,source_age=sample['source_age'],target_age=sample['target_age']))
            except Exception as exc:
                for name in names:
                    path=Path(roots[name])/sample['identity_id']/(sample['sample_id']+'.png')
                    quarantine(path,path.with_suffix('.json'))
                    row=dict(sample,model_name=name,generated_path=None,inference_status='failed',inference_error=str(exc))
                    rows.append(row);failures.append(row)
                persist();continue
            for name in names:
                path=Path(roots[name])/sample['identity_id']/(sample['sample_id']+'.png');marker=path.with_suffix('.json')
                row=dict(sample,model_name=name,generated_path=str(path),inference_status='ok')
                cached=None;valid=False
                if config.resume and marker.exists():
                    try:cached=json.loads(marker.read_text())
                    except (ValueError,OSError):pass
                if cached and cached.get('signature')==key and cached.get('status')!='failed':
                    try:
                        with Image.open(path) as im:im.verify()
                        valid=True
                    except OSError:pass
                if valid:rows.append({**row,**cached.get('selection',{})});continue
                quarantine(path,marker)
                if cached and cached.get('signature')==key and cached.get('status')=='failed' and not config.retry_failed:
                    row.update(generated_path=None,inference_status='failed',inference_error=cached['error'])
                    rows.append(row);failures.append(row);continue
                pending.append((name,path,marker,row))
            if not pending:continue
            error=None;frame=None
            try:
                if not active:
                    active=True  # Also offload after a partially failed device placement.
                    for name in ['unet','vae','text_encoder','age_delta_conditioner','identity_encoder','age_estimator']:
                        module=bundle.get(name)
                        if hasattr(module,'to'):module.to(config.device)
                    bundle['_reuse_loaded_checkpoint']=True
                if sweep_backend is None:
                    from src.inference import diagnose_checkpoint_smart_age_sweep
                    sweep_backend=diagnose_checkpoint_smart_age_sweep
                pair_params={**params,'target_age_strength_map':{int(sample['target_age']):start}}
                print(f'[ours picking variants] {sample["sample_id"]} | initial strength={start:.3f}',flush=True)
                frame=sweep_backend(checkpoint_path=config.checkpoint_path,bundle=bundle,
                    source_image=sample['source_path'],source_age=int(sample['source_age']),target_ages=[int(sample['target_age'])],
                    output_dir=output/'picking'/sample['sample_id'],generate_assisted_prompt_variant=True,**pair_params)
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            for name,path,marker,row in pending:
                try:
                    if error is not None:raise RuntimeError(error)
                    result_key,frame_key=names[name];result=frame.attrs[result_key][0]
                    image=result['image'] if isinstance(result,dict) else result
                    selected=frame if frame_key is None else frame.attrs[frame_key]
                    selection=dict(selected_variant='base' if frame_key is None else 'assisted',strength=float(selected.iloc[0]['strength']))
                    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.png');image.save(temp);temp.replace(path)
                    marker.write_text(json.dumps(dict(signature=key,selection=selection)))
                    row.update(selection)
                except Exception as exc:
                    row.update(generated_path=None,inference_status='failed',inference_error=f'{type(exc).__name__}: {exc}')
                    failures.append(row.copy());marker.parent.mkdir(parents=True,exist_ok=True)
                    marker.write_text(json.dumps(dict(signature=key,status='failed',error=row['inference_error'])))
                rows.append(row)
            persist()
    finally:
        bundle['_reuse_loaded_checkpoint']=previous_reuse
        if active:
            for name in ['unet','vae','text_encoder','age_delta_conditioner','identity_encoder','age_estimator']:
                module=bundle.get(name)
                if hasattr(module,'to'):module.to('cpu')
    persist();panel.to_csv(output/'transitions.csv',index=False)
    (output/'run_config.json').write_text(json.dumps(dict(signature=signature,config=asdict(config)),default=str,indent=2))
    return dict(manifest=pd.DataFrame(rows),generated_roots=roots,failures=pd.DataFrame(failures))
