"""Observed, within-person longitudinal errors using FROZEN module-1 measures.

Positive signed error = more aged than the real target in train-oriented units.
This includes pose/capture/measurement differences; it is not causal evidence of
nationality/ethnicity or a pure biological aging effect. No models are loaded here.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from src.aging_bias.config import WeightingConfig
from src.aging_bias.weighting import transition_weights

ZONE_NAMES=('zone_upper','zone_mid','zone_lower')
METRICS=('signed_error_mean','mae','rmse','model_change_mean','real_change_mean')


def longitudinal_errors(per,reference,models):
    if per.duplicated(['sample_id','model_name']).any():raise ValueError('Duplicate transition/model rows')
    orientations=dict(reference.metadata.get('attribute_orientations',{}))
    transforms=reference.metadata.get('zone_transform') or reference.metadata.get('config',{}).get('features',{}).get('zone_transform',{})
    for spec in transforms.values():
        for name,sign in zip(spec.get('attributes',[]),spec.get('signs',[])):orientations.setdefault(name,sign)
    rows=[]
    for name in reference.attributes:
        kind='zone' if name in ZONE_NAMES else 'latent_component' if name.startswith('embedding_') else 'component'
        sign=1. if kind=='zone' else orientations.get(name,1.)
        # Never interpret PCA axis signs as anatomically oriented aging.
        for row in per.to_dict('records'):
            s,t,g=[row.get(prefix+'__'+name,np.nan) for prefix in ['phi_source','phi_target','phi_generated']]
            finite=bool(np.isfinite([s,t,g,sign]).all())
            real=sign*(t-s);change=sign*(g-s);error=sign*(g-t)
            result={key:row.get(key) for key in ['sample_id','identity_id','model_name','source_age','target_age','delta_age',
                'source_path','target_path','generated_path','reliability_age_supported']}
            result.update(measure=name,measure_kind=kind,orientation_sign=sign,model_change=change,
                real_change=real,error=error if finite else np.nan,absolute_error=abs(error) if finite else np.nan,
                native_standardized_error=g-t if finite else np.nan,
                valid=finite,status='ok' if finite else 'missing_or_quality_excluded',ffhq_expected_gap=np.nan)
            if kind!='latent_component':
                a,b=[row.get('source_age')],[row.get('target_age')]
                j=reference.attributes.index(name)
                cross=reference._c(b,reference.mu_c)[0,j]-reference._c(a,reference.mu_c)[0,j]
                gap=sign*(reference.d_P(a,b)[0,j]-cross)
                result['ffhq_expected_gap']=gap if not row.get('extrapolated',False) else np.nan
            rows.append(result)
    columns=['sample_id','identity_id','model_name','source_age','target_age','delta_age',
        'source_path','target_path','generated_path','reliability_age_supported','measure','measure_kind',
        'orientation_sign','model_change','real_change','error','absolute_error','native_standardized_error','valid','status','ffhq_expected_gap']
    result=pd.DataFrame(rows,columns=columns)
    valid=result[result.model_name.isin(models)].groupby(['sample_id','measure']).agg(n_models=('model_name','nunique'),all_valid=('valid','all'))
    keys=set(valid.index[(valid.n_models==len(models))&valid.all_valid])
    result['in_common_panel']=[(s,m) in keys for s,m in zip(result.sample_id,result.measure)]
    result['direction']=np.where(result.delta_age>0,'aging','rejuvenation')
    result['is_long']=result.delta_age.abs()>=20
    return result


def cluster_weights(base,multiplicity,config):
    """Equivalent to transition_weights on repeated identity DRAW SLOTS.

    Compressed counts avoid constructing thousands of pandas frames per CI.
    Duplicate draws of a person remain independent slots for identity balancing.
    """
    k=np.asarray(multiplicity,float);gaps=abs(base.delta_age.to_numpy(float));edges=np.asarray(config.bins,float)
    if not np.isfinite(k).all() or np.any(k<0) or k.sum()<=0:raise ValueError('Invalid cluster multiplicities')
    if not np.isfinite(gaps).all() or np.any(gaps<edges[0]):raise ValueError('Uncovered transition age delta')
    assignment=np.searchsorted(edges,gaps,side='right')-1
    mapping=np.arange(len(edges))
    while True:
        present=np.unique(assignment[k>0]);small=[b for b in present if k[assignment==b].sum()<config.min_bin_n]
        if not small or len(present)==1:break
        b=small[0];lower=present[present<b];neighbor=lower[-1] if len(lower) else present[present>b][0]
        assignment[assignment==b]=neighbor;mapping[mapping==b]=neighbor
    if config.scheme_primary=='unweighted':return k/k.sum()
    weights=np.zeros(len(k));present=np.unique(assignment[k>0])
    target=config.operational_target
    if config.scheme_primary=='operational':
        target=np.asarray(target,float);target=target/target.sum()
        if sum(target[i] for i in range(len(edges)) if mapping[i] not in present)>1e-12:
            raise ValueError('Bootstrap operational target has unsupported bin')
    ids=base.identity_id.astype(str).to_numpy()
    for b in present:
        mask=assignment==b;v=k[mask].copy()
        if config.identity_balance:
            _,inverse,counts=np.unique(ids[mask],return_inverse=True,return_counts=True);v/=counts[inverse]
        mass=1/len(present) if config.scheme_primary=='bin_balanced' else target[mapping==b].sum()
        weights[mask]=mass*v/v.sum()
    return weights/weights.sum()


def _statistics(error,change,real,weights):
    return np.array([weights@error,weights@abs(error),np.sqrt(weights@(error**2)),weights@change,weights@real])


def _summarize_group(frame,models,config,n_boot,rng):
    base=frame[frame.model_name.eq(models[0])].sort_values('sample_id').reset_index(drop=True)
    ids=base.sample_id.tolist();trans=base[['sample_id','identity_id','delta_age']]
    values=[]
    for model in models:
        aligned=frame[frame.model_name.eq(model)].set_index('sample_id').loc[ids]
        values.append(aligned[['error','model_change','real_change']].to_numpy(float))
    values=np.stack(values,axis=1)
    # Validate configuration/merges against the established weighting contract.
    weighted=transition_weights(trans,config.weighting)
    weights=weighted['w_'+config.weighting.scheme_primary].to_numpy()
    point=_statistics(values[:,:,0],values[:,:,1],values[:,:,2],weights)
    identities,inverse=np.unique(base.identity_id.astype(str),return_inverse=True)
    draws=[]
    if len(identities)>=2:
        for _ in range(n_boot):
            counts=rng.multinomial(len(identities),np.full(len(identities),1/len(identities)))
            try:w=cluster_weights(trans,counts[inverse],config.weighting)
            except ValueError:continue
            draws.append(_statistics(values[:,:,0],values[:,:,1],values[:,:,2],w))
    draws=np.array(draws) if draws else np.empty((0,len(METRICS),len(models)))
    return base,point,draws,weights,weighted.attrs['merges']


def evaluate_zone_errors(per,config,reference):
    """Writes all requested errors, shared-zone panels, CIs and paired contrasts.

    CIs cluster by identity and condition on frozen module-1 transforms. Long
    subsets (>=20 years) remain separate even if original weighting bins merge.
    No ratio/dose identification gate is required for an observed target error.
    """
    root=Path(config.output_dir)
    for name in ['tables','diagnostics','figures']:(root/name).mkdir(parents=True,exist_ok=True)
    errors=longitudinal_errors(per,reference,config.models)
    errors.to_csv(root/'zone_errors_per_transition.csv',index=False)
    summaries=[];paired=[];sd=[];coverage=[];rng=np.random.default_rng(config.seed)
    anchors=['source_noedit','oracle_target'];requested=config.models+anchors
    subsets={'all':lambda f:np.ones(len(f),bool),'aging':lambda f:f.delta_age>0,
        'rejuvenation':lambda f:f.delta_age<0,'long_aging':lambda f:f.delta_age>=20,
        'long_rejuvenation':lambda f:f.delta_age<=-20}
    for measure,group in errors.groupby('measure',sort=False):
        for subset,select in subsets.items():
            subset_frame=group.loc[select(group)]
            for support in ['supported','all_ages_exploratory']:
                frame=subset_frame if support=='all_ages_exploratory' else subset_frame[subset_frame.reliability_age_supported.fillna(True).astype(bool)]
                for panel in ['common','full']:
                    sets=[requested] if panel=='common' else [[m] for m in requested]
                    for models in sets:
                        selected=frame[frame.model_name.isin(models)&frame.valid]
                        if panel=='common':selected=selected[selected.in_common_panel]
                        valid_ids=selected.groupby('sample_id').model_name.nunique()
                        selected=selected[selected.sample_id.isin(valid_ids.index[valid_ids.eq(len(models))])]
                        if selected.empty:continue
                        base,point,draws,w,merges=_summarize_group(selected,models,config,config.n_boot if panel=='common' else 0,rng)
                        kind=base.measure_kind.iloc[0];n_ids=base.identity_id.nunique()
                        metadata=dict(measure=measure,measure_kind=kind,subset=subset,age_support=support,panel=panel,
                            scheme=config.weighting.scheme_primary,n_pairs=len(base),n_identities=n_ids,
                            bin_merges=json.dumps(merges),n_boot_valid=len(draws),
                            status='ok' if n_ids>=2 else 'insufficient_identities')
                        stable=len(draws)>=.9*config.n_boot and config.n_boot>0
                        metadata['ci_status']='ok' if stable else 'not_requested' if config.n_boot==0 else 'full_panel_not_bootstrapped' if panel=='full' else 'insufficient_bootstrap_or_identities'
                        for j,model in enumerate(models):
                            row=dict(metadata,model_name=model)
                            for k,metric in enumerate(METRICS):
                                row[metric]=point[k,j]
                                lo,hi=np.percentile(draws[:,k,j],[2.5,97.5]) if stable else (np.nan,np.nan)
                                row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
                            row['direction_interpretation']='latent axis only, not biological age' if kind=='latent_component' else 'positive = more aged than real target'
                            actual=selected[selected.model_name.eq(model)].set_index('sample_id').loc[base.sample_id]
                            gap=actual.ffhq_expected_gap.to_numpy(float);known=np.isfinite(gap)&(abs(gap)>1e-9)
                            row['ffhq_sign_agreement']=float(np.average((actual.error.to_numpy()[known]*gap[known])>0,weights=w[known])) if known.any() else np.nan
                            summaries.append(row)
                        if panel!='common':continue
                        for j,model in enumerate(models):
                            if model in anchors:continue
                            for k,other in enumerate(models):
                                if j>=k or other in anchors:continue
                                row=dict(metadata,model_a=model,model_b=other)
                                for metric,position in [('signed_error',0),('mae',1)]:
                                    row[metric+'_difference']=point[position,j]-point[position,k]
                                    delta=draws[:,position,j]-draws[:,position,k] if len(draws) else []
                                    lo,hi=np.percentile(delta,[2.5,97.5]) if stable else (np.nan,np.nan)
                                    row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
                                paired.append(row)
                        for before in models:
                            match=re.fullmatch(r'SD15_vanilla@(.+)',before)
                            if not match:continue
                            after='ours@'+match.group(1)
                            if after not in models:continue
                            b,a=models.index(before),models.index(after)
                            row=dict(metadata,model_before=before,model_after=after,strength=float(match.group(1)))
                            for metric,position in [('signed_error',0),('mae',1)]:
                                row[metric+'_difference_after_minus_before']=point[position,a]-point[position,b]
                                delta=draws[:,position,a]-draws[:,position,b] if len(draws) else []
                                lo,hi=np.percentile(delta,[2.5,97.5]) if stable else (np.nan,np.nan)
                                row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
                            sd.append(row)
                for model in requested:
                    f=frame[frame.model_name.eq(model)]
                    coverage.append(dict(measure=measure,measure_kind=group.measure_kind.iloc[0],subset=subset,age_support=support,
                        model_name=model,n_requested=len(f),n_valid=int(f.valid.sum()),n_common=int((f.valid&f.in_common_panel).sum())))
    summary=pd.DataFrame(summaries)
    if summary.empty:
        summary=pd.DataFrame(columns=['measure','measure_kind','subset','age_support','panel','scheme','model_name','n_pairs','n_identities',*METRICS])
    pairs=pd.DataFrame(paired) if paired else pd.DataFrame(columns=['model_a','model_b','measure','measure_kind','subset','age_support','panel','mae_difference','mae_ci_low','mae_ci_high'])
    sdframe=pd.DataFrame(sd) if sd else pd.DataFrame(columns=['model_before','model_after','measure','measure_kind','subset','age_support','panel','strength','mae_difference_after_minus_before','mae_ci_low','mae_ci_high'])
    zones=summary[summary.measure_kind.eq('zone')].copy()
    summary.to_csv(root/'longitudinal_feature_errors.csv',index=False)
    pd.DataFrame(coverage).to_csv(root/'diagnostics/zone_coverage.csv',index=False)
    pairs.to_csv(root/'paired_zone_differences.csv',index=False)
    sdframe.to_csv(root/'sd15_paired_zone_errors.csv',index=False)
    primary=summary[(summary.panel=='common')&(summary.age_support=='supported')]
    for subset in subsets:
        primary[primary.subset.eq(subset)&primary.measure_kind.eq('zone')].to_csv(root/'tables'/f'zone_errors_{subset}.csv',index=False)
        primary[primary.subset.eq(subset)&primary.measure_kind.eq('component')].to_csv(root/'tables'/f'component_errors_{subset}.csv',index=False)
    expected={'wrinkle_forehead':-1,'wrinkle_nasolabial':-1,'geom_jaw_sag':1}
    pattern=[]
    for row in primary[primary.subset.eq('aging')&primary.measure.isin(expected)].to_dict('records'):
        sign=float(errors.loc[errors.measure.eq(row['measure']),'orientation_sign'].iloc[0])
        value=sign*row['signed_error_mean'];lo=sign*row.get('signed_error_mean_ci_low',np.nan);hi=sign*row.get('signed_error_mean_ci_high',np.nan)
        lo,hi=min(lo,hi),max(lo,hi)
        pattern.append(dict(model_name=row['model_name'],measure=row['measure'],n_pairs=row['n_pairs'],
            n_identities=row['n_identities'],native_error=value,ci_low=lo,ci_high=hi,expected_sign=expected[row['measure']],
            expected_direction_matches=bool(value*expected[row['measure']]>0),
            ci_supports_expected=bool(hi<0 if expected[row['measure']]<0 else lo>0) if np.isfinite([lo,hi]).all() else None,
            interpretation='componentwise descriptor errors, not a joint or causal demographic-bias test'))
    pd.DataFrame(pattern,columns=['model_name','measure','n_pairs','n_identities','native_error','ci_low','ci_high',
        'expected_sign','expected_direction_matches','ci_supports_expected','interpretation']).to_csv(root/'tables/ffhq_pattern_aging.csv',index=False)
    transforms=reference.metadata.get('zone_transform',{})
    pd.DataFrame([dict(measure=name,active=name in reference.attributes,
        zone_definition_status=transforms.get(name,{}).get('status','not_a_zone')) for name in (*ZONE_NAMES,*expected)]
        ).to_csv(root/'diagnostics/frozen_measure_availability.csv',index=False)
    oracle=errors[errors.model_name.eq('oracle_target')&errors.valid]
    checks=[dict(check='observed_targets_available',passed=bool(len(oracle)),observed=len(oracle)),
        dict(check='oracle_error_zero',passed=bool(len(oracle) and np.allclose(oracle.error,0,atol=1e-10)),observed=float(oracle.error.abs().max()) if len(oracle) else None),
        dict(check='error_equals_change_residual',passed=bool(np.allclose(errors.loc[errors.valid,'error'],(errors.model_change-errors.real_change)[errors.valid],atol=1e-10)),observed='generated-target'),
        dict(check='common_panel_available',passed=bool((errors.in_common_panel&errors.valid).any()),observed=int(errors.loc[errors.in_common_panel,'sample_id'].nunique()))]
    validation=dict(protocol='observed-longitudinal-error-v1',passed=all(row['passed'] for row in checks),checks=checks,
        note='These checks validate observed errors, not the old D/pi decomposition or causal demographic claims.')
    (root/'diagnostics/zone_validation.json').write_text(json.dumps(validation,indent=2))
    plot_zone_errors(primary,root/'figures',sdframe)
    print('\nERROR POR ZONA — aging; positivo=más envejecido que el destino real; MAE menor es mejor.',flush=True)
    cols=[c for c in ['model_name','measure','n_pairs','n_identities','signed_error_mean','mae','mae_ci_low','mae_ci_high'] if c in primary]
    print(primary[(primary.subset=='aging')&primary.measure_kind.eq('zone')][cols].to_string(index=False),flush=True)
    print('Pares largos se reportan por separado en tables/zone_errors_long_aging.csv. No se fusionan con cortos.',flush=True)
    return dict(zone_errors=errors,zone_summary=zones,feature_error_summary=summary,zone_paired=pairs,sd15_paired=sdframe,zone_validation=validation)


def plot_zone_errors(primary,output,sd):
    import matplotlib.pyplot as plt
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for subset in ['aging','long_aging']:
        frame=primary[primary.subset.eq(subset)&primary.measure_kind.eq('zone')&~primary.model_name.isin(['oracle_target','source_noedit'])]
        if frame.empty:continue
        pivot=frame.pivot(index='model_name',columns='measure',values='signed_error_mean')
        fig,ax=plt.subplots(figsize=(8,max(3,len(pivot)*.45)))
        values=pivot.to_numpy();lim=max(float(np.nanmax(abs(values))),1e-9)
        im=ax.imshow(values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot)),pivot.index)
        ax.set_title(f'{subset}: generated − real target (train-standardized)');fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(root/f'zone_error_{subset}.png',dpi=200);plt.close(fig)
    if not sd.empty:
        frame=sd[(sd.subset=='aging')&(sd.age_support=='supported')&(sd.measure_kind=='zone')]
        if frame.empty:return
        fig,ax=plt.subplots(figsize=(8,4))
        for zone,group in frame.groupby('measure'):
            group=group.sort_values('strength');ax.plot(group.strength,group.mae_difference_after_minus_before,'o-',label=zone)
        ax.axhline(0,color='black',linewidth=.8);ax.set(xlabel='Same strength before/after',ylabel='MAE after − before (negative = improvement)')
        ax.legend();fig.tight_layout();fig.savefig(root/'sd15_before_after_zones.png',dpi=200);plt.close(fig)
