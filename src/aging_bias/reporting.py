"""Paper tables and curves; no model calls, no suppression of failed identification."""
from pathlib import Path
import numpy as np
import pandas as pd


def report_paper(output_dir):
    root=Path(output_dir);tables=root/'tables';tables.mkdir(exist_ok=True)
    indices=pd.read_csv(root/'bias_indices.csv');ops=pd.read_csv(root/'operational_metrics.csv')
    if len(indices):
        t=indices.query("panel=='common' and subset=='aging' and scheme=='bin_balanced'").copy()
        t.to_csv(tables/'table1_bias.csv',index=False)
        homog='homog_excess' if 'homog_excess' in indices else 'homog'
        cols=['model_name','D','pi','drift',homog,'status']
        rendered=t[cols].copy()
        for k in ['D','pi','drift',homog]:
            rendered[k]=[f'{v:.3f} [{lo:.3f}, {hi:.3f}]' if np.isfinite(v) else '--'
                for v,lo,hi in zip(t[k],t[k+'_ci_low'],t[k+'_ci_high'])]
        (tables/'table1_bias.tex').write_text(rendered.to_latex(index=False,escape=True))
        for subset in ['all','aging']:
            view=indices.query("panel=='common' and scheme=='bin_balanced'")
            view=view[view.subset==subset].copy()
            view.to_csv(tables/f'table1_bias_{subset}.csv',index=False)
            formatted=view[cols].copy()
            for k in ['D','pi','drift',homog]:
                formatted[k]=[f'{v:.3f} [{lo:.3f}, {hi:.3f}]' if np.isfinite(v) else '--'
                    for v,lo,hi in zip(view[k],view[k+'_ci_low'],view[k+'_ci_high'])]
            (tables/f'table1_bias_{subset}.tex').write_text(formatted.to_latex(index=False,escape=True))
    if len(ops):
        ops.query("model_name!='oracle_target'").to_csv(tables/'table2_operational.csv',index=False)
        t=ops.query("panel=='common' and subset=='aging' and delta_bin=='all' and model_name!='oracle_target'")
        (tables/'table2_operational.tex').write_text(t.to_latex(index=False,float_format='%.4f'))
        for subset in ['all','aging']:
            view=ops.query("panel=='common' and delta_bin=='all' and model_name!='oracle_target'")
            view=view[view.subset==subset]
            view.to_csv(tables/f'table2_operational_{subset}.csv',index=False)
            (tables/f'table2_operational_{subset}.tex').write_text(view.to_latex(index=False,float_format='%.4f'))
        flags=pd.read_parquet(root/'per_sample.parquet').groupby('model_name')[['metric_contaminated__dex_age_mae','critic_provenance_unknown']].first()
        kid=root/'kid_metrics.csv'
        kid_frame=pd.DataFrame()
        if kid.exists():
            try:kid_frame=pd.read_csv(kid).drop(columns='panel',errors='ignore')
            except pd.errors.EmptyDataError:pass
        for subset in ['all','aging']:
            appendix=ops.query("panel=='common' and scheme=='unweighted' and delta_bin=='all' and model_name!='oracle_target'").copy()
            appendix=appendix[appendix.subset==subset].merge(flags,on='model_name',how='left')
            if not kid_frame.empty:
                if 'subset' in kid_frame:
                    appendix=appendix.merge(kid_frame[kid_frame.subset==subset],on=['model_name','subset'],how='left')
                elif subset=='all':appendix=appendix.merge(kid_frame,on='model_name',how='left')
            appendix.to_csv(tables/f'tableA1_legacy_{subset}.csv',index=False)
            (tables/f'tableA1_legacy_{subset}.tex').write_text(appendix.to_latex(index=False,float_format='%.4f'))
            if subset=='all':
                appendix.to_csv(tables/'tableA1_legacy.csv',index=False)
                (tables/'tableA1_legacy.tex').write_text(appendix.to_latex(index=False,float_format='%.4f'))
    bins=pd.read_csv(root/'bias_indices_by_bin.csv')
    if len(bins):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        f=bins.query("panel=='common' and subset=='aging' and scheme=='bin_balanced'")
        labels=list(dict.fromkeys(f.delta_bin))
        for model,g in f.groupby('model_name',sort=False):
            g=g.set_index('delta_bin').reindex(labels)
            ax.plot(range(len(labels)),g.pi,marker='o',label=model)
            ax.fill_between(range(len(labels)),g.pi_ci_low,g.pi_ci_high,alpha=.12)
        ax.set_xticks(range(len(labels)),labels);ax.set_ylabel('Prior attribution pi (unconstrained)')
        ax.set_xlabel('Absolute age gap');ax.legend();fig.tight_layout();fig.savefig(root/'pi_by_delta_bin.png');plt.close(fig)
    return {'tables':str(tables)}


def plot_sd15_before_after(evaluation_dir,output_dir=None,matched_d=(.3,.5,.7)):
    """Interpolate only inside observed D support; never extrapolate a match."""
    root=Path(evaluation_dir);out=Path(output_dir or root);out.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(root/'bias_indices.csv').query("panel=='common' and subset=='aging' and scheme=='bin_balanced'")
    # New reference policies report noise-calibrated homogenization everywhere,
    # including SD1.5 plots. Raw estimates remain in bias_indices.csv.
    if 'homog_excess' in df:
        for suffix in ['', '_ci_low', '_ci_high']:df['homog'+suffix]=df['homog_excess'+suffix]
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,4));matches=[]
    for prefix,label in [('SD15_vanilla@','vanilla'),('ours@','finetuned')]:
        curve=df[df.model_name.str.startswith(prefix)].copy()
        if curve.empty:continue
        curve['strength']=curve.model_name.str.split('@').str[-1].astype(float);curve=curve.sort_values('strength')
        for ax,k in zip(axes,['pi','drift','homog','D']):
            ax.plot(curve.strength,curve[k],marker='o',label=label)
            ax.fill_between(curve.strength,curve[k+'_ci_low'],curve[k+'_ci_high'],alpha=.15)
            ax.set_xlabel('strength');ax.set_title(k)
        valid=curve.query("status=='ok'").sort_values('D').drop_duplicates('D')
        for d in matched_d:
            if len(valid)<2 or not valid.D.min()<=d<=valid.D.max():continue
            for k in ['pi','drift','homog']:
                matches.append(dict(condition=label,D=d,index_name=k,value=np.interp(d,valid.D,valid[k]),
                    note='Descriptive point interpolation; bootstrap curve-matching CI not estimated'))
    for ax,k in zip(axes,['pi','drift','homog','D']):
        for _,r in df[~df.model_name.str.contains('@',regex=False)&df.model_name.ne('source_noedit')].iterrows():
            ax.axhline(r[k],alpha=.25,linestyle=':',label=f'{r.model_name} (D={r.D:.2f})')
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'sd15_before_after.png');plt.close(fig)
    dosefig,doseaxes=plt.subplots(1,3,figsize=(14,4))
    for prefix,label in [('SD15_vanilla@','vanilla'),('ours@','finetuned')]:
        curve=df[df.model_name.str.startswith(prefix)].query("status=='ok'").sort_values('D')
        for ax,k in zip(doseaxes,['pi','drift','homog']):
            ax.plot(curve.D,curve[k],marker='o',label=label)
            ax.fill_between(curve.D,curve[k+'_ci_low'],curve[k+'_ci_high'],alpha=.12)
    for _,r in df[~df.model_name.str.contains('@',regex=False)&df.model_name.ne('source_noedit')].iterrows():
        for ax,k in zip(doseaxes,['pi','drift','homog']):
            ax.scatter([r.D],[r[k]],marker='x',label=r.model_name)
    for ax,k in zip(doseaxes,['pi','drift','homog']):ax.set_xlabel('Observed dose D');ax.set_ylabel(k);ax.legend(fontsize=7)
    dosefig.tight_layout();dosefig.savefig(out/'sd15_dose_matched.png');plt.close(dosefig)
    result=pd.DataFrame(matches);result.to_csv(out/'matched_d_comparison.csv',index=False)
    return result
