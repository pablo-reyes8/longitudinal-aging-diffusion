"""Closure reports. Mean faces/maps are descriptive, never statistical evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

AGE_BANDS=((30,40,'30-39'),(50,60,'50-59'),(60,70,'60-69'),(70,np.inf,'70+'))
ZONE_POLYGONS={
    'zone_upper':[(.23,.22),(.77,.22),(.79,.43),(.21,.43)],
    'zone_mid':[(.21,.43),(.79,.43),(.75,.69),(.25,.69)],
    'zone_lower':[(.25,.69),(.75,.69),(.63,.86),(.5,.91),(.37,.86)],
}


def zone_consistency(contrasts,attributes,orientations=None):
    rows=[];orientations=orientations or {}
    for name in attributes:
        sub=contrasts[contrasts.attribute.eq(name)]
        counts=[];signs=[]
        for key in ['difference_cross','difference_long']:
            values=sub[key].to_numpy(float) if key in sub else np.array([])
            pos=int((values>0).sum());neg=int((values<0).sum())
            counts.append(max(pos,neg));signs.append(1 if pos>neg else -1 if neg>pos else 0)
        agree=bool(signs[0]!=0 and signs[0]==signs[1]);measured=not sub.empty
        consistent=measured and min(counts)>=2 and agree
        direction='ffhq_more' if signs[0]*orientations.get(name,1)>0 else 'ffhq_less'
        rows.append(dict(attribute=name,n_transitions_same_sign_cross=counts[0],n_transitions_same_sign_long=counts[1],
            designs_agree=agree,direction=direction if consistent else ('inconsistent' if measured else 'not_measured'),
            mean_difference_cross=float(sub.difference_cross.mean()) if measured else np.nan,
            mean_difference_long=float(sub.difference_long.mean()) if measured else np.nan,
            orientation_sign=orientations.get(name,1),
            status='consistent' if consistent else ('inconsistent' if measured else 'not_measured')))
    return pd.DataFrame(rows)


def face_zone_styles(consistency):
    colors={'ffhq_more':'#d6604d','ffhq_less':'#4393c3','inconsistent':'#aaaaaa','not_measured':'#ffffff'}
    result={}
    for zone in ZONE_POLYGONS:
        rows=consistency[consistency.attribute.eq(zone)]
        row=rows.iloc[0].to_dict() if len(rows) else dict(direction='not_measured',status='not_measured',mean_difference_cross=np.nan)
        result[zone]={**row,'color':colors[row['direction']],'hatch':'///' if row['status']=='not_measured' else None}
    return result


def plot_face_zones(consistency,output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Ellipse
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);styles=face_zone_styles(consistency)
    for language in ['en','es']:
        fig,ax=plt.subplots(figsize=(8,6));ax.add_patch(Ellipse((.5,.53),.65,.8,fill=False))
        for zone,points in ZONE_POLYGONS.items():
            style=styles[zone];ax.add_patch(Polygon(points,facecolor=style['color'],edgecolor='black',alpha=.7,hatch=style['hatch']))
            center=np.mean(points,0);ax.text(*center,zone,ha='center',fontsize=9)
        ax.plot([.31,.69],[.4,.4],'ko');ax.plot([.39,.61],[.68,.68],'k-')
        legend='\n'.join(f"{z}: {v['direction']} | Δcross={v['mean_difference_cross']:.3g}" for z,v in styles.items())
        ax.text(.88,.35,legend,fontsize=8,va='center')
        ax.set(xlim=(0,1.6),ylim=(1,0),aspect='equal');ax.axis('off')
        ax.set_title('Zonas faciales — descriptivo; inferencia en tablas' if language=='es' else 'Facial zones — descriptive; inference in tables')
        fig.savefig(root/('face_zones.png' if language=='en' else 'face_zones_es.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    return styles


def age_band(age,group=None,bands=AGE_BANDS):
    """Never split a broad FFHQ label into invented exact-age subgroups."""
    if group is not None and pd.notna(group):
        low,high=map(float,str(group).split('-'))
        return next((label for a,b,label in bands if low>=a and high<b),None)
    return next((label for a,b,label in bands if a<=age<b),None)


def _histogram_normalize(image):
    a=np.asarray(image.convert('L'),np.uint8)
    hist=np.bincount(a.ravel(),minlength=256);cdf=hist.cumsum().astype(float)
    nonzero=cdf[hist>0]
    if not len(nonzero) or nonzero[-1]==nonzero[0]:return a.astype(float)/255
    return np.clip((cdf[a]-nonzero[0])/(nonzero[-1]-nonzero[0]),0,1)


def aggregate_mean_faces(records,min_n=50,normalize_histogram=True,bands=AGE_BANDS):
    # Streaming sums: never retain all aligned photos in memory.
    sums={};counts={}
    for population,age,image in records:
        band=age_band(age,bands=bands)
        if band is None:continue
        key=(population,band)
        a=_histogram_normalize(image) if normalize_histogram else np.asarray(image.convert('L'),float)/255
        hf=(a-gaussian_filter(a,1.2))**2
        if key not in sums:sums[key]=dict(pixels=np.zeros_like(a),hf=np.zeros_like(a));counts[key]=0
        sums[key]['pixels']+=a;sums[key]['hf']+=hf;counts[key]+=1
    means={key:{k:v/counts[key] for k,v in values.items()} for key,values in sums.items() if counts[key]>=min_n}
    coverage=pd.DataFrame([dict(population=p,band=b,n=counts.get((p,b),0),status='ok' if counts.get((p,b),0)>=min_n else 'insufficient_n')
        for p in ['latam','ffhq'] for _,_,b in bands])
    return means,coverage


def aging_maps(means,young='30-39',old='60-69'):
    result={}
    shape=next(iter(means.values()))['pixels'].shape if means else (64,64)
    for kind in ['pixels','hf']:
        for pop in ['latam','ffhq']:
            result[pop+('_hf' if kind=='hf' else '')]=(means[(pop,old)][kind]-means[(pop,young)][kind]
                if (pop,old) in means and (pop,young) in means else np.full(shape,np.nan))
        result['difference'+('_hf' if kind=='hf' else '')]=result['ffhq'+('_hf' if kind=='hf' else '')]-result['latam'+('_hf' if kind=='hf' else '')]
    return result


def mean_faces(config,latam,ffhq,landmark_backend,output_dir,*,min_n=50):
    """Descriptive grayscale means and HF energy, not inferential evidence.

    FFHQ 50–69 is NOT split into 50–59/60–69. Missing exact cells are marked;
    an additional common 50–69 map is reported explicitly as a different band.
    """
    from .preprocessing import preprocess_image
    from src.quantitative_metrics.image_io import ImageResolver
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);(root.parent/'diagnostics').mkdir(parents=True,exist_ok=True)
    resolver=ImageResolver();failures=[];ambiguous=[]
    def records(broad=False):
        bands=((30,40,'30-39'),(50,70,'50-69'),(70,np.inf,'70+')) if broad else AGE_BANDS
        for pop,frame in [('latam',latam),('ffhq',ffhq)]:
            for row in frame.to_dict('records'):
                estimated=not broad and row.get('mean_face_age_source')=='model_prediction'
                band=age_band(row.get('mean_face_age',np.nan) if estimated else row['age'],None if estimated else row.get('age_group'),bands)
                if band is None:
                    if not broad:ambiguous.append(dict(population=pop,image_path=row['image_path'],age_group=row.get('age_group'),reason='outside_or_ambiguous_age_band'))
                    continue
                try:
                    image,_=resolver.load(row['image_path']);resolver._cache.clear()
                    prep=preprocess_image(image,landmark_backend,config)
                    if not prep.covariates['pose_quality_ok'] or not prep.covariates['expression_quality_ok']:continue
                    yield pop,dict((label,(a+b)/2 if np.isfinite(b) else a+5) for a,b,label in bands)[band],prep.image
                except Exception as exc:failures.append(dict(population=pop,image_path=row['image_path'],error=str(exc)))
    means,coverage=aggregate_mean_faces(records(),min_n=min_n)
    estimated='mean_face_age_source' in ffhq and ffhq.mean_face_age_source.eq('model_prediction').any()
    coverage['age_assignment']='FFHQ 50–69 model-estimated; other ages official' if estimated else 'official group / source annotations'
    coverage.to_csv(root.parent/'diagnostics/mean_face_coverage.csv',index=False)
    pd.DataFrame(ambiguous).to_csv(root.parent/'diagnostics/mean_face_ambiguous_ages.csv',index=False)
    _plot_means(means,coverage,root/'mean_faces_grid.png',estimated=estimated)
    maps=aging_maps(means);_plot_maps(maps,root/'aging_difference_maps.png',
        '60–69 minus 30–39'+(' (FFHQ 50–69: model-estimated ages)' if estimated else ''))
    broad_bands=((30,40,'30-39'),(50,70,'50-69'),(70,np.inf,'70+'))
    broad,bcov=aggregate_mean_faces(records(True),min_n=min_n,bands=broad_bands)
    bcov.to_csv(root.parent/'diagnostics/mean_face_common_band_coverage.csv',index=False)
    _plot_maps(aging_maps(broad,old='50-69'),root/'aging_difference_maps_common_50_69.png','50–69 minus 30–39 (common broad bands)')
    pd.DataFrame(failures).to_csv(root.parent/'diagnostics/mean_face_failures.csv',index=False)
    return coverage


def _plot_means(means,coverage,path,*,estimated=False):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for ax,row in zip(axes.ravel(),coverage.to_dict('records')):
        key=(row['population'],row['band']);ax.axis('off')
        if key in means:ax.imshow(means[key]['pixels'],cmap='gray',vmin=0,vmax=1)
        else:ax.text(.5,.5,'Not estimable',ha='center')
        ax.set_title(f"{key[0]} {key[1]} n={row['n']}")
    fig.suptitle('Descriptive means only; inference in tables'+('\nFFHQ 50–69: model-estimated ages' if estimated else ''))
    fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)


def _plot_maps(maps,path,title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    for i,kind in enumerate(['','_hf']):
        arrays=[maps[p+kind] for p in ['latam','ffhq','difference']]
        finite=np.concatenate([a[np.isfinite(a)] for a in arrays]);lim=max(np.max(abs(finite)) if len(finite) else 0,1e-10)
        for ax,p,a in zip(axes[i],['latam','ffhq','difference'],arrays):
            ax.axis('off');ax.set_title(p+kind)
            if np.isfinite(a).any():
                ax.imshow(a,cmap='RdBu_r',vmin=-lim,vmax=lim)
                for points in ZONE_POLYGONS.values():ax.add_patch(Polygon(np.array(points)*[a.shape[1],a.shape[0]],fill=False,edgecolor='black',linewidth=.5))
            else:ax.text(.5,.5,'Not estimable: age labels / n',ha='center',fontsize=8)
    fig.suptitle(title+'\nDescriptive only; inference in tables');fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)


def save_landmark_check(image,landmarks,path):
    """Real image overlay for HUMAN review; not automatic anatomical validation."""
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,6));ax.imshow(image)
    for key,indices in [('malar',[116,345]),('mouth_corners',[61,291]),('zygomatic',[234,454])]:
        if key not in landmarks:continue
        for point,index in zip(landmarks[key],indices):
            ax.plot(*point,'rx');ax.text(*point,str(index),color='yellow')
    ax.set_title('Candidate landmarks — anatomical review required');ax.axis('off');fig.savefig(path,dpi=300);plt.close(fig)
