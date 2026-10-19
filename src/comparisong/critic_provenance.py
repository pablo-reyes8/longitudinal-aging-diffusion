"""Training critics, not inference diagnostics. Unknown is not evidence of independence."""
import re

CRITIC_PROVENANCE={
    'SAM':{'age':['DEX_IMDBWIKI_ft_FFHQAging'],'identity':['ArcFace'],
        'source':'https://github.com/yuval-alaluf/SAM (criteria/aging_loss.py, criteria/id_loss.py)'},
    'HRFAE':{'age':['DEX_IMDBWIKI'],'identity':[],
        'source':'https://github.com/InterDigitalInc/HRFAE ; SAM paper discussion of DEX'},
    'ours':{'age':['MiVOLO_v2'],'identity':['ArcFace_R50'],
        'source':'planing/informe_tecnico_experimento_v5_dual_aging_400x400_epoc6.md'},
    'FRAN':{'age':['unknown'],'identity':['LPIPS_VGG'],
        'source':'https://github.com/timroelofs123/face_reaging/blob/main/model/losses.py',
        'note':'Reproduction loss code: L1 + LPIPS(VGG) + conditional adversarial loss; training-data teacher provenance unknown. LPIPS is perceptual, not an identity classifier.'},
    'FADING':{'age':['unknown'],'identity':['unknown'],
        'source':'https://github.com/MunchkinChen/FADING',
        'note':'Age-aware diffusion specialization documented; exact checkpoint critic/data-teacher provenance not verified.'},
    'SD15_vanilla':{'age':[],'identity':[],
        'source':'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5'},
}


def model_family(name):
    if name.lower().startswith('ours'):return 'ours'
    if name.startswith('SD15'):return 'SD15_vanilla'
    return name.split('@')[0]


def critic_family(name):
    value=re.sub('[^a-z0-9]','',str(name).lower())
    for family in ('dex','mivolo','arcface','adaface','fairface'):
        if family in value:return family
    return value


def assert_independent_age(name,models):
    family=critic_family(name)
    # Forbidden even when a user selects a subset of the comparison.
    if family in {'dex','mivolo'}:raise ValueError(f'{name} is a training critic, not an independent age estimator')
    for model in models:
        entry=CRITIC_PROVENANCE.get(model_family(model),{})
        if family in [critic_family(x) for x in entry.get('age',[])]:
            raise ValueError(f'{name} is a training critic of {model}')


def contamination_flags(name):
    if name in {'source_noedit','oracle_target'}:
        return {'metric_contaminated__dex_age_mae':False,'metric_contaminated__adaface':False,'critic_provenance_unknown':False}
    entry=CRITIC_PROVENANCE.get(model_family(name),{'age':['unknown'],'identity':['unknown']})
    age=entry['age'];identity=entry['identity']
    return {'metric_contaminated__dex_age_mae':any(critic_family(x)=='dex' for x in age),
        'metric_contaminated__adaface':any(critic_family(x)=='adaface' for x in identity),
        'critic_provenance_unknown': 'unknown' in age or 'unknown' in identity}
