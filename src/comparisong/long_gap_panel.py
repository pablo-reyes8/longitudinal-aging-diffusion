"""Select auditable train-source counterfactuals without inventing real targets."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongGapPanelConfig:
    train_manifest_path: str | Path
    output_dir: str | Path
    inventory_manifest_paths: tuple = ()
    source_root: str | Path | None = None
    n_transitions: int = 350
    source_age_min: int = 18
    source_age_max: int = 35
    target_ages: tuple = (60, 65, 70)
    exclusion_margin: float = 10
    min_gap: float = 25
    seed: int = 2026
    reference_bundle_path: str | Path | None = None
    excluded_identity_ids: tuple = ()


def _read_inventory(path, *, train=False, source_root=None):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    identity = next((k for k in ('identity_id', 'person_id', 'id') if k in frame), None)
    if identity is None:
        raise ValueError(f'{path}: missing identity_id')
    pairs = {'source_age', 'source_path', 'target_age', 'target_path'} <= set(frame)
    image_path = next((k for k in ('path', 'image_path', 'relative_path') if k in frame), None)
    if not pairs and not ('age' in frame and image_path):
        raise ValueError(f'{path}: expected image age/path or complete source/target schema')
    records = []
    for number, row in enumerate(frame.to_dict('records'), 2):
        who = row[identity].strip()
        if not who or who in {'.', '..'} or '/' in who or '\\' in who:
            raise ValueError(f'{path}:{number}: invalid identity_id')
        split = row.get('split', '').strip().lower()
        gender = row.get('gender', '').strip().lower()
        aliases = {'h': 'male', 'm': 'female'}
        gender = aliases.get(gender, gender)
        roles = [('source_age', 'source_path'), ('target_age', 'target_path')] if pairs else [('age', image_path)]
        for age_key, path_key in roles:
            raw_age, raw_path = row[age_key].strip(), row[path_key].strip()
            # Counterfactual rows may appear in supplemental inventories: only
            # actual observed photos contribute ages, never requested targets.
            if not raw_path and pairs and age_key == 'target_age':
                continue
            try:
                age = float(raw_age)
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: invalid age') from exc
            if not np.isfinite(age) or not 0 <= age <= 120:
                raise ValueError(f'{path}:{number}: invalid age {raw_age}')
            if not raw_path:
                raise ValueError(f'{path}:{number}: missing observed image path')
            photo = Path(raw_path)
            if train and source_root is not None:
                photo = Path(source_root) / who / photo.name
            elif not photo.is_absolute():
                photo = Path(path).resolve().parent / photo
            records.append(dict(identity_id=who, age=age, path=str(photo.resolve()),
                gender=gender, train_source=bool(train and (not split or split == 'train')),
                inventory_manifest=str(Path(path).resolve()), split=split))
    if not records:
        raise ValueError(f'{path}: empty observed inventory')
    return records


def select_long_gap_panel(config):
    """Return panel/inventory/exclusions/diagnostics/paths and persist audits.

    All supplied observations drive conservative identity-wide exclusion. Only
    train-manifest photos can be sources; one photo is chosen per identity.
    Reference overlap is reported as exposure, never silently removed.
    """
    if isinstance(config.n_transitions, bool) or not isinstance(config.n_transitions, int) or config.n_transitions < 1:
        raise ValueError('n_transitions must be a positive integer')
    targets = tuple(config.target_ages)
    numeric = [config.source_age_min, config.source_age_max, config.exclusion_margin, config.min_gap, *targets]
    if not targets or not all(np.isfinite(v) for v in numeric):
        raise ValueError('Invalid finite age configuration')
    if len(set(targets)) != len(targets) or not all(0 <= t <= 120 for t in targets):
        raise ValueError('target_ages must be distinct ages in [0,120]')
    if any(isinstance(t, (bool, np.bool_)) or float(t) != int(t) for t in targets):
        raise ValueError('target_ages must be integer ages')
    if not 0 <= config.source_age_min <= config.source_age_max <= 120 or config.exclusion_margin < 0 or config.min_gap <= 0:
        raise ValueError('Invalid source age bounds, exclusion margin or minimum gap')
    rows = _read_inventory(config.train_manifest_path, train=True, source_root=config.source_root)
    for path in config.inventory_manifest_paths:
        rows.extend(_read_inventory(path))
    inventory = pd.DataFrame(rows)
    # A shared physical path may not masquerade as multiple people or ages.
    for path, group in inventory.groupby('path'):
        if group.identity_id.nunique() != 1 or group.age.nunique() != 1:
            raise ValueError(f'Conflicting identity/age metadata for image path: {path}')
    for who, group in inventory.groupby('identity_id'):
        genders = group.loc[group.gender.ne(''), 'gender'].unique()
        if len(genders) > 1:
            raise ValueError(f'Conflicting gender metadata for identity: {who}')
        if len(genders):
            inventory.loc[group.index, 'gender'] = genders[0]
    # Preserve train provenance when a photo is repeated in supplemental input.
    inventory = inventory.sort_values(['identity_id', 'age', 'path', 'train_source'], ascending=[True, True, True, False])
    inventory = inventory.drop_duplicates(['identity_id', 'age', 'path']).reset_index(drop=True)
    exclusions, candidates = [], {}
    explicit_exclusions = set(map(str, config.excluded_identity_ids))
    for who, group in inventory.groupby('identity_id', sort=True):
        near = group[group.age.map(lambda age: any(abs(age - t) <= config.exclusion_margin for t in targets))]
        sources = group[group.train_source & group.age.between(config.source_age_min, config.source_age_max)]
        if who in explicit_exclusions:
            exclusions.append(dict(identity_id=who, reason='explicit_identity_exclusion', observed_ages='[]'))
        elif len(near):
            exclusions.append(dict(identity_id=who, reason='near_requested_target',
                observed_ages=json.dumps(sorted(near.age.unique().tolist()))))
        elif sources.empty:
            exclusions.append(dict(identity_id=who, reason='no_train_source_in_age_range', observed_ages='[]'))
        else:
            sources = sources[sources.age.map(lambda age: all(t - age >= config.min_gap for t in targets))]
            if sources.empty:
                exclusions.append(dict(identity_id=who, reason='insufficient_age_gap', observed_ages='[]'))
            else:
                candidates[who] = sources.sort_values(['age', 'path']).to_dict('records')
    capacity = len(candidates) * len(targets)
    if capacity < config.n_transitions:
        raise ValueError(f'Insufficient eligible transitions: {capacity} available from {len(candidates)} identities; requested {config.n_transitions}')
    rng = random.Random(config.seed)
    identities = sorted(candidates)
    rng.shuffle(identities)
    identities = identities[:(config.n_transitions + len(targets) - 1) // len(targets)]
    chosen = {who: rng.choice(candidates[who]) for who in identities}
    observed_ages = {who: sorted(group.age.unique().tolist())
                     for who, group in inventory.groupby('identity_id')}
    selected = []
    # Visit every identity before its second transition; rotate targets so every
    # prefix stays balanced even when 350 is not divisible by three.
    for turn in range(len(targets)):
        for index, who in enumerate(identities):
            source = chosen[who]
            age = targets[(index + turn) % len(targets)]
            key = f'{who}|{source["path"]}|{age}'
            selected.append(dict(sample_id='counterfactual__' + hashlib.sha256(key.encode()).hexdigest()[:20],
                identity_id=who, source_age=source['age'], target_age=age,
                source_path=source['path'], target_path='', target_is_observed=False,
                delta_age=age-source['age'], gender=source['gender'], panel_type='train_source_counterfactual',
                nearest_observed_target_age_distance=min(abs(observed-age) for observed in observed_ages[who]),
                max_observed_age=max(observed_ages[who]), min_observed_age=min(observed_ages[who]),
                observed_ages=json.dumps(observed_ages[who])))
            if len(selected) == config.n_transitions:
                break
        if len(selected) == config.n_transitions:
            break
    panel = pd.DataFrame(selected)
    missing = sorted({p for p in panel.source_path if not Path(p).is_file()})
    if missing:
        raise FileNotFoundError(f'Selected source files do not exist: {missing[:10]}')
    reference_ids = set()
    if config.reference_bundle_path is not None:
        root = Path(config.reference_bundle_path)
        metadata_path = root if root.is_file() else root / 'references.json'
        if not metadata_path.exists() and root.is_dir():
            metadata_path = root / 'metadata.json'
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        reference_ids = set(map(str, meta.get('metadata', meta).get('train_identities', [])))
    caveat = ('Sources come from model training data. This is a train-source counterfactual stress test; '
              'it does not estimate held-out generalization and has no observed target or oracle.')
    diagnostics = dict(n_transitions=len(panel), n_identities=int(panel.identity_id.nunique()),
        n_eligible_identities=len(candidates), eligible_capacity=capacity,
        target_counts={str(k):int(v) for k,v in panel.target_age.value_counts().sort_index().items()},
        gap_counts={str(k):int(v) for k,v in panel.delta_age.value_counts().sort_index().items()},
        reference_overlap_ids=sorted(set(inventory.identity_id) & reference_ids),
        selected_reference_overlap_ids=sorted(set(panel.identity_id) & reference_ids),
        train_exposure_caveat=caveat)
    cfg = json.loads(json.dumps(asdict(config), default=str))
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = dict(panel=root/'transitions.csv', inventory=root/'observed_inventory.csv',
                 exclusions=root/'selection_exclusions.csv', audit=root/'selection_audit.json')
    exclusions = pd.DataFrame(exclusions, columns=['identity_id', 'reason', 'observed_ages'])
    panel.to_csv(paths['panel'], index=False)
    inventory.to_csv(paths['inventory'], index=False)
    exclusions.to_csv(paths['exclusions'], index=False)
    input_paths = [config.train_manifest_path, *config.inventory_manifest_paths]
    hashes = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in input_paths}
    paths['audit'].write_text(json.dumps(dict(config=cfg, config_fingerprint=fingerprint,
        input_manifest_sha256=hashes, **diagnostics), indent=2), encoding='utf-8')
    return dict(panel=panel, inventory=inventory, exclusions=exclusions, diagnostics=diagnostics, paths=paths)
