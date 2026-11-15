"""Frozen-reference counterfactual evaluation, with no observed target image.

Residuals compare generated change to population expectations, not to an
individual's unknown future. Bootstrap intervals condition on module 1.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from src.aging_bias.config import FeatureConfig
from src.aging_bias.features import FeatureStore
from src.aging_bias.references import ReferenceBundle
from src.aging_bias.decomposition import estimate_bias_decomposition
from .evaluate_paper_v2 import OperationalStore, _generated_path


@dataclass
class LongGapEvaluationConfig:
    manifest_path: object
    generated_roots: dict
    models: list
    reference_bundle_path: object
    output_dir: object
    cache_dir: object = None
    n_boot: int = 1000
    seed: int = 2026
    batch_size: int = 16
    d_min: float = .10
    max_cond: float = 1e3
    max_corr: float = .95


def load_long_gap_panel(manifest):
    """Refuse real targets rather than accidentally evaluating an oracle."""
    frame = manifest.copy() if isinstance(manifest, pd.DataFrame) else pd.read_csv(
        manifest, dtype={'sample_id': str, 'identity_id': str})
    required = ['sample_id', 'identity_id', 'source_age', 'target_age', 'source_path']
    if not set(required) <= set(frame):
        raise ValueError(f'Counterfactual manifest requires {required}')
    if 'target_path' in frame and frame.target_path.fillna('').astype(str).str.strip().ne('').any():
        raise ValueError('Counterfactual manifest must not contain a real target_path')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Need nonempty counterfactual transitions without missing required values')
    for key in ['sample_id', 'identity_id', 'source_path']:
        frame[key] = frame[key].astype(str)
        if frame[key].str.strip().eq('').any():
            raise ValueError(f'Empty {key}')
    for key in ['source_age', 'target_age']:
        frame[key] = pd.to_numeric(frame[key], errors='raise')
        if not frame[key].between(0, 120).all():
            raise ValueError('Ages must be finite and in [0,120]')
    frame['delta_age'] = frame.target_age - frame.source_age
    if frame.delta_age.le(0).any():
        raise ValueError('Counterfactual evaluation requires aging transitions')
    if frame.duplicated('sample_id').any() or frame.duplicated(['identity_id', 'source_path', 'target_age']).any():
        raise ValueError('Duplicate counterfactual transition or sample_id')
    return frame.reset_index(drop=True)


def _orientations(ref):
    signs = dict(ref.metadata.get('attribute_orientations', {}))
    transforms = ref.metadata.get('zone_transform') or ref.metadata.get('config', {}).get('features', {}).get('zone_transform', {})
    for spec in transforms.values():
        for name, sign in zip(spec.get('attributes', []), spec.get('signs', [])):
            signs.setdefault(name, sign)
    for name in ref.attributes:
        if name.startswith('zone_'):
            signs[name] = 1.
    return signs


def _cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else np.nan


def _interval(values):
    good = np.asarray(values, float)
    good = good[np.isfinite(good)]
    return np.percentile(good, [2.5, 97.5]) if len(good) else (np.nan, np.nan)


def _summary_tables(measures, per, models, config):
    """Identity means first; shared identity draws preserve paired contrasts."""
    rng = np.random.default_rng(config.seed)
    summary, paired = [], []
    specifications = [('all', 'all', None)]
    specifications += [('target', str(v), ('target_age', v)) for v in sorted(per.target_age.unique())]
    specifications += [('gap', str(v), ('delta_age', v)) for v in sorted(per.delta_age.unique())]
    feature_metrics = ['error', 'absolute_error', 'squared_error', 'model_change', 'expected_change',
                       'ffhq_error', 'latino_cross_error', 'ffhq_absolute_error', 'latino_cross_absolute_error']
    for group, label, selector in specifications:
        for source, is_measure in [(measures, True), (per, False)]:
            subset = source if selector is None else source[source[selector[0]].eq(selector[1])]
            blocks = subset.groupby('measure', sort=False) if is_measure else [('operational', subset)]
            for name, block in blocks:
                metrics = feature_metrics if is_measure else ['adaface_id_source', 'age_signed_indep', 'age_mae_indep',
                    'age_change_indep','age_change_error_indep','age_change_absolute_error_indep','age_source_calibration_indep']
                for metric in metrics:
                    common_column = 'in_common_panel' if is_measure else 'operational_common'
                    eligible = block[block[common_column]]
                    wide = eligible.pivot(index=['sample_id', 'identity_id'], columns='model_name', values=metric)
                    if not set(models) <= set(wide):
                        continue
                    wide = wide[models].dropna()
                    if wide.empty:
                        continue
                    by_id = wide.groupby(level='identity_id').mean()
                    point = by_id.mean().to_numpy()
                    if metric == 'squared_error':
                        point = np.sqrt(point)
                    samples = np.empty((0, len(models)))
                    if len(by_id) >= 2 and config.n_boot:
                        values = by_id.to_numpy()
                        counts = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=config.n_boot)
                        samples = counts @ values / len(values)
                        if metric == 'squared_error':
                            samples = np.sqrt(samples)
                    metric_name = 'rmse' if metric == 'squared_error' else metric
                    for j, model in enumerate(models):
                        lo, hi = _interval(samples[:, j])
                        summary.append(dict(group=group, group_value=label, measure=name, metric=metric_name,
                            model_name=model, value=point[j], ci_low=lo, ci_high=hi,
                            n_pairs=len(wide), n_identities=len(by_id), panel='common',
                            ci_status='ok' if len(samples) else 'insufficient_identities_or_not_bootstrapped'))
                    for j, model in enumerate(models):
                        for k in range(j + 1, len(models)):
                            lo, hi = _interval(samples[:, j] - samples[:, k])
                            paired.append(dict(group=group, group_value=label, measure=name, metric=metric_name,
                                model_a=model, model_b=models[k], difference=point[j]-point[k],
                                ci_low=lo, ci_high=hi, n_pairs=len(wide), n_identities=len(by_id)))
    return (pd.DataFrame(summary, columns=['group','group_value','measure','metric','model_name','value',
                'ci_low','ci_high','n_pairs','n_identities','panel','ci_status']),
            pd.DataFrame(paired, columns=['group','group_value','measure','metric','model_a','model_b',
                'difference','ci_low','ci_high','n_pairs','n_identities']))


def _structural(per, ref, models, config):
    records = []
    policy = ref.metadata.get('reliability_policy', {})
    aw = policy.get('attribute_weights')
    for model in models:
        frame = per[per.model_name.eq(model)].sort_values('sample_id')
        arrays = {key: frame[[key+'__'+a for a in ref.attributes]].to_numpy(float)
                  for key in ['g', 'd_C', 'd_P', 'G_P', 'zeta']}
        mask = frame[['common_valid__'+a for a in ref.attributes]].to_numpy(bool)
        identities = frame.identity_id.astype(str).to_numpy()
        _, inverse, counts = np.unique(identities, return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        options = dict(attribute_weights=aw, d_min=config.d_min, max_cond=config.max_cond,
                       max_corr=config.max_corr, warn=False)
        fit = estimate_bias_decomposition(**arrays, weights=weights, attribute_mask=mask, **options)
        # Actual weighted design diagnostics, including fixed attribute intercepts.
        i, j = np.where(mask & np.logical_and.reduce([np.isfinite(v) for v in arrays.values()]))
        active = np.unique(j)
        design = np.column_stack([(j[:, None] == active).astype(float),
            *[arrays[key][i, j] for key in ['d_C', 'd_P', 'G_P']], -arrays['zeta'][i, j]])
        attribute_weights = np.ones(len(ref.attributes)) if aw is None else np.asarray(aw)
        weighted = design * np.sqrt(weights[i] * attribute_weights[j])[:, None]
        sv = np.linalg.svd(weighted, compute_uv=False) if len(i) else np.array([])
        record = dict(model_name=model, **asdict(fit), rank=int(np.linalg.matrix_rank(weighted)) if len(i) else 0,
            design_columns=design.shape[1], singular_values=json.dumps(sv.tolist()),
            panel='common_supported_joint', reference_uncertainty='conditional_on_frozen_module1')
        good = []
        unique = np.unique(identities)
        rng = np.random.default_rng(config.seed)
        if len(unique) >= 2:
            for _ in range(config.n_boot):
                multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)))
                draw = estimate_bias_decomposition(**arrays, weights=weights*multiplicity[inverse],
                    attribute_mask=mask, compute_diagnostics=False, **options)
                good.append(draw)
        record['share_ok'] = sum(d.status == 'ok' for d in good) / len(good) if good else np.nan
        record['share_identified'] = sum(d.status != 'not_identified' for d in good) / len(good) if good else np.nan
        stable = fit.status == 'ok' and bool(good) and record['share_ok'] > .9
        for metric in ['D', 'pi', 'beta_C', 'beta_P']:
            values = [getattr(d, metric) for d in good if metric != 'pi' or d.status == 'ok']
            lo, hi = _interval(values)
            if metric == 'pi' and not stable:
                lo = hi = np.nan
            record[metric+'_ci_low'], record[metric+'_ci_high'] = lo, hi
        record['pi_ci_status'] = 'ok' if stable else 'not_bootstrapped' if not good else 'unstable_bootstrap'
        records.append(record)
    return pd.DataFrame(records)


def evaluate_long_gap(config, *, bundle, reference_bundle=None, feature_store=None):
    """Evaluate generated images against frozen expectations; never use targets.

    An empty operational bundle permits descriptor-only evaluation. Supplying
    any operational backend requires all three: aligner, identity encoder, and
    independent age backend. Failures remain missing and enter coverage tables.
    """
    if config.n_boot < 0 or config.batch_size < 1:
        raise ValueError('Invalid bootstrap count or batch size')
    panel = load_long_gap_panel(config.manifest_path)
    models = list(dict.fromkeys(config.models))
    if not models or {'source_noedit', 'oracle_target'} & set(models):
        raise ValueError('Supply actual model names; source_noedit is added automatically')
    models += ['source_noedit']
    ref = reference_bundle or ReferenceBundle.load(config.reference_bundle_path, load_training_data=False)
    output = Path(config.output_dir)
    refpath = Path(config.reference_bundle_path).resolve()
    if output.resolve() == refpath or refpath in output.resolve().parents:
        raise ValueError('Module 1 is read-only; choose a separate output directory')
    cache = Path(config.cache_dir or output/'cache')
    if cache.resolve() == refpath or refpath in cache.resolve().parents:
        raise ValueError('Module 1 is read-only; choose a separate cache directory')
    output.mkdir(parents=True, exist_ok=True)
    if feature_store is None:
        raw = dict(ref.metadata['config']['features'])
        raw['new_attributes'] = tuple(raw.get('new_attributes', ()))
        relocated = bundle.get('feature_config')
        relocated_path = relocated.get('landmark_model_path') if isinstance(relocated, dict) else getattr(relocated, 'landmark_model_path', None)
        if relocated_path is not None:
            raw['landmark_model_path'] = relocated_path
        features = FeatureConfig(**raw)
        landmarks = bundle.get('landmark_backend')
        if landmarks is None:
            from src.aging_bias.preprocessing import MediaPipeLandmarks
            landmarks = MediaPipeLandmarks(features.landmark_model_path)
        feature_store = FeatureStore(cache/'features', features, landmarks, bundle.get('feature_backend'))
    if feature_store.config_hash != ref.feature_config_hash:
        raise ValueError('Feature/preprocessing configuration differs from frozen module 1')
    required = ['aligner', 'identity_encoder', 'independent_age_backend']
    operational = None
    if any(bundle.get(key) is not None for key in required):
        if any(bundle.get(key) is None for key in required):
            raise ValueError('Operational metrics require aligner, identity_encoder, independent_age_backend')
        from .critic_provenance import assert_independent_age
        age_backend = bundle['independent_age_backend']
        assert_independent_age(getattr(age_backend, 'name', type(age_backend).__name__), config.models)
        assert_independent_age(type(age_backend).__name__, config.models)
        operational = OperationalStore(bundle, cache/'operational')
    failures, rows = [], []
    memo_features, memo_ops = {}, {}
    policy = ref.metadata.get('reliability_policy', {})
    signs = _orientations(ref)
    def safe_get(path, sample, model, role, ops=False):
        memo = memo_ops if ops else memo_features
        key = str(path)
        if key not in memo:
            try:
                if path is None:
                    raise FileNotFoundError('No generated image path')
                memo[key] = operational.get(path) if ops else feature_store.extract(path)
            except Exception as exc:
                memo[key] = None
                failures.append(dict(sample_id=sample, model_name=model, stage=role+('_operational' if ops else '_features'),
                    path=key, error=f'{type(exc).__name__}: {exc}'))
        return memo[key]
    print(f'[long gap] {len(panel)} transitions, targets {sorted(panel.target_age.unique())}; no observed target/oracle', flush=True)
    if operational is not None and hasattr(operational, 'prefetch'):
        paths = set(panel.source_path)
        for sample in panel.to_dict('records'):
            for model in models[:-1]:
                path = _generated_path(config, model, sample, panel)
                if path is not None and path.is_file():
                    paths.add(str(path))
        operational.prefetch(sorted(paths), config.batch_size)
    for index, sample in enumerate(panel.to_dict('records'), 1):
        sid = sample['sample_id']
        fs = safe_get(sample['source_path'], sid, 'shared', 'source')
        os = safe_get(sample['source_path'], sid, 'shared', 'source', True) if operational else None
        a, b = [sample['source_age']], [sample['target_age']]
        extrapolated = bool(ref.extrapolated(a, b)[0])
        reliability_ok = min(a[0], b[0]) >= (policy.get('retest_min_age') or 0)
        supported = not extrapolated and reliability_ok
        dc, dp, gp = ref.d_C(a, b)[0], ref.d_P(a, b)[0], ref.G_P(a)[0]
        cross = (ref._c(b, ref.mu_c)-ref._c(a, ref.mu_c))[0]
        def vector(features):
            return ref.standardize([features.get('phi__'+name, np.nan) if features else np.nan for name in ref.attributes])
        src = vector(fs)
        zeta = ref.zeta(src[None, :], a)[0]
        for model in models:
            path = sample['source_path'] if model == 'source_noedit' else _generated_path(config, model, sample, panel)
            # Check generated existence even with injected feature backends.
            if model != 'source_noedit' and (path is None or not Path(path).is_file()):
                fg = None
                failures.append(dict(sample_id=sid, model_name=model, stage='generated_path', path=str(path), error='FileNotFoundError'))
            else:
                fg = fs if model == 'source_noedit' else safe_get(path, sid, model, 'generated')
            og = os if model == 'source_noedit' else safe_get(path, sid, model, 'generated', True) if operational and fg is not None else None
            gen = vector(fg)
            row = {**sample, 'model_name':model, 'generated_path':str(path), 'extrapolated':extrapolated,
                'reliability_age_supported':reliability_ok, 'reference_supported':supported,
                'source_valid':fs is not None, 'generated_valid':fg is not None,
                'operational_valid':os is not None and og is not None,
                'adaface_id_source':np.nan, 'age_signed_indep':np.nan, 'age_mae_indep':np.nan,
                'independent_age_source':np.nan, 'independent_age_generated':np.nan,
                'age_change_indep':np.nan,'age_change_error_indep':np.nan,
                'age_change_absolute_error_indep':np.nan,'age_source_calibration_indep':np.nan}
            row['n_finite_source_features'] = int(np.isfinite(src).sum())
            row['n_finite_generated_features'] = int(np.isfinite(gen).sum())
            for role, features in [('source', fs), ('generated', fg)]:
                for key, value in (features or {}).items():
                    if key.startswith(('phi__', 'no_tta__')) or key in {'image_sha256', 'feature_config_hash'}:
                        continue
                    if isinstance(value, (str, bool, int, float, np.generic)) or value is None:
                        row[f'{role}_quality__{key}'] = value
            if os is not None and og is not None:
                signed_age = og['independent_age'] - b[0]
                row.update(adaface_id_source=_cosine(os['embedding'], og['embedding']),
                    age_signed_indep=signed_age, age_mae_indep=abs(signed_age),
                    independent_age_source=os['independent_age'], independent_age_generated=og['independent_age'],
                    age_change_indep=og['independent_age']-os['independent_age'],
                    age_change_error_indep=og['independent_age']-os['independent_age']-sample['delta_age'],
                    age_change_absolute_error_indep=abs(og['independent_age']-os['independent_age']-sample['delta_age']),
                    age_source_calibration_indep=os['independent_age']-a[0])
            for j, name in enumerate(ref.attributes):
                row.update({f'raw_source__{name}':fs.get('phi__'+name, np.nan) if fs else np.nan,
                    f'raw_generated__{name}':fg.get('phi__'+name, np.nan) if fg else np.nan,
                    f'phi_source__{name}':src[j], f'phi_generated__{name}':gen[j],
                    f'g__{name}':gen[j]-src[j], f'd_C__{name}':dc[j] if supported else np.nan,
                    f'd_P__{name}':dp[j] if supported else np.nan, f'G_P__{name}':gp[j] if supported else np.nan,
                    f'zeta__{name}':zeta[j] if supported else np.nan,
                    f'cross__{name}':cross[j] if supported else np.nan})
            rows.append(row)
        if index % 100 == 0 or index == len(panel):
            pd.DataFrame(rows).to_csv(output/'per_sample_progress.csv', index=False)
            pd.DataFrame(failures, columns=['sample_id','model_name','stage','path','error']).to_csv(output/'failures.csv', index=False)
            print(f'[long gap] {index}/{len(panel)} transitions; {len(failures)} failures', flush=True)
    per = pd.DataFrame(rows)
    operational_keys = set(per.groupby('sample_id').operational_valid.all().loc[lambda v:v].index)
    per['operational_common'] = per.sample_id.isin(operational_keys)
    measure_rows = []
    for name in ref.attributes:
        valid = per.reference_supported & np.isfinite(per[['g__'+name, 'd_C__'+name, 'd_P__'+name, 'cross__'+name]]).all(axis=1)
        common = set(per.assign(_valid=valid).groupby('sample_id')._valid.all().loc[lambda v:v].index)
        per['common_valid__'+name] = per.sample_id.isin(common)
        oriented = name in signs and not name.startswith('embedding_')
        sign = float(signs[name]) if oriented else 1.
        if sign not in (-1., 1.):
            raise ValueError(f'Invalid frozen orientation for {name}')
        for (_, row), ok in zip(per.iterrows(), valid):
            change, expected = sign*row['g__'+name], sign*row['d_C__'+name]
            error = change-expected if ok else np.nan
            missing_reason = ('reference_age_out_of_support' if row.extrapolated else
                'reliability_age_unsupported' if not row.reliability_age_supported else
                'source_measure_missing' if not np.isfinite(row['phi_source__'+name]) else
                'generated_measure_missing' if not np.isfinite(row['phi_generated__'+name]) else
                'reference_measure_missing' if not ok else 'ok')
            measure_rows.append({key:row[key] for key in ['sample_id','identity_id','source_age','target_age','delta_age',
                'model_name','source_path','generated_path','extrapolated','reliability_age_supported']} | dict(
                measure=name, measure_kind='zone' if name.startswith('zone_') else 'component',
                orientation_sign=sign, aging_oriented=oriented, model_change=change,
                expected_change=expected, error=error, native_standardized_error=row['g__'+name]-row['d_C__'+name] if ok else np.nan,
                absolute_error=abs(error), squared_error=error**2,
                ffhq_expected_change=sign*row['d_P__'+name], latino_cross_expected_change=sign*row['cross__'+name],
                ffhq_error=change-sign*row['d_P__'+name] if ok else np.nan,
                latino_cross_error=change-sign*row['cross__'+name] if ok else np.nan,
                ffhq_absolute_error=abs(change-sign*row['d_P__'+name]) if ok else np.nan,
                latino_cross_absolute_error=abs(change-sign*row['cross__'+name]) if ok else np.nan,
                interpretation='more_than_expected' if ok and oriented and error>0 else 'less_than_expected' if ok and oriented and error<0 else
                    'matches_expectation' if ok and error==0 else 'native_axis_only' if ok else 'unsupported_or_missing',
                valid=bool(ok), in_common_panel=row.sample_id in common))
            measure_rows[-1]['missing_reason'] = missing_reason
    measures = pd.DataFrame(measure_rows)
    summary, paired = _summary_tables(measures, per, models, config)
    structural = _structural(per, ref, models, config)
    coverage = measures.groupby(['model_name','target_age','measure'], as_index=False).agg(
        n_requested=('sample_id','size'), n_valid=('valid','sum'), n_common=('in_common_panel','sum'),
        n_extrapolated=('extrapolated','sum'), n_reliability_supported=('reliability_age_supported','sum'))
    failures = pd.DataFrame(failures, columns=['sample_id','model_name','stage','path','error'])
    tables = dict(per_sample=per, per_measure=measures, summary=summary, paired=paired, structural_bias=structural,
                  coverage=coverage, failures=failures)
    for key, table in tables.items():
        table.to_csv(output/(key+'.csv'), index=False)
    completeness = dict(complete=True, n_transitions=len(panel), n_models=len(models), failures=len(failures),
        n_supported=int(per[per.model_name.eq('source_noedit')].reference_supported.sum()),
        operational_available=operational is not None, reference_hash=ref.feature_config_hash,
        targets=sorted(map(float,panel.target_age.unique())), reference_support=list(ref.support),
        ffhq_ages=ref.p_ages.tolist(), reference_uncertainty='CIs condition on frozen module 1',
        population_caveat='d_C: Colombian longitudinal FE; d_P: FFHQ cross-sectional; Latino cross: supplemented cross-sectional',
        signs='Positive oriented residual: more change in the frozen aging direction than expected; not years or true future error',
        train_source_caveat='Train-source descriptive evaluation, not independent held-out validation')
    (output/'completeness.json').write_text(json.dumps(completeness, indent=2))
    reference_files = {}
    for filename in ['references.json', 'references.npz']:
        path = refpath/filename
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(chunk)
            reference_files[filename] = digest.hexdigest()
    serial_config = asdict(config)
    if isinstance(config.manifest_path, pd.DataFrame):
        serial_config['manifest_path'] = 'in_memory_dataframe'
    manifest_digest = hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest()
    (output/'run_config.json').write_text(json.dumps(dict(config=serial_config, reference_files_sha256=reference_files,
        manifest_sha256=manifest_digest, feature_config_hash=ref.feature_config_hash,
        operational_signature=operational.signature if operational is not None and hasattr(operational,'signature') else None),
        indent=2,default=str))
    print('[long gap] support/coverage:\n'+coverage.to_string(index=False), flush=True)
    print('[long gap] '+completeness['signs'], flush=True)
    print('[long gap] structural diagnostics:\n'+structural[['model_name','status','D','pi','rank','design_columns','cond_number']].to_string(index=False), flush=True)
    print('[long gap] summaries:\n'+summary.head(18).to_string(index=False), flush=True)
    return {**tables, 'completeness':completeness, 'output_dir':output}
