import json
from dataclasses import replace

import pandas as pd
import pytest

from src.comparisong.long_gap_panel import LongGapPanelConfig, select_long_gap_panel


def setup_pool(tmp_path, count=5):
    rows = []
    for i in range(count):
        folder = tmp_path / f'id_{i}'
        folder.mkdir()
        for age in (20, 30):
            path = folder / f'{age}.jpg'
            path.write_bytes(b'source')
            rows.append(dict(identity_id=f'id_{i}', age=age, path=str(path), split='train'))
    manifest = tmp_path / 'train.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return LongGapPanelConfig(manifest, tmp_path / 'selected', n_transitions=9), rows


def test_balanced_deterministic_and_no_real_targets(tmp_path):
    config, _ = setup_pool(tmp_path)
    first = select_long_gap_panel(config)
    second = select_long_gap_panel(config)
    pd.testing.assert_frame_equal(first['panel'], second['panel'])
    panel = first['panel']
    assert len(panel) == 9
    assert panel.target_age.value_counts().to_dict() == {60: 3, 65: 3, 70: 3}
    assert panel.groupby('identity_id').source_path.nunique().max() == 1
    assert panel.groupby('identity_id').size().max() - panel.groupby('identity_id').size().min() <= 1
    assert panel.target_path.eq('').all()
    assert not panel.target_is_observed.any()
    assert json.loads(first['paths']['audit'].read_text())['config_fingerprint']


def test_default_size_has_350_transitions_from_117_sources(tmp_path):
    config, _ = setup_pool(tmp_path, count=120)
    result = select_long_gap_panel(replace(config, n_transitions=350))
    assert len(result['panel']) == 350
    assert result['panel'].identity_id.nunique() == 117
    assert sorted(result['panel'].target_age.value_counts()) == [116, 117, 117]
    assert result['panel'].source_path.nunique() == 117


def test_excludes_old_target_from_either_side_before_source_filter(tmp_path):
    config, rows = setup_pool(tmp_path)
    inventory = tmp_path / 'known.csv'
    pd.DataFrame([dict(identity_id='id_0', source_age=75, source_path='/old/75.jpg',
                       target_age=22, target_path='/old/22.jpg')]).to_csv(inventory, index=False)
    result = select_long_gap_panel(replace(config, inventory_manifest_paths=(inventory,)))
    assert 'id_0' not in set(result['panel'].identity_id)
    assert 'near_requested_target' in set(result['exclusions'].reason)
    assert 75 in set(result['inventory'].age)


def test_shortage_fails_and_reference_overlap_is_reported(tmp_path):
    config, _ = setup_pool(tmp_path, count=2)
    with pytest.raises(ValueError, match='Insufficient'):
        select_long_gap_panel(config)
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'metadata.json').write_text(json.dumps({'metadata': {'train_identities': ['id_0']}}))
    result = select_long_gap_panel(replace(config, n_transitions=6, reference_bundle_path=reference))
    assert result['diagnostics']['selected_reference_overlap_ids'] == ['id_0']
    assert len(result['panel']) == 6


def test_rejects_path_alias_identity_and_conflicting_age(tmp_path):
    config, rows = setup_pool(tmp_path)
    other = tmp_path / 'aliases.csv'
    pd.DataFrame([dict(identity_id='someone_else', age=20, path=rows[0]['path'])]).to_csv(other, index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        select_long_gap_panel(replace(config, inventory_manifest_paths=(other,)))
    pd.DataFrame([dict(identity_id='id_0', age=21, path=rows[0]['path'])]).to_csv(other, index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        select_long_gap_panel(replace(config, inventory_manifest_paths=(other,)))


def test_supplemental_photos_never_become_train_sources_and_gender_is_retained(tmp_path):
    config, rows = setup_pool(tmp_path, count=2)
    supplement = tmp_path / 'supplement.csv'
    pd.DataFrame([dict(identity_id='id_0', age=20, path=rows[0]['path'], gender='female'),
                  dict(identity_id='outsider', age=25, path='/not_a_train_source.jpg')]).to_csv(supplement, index=False)
    result = select_long_gap_panel(replace(config, n_transitions=6, inventory_manifest_paths=(supplement,)))
    assert set(result['panel'].identity_id) == {'id_0', 'id_1'}
    assert result['panel'].loc[result['panel'].identity_id.eq('id_0'), 'gender'].eq('female').all()
    assert 'outsider' in set(result['exclusions'].identity_id)


def test_nontrain_split_cannot_supply_sources(tmp_path):
    config, rows = setup_pool(tmp_path, count=2)
    for row in rows:
        if row['identity_id'] == 'id_0':
            row['split'] = 'test'
    pd.DataFrame(rows).to_csv(config.train_manifest_path, index=False)
    with pytest.raises(ValueError, match='Insufficient'):
        select_long_gap_panel(replace(config, n_transitions=6))


def test_explicit_unknown_age_identity_exclusion_and_integer_targets(tmp_path):
    config, _ = setup_pool(tmp_path, count=3)
    result = select_long_gap_panel(replace(config, n_transitions=6, excluded_identity_ids=('id_0',)))
    assert 'id_0' not in set(result['panel'].identity_id)
    excluded = result['exclusions'].set_index('identity_id')
    assert excluded.loc['id_0', 'reason'] == 'explicit_identity_exclusion'
    with pytest.raises(ValueError, match='integer'):
        select_long_gap_panel(replace(config, target_ages=(60.5, 65, 70)))


def test_default_ten_year_exclusion_boundaries_across_inventory(tmp_path):
    config, _ = setup_pool(tmp_path, count=4)
    known = tmp_path / 'boundary_observations.csv'
    pd.DataFrame([dict(identity_id=f'id_{i}', age=age, path=f'/known/{i}/{age}.jpg')
                  for i, age in enumerate((49, 50, 80, 81))]).to_csv(known, index=False)
    result = select_long_gap_panel(replace(config, n_transitions=6, inventory_manifest_paths=(known,)))
    assert set(result['panel'].identity_id) == {'id_0', 'id_3'}
    assert result['panel'].nearest_observed_target_age_distance.gt(10).all()
    source_49 = result['panel'].query('identity_id == "id_0" and target_age == 60').iloc[0]
    assert source_49.nearest_observed_target_age_distance == 11
    assert source_49.max_observed_age == 49
    assert source_49.min_observed_age == 20
    assert json.loads(source_49.observed_ages) == [20, 30, 49]
    near = result['exclusions'].query('reason == "near_requested_target"')
    assert set(near.identity_id) == {'id_1', 'id_2'}


@pytest.mark.parametrize('bad_age', [float('inf'), -1, 121])
def test_invalid_inventory_ages_fail(tmp_path, bad_age):
    config, rows = setup_pool(tmp_path)
    rows[0]['age'] = bad_age
    pd.DataFrame(rows).to_csv(config.train_manifest_path, index=False)
    with pytest.raises(ValueError, match='age'):
        select_long_gap_panel(config)
