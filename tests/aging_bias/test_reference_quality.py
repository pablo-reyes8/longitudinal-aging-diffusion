"""Reference controls use train data, not optimistic fallback reliability."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tests.aging_bias.test_features import landmarks


def test_adult_retests_exclude_young_expression_mismatch_and_duplicate_photos():
    from src.aging_bias.config import ReferenceConfig
    from src.aging_bias.reliability import reference_reliability
    frame = pd.DataFrame([
        dict(identity_id='young', age=20, image_sha256='a', phi__a=0, smile_score=.0),
        dict(identity_id='young', age=21, image_sha256='b', phi__a=1, smile_score=.0),
        dict(identity_id='adult', age=40, image_sha256='c', phi__a=2, smile_score=.0),
        dict(identity_id='adult', age=40, image_sha256='d', phi__a=2.1, smile_score=.02),
        dict(identity_id='adult', age=41, image_sha256='e', phi__a=9, smile_score=.8),
        dict(identity_id='adult', age=40, image_sha256='c', phi__a=2, smile_score=.0),
    ])
    config = ReferenceConfig('.', '.', '.', retest_min_age=30, retest_max_expression_delta=.1)
    report = reference_reliability(frame, ['a'], config)
    assert report.iloc[0].n_retest_pairs == 1
    assert report.iloc[0].n_retest_pairs_all_ages == 2
    assert report.iloc[0].exclusion_reason == 'insufficient_retests'
    assert np.isnan(report.iloc[0].icc)  # Never fall back to young retests.


def test_moderate_icc_qualifies_for_pi_but_not_raw_homogenization():
    from src.aging_bias.reliability import reliability_table
    rng = np.random.default_rng(12)
    signal = rng.normal(size=4000) * np.sqrt(.4)
    repeat = signal[:, None] + rng.normal(size=(4000, 2)) * np.sqrt(.6)
    row = reliability_table({'a': repeat}, min_icc=.35, min_homog_icc=.5).iloc[0]
    assert row.active and row.active_pi and not row.active_homog
    assert row.attribute_weight == pytest.approx(row.icc)


def test_pose_and_expression_mask_attributes_without_turning_missing_into_zero(tmp_path):
    from src.aging_bias.config import FeatureConfig
    from src.aging_bias.features import FeatureStore
    path = tmp_path / 'face.png'
    Image.new('RGB', (256, 256), (160, 110, 80)).save(path)
    def backend(yaw, smile):
        return lambda image: {**landmarks(), '_quality': dict(yaw_deg=yaw, smile_score=smile, jaw_open_score=0.)}
    config = FeatureConfig(max_abs_yaw=15, max_smile_score=.25)
    smiling = FeatureStore(tmp_path / 'smile', config, backend(0, .9)).extract(path)
    assert np.isnan(smiling['phi__wrinkle_perioral'])
    assert np.isfinite(smiling['phi__geom_jaw_sag'])
    posed = FeatureStore(tmp_path / 'pose', config, backend(20, 0)).extract(path)
    assert not posed['pose_quality_ok']
    assert all(np.isnan(v) for k, v in posed.items() if k.startswith('phi__'))


def test_local_illumination_correction_preserves_constant_color_and_reduces_shading():
    from src.aging_bias.preprocessing import normalize_local_illumination
    rgb = np.full((128, 128, 3), [160., 110., 80.])
    np.testing.assert_allclose(normalize_local_illumination(rgb), rgb, atol=1e-6)
    shade = np.linspace(.6, 1.4, 128)[None, :, None]
    corrected = normalize_local_illumination(rgb * shade)
    assert corrected[..., 0].std() < (rgb * shade)[..., 0].std() * .5
    # No gray-world white balance: chromatic ratios must remain unchanged.
    assert np.median(corrected[..., 0] / corrected[..., 1]) == pytest.approx(160 / 110)


def test_capture_diagnostic_detects_sign_flip_without_refitting_on_test():
    from src.aging_bias.references import fit_reference_frames
    from src.aging_bias.population_aging import capture_sensitivity
    rng = np.random.default_rng(34)
    ids = np.repeat(np.arange(100), 3)
    age = np.tile([30., 45., 60.], 100)
    capture = age / 20 + rng.normal(0, .4, len(age))
    frame = pd.DataFrame(dict(identity_id=ids, age=age, is_grayscale=capture,
                              log_min_side=0., blur_laplacian_var=0.,
                              phi__geom_brow_eye=.05 * age - 3 * capture + np.repeat(rng.normal(size=100), 3)))
    proxy = pd.DataFrame(dict(age=[30., 60.], phi__geom_brow_eye=[0., 1.]))
    ref = fit_reference_frames(frame, proxy, ['geom_brow_eye'], [30, 45, 60], [0], [1], [.8], 'test')
    report = capture_sensitivity(ref, [(30, 60)])
    row = report.iloc[0]
    assert row.d_C_adjusted == pytest.approx(1.5, abs=.001)
    assert row.d_C_unadjusted < 0 and row.sign_changed


def test_attribute_weighting_downweights_noisy_dimension_in_actual_fit():
    from src.aging_bias.decomposition import estimate_bias_decomposition
    rng = np.random.default_rng(2)
    arrays = {k: rng.normal(size=(300, 2)) for k in ['d_C', 'd_P', 'G_P', 'zeta']}
    g = arrays['d_C'].copy()
    g[:, 1] = 10 * arrays['d_P'][:, 1]
    fit = estimate_bias_decomposition(g, **arrays, weights=np.ones(300), attribute_weights=[1, 0])
    assert fit.beta_C == pytest.approx(1)
    assert fit.beta_P == pytest.approx(0, abs=1e-10)


def test_homogenization_excess_uses_paired_oracle_draws():
    from src.aging_bias.bootstrap import calibrate_homogenization
    from src.aging_bias.decomposition import BiasDecomposition
    indices = pd.DataFrame([
        dict(model_name='model', scheme='unweighted', homog=.5),
        dict(model_name='oracle_target', scheme='unweighted', homog=.2),
    ])
    draws = {('model', 'unweighted'): [BiasDecomposition(homog=.4, status='ok'), BiasDecomposition(homog=.7, status='ok')],
             ('oracle_target', 'unweighted'): [BiasDecomposition(homog=.1, status='ok'), BiasDecomposition(homog=.4, status='ok')]}
    report = calibrate_homogenization(indices, draws)
    row = report[report.model_name == 'model'].iloc[0]
    assert row.homog_excess == pytest.approx(.3)
    assert row.homog_excess_ci_low == pytest.approx(.3)
    assert row.homog_excess_ci_high == pytest.approx(.3)
    assert row.homog == .5  # Raw estimate retained for diagnostics.


def test_strict_homogenization_subset_does_not_remove_moderate_icc_from_pi():
    from src.aging_bias.bootstrap import bootstrap_models
    from src.aging_bias.config import WeightingConfig
    rng=np.random.default_rng(9)
    a={k:rng.normal(size=(200,2)) for k in ['d_C','d_P','G_P','zeta']}
    a['g']=a['d_C']-.8*a['zeta']
    a['g'][:,0]=a['d_C'][:,0]
    panel=pd.DataFrame(dict(identity_id=np.repeat(np.arange(100),2),delta_age=20))
    result=bootstrap_models(panel,{'model':a},WeightingConfig(min_bin_n=1),n_boot=2,
        attribute_weights=[.8,.4],homogenization_mask=[True,False])
    row=result['indices'].iloc[0]
    assert row.homog == pytest.approx(0,abs=1e-10)
    assert row.homog_status=='ok'
    assert row.n_rows == 400  # pi still uses both dimensions.
