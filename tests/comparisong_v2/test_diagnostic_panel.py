import numpy as np


def test_common_panel_includes_real_oracle_and_matches_attribute_cells(tmp_path):
    from test_evaluator import setup
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    original=bundle['feature_backend'].extract
    def extract(image):
        rgb=np.asarray(image)[0,0]
        if tuple(rgb)==(80,120,100):return {'phi__a':np.nan,'phi__b':np.nan}
        if tuple(rgb)==(61,111,100):return {'phi__a':np.nan,'phi__b':.4}
        return original(image)
    bundle['feature_backend'].extract=extract
    result=evaluate_paper_v2(cfg,bundle=bundle)
    assert result['metadata']['n_common_pairs']==29
    per=result['per_sample']
    assert not per[per.sample_id.eq('pair_1')].common_valid__a.any()
    assert per[per.sample_id.eq('pair_1')].common_valid__b.all()
    assert 'g_norm_mean' in result['bias_indices']
