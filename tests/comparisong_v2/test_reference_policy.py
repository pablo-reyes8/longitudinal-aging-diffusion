import pandas as pd


def test_new_reference_policy_is_applied_without_changing_evaluation_call(tmp_path):
    from test_evaluator import setup
    from src.aging_bias.references import ReferenceBundle
    from src.comparisong.evaluate_paper_v2 import evaluate_paper_v2
    cfg,bundle,_=setup(tmp_path);cfg.n_boot=0
    reference=ReferenceBundle.load(cfg.reference_bundle_path)
    reference.metadata['reliability_policy']=dict(weight_by_icc=True,attribute_weights=[.9,.4],
        active_homog=[True,False],homogenization='excess_over_oracle',retest_min_age=30)
    result=evaluate_paper_v2(cfg,bundle=bundle,reference_bundle=reference)
    assert result['metadata']['reliability_policy']['attribute_weights']==[.9,.4]
    assert 'homog_excess' in result['bias_indices']
    assert result['per_sample'].reliability_age_supported.eq(
        (result['per_sample'].source_age>=30)&(result['per_sample'].target_age>=30)).all()
    table=pd.read_csv(tmp_path/'out/tables/table1_bias.csv')
    assert 'homog_excess' in table
    tex=(tmp_path/'out/tables/table1_bias.tex').read_text()
    assert 'homog\\_excess' in tex
