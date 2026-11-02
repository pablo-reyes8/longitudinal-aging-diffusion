import ast
from dataclasses import replace
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('preflight_passes',[True,False])
def test_sd_notebook_preflight_gates_generation_and_passes_exact_returned_roots(tmp_path,monkeypatch,preflight_passes):
    from src.aging_bias.config import SD15Config,PaperV2Config
    import src.comparisong.sd15_counterfactual as sd
    notebook=json.loads(Path('notebooks/Paper_comparison_bias_v2.ipynb').read_text())
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and not ''.join(c['source']).lstrip().startswith('%')]
    source=next(c for c in cells if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_sd15_before_after' for n in ast.walk(ast.parse(c))))
    events=[];roots={'SD15_vanilla@0.1':str(tmp_path/'before'),'ours@0.1':str(tmp_path/'after')}
    validation=tmp_path/'validation.json'
    def preflight(config,**kwargs):
        if not preflight_passes:raise ValueError('synthetic invalid real zones')
        validation.write_text('{"passed":true}')
        events.append('validated');return validation
    def generate(config):
        assert validation.exists() and config.validation_path==validation
        assert config.anchors_reviewed and config.resume
        events.append('generated');return {'generated_roots':roots}
    def evaluate(config,**kwargs):
        assert config.models==list(roots) and config.generated_roots==roots
        assert config.test_manifest_path==tmp_path/'out/sd15_counterfactual/transitions.csv'
        events.append('evaluated');return {'sentinel':'result'}
    monkeypatch.setattr(sd,'run_sd15_before_after',generate)
    namespace=dict(preflight_zone_protocol=preflight,run_paper_analysis=evaluate,SD15Config=SD15Config,
        replace=replace,config=PaperV2Config('manifest',{},['SAM'],'ref',tmp_path/'out'),bundle={},ASSETS=tmp_path/'assets',
        CHECKPOINT_PATH=tmp_path/'adapter.pt',SAVED=tmp_path,TEST_ROOT=tmp_path/'test',OUTPUT=tmp_path/'out',
        display_zone_results=lambda result:events.append('displayed'))
    if preflight_passes:
        exec(compile(source,'sd_notebook_cell','exec'),namespace)
        assert events==['validated','generated','evaluated','displayed']
    else:
        with pytest.raises(ValueError,match='invalid real zones'):exec(compile(source,'sd_notebook_cell','exec'),namespace)
        assert events==[]
