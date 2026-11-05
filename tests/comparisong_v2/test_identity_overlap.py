from dataclasses import replace
from types import SimpleNamespace
import pandas as pd
import pytest
from src.aging_bias.config import PaperV2Config
from src.comparisong.identity_overlap import check_identity_overlap


def test_name_overlap_is_optional_and_default_bypass_survives_replace():
    config=PaperV2Config(test_manifest_path='x',generated_roots={},models=[],reference_bundle_path='r',output_dir='o')
    panel=pd.DataFrame({'identity_id':['id_1']})
    reference=SimpleNamespace(metadata={'train_identities':['id_1']})
    with pytest.warns(UserWarning,match='bypassed'):
        report=check_identity_overlap(replace(config,output_dir='dino'),panel,reference)
    assert report==dict(enabled=False,overlapping_names=['id_1'],status='bypassed')
    with pytest.raises(ValueError,match='overlap'):
        check_identity_overlap(replace(config,check_identity_overlap=True),panel,reference)


def test_disjoint_names_pass():
    result=check_identity_overlap(SimpleNamespace(check_identity_overlap=True),pd.DataFrame({'identity_id':['id_2']}),
        SimpleNamespace(metadata={'train_identities':['id_1']}))
    assert result['status']=='checked' and not result['overlapping_names']
