import torch
from torch import nn
import pytest


def toy():
    from src.model.DoRa import DoRALinear
    from src.model.age_conditioning import AgeConditionerV2
    unet=nn.Module();unet.conv_in=nn.Conv2d(8,3,1);unet.proj=DoRALinear(nn.Linear(3,3),rank=2)
    with torch.no_grad():unet.proj.lora_up.weight.fill_(.2)
    return dict(unet=unet,age_delta_conditioner=AgeConditionerV2(hidden_dim=8,output_dim=3),use_age_delta_conditioning=True)


@pytest.mark.parametrize('force_error',[False,True])
def test_S1_S4_exact_vanilla_and_restore(force_error):
    from src.comparisong.sd15_counterfactual import adapters_disabled
    b=toy();u=b['unet'];c=b['age_delta_conditioner']
    snapshot={k:v.clone() for k,v in u.state_dict().items()};cs={k:v.clone() for k,v in c.state_dict().items()}
    x=torch.randn(2,3);a=torch.randn(1,8,4,4);different=a.clone();different[:,4:]+=10
    try:
        with adapters_disabled(b):
            torch.testing.assert_close(u.proj(x),u.proj.base_layer(x),rtol=0,atol=0)
            torch.testing.assert_close(u.conv_in(a),u.conv_in(different),rtol=0,atol=0)
            delta=c(torch.tensor([20.]),torch.tensor([40.]),torch.tensor([20.]))
            assert torch.count_nonzero(delta)==0
            if force_error:raise RuntimeError('forced')
    except RuntimeError:
        if not force_error:raise
    for k,v in u.state_dict().items():assert torch.equal(v,snapshot[k])
    for k,v in c.state_dict().items():assert torch.equal(v,cs[k])
    assert b['use_age_delta_conditioning'] is True


def test_vanilla_restores_pretrained_first_four_not_trained_conv():
    from src.comparisong.sd15_counterfactual import adapters_disabled
    b=toy();u=b['unet'];w=u.conv_in.weight.clone();bias=u.conv_in.bias.clone()
    b['_sd15_original_conv_in']={'weight':torch.zeros_like(w),'bias':torch.zeros_like(bias)}
    with adapters_disabled(b):
        assert torch.count_nonzero(u.conv_in.weight)==0 and torch.count_nonzero(u.conv_in.bias)==0
    assert torch.equal(u.conv_in.weight,w) and torch.equal(u.conv_in.bias,bias)
