import copy

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.networks import ModularMAPPOActor
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer


def observations(*prefix):
    value=torch.randn(*prefix,52)
    value[...,13]=1;value[...,20]=0;value[...,27]=1
    value[...,33]=1;value[...,39]=0;value[...,45]=1;value[...,51]=0
    return value


def actor(mode=None):
    config={"enabled":True,"entity_dim":32,"attention_heads":2}
    if mode is not None:config["mode"]=mode
    if mode=="gated_residual":config["initial_gate"]=.05
    return ModularMAPPOActor(entity_attention_config=config)


def test_implicit_and_explicit_replacement_are_identical_and_v1_topology_is_unchanged():
    torch.manual_seed(11);implicit=actor()
    torch.manual_seed(11);explicit=actor("replacement")
    assert implicit.entity_attention_mode==explicit.entity_attention_mode=="replacement"
    assert list(implicit.state_dict())==list(explicit.state_dict())
    assert all(torch.equal(implicit.state_dict()[key],explicit.state_dict()[key]) for key in implicit.state_dict())
    assert not hasattr(implicit,"backbone") and not hasattr(implicit,"entity_residual_adapter") and not hasattr(implicit,"entity_gate")


def test_disabled_actor_preserves_exact_legacy_topology_and_values():
    torch.manual_seed(12);plain=ModularMAPPOActor()
    torch.manual_seed(12);disabled=ModularMAPPOActor(entity_attention_config={"enabled":False})
    assert list(plain.state_dict())==list(disabled.state_dict())
    assert all(torch.equal(plain.state_dict()[key],disabled.state_dict()[key]) for key in plain.state_dict())


@pytest.mark.parametrize("mode",["residual","gated_residual"])
def test_zero_adapter_initially_matches_mappo_distribution_exactly(mode):
    obs=observations(3,4)
    torch.manual_seed(13);baseline=ModularMAPPOActor()
    torch.manual_seed(13);candidate=actor(mode)
    base_dist=baseline.distribution(obs);candidate_dist=candidate.distribution(obs)
    assert torch.equal(base_dist.mean,candidate_dist.mean)
    assert torch.equal(base_dist.stddev,candidate_dist.stddev)
    assert torch.count_nonzero(candidate.entity_residual_adapter.weight)==0
    assert torch.count_nonzero(candidate.entity_residual_adapter.bias)==0


def test_gate_initialization_and_diagnostics_are_finite():
    model=actor("gated_residual");obs=observations(2,3,4)
    dist,diag=model.distribution(obs,return_attention=True);gate=diag["entity_gate"]
    assert dist.mean.shape==(2,3,4,3) and torch.isfinite(dist.mean).all() and torch.isfinite(dist.stddev).all()
    assert torch.all((gate>0)&(gate<1)) and torch.allclose(gate,torch.full_like(gate,.05),atol=1e-7)
    for key in ("entity_base_feature_norm","entity_feature_norm","entity_delta_norm","entity_delta_to_base_ratio"):
        assert torch.isfinite(diag[key]).all()


@pytest.mark.parametrize("mode",["residual","gated_residual"])
def test_dead_entity_masks_and_all_dead_groups_remain_finite(mode):
    model=actor(mode);obs=observations(2,4)
    obs[...,13]=obs[...,20]=obs[...,27]=0
    obs[...,33]=obs[...,39]=obs[...,45]=obs[...,51]=0
    dist,diag=model.distribution(obs,return_attention=True)
    assert torch.isfinite(dist.mean).all() and torch.isfinite(dist.stddev).all()
    assert torch.count_nonzero(diag["ally_attention_weights"])==0
    assert torch.count_nonzero(diag["enemy_attention_weights"])==0


@pytest.mark.parametrize("mode",["residual","gated_residual"])
def test_zero_adapter_opens_entity_gradient_path_after_one_step(mode):
    model=actor(mode);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);obs=observations(3,4)
    optimizer.zero_grad();model.distribution(obs).mean.square().mean().backward()
    assert model.entity_residual_adapter.weight.grad is not None
    assert torch.isfinite(model.entity_residual_adapter.weight.grad).all()
    assert torch.count_nonzero(model.entity_residual_adapter.weight.grad)>0
    optimizer.step();optimizer.zero_grad();model.distribution(obs).mean.square().mean().backward()
    entity_grads=[parameter.grad for name,parameter in model.named_parameters() if name.startswith(("self_encoder","ally_encoder","enemy_encoder","ally_attention","enemy_attention","entity_fusion"))]
    assert any(grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)>0 for grad in entity_grads)


@pytest.mark.parametrize("mode",["residual","gated_residual"])
def test_new_checkpoint_round_trip_preserves_outputs(tmp_path,mode):
    modules={"entity_attention":{"enabled":True,"mode":mode,"entity_dim":32,"attention_heads":2}}
    if mode=="gated_residual":modules["entity_attention"]["initial_gate"]=.05
    first=ModularMAPPOTrainer(modules_config=copy.deepcopy(modules),total_sampled_steps=10)
    obs=observations(2,4);expected=first.actor.distribution(obs)
    path=tmp_path/f"{mode}.pt";first.save(path)
    restored=ModularMAPPOTrainer(modules_config=copy.deepcopy(modules),total_sampled_steps=10);restored.load(path)
    actual=restored.actor.distribution(obs)
    assert torch.equal(expected.mean,actual.mean) and torch.equal(expected.stddev,actual.stddev)


def test_invalid_mode_and_gate_are_rejected():
    with pytest.raises(ValueError,match="mode"):actor("unknown")
    for value in (0.,1.,-0.1,1.1):
        with pytest.raises(ValueError,match="initial_gate"):
            ModularMAPPOActor(entity_attention_config={"enabled":True,"mode":"gated_residual","initial_gate":value})


def test_gated_trainer_numeric_diagnostics_do_not_include_mode_string():
    model=ModularMAPPOTrainer(modules_config={"entity_attention":{"enabled":True,"mode":"gated_residual","initial_gate":.05}},total_sampled_steps=10)
    obs=observations(2,1,4);actions=torch.zeros(2,1,4,3);alive=torch.ones(2,1,4);ctx=torch.zeros(2,1,0)
    result=model._policy_diagnostics(type("R",(),{"actor_hidden_before_step":None,"episode_masks":None})(),obs,actions,alive,ctx)
    required=("entity_base_feature_norm","entity_feature_norm","entity_delta_norm","entity_delta_to_base_ratio",
              "entity_gate_mean","entity_gate_std","entity_gate_min","entity_gate_max","entity_gate_p10","entity_gate_p50","entity_gate_p90")
    assert all(key in result and np.isfinite(result[key]) for key in required)
    assert all(isinstance(value,(int,float)) for value in result.values())

