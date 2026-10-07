from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.train_modular_mappo import load_config
from tools.analyze_next_stage_screening import (
    ANCHOR_COEFFICIENTS, classify_candidate, matched_episode_delta,
    validate_diagnostic_seeds, validate_same_source_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CONFIGS = {
    0.001:"pw_m6_m8_anchor_c0001_300k.yaml",
    0.003:"pw_m6_m8_anchor_c0003_300k.yaml",
    0.01:"pw_m6_m8_anchor_c001_300k.yaml",
    0.03:"pw_m6_m8_anchor_c003_300k.yaml",
    0.10:"pw_m6_m8_anchor_c01_300k.yaml",
}


def _without(config, path):
    value=deepcopy(config); cursor=value
    for key in path[:-1]: cursor=cursor[key]
    cursor.pop(path[-1]); return value


def test_alloff_and_m5_are_strictly_matched_except_wave_balancing():
    alloff=load_config(ROOT/"configs/pw_alloff_matched_1p5m.yaml")
    m5=load_config(ROOT/"configs/pw_m5_wave_balance.yaml")
    assert _without(alloff,("modules","wave_balancing"))==_without(m5,("modules","wave_balancing"))
    assert not alloff["modules"]["wave_balancing"]["enabled"]
    assert m5["modules"]["wave_balancing"]["enabled"]
    training=alloff["training"]
    assert (training["gamma"],training["seed"],training["num_train_envs"])==(.999,2023,24)
    assert (training["rollout_steps"],training["ppo_epochs"],training["minibatch_size"])==(256,10,512)
    assert (training["actor_learning_rate"],training["critic_learning_rate"])==(3e-4,3e-4)
    assert (training["evaluation_episodes"],training["evaluation_interval_sampled_steps"])==(20,100000)
    assert alloff["implementation"]["checkpoint_interval_sampled_steps"]==500000
    assert training["total_sampled_steps"]==1500000


def test_all_300k_configs_are_matched_and_have_required_cadence():
    control=load_config(ROOT/"configs/pw_m6_screen_control_300k.yaml")
    normalized_control=deepcopy(control); normalized_control["modules"].pop("policy_anchor")
    for coefficient,name in ANCHOR_CONFIGS.items():
        config=load_config(ROOT/"configs"/name)
        normalized=deepcopy(config); normalized["modules"].pop("policy_anchor")
        assert normalized==normalized_control
        assert config["modules"]["warm_start"]=={"enabled":True,"mode":"actor_only"}
        assert config["modules"]["policy_anchor"]=={"enabled":True,"coefficient":coefficient,"schedule":"constant"}
        assert config["training"]["total_sampled_steps"]==300000
        assert config["training"]["evaluation_interval_sampled_steps"]==50000
        assert config["implementation"]["checkpoint_interval_sampled_steps"]==100000
    assert tuple(ANCHOR_CONFIGS)==ANCHOR_COEFFICIENTS
    assert control["modules"]["warm_start"]["enabled"] and not control["modules"]["policy_anchor"]["enabled"]


def _source_checkpoint(path: Path, bias: float = 0.0):
    source=MAPPOTrainer(hidden_dim=16,seed=4)
    with torch.no_grad(): source.actor.mean.bias.add_(bias)
    source.sampled_steps=123
    source.save(path,{"training_seed":2023,"environment_variant":"direct_v2_3"})
    return source


def _rollout(trainer: ModularMAPPOTrainer):
    rng=np.random.default_rng(8); T,E,A,F=4,2,4,52
    obs=rng.normal(size=(T,E,A,F)).astype("f")
    actions=np.tanh(rng.normal(size=(T,E,A,3))).astype("f")
    raw=np.arctanh(np.clip(actions,-.999,.999)).astype("f")
    alive=np.ones((T,E,A),"f"); rewards=rng.normal(size=(T,E,A)).astype("f")
    with torch.no_grad():
        dist=trainer.actor.distribution(torch.as_tensor(obs))
        old=trainer.actor._squashed_log_prob(dist,torch.as_tensor(raw),torch.as_tensor(actions)).numpy()
    zeros=np.zeros((T,E),"f"); context=np.zeros((T,E,0),"f")
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),zeros,alive,
        obs.copy(),alive.copy(),np.ones((T,E),int),np.ones((T,E),int),context,context,None,None,np.ones((T,E),"f"))


def test_anchor_same_source_initial_invariants_and_short_ppo_update(tmp_path):
    source_path=tmp_path/"direct.pt"; source=_source_checkpoint(source_path)
    modules={"warm_start":{"enabled":True,"mode":"actor_only"},
             "policy_anchor":{"enabled":True,"coefficient":.01,"schedule":"constant"}}
    trainer=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=2,minibatch_size=8,seed=9,modules_config=modules)
    trainer.warm_start.initialize(trainer,str(source_path))
    reference=deepcopy(source.actor); trainer.anchor.attach(reference,str(source_path))
    before={key:value.clone() for key,value in reference.state_dict().items()}
    assert all(torch.equal(current,source_value) for current,source_value in
               zip(trainer.actor.state_dict().values(),reference.state_dict().values()))
    observation=torch.randn(5,52)
    assert torch.equal(trainer.actor.deterministic(observation),reference.deterministic(observation))
    _,metrics=trainer.anchor.loss(trainer.actor.distribution(observation),reference.distribution(observation),0,torch.ones(5))
    assert metrics["anchor_kl"]==pytest.approx(0,abs=1e-8)
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    optimizer_ids={id(parameter) for group in trainer.actor_optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) not in optimizer_ids for parameter in reference.parameters())
    actor_before={key:value.clone() for key,value in trainer.actor.state_dict().items()}
    update=trainer.update(_rollout(trainer))
    assert any(not torch.equal(value,trainer.actor.state_dict()[key]) for key,value in actor_before.items())
    assert np.isfinite(update["anchor_kl"]) and update["anchor_kl"]>=0
    assert all(torch.equal(value,reference.state_dict()[key]) for key,value in before.items())


def test_source_preflight_rejects_different_checkpoint(tmp_path):
    first,second=tmp_path/"a.pt",tmp_path/"b.pt"
    _source_checkpoint(first,0); _source_checkpoint(second,.1)
    assert validate_same_source_checkpoints(first,first)["sha256"]
    with pytest.raises(RuntimeError,match="checkpoint mismatch"):
        validate_same_source_checkpoints(first,second)


def test_analysis_rejects_holdout_and_pairs_by_seed():
    with pytest.raises(ValueError,match="formal holdout"): validate_diagnostic_seeds([20_000_000])
    baseline=pd.DataFrame({"seed":[2,1],"clear_wave_1":[0,1],"clear_wave_2":[0,0],"clear_wave_3":[0,0],
        "waves_cleared":[0,1],"episode_return":[0,1],"red_losses":[4,3],"episode_kill_loss_ratio":[0,1],"red_boundary_exits":[1,0],"red_ground_losses":[0,1]})
    baseline["blue_losses"]=[0,4]; baseline["timeout"]=[0,1]; baseline["episode_length"]=[10,20]
    candidate=baseline.copy(); candidate["episode_return"]+=2
    assert matched_episode_delta(candidate,baseline)["delta_return"]==2


def test_pareto_labels_use_raw_metrics_without_composite_score():
    gains={"W3":.1,"average_waves":.2,"return":1,"red_loss":0,"K_L":0}
    assert classify_candidate(-.11,gains)=="FORGETTING"
    assert classify_candidate(-.05,gains)=="ADAPTATION_CANDIDATE"
    assert classify_candidate(0,{key:0 for key in gains})=="PRESERVATION_ONLY"
