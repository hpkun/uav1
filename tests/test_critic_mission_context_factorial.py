from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules.wave_context import WaveContextModule
from algorithm.modules.wave_survival_pbrs import (
    WaveSurvivalPotentialShapingModule, mission_progress_from_blue_losses,
    mission_progress_from_wave_state,
)
from tools.preflight_critic_mission_context_factorial import (
    CONFIGS, validate_critic_context_factorial_configs,
)
import tools.preflight_critic_mission_context_factorial as factorial_preflight

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return {cell: load_config(path) for cell, path in CONFIGS.items()}


def exact_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def test_old_rich_encoding_exact_unchanged():
    module = WaveContextModule({"enabled":True,"encoding":"rich","context_target":"critic_only","max_waves":3})
    actual = module.encode_numpy(np.array([1,2,3]), np.array([3,3,3]))
    expected = np.array([[1,0,0,0,1],[0,1,0,.5,.5],[0,0,1,1,0]],np.float32)
    assert np.array_equal(actual, expected)


def test_old_scalar_round_encoding_exact_unchanged():
    module = WaveContextModule({"enabled":True,"encoding":"scalar_round","context_target":"actor_critic"})
    assert module.context_dim == 1
    assert np.array_equal(module.encode_numpy([1,2,3],[3,3,3]), np.array([[1],[2],[3]],np.float32))


def test_mission_markov_dim_one_hot_progress_and_horizon():
    module = WaveContextModule({"enabled":True,"encoding":"mission_markov","context_target":"critic_only","max_waves":3})
    context = module.encode_numpy([1,2,3],[3,3,3],mission_progress=[0,1/3,2/3],episode_step=[300,1200,2100],max_steps=3000)
    assert module.context_dim == 5 and module.actor_enabled is False and module.critic_enabled is True
    assert np.array_equal(context[:,:3],np.eye(3,dtype=np.float32))
    assert np.allclose(context[:,3],[0,1/3,2/3])
    assert np.allclose(context[:,4],[.9,.6,.3])


@pytest.mark.parametrize("step,expected",[(0,1),(1,2999/3000),(1000,2/3),(2000,1/3),(2999,1/3000),(3000,0)])
def test_remaining_horizon_exact(step, expected):
    module = WaveContextModule({"enabled":True,"encoding":"mission_markov"})
    context = module.encode_numpy([1],[3],mission_progress=[0],episode_step=[step],max_steps=3000)
    assert context[0,4] == pytest.approx(expected,abs=1e-7)


def test_canonical_progress_all_wave_cases_and_pbrs_match():
    cases = [(1,(1,1,1,1),0),(1,(1,0,0,0),3),(2,(1,1,1,1),4),
             (2,(1,1,0,0),6),(3,(1,1,1,1),8),(3,(1,0,0,0),11)]
    module = WaveSurvivalPotentialShapingModule({"enabled":True,"coefficient":1.,"potential":"progress_times_survival","terminal_zero":True},.999)
    for wave, blue, losses in cases:
        state_progress = mission_progress_from_wave_state(np.array([wave]),np.array([blue],np.float32),np.array([3]))
        loss_progress = mission_progress_from_blue_losses(np.array([losses]),np.array([3]),4)
        assert np.array_equal(state_progress,loss_progress)
        module.adapt(np.zeros((1,4),np.float32),[{"total_waves":3,"blue_losses":losses}],np.array([wave]),
                     np.ones((1,4),np.float32),np.array([blue],np.float32),np.ones((1,4),np.float32),np.array([False]))
        assert np.allclose(module.last_transition["phi_pre"],state_progress[0])


def test_wave_refresh_progress_continuous_and_horizon_not_reset():
    module = WaveContextModule({"enabled":True,"encoding":"mission_markov"})
    pre_p = mission_progress_from_wave_state([1],np.array([[1,0,0,0]],np.float32),[3])
    next_p = mission_progress_from_blue_losses([4],[3],4)
    pre = module.encode_numpy([1],[3],mission_progress=pre_p,episode_step=[700],max_steps=3000)
    nxt = module.encode_numpy([2],[3],mission_progress=next_p,episode_step=[701],max_steps=3000)
    assert pre[0,3] == pytest.approx(3/12) and nxt[0,3] == pytest.approx(4/12)
    assert nxt[0,4] == pytest.approx((3000-701)/3000) and nxt[0,4] < pre[0,4]


def test_factorial_config_validator_and_forbidden_modules():
    resolved = configs()
    rows = validate_critic_context_factorial_configs(resolved)
    assert set(rows) == {"C0R0","C0R1","C1R0","C1R1"}
    bad = deepcopy(resolved);bad["C1R1"]["modules"]["wave_balancing"]["enabled"] = True
    with pytest.raises(RuntimeError):validate_critic_context_factorial_configs(bad)


def test_actor_exact_initialization_and_critic_pairing_all_cells():
    resolved=configs();trainers={}
    for cell,cfg in resolved.items():
        value=deepcopy(cfg);value["training"]["seed"]=5201
        trainers[cell]=build_modular_mappo_trainer(value,"cpu")
    assert all(exact_state(trainers["C0R0"].actor.state_dict(),trainers[cell].actor.state_dict()) for cell in trainers)
    assert exact_state(trainers["C0R0"].critic.state_dict(),trainers["C0R1"].critic.state_dict())
    assert exact_state(trainers["C1R0"].critic.state_dict(),trainers["C1R1"].critic.state_dict())
    assert trainers["C0R0"].critic.context_dim==0 and trainers["C1R0"].critic.context_dim==5
    assert trainers["C1R0"].actor.context_dim==0
    assert trainers["C0R0"].critic.embedding[0].in_features==52
    assert trainers["C1R0"].critic.embedding[0].in_features==52
    assert trainers["C1R0"].critic.context_injection=="additive_zero"
    common=set(trainers["C0R0"].critic.state_dict())
    assert all(torch.equal(trainers["C0R0"].critic.state_dict()[key],trainers["C1R0"].critic.state_dict()[key]) for key in common)
    assert torch.count_nonzero(trainers["C1R0"].critic.mission_context_projection)==0
    assert sum(p.numel() for p in trainers["C1R0"].critic.parameters())-sum(p.numel() for p in trainers["C0R0"].critic.parameters())==5*256


def test_actor_distribution_actions_and_post_init_rng_sample_exact_without_reseed():
    resolved=configs();obs=np.random.default_rng(2).normal(size=(1,4,52)).astype("f");alive=np.ones((1,4),"f")
    outputs=[]
    for cell,cfg in resolved.items():
        value=deepcopy(cfg);value["training"]["seed"]=5202
        trainer=build_modular_mappo_trainer(value,"cpu")
        dist=trainer.actor.distribution(torch.as_tensor(obs));det=trainer.act(obs,alive,True)[0]
        sample=trainer.act(obs,alive,False)[0]
        outputs.append((dist.mean.detach(),dist.stddev.detach(),det,sample))
    assert all(torch.equal(outputs[0][0],x[0]) and torch.equal(outputs[0][1],x[1]) and np.array_equal(outputs[0][2],x[2]) and np.array_equal(outputs[0][3],x[3]) for x in outputs[1:])


def test_same_observation_different_context_reaches_critic_graph_only():
    cfg=configs()["C1R0"];trainer=build_modular_mappo_trainer(cfg,"cpu",hidden_dim=16)
    obs=torch.zeros((1,4,52));alive=torch.ones((1,4));context=torch.tensor([[1.,0,0,0,.9]],requires_grad=True)
    value,_=trainer.critic.forward_step(obs,alive,context)
    baseline=trainer.values_step(np.zeros((1,4,52),np.float32),np.ones((1,4),np.float32),np.array([[1,0,0,0,.9]],np.float32))[0]
    initial_other=trainer.values_step(np.zeros((1,4,52),np.float32),np.ones((1,4),np.float32),np.array([[0,1,0,.5,.5]],np.float32))[0]
    assert np.array_equal(baseline,initial_other)
    value.sum().backward()
    projection=trainer.critic.mission_context_projection
    assert projection.grad is not None and torch.count_nonzero(projection.grad)>0
    trainer.critic_optimizer.step()
    changed=trainer.values_step(np.zeros((1,4,52),np.float32),np.ones((1,4),np.float32),np.array([[0,1,0,.5,.5]],np.float32))[0]
    assert not np.array_equal(baseline,changed)
    contexts=np.array([[1,0,0,0,.9],[0,1,0,1/3,.6],[0,0,1,2/3,.3]],np.float32)
    actor_outputs=[trainer.act(np.zeros((1,4,52),np.float32),np.ones((1,4),np.float32),True,context=row[None])[0] for row in contexts]
    assert all(np.array_equal(actor_outputs[0],row) for row in actor_outputs[1:])


def test_runner_current_next_context_and_terminal_reset_alignment(tmp_path):
    cfg=configs()["C1R0"];cfg["training"]["seed"]=9911
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));env["simulation"]["max_steps"]=2
    runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=3,device="cpu",seed=9911,output_dir=tmp_path/"runner",smoke=True)
    try:
        rollout=runner.collect_rollout(3)
        assert rollout.contexts.shape==(3,1,5) and rollout.next_contexts.shape==(3,1,5)
        assert rollout.contexts[0,0,4]==1 and rollout.next_contexts[0,0,4]==pytest.approx(.5)
        assert rollout.dones[1,0]==1 and rollout.next_contexts[1,0,4]==0
        assert rollout.contexts[2,0,4]==1 and rollout.contexts[2,0,3]==0
    finally:runner.vector.close()


def test_pbrs_disabled_identity_and_evaluation_source_is_raw():
    module=WaveSurvivalPotentialShapingModule({"enabled":False},.999);raw=np.arange(4,dtype=np.float32)[None]
    out,_=module.adapt(raw,[{"total_waves":3,"blue_losses":1}],np.array([1]),np.ones((1,4),"f"),np.ones((1,4),"f"),np.ones((1,4),"f"),np.array([False]))
    assert np.array_equal(raw,out)
    source=inspect.getsource(evaluate_modular_episode)
    assert "ret+=reward" in source and "wave_survival_pbrs" not in source


@pytest.mark.parametrize("relative",[
    "outputs/pw_alloff_matched_1p5m_seed2024/latest.pt",
    "outputs/pw_m5_wave_balance_1p5m_seed2024/latest.pt",
    "outputs/formal_eawb/ea_seed3101/latest.pt",
    "outputs/dev_fbmr_stage2/fbmr_seed3101/latest.pt",
    "outputs/dev_ws_pbrs/baseline_seed5101/latest.pt",
])
def test_representative_old_modular_checkpoint_loads(relative):
    path=ROOT/relative
    if not path.exists():pytest.skip(f"missing local checkpoint {relative}")
    state=torch.load(path,map_location="cpu",weights_only=False);cfg=state["extra"]["algorithm_config"]
    trainer=build_modular_mappo_trainer(cfg,"cpu")
    trainer.load(path,restore_rng=False)


def test_reserved_and_historical_seeds_not_reused():
    for cfg in configs().values():
        protocol=cfg["development_protocol"]
        assert cfg["training"]["seed"] not in (5101,5102,5103)
        assert cfg["implementation"]["evaluation_seed_base"]==42000000
        assert protocol["historical_exposed"]["evaluation_seed_start"]==39000000
        assert protocol["reserved_future_final_test"]=={"seed_start":33000000,"seed_end":33000199,"executed":False}


def test_launch_check_accepts_checkout_without_archived_history(monkeypatch, tmp_path):
    """Deployment preflight trusts frozen provenance but still rejects fresh-seed hits."""
    monkeypatch.setattr(factorial_preflight,"AUDIT",tmp_path/"audit")
    monkeypatch.setattr(factorial_preflight.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(factorial_preflight,"text_freshness_scan",lambda:{
        "training_5201_5203":[],"candidate_40m":[],"exposed_41m":[],"selected_42m":[]})
    monkeypatch.setattr(factorial_preflight,"checkpoint_freshness_scan",lambda:{"checkpoint_count":0,"hits":[]})
    result=factorial_preflight.validate(check_outputs=False,launch_check=True)
    assert result["status"]=="CRITIC_MISSION_CONTEXT_FACTORIAL_HARDENED_READY"
    freshness=yaml.safe_load((tmp_path/"audit"/"seed_freshness.json").read_text(encoding="utf-8"))
    assert freshness["historical_evidence_policy"]=="FROZEN_MANIFEST_PROVENANCE_FOR_DEPLOYMENT_CHECKOUT"
