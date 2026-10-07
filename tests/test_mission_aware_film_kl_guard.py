"""Focused protocol tests for the actor-only KL epoch guard."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules import ActorKLEpochGuardModule,PPOStabilizationModule
from algorithm.train_modular_mappo import load_config
from tools.preflight_mission_aware_film_kl_guard import (GUARD,MANIFEST,ORIGINAL,REGISTRY,ROOT,
    _equal,_no_trigger_exact_match,_rollout,_simulated_update,validate)

def configs():return load_config(ORIGINAL),load_config(GUARD)

def test_01_guard_validation_and_version():
    module=ActorKLEpochGuardModule({"enabled":True,"hard_kl":.05,"actor_early_stop":True})
    assert module.version==1 and module.enabled and not isinstance(module,torch.nn.Module)
    assert not any(isinstance(v,(torch.Tensor,torch.nn.Parameter)) for v in vars(module).values())
    with pytest.raises(ValueError):ActorKLEpochGuardModule({"enabled":True,"hard_kl":0,"actor_early_stop":True})
    with pytest.raises(ValueError):ActorKLEpochGuardModule({"enabled":True,"hard_kl":.05,"actor_early_stop":False})

def test_02_guard_threshold_is_inclusive():
    module=ActorKLEpochGuardModule({"enabled":True,"hard_kl":.05,"actor_early_stop":True})
    assert not module.should_stop_actor(.049999) and module.should_stop_actor(.05)

def test_03_old_ppo_stabilization_threshold_remains_strict():
    module=PPOStabilizationModule({"enabled":True,"hard_kl":.05})
    assert not module.should_stop_actor(.05) and module.should_stop_actor(.050001)

def test_04_resolved_config_only_adds_guard():
    original,guard=configs();modules=deepcopy(guard["modules"]);assert modules.pop("actor_kl_guard")=={"enabled":True,"hard_kl":.05,"actor_early_stop":True}
    assert modules==original["modules"] and guard["development_method"]=="mission_aware_film_kl_guard"

def test_05_enabled_module_set_is_exact():
    _,guard=configs();enabled=sorted(k for k,v in guard["modules"].items() if v.get("enabled",False))
    assert enabled==["actor_kl_guard","actor_lr_decay","mission_film","wave_context"]

def test_06_matched_initialization_and_optimizers():
    original,guard=configs()
    for seed in (5301,5302,5303):
        original["training"]["seed"]=guard["training"]["seed"]=seed
        a=build_modular_mappo_trainer(original,"cpu",64,3_000_000);b=build_modular_mappo_trainer(guard,"cpu",64,3_000_000)
        assert _equal(a.actor.state_dict(),b.actor.state_dict()) and _equal(a.critic.state_dict(),b.critic.state_dict())
        assert _equal(a.actor_optimizer.state_dict(),b.actor_optimizer.state_dict()) and _equal(a.critic_optimizer.state_dict(),b.critic_optimizer.state_dict())

def test_07_actor_lr_schedule_is_unchanged():
    original,guard=configs()
    for step,expected in ((0,.0003),(600000,.0003),(750000,.0002),(900000,.0001),(3000000,.0001)):
        a=build_modular_mappo_trainer(original,"cpu",32,3_000_000).actor_lr_decay.learning_rate(step,.0003)
        b=build_modular_mappo_trainer(guard,"cpu",32,3_000_000).actor_lr_decay.learning_rate(step,.0003)
        assert a==pytest.approx(expected) and b==a

def test_08_low_kl_runs_full_actor_and_critic_epochs():
    _,guard=configs();m=_simulated_update(guard,.01)
    assert m["actor_epochs_used"]==10 and m["critic_epochs_used"]==10 and m["kl_hard_stop_triggered"]==0
    assert m["actor_optimizer_steps_this_update"]==10 and m["critic_optimizer_steps_this_update"]==10

def test_09_high_kl_stops_actor_only_after_complete_epoch():
    _,guard=configs();m=_simulated_update(guard,.06)
    assert m["actor_epochs_used"]==1 and m["critic_epochs_used"]==10 and m["kl_hard_stop_triggered"]==1
    assert m["actor_kl_guard_hard_stop_count"]==1 and m["actor_kl_guard_hard_stop_fraction"]==1
    assert m["actor_optimizer_steps_this_update"]==1 and m["critic_optimizer_steps_this_update"]==10


def test_09a_third_epoch_stop_keeps_interleaved_critic_schedule():
    _,guard=configs();m=_simulated_update(guard,[.01,.01,.06])
    assert m["actor_epochs_used"]==3 and m["critic_epochs_used"]==10 and m["kl_hard_stop_triggered"]==1
    assert m["actor_optimizer_steps_this_update"]==3 and m["critic_optimizer_steps_this_update"]==10


def test_09b_no_trigger_update_is_exact_matched_intervention():
    original,guard=configs();result=_no_trigger_exact_match(original,guard)
    assert all(result.values())


def test_09c_full_rollout_kl_does_not_consume_torch_rng():
    _,guard=configs();trainer=build_modular_mappo_trainer(guard,"cpu",32,1000);batch=_rollout(trainer)
    tensors=[torch.as_tensor(value,dtype=torch.float32) for value in
             (batch.observations,batch.raw_actions,batch.old_log_probs,batch.alive_masks,batch.contexts)]
    obs,raw,oldlog,alive,ctx=[value.reshape(value.shape[0]*value.shape[1],*value.shape[2:]) for value in tensors]
    before=torch.get_rng_state();value=trainer._full_rollout_kl(obs,raw,oldlog,alive,ctx);after=torch.get_rng_state()
    assert np.isfinite(value) and torch.equal(before,after)

def test_10_guard_conflicts_are_enforced():
    _,guard=configs()
    for module in ("recurrent_memory","ppo_stabilization"):
        bad=deepcopy(guard);bad["modules"][module]["enabled"]=True
        with pytest.raises(ValueError):build_modular_mappo_trainer(bad,"cpu",32,10)

def test_11_guard_checkpoint_roundtrip_and_counters(tmp_path):
    _,guard=configs();trainer=build_modular_mappo_trainer(guard,"cpu",32,1000);trainer._full_rollout_kl=lambda *_:.06
    trainer.update(__import__("tools.preflight_mission_aware_film_kl_guard",fromlist=["_rollout"])._rollout(trainer))
    path=tmp_path/"guard.pt";trainer.save(path);restored=build_modular_mappo_trainer(guard,"cpu",32,1000);restored.load(path)
    state=torch.load(path,map_location="cpu",weights_only=False)
    assert state["development_feature_versions"]["actor_kl_guard"]==1 and restored.actor_kl_guard_hard_stop_count==1
    assert restored.actor_kl_guard_actor_epochs_total==1 and restored.actor_kl_guard_actor_epochs_min==1

def test_12_old_film_checkpoint_roundtrip_is_compatible(tmp_path):
    original,_=configs();trainer=build_modular_mappo_trainer(original,"cpu",32,1000);path=tmp_path/"film.pt";trainer.save(path)
    build_modular_mappo_trainer(original,"cpu",32,1000).load(path)

def test_13_manifest_seed_guard_and_serial_launcher():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));registry=json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert manifest["training"]["seeds"]==[5301,5302,5303] and manifest["future_final"]["executed"] is False
    assert registry["evaluation_ranges"]["45000000..45000199"]["executed"] is False
    launcher=(ROOT/"tools/run_mission_aware_film_kl_guard_3m.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in launcher and "for seed in 5301 5302 5303" in launcher and "nohup" not in launcher

def test_14_preflight_and_analyzer_are_focused_and_offline():
    result=validate(check_outputs=False);assert result["status"]=="READY_FOR_MISSION_AWARE_FILM_KL_GUARD_3M"
    assert result["NO_TRIGGER_UPDATE_EXACT_MATCH"]=="PASS"
    source=(ROOT/"tools/analyze_mission_aware_film_kl_guard.py").read_text(encoding="utf-8")
    assert "evaluate_modular" not in source and "torch.cuda" not in source and len(source.splitlines())<180
