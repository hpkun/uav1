from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.protocol import checkpoint_architecture, validate_modular_branch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modules import ActorLRDecayModule
from algorithm.train_mappo import ensure_fresh_output_directory
from algorithm.train_modular_mappo import (
    file_sha256, load_config, preserve_resume_start_checkpoint,
    resolve_branch_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rollout(trainer: ModularMAPPOTrainer) -> ModularRolloutBatch:
    rng=np.random.default_rng(71);t,e,a,f=4,2,4,52
    obs=rng.normal(size=(t,e,a,f)).astype("f");alive=np.ones((t,e,a),"f")
    raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad():
        old=trainer.actor._squashed_log_prob(
            trainer.actor.distribution(torch.as_tensor(obs)),torch.as_tensor(raw),torch.as_tensor(actions)
        ).numpy()
    rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,obs.copy(),alive.copy(),np.ones((t,e),int),np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"))


def formal_source_state() -> tuple[dict,dict,dict]:
    config=load_config("configs/ea_wb_1p5m.yaml")
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    trainer=ModularMAPPOTrainer(hidden_dim=256,modules_config=config["modules"],seed=2023)
    trainer.sampled_steps=502752;trainer.vector_steps=20948
    extra={"environment_version":env["environment_version"],"environment_variant":env["environment_variant"],"environment_config_sha256":config_sha256(env),"algorithm_config_sha256":config_sha256(config),"environment_config":env,"algorithm_config":config,"network_architecture":checkpoint_architecture(trainer),"observation_dim":52,"action_dim":3,"num_agents":4,"training_seed":2023,"training_gamma":.999,"training_num_envs":24,"training_total_sampled_steps":600000,"training_smoke":False,"curriculum_stage":3,"current_total_waves":3,"episode_indices":[0]*24}
    return trainer.checkpoint_state(extra),env,config


def test_actor_lr_decay_endpoints_and_disabled_path():
    disabled=ActorLRDecayModule({"enabled":False})
    for step in (0,502752,600000,750000,900000,999999):assert disabled.learning_rate(step,3e-4)==pytest.approx(3e-4)
    enabled=ActorLRDecayModule({"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4})
    expected={0:3e-4,502752:3e-4,600000:3e-4,750000:2e-4,900000:1e-4,999999:1e-4}
    for step,lr in expected.items():assert enabled.learning_rate(step,9e-4)==pytest.approx(lr)


def test_decay_is_actor_only_global_step_and_not_stabilized_path():
    modules={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    trainer=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,actor_learning_rate=3e-4,critic_learning_rate=3e-4,modules_config=modules)
    trainer.sampled_steps=750000;trainer._update_flat_stabilized=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("stabilized path used"))
    metrics=trainer.update(synthetic_rollout(trainer))
    assert metrics["actor_learning_rate"]==pytest.approx(2e-4)
    assert metrics["critic_learning_rate"]==pytest.approx(3e-4)
    trainer.sampled_steps=900001;trainer.update(synthetic_rollout(trainer))
    assert trainer.actor_optimizer.param_groups[0]["lr"]==pytest.approx(1e-4)
    assert trainer.critic_optimizer.param_groups[0]["lr"]==pytest.approx(3e-4)


def test_disabled_decay_is_exact_update_semantics():
    common={"wave_balancing":{"enabled":True,"loss_target":"actor_critic","frequency_basis":"alive_agent","max_weight":3.0,"epsilon":1e-6}}
    first=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=19,modules_config=deepcopy(common))
    second=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=19,modules_config={**deepcopy(common),"actor_lr_decay":{"enabled":False}})
    batch=synthetic_rollout(first);second._update_flat_stabilized=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("stabilized path used"))
    rng=torch.get_rng_state();first_metrics=first.update(batch);torch.set_rng_state(rng);second_metrics=second.update(batch)
    for key in ("actor_loss","value_loss","entropy","approx_kl","actor_optimizer_steps_this_update","critic_optimizer_steps_this_update"):
        assert second_metrics[key]==pytest.approx(first_metrics[key],abs=0,rel=0)
    for left,right in zip(first.actor.parameters(),second.actor.parameters()):assert torch.equal(left,right)
    for left,right in zip(first.critic.parameters(),second.critic.parameters()):assert torch.equal(left,right)
    assert second.actor_optimizer.param_groups[0]["lr"]==pytest.approx(3e-4)


def test_decay_resume_lr_is_derived_from_global_step(tmp_path):
    modules={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    source=ModularMAPPOTrainer(hidden_dim=16,modules_config=modules);source.sampled_steps=750000
    path=tmp_path/"decay.pt";source.save(path)
    restored=ModularMAPPOTrainer(hidden_dim=16,modules_config=modules);restored.load(path)
    assert restored.actor_optimizer.param_groups[0]["lr"]==pytest.approx(2e-4)
    assert restored.critic_optimizer.param_groups[0]["lr"]==pytest.approx(3e-4)


def test_branch_whitelist_accepts_only_decay_and_target_budget():
    state,env,source=formal_source_state();destination=load_config("configs/ea_wb_actor_lr_decay_900k.yaml")
    result=validate_modular_branch(state,env,destination,{"training_seed":2023,"training_num_envs":24,"training_smoke":False})
    assert result["intervention"]=="actor_lr_decay"
    control=deepcopy(source);control["training"]["total_sampled_steps"]=900000
    assert validate_modular_branch(state,env,control)["intervention"]=="fixed_lr_control"
    mutations=[
        ("wave balance",lambda c:c["modules"]["wave_balancing"].update({"max_weight":2.0})),
        ("network",lambda c:c["network"].update({"attention_heads":1})),
        ("gamma",lambda c:c["training"].update({"gamma":.99})),
        ("critic lr",lambda c:c["training"].update({"critic_learning_rate":1e-4})),
    ]
    for _,mutate in mutations:
        bad=deepcopy(destination);mutate(bad)
        with pytest.raises(RuntimeError,match="outside"):validate_modular_branch(state,env,bad)
    bad_env=deepcopy(env);bad_env["reward"]["kill_reward"]=99
    with pytest.raises(RuntimeError,match="environment"):validate_modular_branch(state,bad_env,destination)


def test_branch_runtime_and_parent_are_isolated(tmp_path):
    state,env,config=formal_source_state();source=tmp_path/"source.pt";torch.save(state,source);before=file_sha256(source)
    runtime=resolve_branch_runtime(config,state,seed=2023,num_envs=24,total_sampled_steps=900000,device="cpu",smoke=None)
    assert runtime["total_sampled_steps"]==900000 and runtime["seed"]==2023
    output=tmp_path/"branch";ensure_fresh_output_directory(output);(output/"branch_only.txt").write_text("new",encoding="utf-8")
    assert file_sha256(source)==before and source.parent!=output
    with pytest.raises(RuntimeError,match="seed"):resolve_branch_runtime(config,state,seed=2024,num_envs=24,total_sampled_steps=900000,device="cpu",smoke=None)
    with pytest.raises(RuntimeError,match="explicit"):resolve_branch_runtime(config,state,seed=2023,num_envs=24,total_sampled_steps=None,device="cpu",smoke=None)


def test_pre_resume_checkpoint_is_content_addressed_and_not_overwritten(tmp_path):
    source=tmp_path/"latest.pt";torch.save({"sampled_steps":600000,"actor":{"x":torch.ones(1)}},source);before=file_sha256(source)
    first=preserve_resume_start_checkpoint(tmp_path,source,{"sampled_steps":600000});second=preserve_resume_start_checkpoint(tmp_path,source,{"sampled_steps":600000})
    backups=list((tmp_path/"resume_points").glob("*.pt"))
    assert len(backups)==1 and file_sha256(backups[0])==before and file_sha256(source)==before
    assert first["created"] is True and second["created"] is False
    metadata=json.loads(backups[0].with_suffix(".json").read_text(encoding="utf-8"));assert metadata["source_sampled_steps"]==600000


def test_old_checkpoint_without_rng_is_backward_compatible(tmp_path):
    trainer=ModularMAPPOTrainer(hidden_dim=16);state=trainer.checkpoint_state();state.pop("rng_state");state.pop("rng_state_available");state.pop("rng_state_restored");state.pop("cuda_rng_state_restored")
    state["development_feature_versions"].pop("actor_lr_decay")
    path=tmp_path/"old.pt";torch.save(state,path);restored=ModularMAPPOTrainer(hidden_dim=16);restored.load(path)
    assert restored.rng_restore_metadata=={"rng_state_available":False,"rng_state_restored":False,"cuda_rng_state_restored":False}


def test_new_checkpoint_restores_cpu_rng_streams(tmp_path):
    source=ModularMAPPOTrainer(hidden_dim=16,seed=33);random.seed(7);np.random.seed(8);path=tmp_path/"rng.pt";source.save(path)
    expected=(random.random(),np.random.random(),torch.rand(4),source.rng.random(4))
    restored=ModularMAPPOTrainer(hidden_dim=16,seed=999);restored.load(path)
    actual=(random.random(),np.random.random(),torch.rand(4),restored.rng.random(4))
    assert actual[0]==expected[0] and actual[1]==expected[1]
    assert torch.equal(actual[2],expected[2]) and np.array_equal(actual[3],expected[3])
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    assert checkpoint["rng_state_available"] is True
    assert set(checkpoint["rng_state"])=={"python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state"}


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_new_checkpoint_restores_cuda_rng_stream(tmp_path):
    source=ModularMAPPOTrainer(hidden_dim=16,device="cuda",seed=44);path=tmp_path/"cuda_rng.pt";source.save(path);expected=torch.rand(4,device="cuda")
    restored=ModularMAPPOTrainer(hidden_dim=16,device="cuda",seed=999);restored.load(path);actual=torch.rand(4,device="cuda")
    assert torch.equal(actual,expected) and restored.rng_restore_metadata["cuda_rng_state_restored"] is True
