import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.networks import ModularMAPPOActor
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.evaluation import per_wave_episode_diagnostics
from algorithm.evaluate_modular_mappo import checkpoint_hidden_dim
from algorithm.modules import AdvantagePriorityModule, PPOStabilizationModule


def entity_observations(*prefix):
    obs=torch.randn(*prefix,52)
    obs[...,13]=1;obs[...,20]=0;obs[...,27]=1
    obs[...,33]=1;obs[...,39]=0;obs[...,45]=1;obs[...,51]=0
    return obs


def rollout(trainer,T=4,E=2):
    rng=np.random.default_rng(12);obs=rng.normal(size=(T,E,4,52)).astype("f")
    obs[...,13]=1;obs[...,20]=0;obs[...,27]=1;obs[...,33]=1;obs[...,39]=0;obs[...,45]=1;obs[...,51]=0
    alive=np.ones((T,E,4),"f");raw=rng.normal(size=(T,E,4,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad():
        o=torch.as_tensor(obs,device=trainer.device);r=torch.as_tensor(raw,device=trainer.device);a=torch.as_tensor(actions,device=trainer.device)
        old=trainer.actor._squashed_log_prob(trainer.actor.distribution(o),r,a).cpu().numpy()
    rewards=rng.normal(size=(T,E,4)).astype("f");ctx=np.zeros((T,E,0),"f")
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((T,E),"f"),alive,obs.copy(),alive.copy(),np.tile(np.array([[1],[2],[3],[3]]),(1,E)),np.full((T,E),3),ctx,ctx,episode_masks=np.ones((T,E),"f"))


def test_entity_split_exact_and_shared_encoders():
    obs=torch.arange(52.).reshape(1,52);own,ally,am,enemy,em=ModularMAPPOActor.split_entities(obs)
    assert own.shape==(1,7) and ally.shape==(1,3,6) and enemy.shape==(1,4,5)
    assert torch.equal(ally[0,0],obs[0,7:13]) and am[0,0]==obs[0,13]
    assert torch.equal(enemy[0,3],obs[0,46:51]) and em[0,3]==obs[0,51]
    actor=ModularMAPPOActor(entity_attention_config={"enabled":True})
    assert len([name for name,_ in actor.named_modules() if name=="ally_encoder"])==1
    assert not hasattr(actor,"backbone")


@pytest.mark.parametrize("prefix",[(2,4),(3,2,4)])
def test_entity_attention_masks_dead_slots_and_is_finite(prefix):
    actor=ModularMAPPOActor(entity_attention_config={"enabled":True});obs=entity_observations(*prefix)
    dist,diag=actor.distribution(obs,return_attention=True)
    assert dist.mean.shape==(*prefix,3) and torch.isfinite(dist.mean).all()
    assert torch.all(diag["ally_attention_weights"][...,1]==0)
    assert torch.all(diag["enemy_attention_weights"][...,1]==0) and torch.all(diag["enemy_attention_weights"][...,3]==0)
    obs[...,13]=obs[...,20]=obs[...,27]=0;obs[...,33]=obs[...,39]=obs[...,45]=obs[...,51]=0
    dist,diag=actor.distribution(obs,return_attention=True)
    assert torch.isfinite(dist.mean).all() and torch.count_nonzero(diag["ally_attention_weights"])==0 and torch.count_nonzero(diag["enemy_attention_weights"])==0


def test_entity_actions_and_exact_squashed_log_probability():
    actor=ModularMAPPOActor(entity_attention_config={"enabled":True});obs=entity_observations(5,4);dist=actor.distribution(obs)
    raw=dist.rsample();action=torch.tanh(raw);actual=actor._squashed_log_prob(dist,raw,action)
    expected=(dist.log_prob(raw)-2*(np.log(2.)-raw-F.softplus(-2*raw))).sum(-1)
    assert torch.allclose(actual,expected) and torch.isfinite(actual).all()
    assert torch.isfinite(torch.tanh(dist.mean)).all() and torch.isfinite(action).all()


def test_disabled_entity_actor_has_exact_legacy_state_dict():
    torch.manual_seed(4);a=ModularMAPPOActor();torch.manual_seed(4);b=ModularMAPPOActor(entity_attention_config={"enabled":False})
    assert list(a.state_dict())==list(b.state_dict()) and all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


def test_within_wave_priority_mean_cap_dead_and_combination():
    module=AdvantagePriorityModule({"enabled":True,"alpha":.5,"z_clip":2.,"final_weight_cap":4.})
    adv=torch.tensor([[[ -3.,0.,1.,7.]],[[1.,2.,3.,4.]],[[0.,0.,9.,9.]]]);waves=torch.tensor([[1],[2],[3]]);alive=torch.tensor([[[1.,1.,1.,1.]],[[1.,1.,0.,0.]],[[0.,0.,0.,0.]]]);base=torch.tensor([[.5],[1.5],[3.]])
    priority,combined,metrics=module.compute_tensor(adv,waves,alive,base)
    for wave in (1,2):
        chosen=((waves==wave).unsqueeze(-1)&(alive>.5));assert priority[chosen].mean()==pytest.approx(1.)
    assert torch.count_nonzero(combined[alive==0])==0 and combined[alive>.5].mean()==pytest.approx(1.,abs=1e-5)
    assert combined.max()<=4 and priority[0,0,3]>priority[0,0,0]
    assert metrics["combined_actor_weight_mean"]==pytest.approx(1.,abs=1e-5)


def test_disabled_priority_returns_base_exactly():
    module=AdvantagePriorityModule({"enabled":False});adv=torch.randn(2,1,4);waves=torch.ones(2,1,dtype=torch.long);alive=torch.ones_like(adv);base=torch.tensor([[.5],[1.5]])
    priority,combined,metrics=module.compute_tensor(adv,waves,alive,base)
    assert torch.equal(priority,torch.ones_like(adv));assert torch.equal(combined,base.unsqueeze(-1).expand_as(adv));assert metrics=={}


def test_priority_is_actor_only_and_entropy_is_unweighted():
    trainer=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,modules_config={})
    batch=rollout(trainer);to=lambda x:torch.as_tensor(x);args=(to(batch.observations[0]),to(batch.actions[0]),to(batch.raw_actions[0]),to(batch.old_log_probs[0]),to(batch.alive_masks[0]),torch.randn(2,4),torch.zeros(2,4),torch.randn(2,4),torch.ones(2),to(batch.contexts[0]))
    torch.manual_seed(88);base=trainer._loss_step(*args)
    torch.manual_seed(88);weighted=trainer._loss_step(*args,actor_weights=torch.full((2,4),2.))
    assert weighted[0]!=base[0] and torch.equal(weighted[1],base[1]) and torch.equal(weighted[2],base[2])


def test_dead_values_do_not_change_priority_statistics():
    module=AdvantagePriorityModule({"enabled":True});waves=torch.ones(1,1,dtype=torch.long);alive=torch.tensor([[[1.,1.,0.,0.]]]);base=torch.ones(1,1)
    first=module.compute_tensor(torch.tensor([[[0.,2.,1e6,-1e6]]]),waves,alive,base)[0]
    second=module.compute_tensor(torch.tensor([[[0.,2.,-7.,19.]]]),waves,alive,base)[0]
    assert torch.equal(first[alive>.5],second[alive>.5])


def test_priority_receives_raw_gae_before_global_normalization():
    trainer=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,modules_config={"advantage_priority":{"enabled":True}})
    batch=rollout(trainer);captured={};original=trainer.advantage_priority.compute_tensor
    def priority(raw,*args):captured["raw"]=raw.clone();return original(raw,*args)
    def update_flat(obs,act,raw,oldlog,alive,adv,*args,**kwargs):captured["normalized"]=adv.clone();return {}
    trainer.advantage_priority.compute_tensor=priority;trainer._update_flat=update_flat;trainer.update(batch)
    live=torch.as_tensor(batch.alive_masks)> .5;normalized=captured["normalized"][live]
    assert normalized.mean()==pytest.approx(0.,abs=1e-5) and normalized.std(unbiased=False)==pytest.approx(1.,abs=1e-5)
    assert not torch.allclose(captured["raw"][live],normalized)


def test_linear_actor_learning_rate_endpoints():
    module=PPOStabilizationModule({"enabled":True})
    assert module.actor_learning_rate(0,100)==pytest.approx(3e-4)
    assert module.actor_learning_rate(50,100)==pytest.approx(2e-4)
    assert module.actor_learning_rate(100,100)==pytest.approx(1e-4)


def test_stabilized_hard_stop_keeps_all_critic_epochs(tmp_path):
    modules={"ppo_stabilization":{"enabled":True}}
    trainer=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=4,minibatch_size=8,modules_config=modules,total_sampled_steps=100)
    trainer._full_rollout_kl=lambda *args:.1
    metrics=trainer.update(rollout(trainer))
    assert metrics["actor_epochs_used"]==1 and metrics["critic_epochs_used"]==4 and metrics["kl_hard_stop_triggered"]==1
    assert metrics["actor_optimizer_steps_this_update"]==1 and metrics["critic_optimizer_steps_this_update"]==4
    trainer.sampled_steps=50;path=tmp_path/"stable.pt";trainer.save(path)
    restored=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=4,minibatch_size=8,modules_config=modules,total_sampled_steps=100);restored.load(path)
    assert restored.actor_optimizer.param_groups[0]["lr"]==pytest.approx(2e-4) and restored.kl_hard_stop_count==1


def test_stabilization_disabled_keeps_configured_lr_and_full_epochs():
    trainer=ModularMAPPOTrainer(hidden_dim=32,actor_learning_rate=7e-4,ppo_epochs=10,minibatch_size=8,modules_config={},total_sampled_steps=100)
    metrics=trainer.update(rollout(trainer));assert metrics["actor_optimizer_steps_this_update"]==10 and metrics["critic_optimizer_steps_this_update"]==10
    assert trainer.actor_optimizer.param_groups[0]["lr"]==pytest.approx(7e-4)


@pytest.mark.parametrize("modules",[
    {"entity_attention":{"enabled":True}},
    {"wave_balancing":{"enabled":True}},
    {"wave_balancing":{"enabled":True},"advantage_priority":{"enabled":True}},
    {"entity_attention":{"enabled":True},"wave_balancing":{"enabled":True}},
    {"entity_attention":{"enabled":True},"wave_balancing":{"enabled":True},"advantage_priority":{"enabled":True}},
])
def test_module_combinations_one_update_are_finite(modules):
    hidden=256 if modules.get("entity_attention",{}).get("enabled") else 32
    trainer=ModularMAPPOTrainer(hidden_dim=hidden,ppo_epochs=1,minibatch_size=8,modules_config=copy.deepcopy(modules),total_sampled_steps=100)
    metrics=trainer.update(rollout(trainer));assert all(np.isfinite(list(metrics.values())))


def test_ea_hwb_stable_integration_and_diagnostics_are_finite():
    modules={"entity_attention":{"enabled":True},"wave_balancing":{"enabled":True},"advantage_priority":{"enabled":True},"ppo_stabilization":{"enabled":True}}
    trainer=ModularMAPPOTrainer(hidden_dim=256,ppo_epochs=2,minibatch_size=8,modules_config=modules,total_sampled_steps=100)
    metrics=trainer.update(rollout(trainer));required=("attention_dead_mass","combined_actor_weight_mean","epoch_kl_max","actor_grad_norm","critic_grad_norm","ratio_mean")
    assert all(np.isfinite(metrics[key]) for key in required) and metrics["attention_dead_mass"]==0


def test_per_wave_diagnostics_use_existing_records_and_do_not_guess():
    info={"per_wave_metrics":[{"wave_index":1,"red_survivors_end":3,"blue_survivors_end":0,"red_attack_kills":4,"red_boundary_exits":1,"red_ground_losses":0,"start_step":0,"duration_steps":25}]}
    result=per_wave_episode_diagnostics(info,3)
    assert result["red_survivors_after_wave_1"]==3 and result["wave_1_entry_step"]==0
    assert result["red_survivors_after_wave_2"] is None and result["wave_3_duration"] is None


def test_ea_checkpoint_hidden_metadata_never_requires_legacy_backbone():
    assert checkpoint_hidden_dim({"actor":{}},{"network_architecture":{"hidden_dim":256}})==256
