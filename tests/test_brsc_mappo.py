from copy import deepcopy
import math
import numpy as np
import pytest
import torch

from algorithm.modules import (BoundaryRedistributedSegmentCreditModule,W1_BOUNDARY_TO_W2,
 W1_BOUNDARY_TO_W3,W2_BOUNDARY_TO_W3,redistribute_boundary_credit)
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.networks import BoundaryStateOutcomeCritic
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.train_modular_mappo import load_config
from tools.analyze_brsc_mappo import mechanism

BRSC={"enabled":True,"max_waves":3,"outcome_critic_learning_rate":3e-4,
 "train_boundaries_per_class":16,"validation_boundaries_per_wave":80,
 "prior_window_boundaries_per_wave":80,"critic_samples_per_class_per_task":2,
 "critic_updates_per_rollout":1,"min_train_boundaries_per_class":1,
 "validation_interval_updates":1,"min_validation_boundaries":4,"min_validation_positive":2,
 "min_validation_negative":2,"min_validation_auroc":.6,"min_validation_brier_skill":0.,
 "readiness_consecutive_passes":3,"auxiliary_gradient_ratio_cap":.25,"prior_probability_epsilon":1e-4}


def pending(wave=1,value=0.,steps=100):
    return {"source_wave":wave,"entry_observation":np.full((4,52),value,np.float32),
            "entry_alive_mask":np.ones(4,np.float32),"entry_remaining_horizon":.8,
            "collection_sampled_steps":steps,"collection_ppo_update_id":2,"post_spawn_entry_state":True}


def rollout(trainer,boundaries=(),with_boundary=False):
    rng=np.random.default_rng(19);t,e,a=4,2,4;obs=rng.normal(size=(t,e,a,52)).astype("f")
    alive=np.ones((t,e,a),"f");raw=rng.normal(size=(t,e,a,3)).astype("f");act=np.tanh(raw).astype("f")
    with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),torch.tensor(raw),torch.tensor(act)).numpy()
    flags=np.zeros((t,e),"f");waves=np.ones((t,e),int)
    if with_boundary:flags[-1,0]=1
    ctx=np.zeros((t,e,0),"f")
    return ModularRolloutBatch(obs,act,raw,old,rng.normal(size=(t,e,a)).astype("f"),np.zeros((t,e,a),"f"),
        np.zeros((t,e),"f"),alive,obs+.01,alive.copy(),waves,np.full((t,e),3),ctx,ctx,
        episode_masks=np.ones((t,e),"f"),remaining_horizons=np.full((t,e),.8,"f"),
        next_remaining_horizons=np.full((t,e),.79,"f"),wave_transition_flags=flags,
        brsc_supervision_boundaries=list(boundaries))


def test_boundary_labels_split_and_train_validation_prior_isolation():
    module=BoundaryRedistributedSegmentCreditModule(BRSC)
    b1=module.complete_boundary(pending(1),1,1,0);b2=module.complete_boundary(pending(1,1),2,2,0)
    b3=module.complete_boundary(pending(1,2),3,5,0);b4=module.complete_boundary(pending(2,3),3,5,0)
    assert (b1["label_c2"],b1["label_c3"])==(0,0)
    assert (b2["label_c2"],b2["label_c3"])==(1,0)
    assert (b3["label_c2"],b3["label_c3"])==(1,1) and b3["split"]==b4["split"]=="validation"
    module.ingest([b1,b2,b3,b4])
    assert len(module.validation[1])==len(module.validation[2])==1 and len(module.prior_window[1])==2
    assert module.recent_prior(W1_BOUNDARY_TO_W2)==.5 and module.recent_prior(W1_BOUNDARY_TO_W3)==0
    b3["label_c2"]=0;b3["label_c3"]=0
    assert module.recent_prior(W1_BOUNDARY_TO_W2)==.5 and module.recent_prior(W1_BOUNDARY_TO_W3)==0


def test_runner_records_only_post_spawn_state_and_rejects_duplicate():
    runner=ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    runner.trainer=type("T",(),{})();runner.trainer.boundary_redistributed_segment_credit=BoundaryRedistributedSegmentCreditModule(BRSC)
    runner.trainer.ppo_update_count=7;runner.brsc_pending_episode=[{1:None,2:None}];runner.brsc_completed_episode_counter=0
    pre=np.zeros((4,52),np.float32);post=np.ones((4,52),np.float32);alive=np.ones(4,np.float32)
    runner._brsc_record_boundary(0,1,pre,alive,.9,False,10);assert runner.brsc_pending_episode[0][1] is None
    runner._brsc_record_boundary(0,1,post,alive,.8,True,11)
    assert np.array_equal(runner.brsc_pending_episode[0][1]["entry_observation"],post)
    with pytest.raises(RuntimeError):runner._brsc_record_boundary(0,1,post+1,alive,.7,True,12)
    completed=runner._brsc_finalize_episode(0,1)
    assert len(completed)==1 and completed[0]["label_c2"]==completed[0]["label_c3"]==0


def test_boundary_critic_is_action_free_raw_logits_and_task_heads():
    critic=BoundaryStateOutcomeCritic(52,16,2);obs=torch.randn(5,4,52);alive=torch.ones(5,4)
    out=critic(obs,alive,torch.tensor([1,1,2,2,1]),torch.ones(5))
    assert out.shape==(5,2) and "actions" not in critic.forward.__code__.co_varnames
    assert {W1_BOUNDARY_TO_W2:0,W1_BOUNDARY_TO_W3:1,W2_BOUNDARY_TO_W3:1}==__import__("algorithm.modules",fromlist=["BRSC_TASK_HEAD"]).BRSC_TASK_HEAD


def test_redistribution_exact_and_does_not_cross_wave_episode_or_rollout():
    decay=.999*.95;waves=np.asarray([[1],[1],[1],[1],[1],[1]],int);dones=np.zeros((6,1));flags=np.zeros((6,1));flags[5]=1;credit=np.zeros((6,1),np.float32);credit[5]=.4
    adv,active,lengths=redistribute_boundary_credit(waves,dones,flags,credit,decay)
    assert lengths==[6] and adv[5,0]==pytest.approx(.4) and adv[4,0]==pytest.approx(.4*decay) and adv[3,0]==pytest.approx(.4*decay**2)
    waves=np.asarray([[1],[1],[1],[1],[1],[2],[2],[2],[2],[2]],int);flags=np.zeros((10,1));flags[4]=flags[9]=1;credit=np.zeros((10,1),np.float32);credit[4]=.2;credit[9]=.3
    adv,_,lengths=redistribute_boundary_credit(waves,np.zeros((10,1)),flags,credit,decay)
    assert lengths==[5,5] and adv[0,0]==pytest.approx(.2*decay**4) and adv[5,0]==pytest.approx(.3*decay**4)
    waves=np.ones((6,1),int);dones=np.zeros((6,1));dones[2]=1;flags=np.zeros((6,1));flags[5]=1;credit=np.zeros((6,1));credit[5]=.3
    adv,_,lengths=redistribute_boundary_credit(waves,dones,flags,credit,decay)
    assert lengths==[3] and np.all(adv[:3]==0)  # rollout input itself is the hard outer bound


def test_raw_credit_has_no_normalization_and_active_update_is_single_step():
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=91,ppo_epochs=1,minibatch_size=8,modules_config={"boundary_redistributed_segment_credit":BRSC})
    module=trainer.boundary_redistributed_segment_credit;module.task_ready[W1_BOUNDARY_TO_W2]=True
    class FakeCritic(torch.nn.Module):
        def forward(self,observations,alive_mask,source_wave,remaining_horizon):
            logit=torch.full((len(observations),),math.log(.503/.497),device=observations.device)
            return torch.stack((logit,torch.zeros_like(logit)),-1)
    trainer.brsc_critic=FakeCritic().to(trainer.device);trainer._train_brsc_critic=lambda:{"brsc_critic_loss":0.};trainer._validate_brsc_tasks=lambda:{}
    batch=rollout(trainer,with_boundary=True);before_a=[p.detach().clone() for p in trainer.actor.parameters()];before_c=[p.detach().clone() for p in trainer.critic.parameters()]
    metrics=trainer.update(batch)
    assert metrics["brsc_actor_active"]==1 and metrics["brsc_boundary_credit_mean_wave1"]==pytest.approx(.003,abs=1e-6)
    assert metrics["brsc_adv_abs_max"]==pytest.approx(.003,abs=1e-6)
    assert metrics["actor_optimizer_steps_this_update"]==metrics["critic_optimizer_steps_this_update"]==1
    assert any(not torch.equal(a,b) for a,b in zip(before_a,trainer.actor.parameters())) and any(not torch.equal(a,b) for a,b in zip(before_c,trainer.critic.parameters()))
    assert metrics["brsc_aux_to_tactical_ratio_post"]<=.25+1e-6


def test_formal_shape_multi_boundary_credit_uses_rollout_alive_coverage():
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=92,modules_config={"boundary_redistributed_segment_credit":BRSC})
    module=trainer.boundary_redistributed_segment_credit
    for task in (W1_BOUNDARY_TO_W2,W1_BOUNDARY_TO_W3,W2_BOUNDARY_TO_W3):module.task_ready[task]=True
    class FakeCritic(torch.nn.Module):
        def forward(self,observations,alive_mask,source_wave,remaining_horizon):
            logit=torch.full((len(observations),),.2,device=observations.device)
            return torch.stack((logit,logit),-1)
    trainer.brsc_critic=FakeCritic().to(trainer.device)
    t,e,a=256,24,4;flags=np.zeros((t,e),np.float32)
    for index in ((20,0),(60,1),(100,2),(140,3),(200,4)):flags[index]=1
    waves=np.ones((t,e),np.int64);waves[120:,3]=2;waves[180:,4]=2
    dones=np.zeros((t,e),np.float32);dones[40,1]=1
    rollout_alive=np.ones((t,e,a),np.float32);rollout_alive[::3,:,3]=0;rollout_alive[::5,:,2]=0
    next_alive=rollout_alive.copy();next_obs=np.zeros((t,e,a,52),np.float32)
    r=type("R",(),{"wave_transition_flags":flags,"next_remaining_horizons":np.full((t,e),.5,np.float32)})()
    advantage,active,metrics=trainer._brsc_rollout_credit(
        r,torch.as_tensor(next_obs),torch.as_tensor(rollout_alive),torch.as_tensor(next_alive),
        torch.as_tensor(dones),torch.as_tensor(waves))
    assert advantage.shape==active.shape==(256,24)
    assert 0<=metrics["brsc_credited_alive_agent_fraction"]<=1 and 0<=metrics["brsc_credited_transition_fraction"]<=1
    expected=((active.cpu().numpy()[...,None])*(rollout_alive>.5)).sum()/(rollout_alive>.5).sum()
    assert metrics["brsc_credited_alive_agent_fraction"]==pytest.approx(expected)
    assert not active[:120,3].any()  # Wave2 boundary never credits preceding Wave1.
    assert not active[:41,1].any()  # Boundary after reset never crosses episode reset.


def test_multi_boundary_trainer_update_completes_with_finite_trusted_gradients():
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=93,ppo_epochs=1,minibatch_size=8,modules_config={"boundary_redistributed_segment_credit":BRSC})
    module=trainer.boundary_redistributed_segment_credit;module.task_ready[W1_BOUNDARY_TO_W2]=True
    class FakeCritic(torch.nn.Module):
        def forward(self,observations,alive_mask,source_wave,remaining_horizon):
            logit=torch.full((len(observations),),.2,device=observations.device)
            return torch.stack((logit,torch.zeros_like(logit)),-1)
    trainer.brsc_critic=FakeCritic().to(trainer.device);trainer._train_brsc_critic=lambda:{"brsc_critic_loss":0.};trainer._validate_brsc_tasks=lambda:{}
    batch=rollout(trainer);batch.wave_transition_flags[1,0]=batch.wave_transition_flags[3,1]=1
    before_a=[p.detach().clone() for p in trainer.actor.parameters()];before_c=[p.detach().clone() for p in trainer.critic.parameters()]
    metrics=trainer.update(batch)
    assert metrics["brsc_actor_active"]==1 and metrics["actor_optimizer_steps_this_update"]==metrics["critic_optimizer_steps_this_update"]==1
    assert any(not torch.equal(a,b) for a,b in zip(before_a,trainer.actor.parameters())) and any(not torch.equal(a,b) for a,b in zip(before_c,trainer.critic.parameters()))
    assert np.isfinite([float(v) for v in metrics.values() if isinstance(v,(int,float))]).all()
    assert metrics["brsc_aux_to_tactical_ratio_post"]<=.25+1e-6 and 0<=metrics["brsc_credited_alive_agent_fraction"]<=1


def test_preupdate_causality_and_cold_gate_exact_plain(tmp_path):
    common={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    plain=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config=common)
    brsc=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config={**common,"boundary_redistributed_segment_credit":BRSC})
    batch=rollout(plain);state=torch.random.get_rng_state();plain.update(deepcopy(batch));torch.random.set_rng_state(state);metrics=brsc.update(deepcopy(batch))
    assert metrics["brsc_actor_active"]==0 and all(torch.equal(a,b) for a,b in zip(plain.actor.parameters(),brsc.actor.parameters())) and all(torch.equal(a,b) for a,b in zip(plain.critic.parameters(),brsc.critic.parameters()))
    assert plain.rng.bit_generator.state==brsc.rng.bit_generator.state
    # A current-rollout label may make the task ready after PPO, never before it.
    current=brsc.boundary_redistributed_segment_credit.complete_boundary(pending(1),1,1,0);batch=rollout(brsc,[current],True)
    brsc._train_brsc_critic=lambda:{}
    def validate():brsc.boundary_redistributed_segment_credit.task_ready[W1_BOUNDARY_TO_W2]=True;return {}
    brsc._validate_brsc_tasks=validate;metrics=brsc.update(batch);assert metrics["brsc_actor_active"]==0
    brsc._train_brsc_critic=lambda:{};brsc._validate_brsc_tasks=lambda:{};assert brsc.update(rollout(brsc,with_boundary=True))["brsc_actor_active"]==1
    path=tmp_path/"brsc.pt";brsc.save(path);restored=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config={**common,"boundary_redistributed_segment_credit":BRSC});restored.load(path)
    original=brsc.boundary_redistributed_segment_credit;loaded=restored.boundary_redistributed_segment_credit
    assert loaded.task_ready==original.task_ready and loaded.validation_pass_streak==original.validation_pass_streak
    assert loaded.next_boundary_id==original.next_boundary_id and len(loaded.train_replay[1]["00"])==len(original.train_replay[1]["00"])
    assert restored.brsc_rng.bit_generator.state==brsc.brsc_rng.bit_generator.state


def test_readiness_three_passes_and_failure_deactivates():
    module=BoundaryRedistributedSegmentCreditModule(BRSC);good={"validation_segments":64,"validation_positive":32,"validation_negative":32,"validation_auroc":.8,"validation_brier_skill":.1}
    assert module.apply_validation(W1_BOUNDARY_TO_W2,good,1) and not module.task_ready[W1_BOUNDARY_TO_W2]
    assert module.apply_validation(W1_BOUNDARY_TO_W2,good,2) and not module.task_ready[W1_BOUNDARY_TO_W2]
    assert module.apply_validation(W1_BOUNDARY_TO_W2,good,3) and module.task_ready[W1_BOUNDARY_TO_W2]
    assert not module.apply_validation(W1_BOUNDARY_TO_W2,{**good,"validation_brier_skill":0},4) and not module.task_ready[W1_BOUNDARY_TO_W2]


def test_formal_config_protocol_metadata_and_resume_fields_are_frozen():
    config=load_config("configs/dev_brsc_mappo_v1_3m.yaml");enabled={k for k,v in config["modules"].items() if v.get("enabled",False)}
    assert enabled=={"actor_lr_decay","boundary_redistributed_segment_credit"}
    assert config["training"]["seed"]==5301 and config["training"]["rollout_steps"]==256 and config["training"]["num_train_envs"]==24 and config["training"]["total_sampled_steps"]==3000000
    trainer=ModularMAPPOTrainer(hidden_dim=16,modules_config={"boundary_redistributed_segment_credit":BRSC});metadata=checkpoint_architecture(trainer)
    assert metadata["action_input"] is False and metadata["post_spawn_entry_state"] is True and metadata["advantage_normalization"]=="none" and metadata["gradient_ratio_cap"]==.25
    source=__import__("pathlib").Path("algorithm/modular_mappo/runner.py").read_text(encoding="utf-8")
    assert '"brsc_completed_episode_counter":self.brsc_completed_episode_counter' in source
    assert 'extra.get("brsc_completed_episode_counter",0)' in source and 'brsc_pending_boundaries_dropped_on_resume' in source


def test_analyzer_uses_only_real_validation_rows(tmp_path):
    path=tmp_path/"optimization_metrics.jsonl";rows=[
      {"sampled_steps":10,"brsc_w1_to_w2_validation_segments":0,"brsc_w1_to_w2_validation_auroc":0,"brsc_w1_to_w2_validation_brier_skill":0},
      {"sampled_steps":20,"brsc_w1_to_w2_validation_segments":64,"brsc_w1_to_w2_validation_auroc":.8,"brsc_w1_to_w2_validation_brier_skill":.2}]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows),encoding="utf-8")
    result=mechanism(path);assert result["w1_to_w2"]["validation_checks"]==1
    assert result["w1_to_w2"]["validation_auroc_mean"]==pytest.approx(.8) and result["w1_to_w2"]["validation_brier_skill_mean"]==pytest.approx(.2)
