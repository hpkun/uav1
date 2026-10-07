from copy import deepcopy
import math
import numpy as np
import pytest
import torch

from algorithm.modules import (CounterfactualInterWaveCreditModule,W1_TO_W2,W1_TO_W3,W2_TO_W3,
 binary_auroc,prior_corrected_probability,freshness_mask)
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.networks import InterWaveActionOutcomeCritic
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,antithetic_latent_actions,caiw_trust_cap,asymmetric_tactical_projection

CAIW={"enabled":True,"max_waves":3,"outcome_critic_learning_rate":3e-4,"train_segments_per_class":16,"validation_segments_per_wave":80,"prior_window_segments_per_wave":80,"max_states_per_segment":8,"train_states_per_class_per_task":2,"critic_updates_per_rollout":1,"min_train_segments_per_class":1,"freshness_ratio_low":.8,"freshness_ratio_high":1.2,"min_fresh_states_per_validation_segment":1,"validation_interval_updates":1,"min_validation_segments":4,"min_validation_positive_segments":2,"min_validation_negative_segments":2,"min_validation_auroc":.6,"min_validation_brier_skill":0.,"readiness_consecutive_passes":3,"counterfactual_samples":4,"counterfactual_batch_size":32,"auxiliary_gradient_ratio_cap":.25,"gradient_projection":"asymmetric_tactical_preserving","prior_probability_epsilon":1e-4}

def rows(n=10,offset=0.):
    return [{"observation":np.full((4,52),offset+i,np.float32),"alive_mask":np.ones(4,np.float32),"action":np.full((4,3),.1,np.float32),"raw_action":np.full((4,3),np.arctanh(.1),np.float32),"behavior_log_prob":np.zeros(4,np.float32),"remaining_horizon":1-i/20,"collection_ppo_update_id":2,"collection_sampled_steps":100+i,"wave_cleared_this_step":i==n-1,"spawned_next_wave":i==n-1} for i in range(n)]

def rollout(trainer,segments=()):
    rng=np.random.default_rng(9);t,e,a=4,2,4;obs=rng.normal(size=(t,e,a,52)).astype("f");alive=np.ones((t,e,a),"f");raw=rng.normal(size=(t,e,a,3)).astype("f");act=np.tanh(raw).astype("f")
    with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),torch.tensor(raw),torch.tensor(act)).numpy()
    ctx=np.zeros((t,e,0),"f");waves=np.asarray([[1,2]]*t)
    return ModularRolloutBatch(obs,act,raw,old,rng.normal(size=(t,e,a)).astype("f"),np.zeros((t,e,a),"f"),np.zeros((t,e),"f"),alive,obs+.01,alive.copy(),waves,np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"),remaining_horizons=np.full((t,e),.8,"f"),next_remaining_horizons=np.full((t,e),.79,"f"),wave_transition_flags=np.zeros((t,e),"f"),caiw_supervision_segments=list(segments))

def test_target_semantics_split_and_segment_action_preservation():
    m=CounterfactualInterWaveCreditModule(CAIW)
    assert [m.labels(x) for x in range(4)]==[{"c2":0,"c3":0},{"c2":0,"c3":0},{"c2":1,"c3":0},{"c2":1,"c3":1}]
    s1=m.cap_segment(rows(),1,3,5,0);s2=m.cap_segment(rows(offset=2),2,3,5,0)
    assert s1["split"]==s2["split"]=="validation" and np.array_equal(s1["actions"][0],rows()[0]["action"])
    assert np.array_equal(s1["raw_actions"][0],rows()[0]["raw_action"]) and s1["wave_cleared_flags"].any()

def test_action_path_zero_init_and_output_shape():
    critic=InterWaveActionOutcomeCritic(52,3,16,2);obs=torch.randn(5,4,52);alive=torch.ones(5,4);wave=torch.tensor([1,1,2,2,1]);h=torch.ones(5)
    a=torch.randn(5,4,3);b=torch.randn(5,4,3);x=critic(obs,a,alive,wave,h);y=critic(obs,b,alive,wave,h)
    assert x.shape==(5,2) and torch.allclose(x,y,atol=1e-7,rtol=0)

def test_binary_auroc_ties_single_class_and_prior_correction():
    assert binary_auroc([0,0,1,1],[0,1,2,3])==1 and binary_auroc([0,0,1,1],[3,2,1,0])==0
    assert binary_auroc([0,1],[.5,.5])==.5 and binary_auroc([1,1],[.2,.8]) is None
    z=torch.tensor(0.);assert prior_corrected_probability(z,.2).item()==pytest.approx(.2) and prior_corrected_probability(z,.5).item()==pytest.approx(.5)

def test_freshness_log_space_boundaries_and_dead_agent():
    ratios=np.log(np.asarray([[.8,1.,1.2,99.],[.79,1.,1.,1.],[1.21,1.,1.,1.]]));alive=np.asarray([[1,1,1,0],[1,1,1,0],[1,1,1,0]])
    assert freshness_mask(ratios,alive).tolist()==[True,False,False]

def test_class_replay_balance_and_task_routing():
    m=CounterfactualInterWaveCreditModule(CAIW)
    for i,c in enumerate([0]*10+[2]*3+[3]*2):m.ingest([m.cap_segment(rows(offset=i),1,c,i*5+1,0)])
    p,n=m.task_classes(W1_TO_W2);assert len(p)==5 and len(n)==10
    p,n=m.task_classes(W1_TO_W3);assert len(p)==2 and len(n)==13
    assert {W1_TO_W2:0,W1_TO_W3:1,W2_TO_W3:1}==__import__("algorithm.modules",fromlist=["TASK_HEAD"]).TASK_HEAD

def test_validation_labels_do_not_leak_into_train_prior():
    m=CounterfactualInterWaveCreditModule(CAIW)
    train=m.cap_segment(rows(),1,0,1,0); validation=m.cap_segment(rows(offset=3),1,3,5,0)
    m.ingest([train,validation])
    assert len(m.validation[1])==1 and len(m.prior_window[1])==1
    assert m.recent_prior(W1_TO_W2)==0 and m.recent_prior(W1_TO_W3)==0
    m2=CounterfactualInterWaveCreditModule(CAIW);m2.ingest([validation,train])
    assert m2.recent_prior(W1_TO_W2)==0 and m2.recent_prior(W1_TO_W3)==0

def test_heldout_gate_requires_three_consecutive_and_resets():
    m=CounterfactualInterWaveCreditModule(CAIW);good={"validation_segments":64,"validation_positive":32,"validation_negative":32,"validation_auroc":.8,"validation_brier_skill":.1}
    assert m.apply_validation(W1_TO_W2,good,1) and not m.task_ready[W1_TO_W2];assert m.apply_validation(W1_TO_W2,good,2) and not m.task_ready[W1_TO_W2];assert m.apply_validation(W1_TO_W2,good,3) and m.task_ready[W1_TO_W2]
    bad={**good,"validation_brier_skill":0};assert not m.apply_validation(W1_TO_W2,bad,4) and not m.task_ready[W1_TO_W2] and m.validation_pass_streak[W1_TO_W2]==0

def test_antithetic_sampling_rng_isolation_raw_scale_and_trust_cap():
    mean=torch.zeros(3,3);std=torch.ones(3,3);rng=np.random.default_rng(8);global_before=torch.random.get_rng_state();samples=antithetic_latent_actions(mean,std,rng)
    assert torch.equal(global_before,torch.random.get_rng_state()) and torch.allclose(samples[:,0]+samples[:,1],2*mean) and torch.allclose(samples[:,2]+samples[:,3],2*mean)
    assert (.503-.500)==pytest.approx(.003)
    tactical=[torch.tensor([2.,0.])];aux=[torch.tensor([4.,0.])];trusted,scale,nt,na=caiw_trust_cap(tactical,aux,.25)
    assert torch.linalg.vector_norm(trusted[0])<=.25*nt+1e-7 and scale.item()==pytest.approx(.125)
    trusted,scale,_,_=caiw_trust_cap([torch.zeros(2)],aux,.25);assert scale==0 and torch.equal(trusted[0],torch.zeros(2))
    projected,dot=asymmetric_tactical_projection([torch.tensor([1.,0.])],[torch.tensor([-2.,1.])]);assert dot<0 and torch.dot(projected[0],torch.tensor([1.,0.]))==pytest.approx(0)

def test_same_seed_initialization_cold_gate_and_checkpoint_roundtrip(tmp_path):
    common={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    plain=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config=common)
    caiw=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config={**common,"counterfactual_inter_wave_credit":CAIW})
    assert all(torch.equal(v,caiw.actor.state_dict()[k]) for k,v in plain.actor.state_dict().items()) and all(torch.equal(v,caiw.critic.state_dict()[k]) for k,v in plain.critic.state_dict().items())
    batch=rollout(plain);state=torch.random.get_rng_state();plain.update(deepcopy(batch));torch.random.set_rng_state(state);metrics=caiw.update(deepcopy(batch))
    assert metrics["caiw_actor_active"]==0 and all(torch.equal(a,b) for a,b in zip(plain.actor.parameters(),caiw.actor.parameters())) and plain.rng.bit_generator.state==caiw.rng.bit_generator.state
    segment=caiw.counterfactual_inter_wave_credit.cap_segment(rows(),1,3,1,0);caiw.counterfactual_inter_wave_credit.ingest([segment]);path=tmp_path/"caiw.pt";caiw.save(path)
    restored=ModularMAPPOTrainer(hidden_dim=16,seed=77,ppo_epochs=1,minibatch_size=8,modules_config={**common,"counterfactual_inter_wave_credit":CAIW});restored.load(path)
    assert restored.counterfactual_inter_wave_credit.next_segment_id==caiw.counterfactual_inter_wave_credit.next_segment_id and restored.caiw_rng.bit_generator.state==caiw.caiw_rng.bit_generator.state

def test_active_caiw_trainer_update_uses_per_agent_credit_and_single_steps():
    cfg={"counterfactual_inter_wave_credit":CAIW}
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=91,ppo_epochs=1,minibatch_size=8,modules_config=cfg)
    module=trainer.counterfactual_inter_wave_credit;module.task_ready[W1_TO_W2]=True
    class FakeCritic(torch.nn.Module):
        def forward(self,observations,actions,alive_mask,source_wave,remaining_horizon):
            q=0.5+0.1*actions[:,0,0]
            return torch.stack((q,q*0),-1)
    trainer.caiw_critic=FakeCritic().to(trainer.device)
    trainer._train_caiw_critic=lambda: {"caiw_critic_loss":0.1}
    trainer._validate_caiw_tasks=lambda: {}
    batch=rollout(trainer)
    before_actor=[p.detach().clone() for p in trainer.actor.parameters()];before_critic=[p.detach().clone() for p in trainer.critic.parameters()]
    metrics=trainer.update(batch)
    assert metrics["caiw_actor_active"]==1 and metrics["actor_optimizer_steps_this_update"]==1 and metrics["critic_optimizer_steps_this_update"]==1
    assert any(not torch.equal(a,b) for a,b in zip(before_actor,trainer.actor.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(before_critic,trainer.critic.parameters()))
    assert np.isfinite([float(v) for v in metrics.values() if isinstance(v,(int,float))]).all()
    assert metrics["caiw_aux_to_tactical_ratio_post"]<=.25+1e-6
    assert metrics["caiw_agent0_advantage_abs_mean"]>0 and metrics["caiw_agent1_advantage_abs_mean"]==0

def test_caiw_path_has_no_v1_delta_normalization():
    source=__import__("pathlib").Path("algorithm/modular_mappo/trainer.py").read_text(encoding="utf-8")
    body=source[source.index("def _caiw_default_metrics"):source.index("def _loss_step",source.index("def _caiw_default_metrics"))]
    assert "normalize_iw_deltas(" not in body and "values-values.mean()" not in body
