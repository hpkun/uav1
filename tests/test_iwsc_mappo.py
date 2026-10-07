from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.modules import InterWaveCreditModule
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.networks import InterWaveStateQualityCritic
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,asymmetric_tactical_projection,normalize_iw_deltas,balanced_iw_losses,combine_iw_actor_loss
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from tools.analyze_iwsc_mappo import exact_row,nearest,primary_endpoint
import tools.smoke_iwsc_persistent_integration as real_smoke


IW={"enabled":True,"max_waves":3,"quality_critic_learning_rate":3e-4,"actor_credit_coefficient":1.,
    "replay_segments_per_wave":128,"max_states_per_segment":64,"critic_minibatch_size":8,
    "critic_updates_per_rollout":1,"min_completed_segments_per_wave":2,"min_target_std":.05,
    "huber_delta":1.,"wave_balanced_supervision":True,"wave_balanced_actor_loss":True,
    "gradient_projection":"asymmetric_tactical_preserving"}


def segment(module,wave,target,offset=0.):
    states=[{"observation":np.full((4,52),offset+i,dtype=np.float32),"alive_mask":np.ones(4,np.float32),
             "horizon":1-i/70,"boundary":i==69} for i in range(70)]
    return module.cap_segment(states,target,wave)


def rollout(trainer,segments=()):
    rng=np.random.default_rng(4);t,e,a=4,2,4
    obs=rng.normal(size=(t,e,a,52)).astype("f");nobs=obs+.01;alive=np.ones((t,e,a),"f")
    raw=rng.normal(size=(t,e,a,3)).astype("f");act=np.tanh(raw).astype("f")
    with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),torch.tensor(raw),torch.tensor(act)).numpy()
    waves=np.asarray([[1,2]]*t);ctx=np.zeros((t,e,0),"f")
    return ModularRolloutBatch(obs,act,raw,old,rng.normal(size=(t,e,a)).astype("f"),np.zeros((t,e,a),"f"),
        np.zeros((t,e),"f"),alive,nobs,alive.copy(),waves,np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"),
        remaining_horizons=np.full((t,e),.8,"f"),next_remaining_horizons=np.full((t,e),.79,"f"),
        wave_transition_flags=np.zeros((t,e),"f"),iw_supervision_segments=list(segments))


def test_targets_segment_cap_and_boundary_preservation():
    m=InterWaveCreditModule(IW)
    assert [m.target(1,x) for x in (0,2,3)]==[0,.5,1] and [m.target(2,x) for x in (2,3)]==[0,1]
    s=segment(m,1,.5);assert len(s["horizons"])<=64 and s["boundary_flags"][-1] and s["target"]==.5


def test_quality_critic_shape_range_and_wave_semantics():
    c=InterWaveStateQualityCritic(52,16,2);obs=torch.randn(5,4,52);alive=torch.ones(5,4)
    out=c(obs,alive,torch.tensor([1,1,2,2,1]),torch.linspace(0,1,5))
    assert out.shape==(5,) and torch.all((out>=0)&(out<=1))


def test_segment_balanced_sampling_and_readiness():
    m=InterWaveCreditModule(IW);m.ingest([segment(m,1,0),segment(m,1,1),segment(m,2,0),segment(m,2,1)])
    assert not m.ready(1);m.valid_update_count=1;assert m.ready(1) and m.ready(2)
    sample=m.sample_states(1,100,np.random.default_rng(8));assert sample["observations"].shape==(100,4,52)


def test_runner_pending_cache_cross_rollout_boundary_and_failure_labels():
    runner=ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    runner.trainer=type("T",(),{})();runner.trainer.inter_wave_credit=InterWaveCreditModule(IW)
    runner.iw_pending_episode=[{1:[],2:[]}];runner.iw_completed_episode_counter=0
    z=np.zeros((4,52),np.float32);alive=np.ones(4,np.float32)
    runner._iw_record_transition(0,1,z,alive,.9,z+1,alive,.8,False)
    assert len(runner.iw_pending_episode[0][1])==1  # survives an arbitrary rollout boundary
    runner._iw_record_transition(0,1,z+2,alive,.7,z+3,alive,.6,True)
    completed=runner._iw_finalize_episode(0,1)
    assert len(completed)==1 and completed[0]["target"]==0 and completed[0]["boundary_flags"].sum()==1
    assert runner.iw_pending_episode[0]=={1:[],2:[]}


def test_terminal_zero_source_wave_next_semantics_and_wave3_mask():
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=1,modules_config={"inter_wave_credit":IW})
    class Fake(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=[]
        def forward(self,o,a,w,h):self.calls.append(w.detach().clone());return .1*w.float()+h
    fake=Fake();trainer.iw_critic=fake
    batch=rollout(trainer);batch.dones[0,0]=1
    obs=torch.tensor(batch.observations);nobs=torch.tensor(batch.next_observations);alive=torch.tensor(batch.alive_masks);waves=torch.tensor(batch.wave_indices)
    delta,active,_=trainer._iw_rollout_credit(batch,obs,nobs,alive,alive,torch.tensor(batch.dones),waves)
    assert delta[0,0].item()==pytest.approx(-.9,abs=1e-6) and not active.any()
    assert torch.equal(fake.calls[0],fake.calls[1])  # q_next uses source, not destination, wave identity
    batch.wave_indices[:]=3;_,active,_=trainer._iw_rollout_credit(batch,obs,nobs,alive,alive,torch.tensor(batch.dones),torch.tensor(batch.wave_indices))
    assert not active.any()


def test_asymmetric_projection_preserves_tactical_and_removes_conflict():
    tactical=[torch.tensor([1.,0.])];aux=[torch.tensor([-2.,1.])]
    projected,dot=asymmetric_tactical_projection(tactical,aux)
    assert dot<0 and torch.dot(tactical[0],projected[0])==pytest.approx(0,abs=1e-6)
    assert torch.equal(tactical[0],torch.tensor([1.,0.]))


def test_delta_telescoping_per_wave_normalization_and_equal_wave_loss():
    q=torch.tensor([.2,.4,.7]);assert torch.diff(q).sum().item()==pytest.approx((q[-1]-q[0]).item())
    delta=torch.tensor([[1.,10.],[3.,14.]]);waves=torch.tensor([[1,2],[1,2]])
    normalized,active=normalize_iw_deltas(delta,waves,{1:True,2:True})
    assert active.all() and normalized[waves==1].mean()==pytest.approx(0,abs=1e-6) and normalized[waves==2].std(unbiased=False)==pytest.approx(1,abs=1e-6)
    assert balanced_iw_losses([torch.tensor(2.),torch.tensor(8.)]).item()==5


def test_actor_balance_true_false_and_single_wave_semantics():
    surrogate=torch.cat((torch.ones(100,1),torch.full((10,1),10.0)));alive=torch.ones_like(surrogate)
    waves=torch.cat((torch.ones(100,dtype=torch.long),torch.full((10,),2,dtype=torch.long)));active=torch.ones(110,dtype=torch.bool)
    balanced=combine_iw_actor_loss(surrogate,alive,active,waves,True)
    pooled=combine_iw_actor_loss(surrogate,alive,active,waves,False)
    assert balanced.item()==pytest.approx(-.5*(1+10))
    assert pooled.item()==pytest.approx(-(100+100)/110) and pooled.item()!=balanced.item()
    only1=waves==1
    assert combine_iw_actor_loss(surrogate,alive,only1,waves,True)==combine_iw_actor_loss(surrogate,alive,only1,waves,False)


def test_exact_primary_endpoint_and_intermediate_nearest(tmp_path):
    values=[{"sampled_steps":"2900000"},{"sampled_steps":"3000000"},{"sampled_steps":"3100000"}]
    assert exact_row(values,3000000)["sampled_steps"]=="3000000"
    assert nearest(values,2920000)["sampled_steps"]=="2900000"
    with pytest.raises(RuntimeError,match="IWSC_PRIMARY_ENDPOINT_3M_MISSING"):exact_row([values[0],values[2]],3000000)
    (tmp_path/"run_summary.json").write_text('{"sampled_steps":3000000}',encoding="utf-8")
    with pytest.raises(RuntimeError,match="IWSC_DEVELOPMENT_INCOMPLETE"):primary_endpoint(tmp_path,[values[0]])
    (tmp_path/"run_summary.json").write_text('{"sampled_steps":2900000}',encoding="utf-8")
    with pytest.raises(RuntimeError,match="IWSC_DEVELOPMENT_INCOMPLETE"):primary_endpoint(tmp_path,[values[1]])


def test_real_smoke_static_protocol_guards():
    assert real_smoke.ENV_CONFIG_PATH=="configs/persistent_wave_v2_environment.yaml"
    env=__import__("yaml").safe_load(Path(real_smoke.ENV_CONFIG_PATH).read_text(encoding="utf-8"))
    assert env["persistent_waves"]["total_waves"]==3 and env["simulation"]["max_steps"]==3000
    assert all("best" not in path.lower() for path in real_smoke.BEHAVIOR_CHECKPOINT_CANDIDATES)
    assert real_smoke.ENGINEERING_SMOKE_SEED>=8801000 and all(not lo<=real_smoke.ENGINEERING_SMOKE_SEED<=hi for lo,hi in real_smoke.FORBIDDEN_SEED_RANGES)
    assert real_smoke.ACTOR_OPTIMIZER_STEPS_ALLOWED is False and real_smoke.MAX_VECTOR_STEPS==4096


def test_same_seed_plain_initialization_and_rng_isolation_exact():
    common={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    plain=ModularMAPPOTrainer(hidden_dim=16,seed=5301,modules_config=common)
    iw=ModularMAPPOTrainer(hidden_dim=16,seed=5301,modules_config={**common,"inter_wave_credit":IW})
    assert all(torch.equal(v,iw.actor.state_dict()[k]) for k,v in plain.actor.state_dict().items())
    assert all(torch.equal(v,iw.critic.state_dict()[k]) for k,v in plain.critic.state_dict().items())
    assert plain.rng.bit_generator.state==iw.rng.bit_generator.state


def test_cold_gate_uses_exact_plain_update_path():
    common={"actor_lr_decay":{"enabled":True,"start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
    plain=ModularMAPPOTrainer(hidden_dim=16,seed=9,ppo_epochs=1,minibatch_size=8,modules_config=common)
    iw=ModularMAPPOTrainer(hidden_dim=16,seed=9,ppo_epochs=1,minibatch_size=8,modules_config={**common,"inter_wave_credit":IW})
    batch=rollout(plain);batch_iw=deepcopy(batch);state=torch.random.get_rng_state();plain.update(batch);torch.random.set_rng_state(state);metrics=iw.update(batch_iw)
    assert metrics["iw_actor_active"]==0
    assert all(torch.equal(a,b) for a,b in zip(plain.actor.parameters(),iw.actor.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(plain.critic.parameters(),iw.critic.parameters()))


def test_ready_actor_update_wave3_mask_terminal_and_checkpoint_roundtrip(tmp_path):
    iw=ModularMAPPOTrainer(hidden_dim=16,seed=3,ppo_epochs=1,minibatch_size=8,modules_config={"inter_wave_credit":IW})
    segments=[segment(iw.inter_wave_credit,w,t,i) for w in (1,2) for i,t in enumerate((0.,1.))]
    batch=rollout(iw,segments);metrics=iw.update(batch)
    assert metrics["iw_actor_active"]==1 and metrics["iw_ready_wave1"]==1 and metrics["iw_ready_wave2"]==1
    assert np.isfinite(list(metrics.values())).all()
    path=tmp_path/"iw.pt";iw.save(path);restored=ModularMAPPOTrainer(hidden_dim=16,seed=3,ppo_epochs=1,minibatch_size=8,modules_config={"inter_wave_credit":IW});restored.load(path)
    assert len(restored.inter_wave_credit.replay[1])==2 and restored.inter_wave_credit.valid_update_count>0
    assert restored.iw_rng.bit_generator.state==iw.iw_rng.bit_generator.state


def test_future_final_range_is_never_an_execution_seed():
    manifest=Path("experiments/iwsc_mappo_manifest.json").read_text(encoding="utf-8")
    assert '"future_final_executed": false' in manifest and "45000000" in manifest
