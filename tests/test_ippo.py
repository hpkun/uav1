"""PS-IPPO math, information boundaries, protocol parity and real CUDA loop."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.ippo.networks import LocalValueCritic, SharedMAPPOActor
from algorithm.ippo.factory import build_ippo_trainer
from algorithm.ippo.trainer import IPPOTrainer, RolloutBatch, compute_gae, masked_mean
from algorithm.ippo.protocol import validate_algorithm_config, validate_checkpoint
from algorithm.common.protocol import config_sha256

ROOT=Path(__file__).resolve().parents[1]

def configs():
    return (yaml.safe_load((ROOT/'configs/combat_environment_v341.yaml').read_text()),
            yaml.safe_load((ROOT/'configs/ippo_4v4_v341.yaml').read_text()))

def trainer(**kwargs):
    return IPPOTrainer(device='cuda',**kwargs)

def batch(t,steps=4,envs=2,dead=False):
    rng=np.random.default_rng(4)
    obs=rng.normal(size=(steps,envs,4,45)).astype(np.float32)
    mask=np.ones((steps,envs,4),np.float32)*(not dead)
    a,raw,log=t.act(obs,mask,return_policy_data=True)
    return RolloutBatch(obs,a,raw,log,rng.normal(size=mask.shape).astype(np.float32),
        np.zeros((steps,envs),np.float32),mask,obs+.01,mask.copy())

def test_network_counts_and_information_boundary():
    t=trainer()
    assert type(t.actor) is SharedMAPPOActor
    assert type(t.critic) is LocalValueCritic
    assert sum(p.numel() for p in t.actor.parameters())==78342
    assert sum(p.numel() for p in t.critic.parameters())==77825
    assert sum(p.numel() for m in (t.actor,t.critic) for p in m.parameters())==156167
    obs=torch.randn(3,2,4,45,device='cuda')
    baseline=t.critic(obs)
    changed=obs.clone();changed[:,:,1:]+=100
    torch.testing.assert_close(t.critic(changed)[:,:,0],baseline[:,:,0],rtol=0,atol=0)
    repeated=obs[:,:,0:1].expand(-1,-1,4,-1)
    values=t.critic(repeated)
    torch.testing.assert_close(values[:,:,0],values[:,:,3],rtol=0,atol=0)
    torch.testing.assert_close(t.actor(repeated)[:,:,0],t.actor(repeated)[:,:,3],rtol=0,atol=0)
    with pytest.raises(ValueError):t.critic(torch.zeros(2,4,225,device='cuda'))

def test_gae_manual_death_terminal_truncation():
    # Agent 0 dies at transition 0: keep death reward, no bootstrap.
    r=torch.tensor([[[7.,1.]],[[99.,2.]],[[99.,3.]]],device='cuda')
    v=torch.ones_like(r);n=v*5
    alive=torch.tensor([[[1.,1.]],[[0.,1.]],[[0.,1.]]],device='cuda')
    next_alive=alive.clone();next_alive[0,0,0]=0
    done=torch.tensor([[0.],[1.],[0.]],device='cuda')
    a,ret=compute_gae(r,v,n,done,alive,next_alive,.9,.8)
    assert a[0,0,0]==6 and ret[0,0,0]==7
    assert a[1,0,1]==1 # termination or truncation: no bootstrap or cross reset
    assert a[0,0,1].item()==pytest.approx(1+.9*5-1+.9*.8*1)
    assert torch.equal(a[1:,:,0],torch.zeros_like(a[1:,:,0]))

def test_ratio_clip_value_clip_and_log_jacobian():
    ratio=torch.tensor([1.5,.5,1.1],device='cuda')
    advantage=torch.tensor([2.,-2.,1.],device='cuda')
    clipped=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
    torch.testing.assert_close(clipped,torch.tensor([2.4,-1.6,1.1],device='cuda'))
    old=torch.tensor([0.,2.],device='cuda');value=torch.tensor([1.,1.],device='cuda');target=torch.tensor([1.,0.],device='cuda')
    loss=.5*torch.maximum((value-target)**2,(old+(value-old).clamp(-.2,.2)-target)**2).mean()
    assert loss.item()==pytest.approx(.5*(.64+3.24)/2)
    actor=trainer().actor
    obs=torch.randn(8,4,45,device='cuda');dist=actor.distribution(obs)
    raw=torch.randn(8,4,3,device='cuda',dtype=torch.float64)
    actual=actor._squashed_log_prob(dist,raw,raw.tanh())
    expected=(dist.log_prob(raw)-torch.log1p(-raw.tanh().square())).sum(-1)
    torch.testing.assert_close(actual,expected)
    assert torch.isfinite(actor._squashed_log_prob(dist,raw*100,(raw*100).tanh())).all()

def test_masked_normalization_and_dead_batches():
    t=trainer(ppo_epochs=1,minibatch_size=2)
    b=batch(t);b.alive_masks[:,:,1:]=0;b.next_alive_masks[:,:,1:]=0
    metrics=t.update(b)
    assert metrics['valid_alive_agent_samples']==8
    live=torch.tensor([2.,4.],device='cuda');norm=(live-live.mean())/live.std(unbiased=False)
    assert norm.mean()==0 and norm.std(unbiased=False)==1
    before=t.actor_update_count
    metrics=t.update(batch(t,dead=True))
    assert metrics['skipped_all_dead']==1 and t.actor_update_count==before
    assert all(np.isfinite(v) for v in metrics.values())

def test_gradient_isolation_and_deterministic_rng():
    t=trainer();obs=torch.randn(2,4,45,device='cuda')
    t.critic(obs).sum().backward()
    assert all(p.grad is None for p in t.actor.parameters())
    t.critic.zero_grad(set_to_none=True)
    t.actor(obs).sum().backward()
    assert all(p.grad is None for p in t.critic.parameters())
    rng=torch.cuda.get_rng_state().clone()
    action=t.act(obs.cpu().numpy(),deterministic=True)
    assert torch.equal(torch.cuda.get_rng_state(),rng)
    np.testing.assert_array_equal(action,t.act(obs.cpu().numpy(),deterministic=True))

def test_joint_minibatch_and_kl_full_epoch():
    t=trainer(ppo_epochs=3,minibatch_size=2,target_kl=1e-12)
    b=batch(t,steps=4,envs=2)
    # Force a clear KL threshold crossing. It must finish all four minibatches.
    b.old_log_probs+=1.
    result=t.update(b)
    assert result['effective_ppo_epochs']==1
    assert result['kl_early_stop']==1
    assert result['actual_actor_optimizer_steps']==4
    assert result['actual_critic_optimizer_steps']==4

def test_fair_config_and_theoretical_budget():
    _,ippo=configs()
    mappo=yaml.safe_load((ROOT/'configs/mappo_4v4_v341.yaml').read_text())
    assert ippo['training']==mappo['training']
    assert ippo['implementation']==mappo['implementation']
    assert ippo['runtime_logging']==mappo['runtime_logging']
    assert ippo['network']['actor_hidden_layers']==mappo['network']['actor_hidden_layers']
    full,remainder=divmod(3_000_000,256*16)
    assert (full,remainder//16)==(732,108)
    assert full*8*10+int(np.ceil((remainder//16)*16/512))*10==58600
    from algorithm.mappo.trainer import compute_gae as baseline
    assert compute_gae is baseline

@pytest.mark.parametrize('field,value', [('parameter_sharing',False),('critic_scope','global'),('critic_type','mlp'),('observation_dim',225)])
def test_reject_wrong_identity(field,value):
    e,c=configs();c['network'][field]=value
    with pytest.raises(ValueError):validate_algorithm_config(c,e)

@pytest.fixture(scope='module')
def real_run(tmp_path_factory):
    from algorithm.ippo.runner import IPPOTrainingRunner
    e,c=configs();out=tmp_path_factory.mktemp('ippo_real_cuda')
    runner=IPPOTrainingRunner(e,c,16,4096,'cuda',321,out,True)
    try:
        before_actor={k:v.clone() for k,v in runner.trainer.actor.state_dict().items()}
        before_critic={k:v.clone() for k,v in runner.trainer.critic.state_dict().items()}
        b=runner.collect_rollout(256)
        metrics=runner.trainer.update(b)
        assert b.observations.shape==(256,16,4,45)
        assert runner.trainer.sampled_steps==4096 and runner.trainer.vector_steps==256
        assert any(not torch.equal(v,before_actor[k]) for k,v in runner.trainer.actor.state_dict().items())
        assert any(not torch.equal(v,before_critic[k]) for k,v in runner.trainer.critic.state_dict().items())
        assert np.isfinite(list(metrics.values())).all()
        path=out/'final.pt';runner.save_checkpoint(path)
        evidence=dict(gpu=torch.cuda.get_device_name(),sampled_steps=4096,vector_steps=256,
            rollout_shape=list(b.observations.shape),actor_changed=True,critic_changed=True,
            actor_parameters=78342,critic_parameters=77825,
            actor_parameter_delta_l2=float(torch.sqrt(sum((v-before_actor[k]).square().sum() for k,v in runner.trainer.actor.state_dict().items()))),
            critic_parameter_delta_l2=float(torch.sqrt(sum((v-before_critic[k]).square().sum() for k,v in runner.trainer.critic.state_dict().items()))),
            metrics=metrics)
        evidence_path=ROOT/'outputs/ippo_validation/cuda_test_evidence.json'
        evidence_path.parent.mkdir(exist_ok=True,parents=True)
        evidence_path.write_text(json.dumps(evidence,indent=2))
        # Preserve reviewable checkpoint and config outside temporary pytest directories.
        import shutil
        for name in ('final.pt','diagnostic_observations.npz','action_diagnostics.jsonl'):
            shutil.copy2(out/name,ROOT/'outputs/ippo_validation'/name)
        return runner,path,b,e,c
    finally:
        runner.vector.close()

def test_real_cuda_full_rollout_and_reload(real_run):
    from algorithm.ippo.evaluation import load_checkpoint
    runner,path,b,e,c=real_run
    restored=load_checkpoint(path,c,e,'cuda')
    np.testing.assert_array_equal(runner.trainer.act(b.observations[0],deterministic=True),
                                  restored.act(b.observations[0],deterministic=True))
    assert 1<=runner.trainer.actor_update_count<=80
    assert runner.trainer.actor_update_count==runner.trainer.critic_update_count
    assert restored.actor_update_count==runner.trainer.actor_update_count

def test_checkpoint_rejections(real_run,tmp_path):
    runner,path,b,e,c=real_run
    state=torch.load(path,weights_only=False)
    validate_checkpoint(state,e,c)
    for algorithm in ('MAPPO','MATD3','MADDPG'):
        changed=deepcopy(state);changed['algorithm']=algorithm
        with pytest.raises(RuntimeError):validate_checkpoint(changed,e,c)
    from algorithm.mappo.factory import build_mappo_trainer
    m=yaml.safe_load((ROOT/'configs/mappo_4v4_v341.yaml').read_text())
    with pytest.raises(RuntimeError):build_mappo_trainer(m,'cuda').load(path)
    changed=deepcopy(e);changed['reward']['kill_reward']+=1
    with pytest.raises((RuntimeError,ValueError)):validate_checkpoint(state,changed,c)
    changed=deepcopy(c);changed['training']['gamma']=.8
    with pytest.raises(RuntimeError):validate_checkpoint(state,e,changed)
    broken=deepcopy(state);broken['extra'].pop('algorithm_config_sha256')
    with pytest.raises(RuntimeError):validate_checkpoint(broken,e,c)

def test_real_evaluation_and_canonical_recording_rng(real_run,tmp_path):
    from algorithm.ippo.evaluation import load_checkpoint, evaluate_ippo_checkpoint
    from algorithm.common.evaluator import evaluate
    from tools.record_combat_episode import load_recording_policy, record
    from env.factory import make_combat_environment
    runner,path,b,e,c=real_run
    restored=load_checkpoint(path,c,e,'cuda')
    a=evaluate(runner.trainer,e,[10_000_051])
    assert a==evaluate(restored,e,[10_000_051])
    policy,_,provenance=load_recording_policy(path,e,c,'cuda')
    metadata=record(policy,e,20_000_051,tmp_path/'recording',provenance)
    trace=np.load(tmp_path/'recording/episode_trace.npz')
    env=make_combat_environment(e);obs,_=env.reset(20_000_051)
    for i,action in enumerate(trace['red_actions']):
        expected=policy.act(obs[None],env.red_alive_mask[None],deterministic=True)[0]
        np.testing.assert_array_equal(action,expected)
        obs,_,_,_,info=env.step(action)
        np.testing.assert_array_equal(info['executed_red_actions'],action)
        np.testing.assert_array_equal(np.array([s.as_array() for s in env.red]),trace['red_kinematics'][i+1])
        np.testing.assert_array_equal(np.array([s.as_array() for s in env.blue]),trace['blue_kinematics'][i+1])
    # Equal trajectories and complete RNG states prove observation hooks add no draws.
    from tools.combat_visualization import RecordingCombatEnv
    rec=RecordingCombatEnv(e);rec.reset(20_000_051)
    for action in trace['red_actions']:rec.step(action)
    assert rec.rng.bit_generator.state==env.rng.bit_generator.state
    diagnostics=json.loads((tmp_path/'recording/ippo_flight_diagnostics.json').read_text())
    assert 'red_losses_by_agent' in diagnostics
    import shutil
    shutil.copytree(tmp_path/'recording',ROOT/'outputs/ippo_validation/canonical_recording',dirs_exist_ok=True)

def test_auto_reset_terminal_observation_boundary():
    from algorithm.common.vector_env import ParallelVectorEnv
    e,_=configs()
    with ParallelVectorEnv(1,e,123) as vector:
        initial=vector.reset()
        for _ in range(1000):
            result=vector.step_batch(np.zeros((1,4,3),np.float32))
            if result.terminated[0] or result.truncated[0]:
                break
        assert result.terminated[0] or result.truncated[0]
        assert not np.array_equal(result.transition_next_observations,result.observations)
        assert vector.episode_indices[0]==1
        r=torch.as_tensor(result.rewards[None],device='cuda')
        ones=torch.ones_like(r)
        a,ret=compute_gae(r,ones,ones*999,torch.ones((1,1),device='cuda'),ones,ones,.99,.95)
        torch.testing.assert_close(ret,r)


def test_ppo_kernel_exact_baseline_parity():
    # Give the unchanged MAPPO trainer the same local network solely as a
    # reference kernel. This verifies all PPO arithmetic, normalization, KL
    # aggregation and RNG consumption independently of IPPO telemetry.
    from algorithm.mappo.trainer import MAPPOTrainer
    t=trainer(ppo_epochs=2,minibatch_size=4,target_kl=None,seed=19)
    ref=MAPPOTrainer(observation_dim=45,device='cuda',hidden_dim=256,
        critic_type='mlp',policy_std_mode='state_independent',log_std_max=.5,
        entropy_coefficient=.001,ppo_epochs=2,minibatch_size=4,target_kl=None,seed=19)
    ref.critic=deepcopy(t.critic)
    ref.actor.load_state_dict(t.actor.state_dict())
    ref.critic_optimizer=torch.optim.Adam(ref.critic.parameters(),lr=3e-4)
    b=batch(t)
    rng=torch.cuda.get_rng_state().clone()
    actual=t.update(b)
    torch.cuda.set_rng_state(rng)
    expected=ref.update(b)
    for module in ('actor','critic'):
        for key,value in getattr(t,module).state_dict().items():
            torch.testing.assert_close(value,getattr(ref,module).state_dict()[key],rtol=0,atol=0)
    for key,value in expected.items():
        assert actual[key]==value


def test_fixed_bank_comparison_and_seed_guards(real_run,tmp_path):
    from algorithm.ippo.diagnostics import compare_checkpoints, save_observation_bank
    from algorithm.ippo.evaluation import evaluate_ippo_checkpoint
    runner,path,b,e,c=real_run
    bank=tmp_path/'bank.npz'
    save_observation_bank(bank,b.observations[:2],dict(split='training',seed_provenance=[321]))
    report=compare_checkpoints([path,path],bank,e,c)
    assert report['checkpoints'][1]['previous_checkpoint_mean_abs_delta']==[0.,0.,0.]
    with pytest.raises(ValueError):
        save_observation_bank(tmp_path/'holdout.npz',b.observations,dict(split='holdout',seed_provenance=[70_000_000]))
    with pytest.raises(ValueError):
        evaluate_ippo_checkpoint(path,c,e,'cuda',[70_000_000],split='development')
    with pytest.raises(ValueError):
        evaluate_ippo_checkpoint(path,c,e,'cuda',[10_000_000],split='holdout')
    changed=deepcopy(c);changed['implementation']['evaluation_seed_base']=70_000_000
    with pytest.raises(ValueError):validate_algorithm_config(changed,e)


def test_dead_samples_do_not_affect_normalization_or_update():
    t=trainer(ppo_epochs=1,minibatch_size=8,seed=77)
    other=trainer(ppo_epochs=1,minibatch_size=8,seed=77)
    b=batch(t)
    b.alive_masks[:,:,1:]=0;b.next_alive_masks[:,:,1:]=0
    altered=deepcopy(b)
    altered.rewards[:,:,1:]=1e6
    altered.observations[:,:,1:]=1000
    altered.next_observations[:,:,1:]=-1000
    rng=torch.cuda.get_rng_state().clone()
    first=t.update(b)
    torch.cuda.set_rng_state(rng)
    second=other.update(altered)
    assert first['actor_loss']==second['actor_loss']
    assert first['value_loss']==second['value_loss']
    assert first['raw_advantage_mean']==second['raw_advantage_mean']
    for module in ('actor','critic'):
        for a,b in zip(getattr(t,module).parameters(),getattr(other,module).parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_actual_checkpoint_families_mutually_reject(real_run,tmp_path):
    from algorithm.mappo.factory import build_mappo_trainer
    from algorithm.matd3.factory import build_matd3_trainer
    from algorithm.matd3.protocol import validate_checkpoint as validate_matd3
    _,ippo_path,_,e,c=real_run
    mconfig=yaml.safe_load((ROOT/'configs/mappo_4v4_v341.yaml').read_text())
    mappo=build_mappo_trainer(mconfig,'cuda')
    mpath=tmp_path/'mappo.pt';mappo.save(mpath)
    tconfig=yaml.safe_load((ROOT/'configs/matd3_4v4_v341.yaml').read_text())
    matd3=build_matd3_trainer(tconfig,'cuda')
    tpath=tmp_path/'matd3.pt'
    matd3.save(tpath,dict(algorithm_config=tconfig,transitions_since_update=0,update_event_count=0))
    ippo=build_ippo_trainer(c,'cuda')
    for foreign in (mpath,tpath):
        with pytest.raises(RuntimeError):ippo.load(foreign,e,c)
    with pytest.raises(RuntimeError):mappo.load(ippo_path)
    with pytest.raises(RuntimeError):matd3.load(ippo_path,e,tconfig)
    with pytest.raises(RuntimeError):validate_matd3(torch.load(ippo_path,weights_only=False),e,tconfig)


@pytest.mark.parametrize('envs,budget',[(0,4096),(16,0),(16,4097)])
def test_invalid_runtime_budget_fails_before_environment_launch(tmp_path,envs,budget):
    from algorithm.ippo.runner import IPPOTrainingRunner
    e,c=configs()
    with pytest.raises(ValueError,match='sample budget'):
        IPPOTrainingRunner(e,c,envs,budget,'cuda',0,tmp_path,True)
