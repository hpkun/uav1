"""Real-width MATD3 math, CUDA optimization, protocol and rollout integration."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import yaml
from algorithm.matd3.trainer import MATD3Trainer, critic_target, polyak_update
from algorithm.matd3.factory import build_matd3_trainer
from algorithm.matd3.protocol import validate_config, validate_checkpoint, require_cuda
from algorithm.matd3.runner import MATD3TrainingRunner
from algorithm.matd3.evaluation import evaluate, evaluate_checkpoint
from algorithm.maddpg.replay_buffer import ReplayBatch, JointReplayBuffer
from algorithm.maddpg.noise import OUNoise
from algorithm.maddpg.schedule import SampledTransitionSchedule
from algorithm.common.vector_env import ParallelVectorEnv, VectorStep
from env.factory import make_combat_environment

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return tuple(yaml.safe_load((ROOT/'configs'/name).read_text(encoding='utf-8')) for name in
                 ('combat_environment_v341.yaml', 'matd3_4v4_v341.yaml'))


def batch(size=16):
    rng = np.random.default_rng(17)
    return ReplayBatch(rng.normal(size=(size,4,45)).astype('f'), rng.uniform(-1,1,(size,4,3)).astype('f'),
        rng.normal(size=(size,4)).astype('f'), rng.normal(size=(size,4,45)).astype('f'),
        np.zeros(size,bool), np.ones((size,4),'f'), np.ones((size,4),'f'))


@pytest.fixture
def trainer():
    require_cuda('cuda')  # Explicit failure, never silently substitute CPU or skip.
    return MATD3Trainer(45,3,4,device='cuda',seed=19)


def vector(module):
    return torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()


def test_real_network_architecture_independence_and_initial_targets(trainer):
    t = trainer
    groups = [t.actors,t.critics,t.critics2,t.target_actors,t.target_critics,t.target_critics2]
    modules = [m for group in groups for m in group]
    pointers = [p.data_ptr() for m in modules for p in m.parameters()]
    assert len(pointers) == len(set(pointers))
    assert len(modules) == 24
    assert not torch.equal(vector(t.critics[0]),vector(t.critics2[0]))
    assert len({id(o) for o in t.actor_optimizers+t.critic_optimizers+t.critic2_optimizers}) == 12
    for group, targets in ((t.actors,t.target_actors),(t.critics,t.target_critics),(t.critics2,t.target_critics2)):
        for online,target in zip(group,targets):
            assert torch.equal(vector(online),vector(target))
            assert all(not p.requires_grad for p in target.parameters())
    b = t.tensors(batch())
    assert t.actors[0](b['observations'][:,0]).shape == (16,3)
    assert t.critics[0].joint_input(b['observations'],b['actions'],b['alive_masks']).shape == (16,192)
    assert t.critics2[0](b['observations'],b['actions'],b['alive_masks']).shape == (16,)
    assert t.parameter_counts() == dict(actor_per_agent=78339,actors_trainable=313356,
        critic_per_agent=115457,critics_trainable=923656,trainable_total=1237012,target_total_nontrainable=1237012)


def test_same_agent_min_target_terminal_and_death_reward(trainer):
    t = trainer
    with torch.no_grad():
        for group, values in ((t.target_critics,(1,8,3,10)),(t.target_critics2,(5,2,7,4))):
            for q,value in zip(group,values):
                for p in q.parameters(): p.zero_()
                q.network[-1].bias.fill_(value)
    b = t.tensors(batch(4)); b['rewards'].fill_(6)
    b['next_alive_masks'][1,0] = 0
    b['dones'][2:] = 1  # terminated and truncated share the project's done convention
    result = t.targets(b)
    expected = 6+t.gamma*t.tensor([1,2,3,4])
    torch.testing.assert_close(result[0],expected)
    assert result[1,0] == result[2,0] == result[3,0] == 6
    assert not result.requires_grad
    assert all(p.grad is None for group in (t.target_actors,t.target_critics,t.target_critics2) for p in group.parameters())


def test_smoothing_gaussian_clip_order_rng_isolation_and_reproducibility(trainer):
    t = trainer; b = t.tensors(batch(64)); alive=b['next_alive_masks']; alive[:,3]=0
    with torch.no_grad():
        for actor in t.target_actors:
            for p in actor.parameters(): p.zero_()
            actor.network[-2].bias.fill_(2)
    state = t.target_smoothing_rng.get_state()
    oracle = torch.Generator(device='cuda'); oracle.set_state(state)
    raw = torch.randn((64,4,3),generator=oracle,device='cuda')*.2
    means = t.target_actors[0](b['next_observations'][:,0])[:,None].expand(-1,4,-1)
    expected = (means+raw.clamp(-.5,.5)).clamp(-1,1); expected[:,3]=0
    global_cpu = torch.get_rng_state().clone(); global_cuda=torch.cuda.get_rng_state().clone()
    np_state=deepcopy(np.random.get_state()); ou=OUNoise(2,4,3,5); ou_state=deepcopy(ou.state_dict())
    actual=t.smoothed_target_actions(b['next_observations'],alive)
    assert torch.equal(actual,expected)
    assert torch.equal(global_cpu,torch.get_rng_state()) and torch.equal(global_cuda,torch.cuda.get_rng_state())
    assert np.array_equal(np_state[1],np.random.get_state()[1])
    assert np.array_equal(ou_state['state'],ou.state_dict()['state'])
    assert ou_state['rng_states']==ou.state_dict()['rng_states']
    t.target_smoothing_rng.set_state(state)
    assert torch.equal(actual,t.smoothed_target_actions(b['next_observations'],alive))
    assert torch.all(actual.abs()<=1) and actual[:,3].count_nonzero()==0
    assert t._smoothing_metrics['target_noise_clip_fraction']>0


def test_delay_two_actual_steps_frozen_actors_and_polyak(trainer):
    t=trainer; b=batch()
    actors=vector(t.actors); targets=[vector(g) for g in (t.target_actors,t.target_critics,t.target_critics2)]
    q1=vector(t.critics); q2=vector(t.critics2)
    m1=t.update(b)
    assert (t.critic_update_count,t.actor_update_count)==(1,0)
    assert torch.equal(actors,vector(t.actors))
    assert not torch.equal(q1,vector(t.critics)) and not torch.equal(q2,vector(t.critics2))
    assert all(torch.equal(old,vector(g)) for old,g in zip(targets,(t.target_actors,t.target_critics,t.target_critics2)))
    m2=t.update(b)
    assert (t.critic_update_count,t.actor_update_count)==(2,1)
    assert not torch.equal(actors,vector(t.actors))
    for old,online,target in zip(targets,(t.actors,t.critics,t.critics2),(t.target_actors,t.target_critics,t.target_critics2)):
        torch.testing.assert_close(vector(target),old*(1-t.tau)+vector(online)*t.tau,rtol=1e-5,atol=1e-8)
    assert t.actor_updates_per_agent==[1]*4
    assert t.critic_updates_per_agent==t.critic2_updates_per_agent==[2]*4
    assert m1['actor_loss'] is None and np.isfinite(m2['actor_loss'])
    for m in (m1,m2):
        for name in ('q1','q2'):
            assert max(m[name+'_grad_norm_postclip']) <= .500001


def test_actor_uses_only_q1_replay_others_and_freezes_both_critics(trainer):
    t=trainer; b=t.tensors(batch()); before=[vector(a) for a in t.actors]
    qs=[vector(t.critics),vector(t.critics2)]
    original=t.critics[1].forward; captured=[]
    def checked(obs,actions,alive):
        captured.append(actions.detach().clone())
        assert all(not p.requires_grad for q in (t.critics[1],t.critics2[1]) for p in q.parameters())
        return original(obs,actions,alive)
    t.critics[1].forward=checked
    t.critics2[1].forward=lambda *a: pytest.fail('Q2 must never enter actor loss')
    t.update_actor(1,b)
    for i,a in enumerate(t.actors):
        assert torch.equal(before[i],vector(a)) == (i!=1)
        if i!=1: assert all(p.grad is None for p in a.parameters())
    for i in (0,2,3): assert torch.equal(captured[0][:,i],b['actions'][:,i])
    assert torch.equal(qs[0],vector(t.critics)) and torch.equal(qs[1],vector(t.critics2))
    assert all(p.grad is None for group in (t.critics,t.critics2) for p in group.parameters())


def test_frozen_q1_actor_gradient_direction_real_network(trainer):
    t=trainer; b=t.tensors(batch()); q=t.critics[0]
    # Configure a positive path from own action[0] through the full 256-wide Q1 MLP.
    with torch.no_grad():
        for p in q.parameters():p.zero_()
        q.network[0].weight[0,180]=1; q.network[0].bias[0]=2
        q.network[2].weight[0,0]=1; q.network[4].weight[0,0]=1
    before=t.actors[0](b['observations'][:,0]).detach()[:,0].mean()
    t.update_actor(0,b)
    after=t.actors[0](b['observations'][:,0]).detach()[:,0].mean()
    assert after>before


@pytest.mark.parametrize('twin',[1,2])
def test_fixed_batch_critic_optimization_independent_gradients_and_clipping_before_step(trainer,twin):
    t=trainer; b=t.tensors(batch()); target=torch.full((16,4),5.,device='cuda')
    q=(t.critics if twin==1 else t.critics2)[0]
    other=t.critics2 if twin==1 else t.critics
    other_before=vector(other); actors=vector(t.actors)
    optimizer=(t.critic_optimizers if twin==1 else t.critic2_optimizers)[0]
    step=optimizer.step; norms=[]
    def checked_step(*a,**kw):
        norms.append(float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in q.parameters()]))))
        return step(*a,**kw)
    optimizer.step=checked_step
    initial=float(((q(b['observations'],b['actions'],b['alive_masks'])-5)**2).mean().detach())
    for _ in range(12):t.update_critic(0,b,target,twin)
    final=float(((q(b['observations'],b['actions'],b['alive_masks'])-5)**2).mean().detach())
    assert final<initial and len(norms)==12 and max(norms)<=.500001
    assert torch.equal(actors,vector(t.actors)) and torch.equal(other_before,vector(other))


def test_dead_slots_masks_empty_batches_and_death_step_optimized(trainer):
    t=trainer; b=batch(); b.next_alive_masks[:,0]=0
    b.rewards[:,0]=8; b.alive_masks[:,3]=0; b.next_alive_masks[:,3]=0
    b.observations[:,3]=np.nan; b.actions[:,3]=np.nan; b.next_observations[:,3]=np.nan
    dead_actor=vector(t.actors[3]); dead_q1=vector(t.critics[3]); dead_q2=vector(t.critics2[3])
    assert torch.all(t.targets(t.tensors(b))[:,0]==8)
    t.update(b); t.update(b)
    assert t.critic_updates_per_agent==t.critic2_updates_per_agent==[2,2,2,0]
    assert t.actor_updates_per_agent==[1,1,1,0]
    assert all(torch.equal(a,vector(m)) for a,m in zip((dead_actor,dead_q1,dead_q2),(t.actors[3],t.critics[3],t.critics2[3])))
    b.alive_masks[:]=0; b.next_alive_masks[:]=0
    for _ in range(2):
        m=t.update(b); assert m['critic_loss']==0
    assert t.critic_updates_per_agent==[2,2,2,0]


def test_polyak_direction_exact(trainer):
    with torch.no_grad():
        for p in trainer.actors.parameters():p.fill_(4)
        for p in trainer.target_actors.parameters():p.fill_(2)
    polyak_update(trainer.actors,trainer.target_actors,.001)
    assert all(torch.allclose(p,torch.full_like(p,2.002)) for p in trainer.target_actors.parameters())


def test_delay_counter_continues_across_events_and_replay_is_resampled(trainer):
    _,c=configs();c['training'].update(learning_starts=16,minibatch_size=16,gradient_updates_per_event=3)
    r=object.__new__(MATD3TrainingRunner);r.algorithm_config=c;r.trainer=trainer
    r.schedule=SampledTransitionSchedule(16,100);r.replay=JointReplayBuffer(100,4,45,seed=5)
    b=batch(64);r.replay.add_batch(*vars(b).values());r.write_metrics=lambda *a:None
    r.current_exploration_scale=lambda:1.
    sample=r.replay.sample;draws=[]
    def sampled(size):
        b=sample(size);draws.append(b.observations.copy());return b
    r.replay.sample=sampled
    r.schedule.add(0,116);r.update_from_credit()
    assert (trainer.critic_update_count,trainer.actor_update_count)==(3,1)
    r.schedule.add(116,216);r.update_from_credit()
    assert (trainer.critic_update_count,trainer.actor_update_count)==(6,3)
    assert len(draws)==6 and all(not np.array_equal(a,b) for a,b in zip(draws,draws[1:]))


def test_actor_clipping_occurs_before_optimizer_step(trainer):
    t=trainer;b=t.tensors(batch());q=t.critics[0]
    with torch.no_grad():q.network[-1].weight.mul_(10000)
    optimizer=t.actor_optimizers[0];step=optimizer.step;norms=[]
    def checked(*a,**kw):
        norms.append(float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in t.actors[0].parameters()]))))
        return step(*a,**kw)
    optimizer.step=checked;t.update_actor(0,b)
    assert len(norms)==1 and norms[0]<=.500001
    assert t._actor_diagnostics[0]['pre']>.5


@pytest.mark.parametrize('num_envs',[1,16,32])
def test_transition_schedule_three_million_counts(num_envs):
    schedule=SampledTransitionSchedule(10000,100)
    for previous in range(0,3000000,num_envs):
        schedule.add(previous,min(previous+num_envs,3000000))
        while schedule.credit>=100:schedule.consume()
    assert (schedule.event_count,schedule.event_count//2,schedule.credit)==(29900,14950,0)


def test_runner_stores_executed_actions_true_transition_next_and_both_done_kinds():
    b=batch(2); runner=object.__new__(MATD3TrainingRunner)
    runner.replay=JointReplayBuffer(10,4,45)
    result=VectorStep(np.full_like(b.observations,99),b.next_observations,b.rewards,
        np.array([True,False]),np.array([False,True]),
        [{'executed_red_actions':a.copy()} for a in b.actions],b.alive_masks,b.next_alive_masks)
    runner.store_vector_step(b.observations,b.actions,result)
    saved=runner.replay.get_batch([0,1])
    assert np.array_equal(saved.actions,b.actions) and np.array_equal(saved.next_observations,b.next_observations)
    assert saved.dones.all()


def test_ou_behavior_clip_and_dead_actions(trainer):
    r=object.__new__(MATD3TrainingRunner); _,r.algorithm_config=configs();r.trainer=trainer;r.noise=OUNoise(2,4,3,3)
    r.noise.state.fill(100)
    b=batch(2); b.alive_masks[:,2]=0
    a=r.behavior_actions(b.observations,b.alive_masks)
    assert np.all(np.abs(a)<=1) and np.all(a[:,2]==0)


def test_real_vector_reset_boundary_observation():
    e,_=configs()
    with_env=ParallelVectorEnv(2,e,1)
    try:
        with_env.reset()
        for _ in range(e['simulation']['max_steps']):
            result=with_env.step_batch(np.zeros((2,4,3),'f'))
            done=result.terminated|result.truncated
            if done.any():break
        assert done.any()
        assert not np.array_equal(result.observations[done],result.transition_next_observations[done])
        assert result.alive_masks.shape==result.next_alive_masks.shape==(2,4)
    finally:with_env.close()


def test_maddpg_v341_identity_only_delta():
    from algorithm.maddpg.protocol import validate_config as validate
    e,_=configs()
    c=yaml.safe_load((ROOT/'configs/maddpg_4v4_v341.yaml').read_text())
    old=yaml.safe_load((ROOT/'configs/maddpg_4v4_v339.yaml').read_text())
    validate(e,c);c['environment']['environment_version']='3.39'; assert c==old


def test_maddpg_v341_cuda_checkpoint_roundtrip_and_fingerprints(tmp_path):
    from algorithm.maddpg.factory import build_maddpg_trainer
    from algorithm.maddpg.protocol import schedule_contract,validate_checkpoint as validate
    from algorithm.common.protocol import config_sha256
    e,_=configs();c=yaml.safe_load((ROOT/'configs/maddpg_4v4_v341.yaml').read_text())
    t=build_maddpg_trainer(c);sample=batch();t.update(sample)
    # Synthetic schedule-consistent provenance for this one bounded optimizer round.
    t.sampled_steps=10112;t.vector_steps=632
    extra={'environment_version':'3.41','environment_config':e,'algorithm_config':c,
        'environment_config_sha256':config_sha256(e),'algorithm_config_sha256':config_sha256(c),
        **{k:getattr(t,k) for k in ('observation_dim','action_dim','num_agents','gamma','tau')},
        'training_seed':1,'training_num_envs':16,'training_total_sampled_steps':3000000,'training_smoke':False,
        'network_architecture':t.network_architecture,'parameter_counts':t.parameter_counts(),
        'ou_noise_state':OUNoise(16,4,3,4).state_dict(),'transitions_since_update':12,'update_event_count':1,
        **schedule_contract(c)}
    t.save(tmp_path/'maddpg.pt',extra)
    loaded=build_maddpg_trainer(c);loaded.load(tmp_path/'maddpg.pt',e,c)
    np.testing.assert_array_equal(t.act(sample.observations,sample.alive_masks),loaded.act(sample.observations,sample.alive_masks))
    state=torch.load(tmp_path/'maddpg.pt',map_location='cpu',weights_only=False)
    validate(state,e,c)
    for key in ('environment_config_sha256','algorithm_config_sha256'):
        corrupted=deepcopy(state);corrupted['extra'][key]='0'*64
        with pytest.raises(RuntimeError,match=key):validate(corrupted,e,c)


@pytest.mark.parametrize('case',['algorithm','identity','dimensions','hidden','policy_delay','noise','optimizer','entropy','attention'])
def test_invalid_config_rejected(case):
    e,c=configs()
    if case=='algorithm':c['algorithm']='maddpg'
    elif case=='identity':c.pop('environment')
    elif case=='dimensions':c['network']['observation_dim']=44
    elif case=='hidden':c['network']['actor_hidden_layers']=[128,128]
    elif case=='policy_delay':c['training']['policy_delay']=0
    elif case=='noise':c['training']['target_policy_noise']=float('nan')
    elif case=='optimizer':c['training']['optimizer']='SGD'
    elif case=='entropy':c['training']['alpha']=.1
    else:c['network']['attention_heads']=2
    with pytest.raises((ValueError,RuntimeError)):validate_config(e,c)


def test_cuda_required_and_exact_resume_refused(trainer):
    with pytest.raises(RuntimeError,match='CUDA'):require_cuda('cpu')
    with pytest.raises(RuntimeError,match='unsupported'):trainer.resume()


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    e,c=configs();out=tmp_path_factory.mktemp('matd3_cuda_smoke')/'run'
    r=MATD3TrainingRunner(e,c,num_envs=2,total_sampled_steps=512,output_dir=out,smoke=True)
    summary=r.run()
    return e,r.algorithm_config,out,summary


def test_complete_cuda_smoke_artifacts_counts_drift_and_finite_metrics(smoke):
    e,c,out,summary=smoke
    assert summary['sampled_steps']==summary['replay_size']==512
    assert summary['vector_steps']==256
    assert summary['critic_update_count']==3 and summary['actor_update_count']==1
    assert summary['critic_updates_per_agent']==summary['critic2_updates_per_agent']==[3]*4
    assert summary['actor_updates_per_agent']==[1]*4
    for key in ('actor_parameter_drift_l2','critic_parameter_drift_l2','critic2_parameter_drift_l2'):
        assert min(summary['final_parameter_diagnostics'][key])>0
    for name in ('env_config.yaml','algorithm_config.yaml','requested_algorithm_config.yaml','run_config.json',
                 'training_metrics.jsonl','optimization_metrics.jsonl','evaluation_history.csv',
                 'best_eval.pt','latest.pt','final.pt','run_summary.json'):
        assert (out/name).is_file()
    assert c['network']['actor_hidden_layers']==[256,256] and c['training']['num_train_envs']==2
    assert json.loads((out/'run_config.json').read_text())['requested_algorithm_config_sha256'] != summary['protocol']['algorithm_config_sha256']
    rows=[json.loads(line) for line in (out/'optimization_metrics.jsonl').read_text().splitlines()]
    assert [row['actor_updated'] for row in rows]==[False,True,False]
    assert all(np.isfinite(row['critic_loss']) for row in rows)


def test_checkpoint_exact_actions_q_rng_and_noise_free_repeated_evaluation(smoke):
    e,c,out,summary=smoke
    a=build_matd3_trainer(c,'cuda');a.load(out/'final.pt',e,c)
    b=build_matd3_trainer(c,'cuda');b.load(out/'latest.pt',e,c)
    sample=batch(); tb=a.tensors(sample)
    assert np.array_equal(a.act(sample.observations,sample.alive_masks),b.act(sample.observations,sample.alive_masks))
    for group in ('critics','critics2'):
        for qa,qb in zip(getattr(a,group),getattr(b,group)):
            assert torch.equal(qa(tb['observations'],tb['actions'],tb['alive_masks']),qb(tb['observations'],tb['actions'],tb['alive_masks']))
    assert torch.equal(a.targets(tb),b.targets(tb))
    rng=a.target_smoothing_rng.get_state().clone()
    result=evaluate(a,e,[10000000]);assert result==evaluate(b,e,[10000000])
    assert result['average_return']==summary['latest_evaluation']['average_return']
    assert torch.equal(rng,a.target_smoothing_rng.get_state())
    audited=evaluate_checkpoint(out/'final.pt',e,c,'cuda',[10000000])
    assert audited['protocol_complete'] and audited['average_return']==result['average_return']


@pytest.mark.parametrize('case',['maddpg','madsac','version','env_hash','config_hash','architecture','q2_shape','delay_count','q2_count','rng','resume'])
def test_strict_checkpoint_corruption_rejected(smoke,case):
    e,c,out,_=smoke;e,c=deepcopy(e),deepcopy(c)
    state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    if case in ('maddpg','madsac'):state['algorithm']=case
    elif case=='version':state['implementation_version']=99
    elif case=='env_hash':state['extra']['environment_config_sha256']='0'*64
    elif case=='config_hash':c=deepcopy(c);c['training']['gamma']=.98
    elif case=='architecture':state['network_architecture']['critics_per_agent']=1
    elif case=='q2_shape':state['critics2'][3]['network.0.weight']=torch.zeros(256,191)
    elif case=='delay_count':state['actor_update_count']+=1
    elif case=='q2_count':state['critic2_updates_per_agent'][0]=0
    elif case=='rng':state.pop('target_smoothing_rng_state')
    else:state['formal_exact_resume_supported']=True
    with pytest.raises(RuntimeError):validate_checkpoint(state,e,c)


def test_other_algorithms_reject_matd3_checkpoint(smoke):
    from algorithm.maddpg.protocol import validate_checkpoint as maddpg_validate
    from algorithm.madsac.protocol import validate_madsac_checkpoint
    e,_,out,_=smoke;state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    for algorithm,validator in (('maddpg',maddpg_validate),('madsac',validate_madsac_checkpoint)):
        c=yaml.safe_load((ROOT/f'configs/{algorithm}_4v4_v341.yaml').read_text())
        with pytest.raises(RuntimeError):validator(state,e,c)


def test_canonical_recording_actions_and_environment_rng_unchanged(smoke,tmp_path):
    from tools.record_combat_episode import load_recording_policy,record
    from tools.combat_visualization import RecordingCombatEnv
    e,c,out,_=smoke;trainer,_,metadata=load_recording_policy(out/'final.pt',e,c)
    noise=trainer.target_smoothing_rng.get_state().clone()
    result=record(trainer,e,10000000,tmp_path/'trace',metadata)
    saved=np.load(tmp_path/'trace/episode_trace.npz')
    env=make_combat_environment(e);obs,_=env.reset(10000000);actions=[]
    recorded_env=RecordingCombatEnv(e);recorded_env.reset(10000000)
    assert env.rng.bit_generator.state==recorded_env.rng.bit_generator.state
    while True:
        a=trainer.act(obs,env.red_alive_mask);actions.append(a)
        obs,reward,terminated,truncated,info=env.step(a)
        robs,rreward,rterminated,rtruncated,rinfo=recorded_env.step(a)
        assert env.rng.bit_generator.state==recorded_env.rng.bit_generator.state
        np.testing.assert_array_equal(obs,robs);np.testing.assert_array_equal(reward,rreward)
        assert (terminated,truncated)==(rterminated,rtruncated)
        if terminated or truncated:break
    np.testing.assert_array_equal(saved['red_actions'],np.asarray(actions))
    assert result['episode_steps']==env.steps and result['red_weapon_hits']==info['red_weapon_hits']
    assert torch.equal(noise,trainer.target_smoothing_rng.get_state())
