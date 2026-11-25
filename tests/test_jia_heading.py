"""Speed-heading radial dynamics and MADDPG contract checks."""
from dataclasses import replace
import json
import numpy as np
import pytest

from algorithm.maddpg import BoxActor, DiskActor
from algorithm.maddpg_config import NormalizedMADDPGConfig
from env import make_environment
from env.jia_heading import rk4_pose, wrap_heading
from experiment.interception import ReplayBuffer, evaluate_policy, load_run, make_policy, read_config, train
from tools.replay import collect, validate_trace

ENV='configs/interception_radial_5v5_jia_heading.yaml'
ALG='configs/maddpg_jia_heading_interval10.yaml'


def test_rk4_straight_and_circle():
    pos=np.zeros(2);heading=0.0
    for _ in range(10):pos,heading=rk4_pose(pos,heading,10,0,.1)
    np.testing.assert_allclose(pos,[10,0],atol=1e-12)
    assert heading==pytest.approx(0)
    pos=np.zeros(2);heading=0.0
    for _ in range(10):pos,heading=rk4_pose(pos,heading,10,.5,.1)
    np.testing.assert_allclose(pos,[20*np.sin(.5),20*(1-np.cos(.5))],atol=1e-8)
    assert heading==pytest.approx(.5)


def test_action_mapping_masks_and_observation_contract():
    config,parameters,_=read_config(ENV,ALG)
    env=make_environment(config);obs,info=env.reset(seed=7)
    assert obs.shape==(5,26) and info['central_state'].shape==(85,)
    assert env.contract['action_schema']=='normalized_speed_turn_box'
    assert env.contract['version']=='interception_13'
    assert np.allclose(np.linalg.norm(env.velocities[0],axis=1),6)
    assert np.allclose(np.linalg.norm(env.velocities[1],axis=1),6)
    np.testing.assert_allclose(obs[:,-2:],np.column_stack((np.cos(env.headings[0]),np.sin(env.headings[0]))))
    actions=np.array([[-1,-1],[0,0],[1,1],[0,-1],[1,0]],np.float32)
    old_position=env.positions[0,4].copy()
    env.defender_alive[4]=False
    env.velocities[0,4]=0
    obs,individual,_,_,info=env.step(actions)
    np.testing.assert_allclose(info['physical_commands'][0,:3],[[6,-.5],[8,0],[10,.5]])
    np.testing.assert_array_equal(info['physical_commands'][0,4],0)
    np.testing.assert_array_equal(env.positions[0,4],old_position)
    np.testing.assert_array_equal(obs[4],0)
    np.testing.assert_array_equal(info['central_state'][-2:],0)
    assert info['local_rewards'][4]==0
    np.testing.assert_array_equal(info['executed_actions'][0,4],0)
    assert np.isclose(sum(info['reward_components'].values()),info['team_reward'])
    np.testing.assert_allclose(individual,info['individual_rewards'],atol=1e-5)
    with pytest.raises(ValueError,match='normalized'):
        env.step(np.full((5,2),1.1))
    policy=make_policy('maddpg',config,parameters,'cpu',7)
    assert isinstance(policy.actors[0],BoxActor)
    assert policy.critics[0].net[0].in_features==95
    sampled=policy.policy_actions(obs[None],info['active_masks'][None],deterministic=False)
    assert np.max(np.abs(sampled))<=1
    np.testing.assert_array_equal(sampled[0,4],0)


def test_heading_reward_and_capture_event():
    config,_,_=read_config(ENV,ALG)
    env=make_environment(config);env.reset(seed=7)
    env.positions[0,0]=[0,0];env.positions[1,0]=[100,0]
    for angle,expected in ((0,.3),(np.pi/2,0),(np.pi,-.3)):
        env.headings[0,0]=angle
        assert config.heading_weight*env._heading_alignment(0,0,None,None)==pytest.approx(expected,abs=1e-12)
    env.reset(seed=7)
    env.positions[0,0]=env.positions[1,0].copy()
    _,_,_,_,info=env.step(np.zeros((5,2)))
    assert any(e['type']=='capture' and e['defender']==0 for e in info['events'])
    assert not env.defender_alive[0] and not env.attacker_alive[0]
    assert info['reward_components']['capture']==20
    assert 'heading' in info['reward_components']
    assert np.isclose(sum(info['reward_components'].values()),info['team_reward'])


def test_normalized_replay_actions_and_legacy_disk():
    config,parameters,_=read_config(ENV,ALG)
    env=make_environment(config);obs,info=env.reset(seed=7)
    policy=make_policy('maddpg',config,parameters,'cpu',7)
    actions=policy.policy_actions(obs[None],info['active_masks'][None],deterministic=False)[0]
    state=info['central_state'].copy();mask=info['active_masks'].copy()
    next_obs,reward,terminated,truncated,info=env.step(actions)
    np.testing.assert_allclose(info['executed_actions'][0],actions,atol=1e-6)
    np.testing.assert_allclose(info['physical_commands'][0,:,0],6+2*(actions[:,0]+1),atol=1e-6)
    np.testing.assert_allclose(info['physical_commands'][0,:,1],.5*actions[:,1],atol=1e-6)
    buffer=ReplayBuffer(8,7)
    buffer.add(observations=obs,next_observations=next_obs,states=state,
               next_states=info['central_state'],actions=actions,rewards=reward,
               dones=float(terminated or truncated),active_masks=mask,next_active_masks=info['active_masks'])
    batch=buffer.sample(1)
    np.testing.assert_allclose(batch['actions'][0],actions)
    checked=policy._batch(batch)
    np.testing.assert_allclose(checked['actions'].cpu().numpy()[0],actions)
    assert np.isfinite(policy.critics[0](checked['states'],checked['actions']).detach().numpy()).all()
    bad={k:v.copy() for k,v in batch.items()};bad['actions'][0,0,0]=1.1
    with pytest.raises(ValueError,match='normalized'):
        policy._batch(bad)
    old,old_parameters,_=read_config('configs/interception_ideal_radial_straight.yaml',
                                      'configs/maddpg_low_level_interval10.yaml')
    old_env=make_environment(old);old_env.reset(seed=7)
    old_policy=make_policy('maddpg',old,old_parameters,'cpu',7)
    assert old_env.contract['action_schema']=='world_acceleration_disk_si'
    assert isinstance(old_policy.actors[0],DiskActor)
    assert old_env.observation_dim==24 and old_env.state_dim==75
    old_env.step(np.zeros((5,2)))


def test_complete_deterministic_pursuit_and_reward_ledger():
    config,_,_=read_config(ENV,ALG)
    results=[]
    for seed in (7,17,27,37,47):
        env=make_environment(config);env.reset(seed=seed)
        totals={}
        while not env.finished:
            actions=np.zeros((5,2))
            for i in np.flatnonzero(env.defender_alive):
                target=env.assignment.targets[i]
                if target<0:continue
                delta=env.positions[1,target]-env.positions[0,i]
                error=wrap_heading(np.arctan2(delta[1],delta[0])-env.headings[0,i])
                actions[i,0]=-1 if abs(error)>.35 else 1
                actions[i,1]=np.clip(4*error,-1,1)
            _,_,_,_,info=env.step(actions)
            for name,value in info['reward_components'].items():totals[name]=totals.get(name,0)+value
        assert (env.captured_count,env.penetrated_count)==(5,0)
        assert np.isclose(sum(totals.values()),env.team_return)
        assert np.isclose(env.team_return,info['episode_summary']['team_return'])
        results.append((env.first_capture_time_s,env.step_count,totals['heading']))
    assert len(results)==5


def test_unopposed_breach_once_per_attacker():
    config,_,_=read_config(ENV,ALG)
    env=make_environment(config);env.reset(seed=7)
    components={};events=[]
    while not env.finished:
        _,_,_,_,info=env.step(np.zeros((5,2)))
        events.extend(info['events'])
        for key,value in info['reward_components'].items():components[key]=components.get(key,0)+value
    assert env.captured_count==0 and env.penetrated_count==5
    assert sum(event['type']=='penetration' for event in events)==5
    assert components['penetration']==-250
    assert np.isclose(sum(components.values()),env.team_return)


def test_short_train_checkpoint_eval_and_replay(tmp_path):
    config,_,_=read_config(ENV,ALG)
    parameters=NormalizedMADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=8,
                                      learning_starts=2,update_interval_steps=2,updates_per_interval=1,
                                      exploration_std_mps2=0)
    output=tmp_path/'jia_heading'
    train(replace(config,episode_steps=2),parameters,2,'maddpg',steps=4,num_envs=1,
          seed=7,device='cpu',attack_mode='radial_straight',output=output,
          eval_interval=2,eval_episodes=1,log_every_steps=4)
    restored,policy,_,_,settings=load_run(output,'Last','cpu')
    assert settings['environment_contract']['version']=='interception_13'
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['updates']==2
    for line in (output/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        episode=json.loads(line)
        assert np.isclose(sum(episode['reward_components'].values()),episode['team_return'])
    rows=evaluate_policy(restored,policy,'radial_straight',1,900001)
    assert np.isclose(sum(rows[0]['reward_components'].values()),rows[0]['team_return'])
    trace=validate_trace(collect(restored,policy,'radial_straight',900001))
    assert trace['schema']=='InterceptionReplay/3'
    assert trace['frames'][1]['physical_commands'] is not None
