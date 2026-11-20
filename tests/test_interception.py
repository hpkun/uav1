import numpy as np
import pytest

from env.interception import InterceptionConfig, InterceptionEnv
from env.parallel import ParallelInterceptionEnvs


def test_seed_observation_and_kinematics():
    config=InterceptionConfig()
    left=InterceptionEnv(config)
    right=InterceptionEnv(config)
    obs,info=left.reset(seed=42)
    other,_=right.reset(seed=42)
    np.testing.assert_array_equal(obs,other)
    assert obs.shape==(5,36) and info['central_state'].shape==(180,)
    assert np.all(left.positions[0,:,1]>=-95) and np.all(left.positions[0,:,1]<=-75)
    assert np.all(left.positions[1,:,1]>=75) and np.all(left.positions[1,:,1]<=90)
    before=left.positions.copy()
    left.step(np.full((5,2),[2.,0.]))
    np.testing.assert_allclose(left.positions,before)
    assert np.all(np.linalg.norm(left.velocities[0],axis=1)<=config.defender_speed_max)


def test_capture_precedes_breach_and_reward_balances():
    env=InterceptionEnv()
    env.reset(seed=7)
    env.positions[0,0]=[0,-65]
    env.positions[1,0]=[0,-65]
    env.velocities.fill(0)
    _,_,terminated,_,info=env.step(np.zeros((5,2)))
    assert not terminated
    assert not any(event['type']=='penetration' and event['attacker']==0 for event in info['events'])
    assert any(event['type']=='capture' and event['attacker']==0 for event in info['events'])
    assert np.isclose(info['team_reward'],np.mean(info['individual_rewards']))
    assert np.isclose(info['team_reward'],sum(info['reward_components'].values()))


def test_all_capture_ends_episode_once():
    env=InterceptionEnv()
    env.reset(seed=5)
    env.positions[0,0]=[0,-65]
    env.positions[1]=[0,-65]
    env.velocities.fill(0)
    _,_,terminated,truncated,info=env.step(np.zeros((5,2)))
    assert terminated and not truncated
    assert info['episode_summary']['outcome']=='all_captured'
    assert info['episode_summary']['captured_count']==5
    assert len([event for event in info['events'] if event['type']=='capture'])==5
    assert not any(event['type']=='penetration' for event in info['events'])


def test_parallel_sampler_preserves_terminal_state():
    config=InterceptionConfig(episode_steps=1)
    with ParallelInterceptionEnvs(2,config,seed=3) as sampler:
        obs,states,masks,_=sampler.reset()
        result=sampler.step(np.zeros((2,5,2)))
    assert result.dones.tolist()==[1,1]
    assert result.truncated.tolist()==[True,True]
    assert result.rewards.shape==(2,)
    assert result.individual_rewards.shape==(2,5)
    np.testing.assert_allclose(result.rewards,result.individual_rewards.mean(axis=1),atol=1e-5)
    assert all(info['episode_summary']['outcome']=='time_limit' for info in result.infos)
    assert np.any(result.observations!=result.transition_observations)


def test_unopposed_attackers_reach_the_lower_target():
    for mode in ('direct_rush','split_attack'):
        env=InterceptionEnv(InterceptionConfig(attack_mode=mode))
        env.reset(seed=7)
        for _ in range(500):
            _,_,terminated,truncated,info=env.step(np.zeros((5,2)))
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info['episode_summary']['outcome']=='penetrated'
        assert 0<info['episode_summary']['steps']<500


def test_mappo_team_and_maddpg_individual_interfaces():
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import make_policy

    config=InterceptionConfig()
    env=InterceptionEnv(config)
    observations,info=env.reset(seed=9)
    batch_obs=observations[None]
    masks=info['active_masks'][None]
    state=info['central_state'][None]
    mappo=make_policy('mappo',config,MAPPOConfig(hidden_size=16),'cpu',9)
    assert mappo.policy_actions(batch_obs,masks).shape==(1,5,2)
    assert mappo.values(state).shape==(1,)
    maddpg=make_policy('maddpg',config,MADDPGConfig(hidden_dim=16,batch_size=2,learning_starts=2),'cpu',9)
    assert maddpg.policy_actions(batch_obs,masks).shape==(1,5,2)
    transition=dict(observations=np.repeat(batch_obs,2,axis=0),next_observations=np.repeat(batch_obs,2,axis=0),
                    states=np.repeat(state,2,axis=0),next_states=np.repeat(state,2,axis=0),
                    actions=np.zeros((2,5,2),np.float32),rewards=np.zeros((2,5),np.float32),
                    dones=np.zeros(2,np.float32),active_masks=np.ones((2,5),np.float32),
                    next_active_masks=np.ones((2,5),np.float32))
    maddpg._batch(transition)
    with pytest.raises(ValueError,match='Invalid MADDPG joint replay batch'):
        maddpg._batch(dict(transition,rewards=np.zeros(2,np.float32)))


def test_evaluation_rows_use_environment_seed_once():
    from experiment.interception import evaluate_policy

    class StationaryPolicy:
        def policy_actions(self,observations,active_masks,*,deterministic):
            return np.zeros((len(observations),5,2),np.float32)

    rows=evaluate_policy(InterceptionConfig(episode_steps=2),StationaryPolicy(),
                         'direct_rush',episodes=2,seed_start=600000)
    assert [row['seed'] for row in rows]==[600000,600001]
    assert all(row['outcome']=='time_limit' for row in rows)
