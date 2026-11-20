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
    assert obs.shape==(5,49) and info['central_state'].shape==(245,)
    assert np.all(left.positions[0,:,1]>=-95) and np.all(left.positions[0,:,1]<=-75)
    assert np.all(left.positions[1,:,1]>=75) and np.all(left.positions[1,:,1]<=90)
    before=left.positions.copy()
    left.step(np.full((5,2),[2.,0.]))
    np.testing.assert_allclose(left.positions,before)
    assert np.all(np.linalg.norm(left.velocities[0],axis=1)<=config.defender_speed_max)


def test_all_in_range_agents_have_fixed_relative_slots():
    env=InterceptionEnv()
    env.reset(seed=11)
    env.positions[0]=[[0,0],[5,0],[10,0],[15,0],[20,0]]
    env.positions[1]=[[0,25],[0,30],[0,35],[0,40],[0,45]]
    env.velocities.fill(0)
    observation=env.observations()[0]
    assert observation.shape==(49,)
    assert observation[4:24:5].tolist()==pytest.approx([.05,.1,.15,.2])
    assert observation[24:49:5].tolist()==pytest.approx([0]*5)
    assert observation[25:49:5].tolist()==pytest.approx([.25,.3,.35,.4,.45])
    assert observation[8:24:5].tolist()==[1]*4
    assert observation[28:49:5].tolist()==[1]*5
    env.positions[1,4]=[0,90]
    assert not env.observations()[0,44:49].any()
    env.positions[0,0]=[10,0]
    env.velocities[0,0]=[1,2]
    env.velocities[0,2]=[3,4]
    absolute=env.observations()[0]
    np.testing.assert_allclose(absolute[:4],[.1,0,.125,.25])
    np.testing.assert_allclose(absolute[4:9],[.1,0,.375,.5,1])
    legacy=InterceptionEnv(InterceptionConfig(visible_neighbors=3,visible_attackers=3,
                                               observation_layout='relative_with_base',communication_radius=60))
    assert legacy.reset(seed=11)[0].shape==(5,36)


def test_one_sensing_radius_applies_to_both_sides():
    env=InterceptionEnv()
    env.reset(seed=3)
    env.positions[0]=[[0,0],[70,0],[90,0],[0,90],[-90,0]]
    env.positions[1]=[[0,70],[90,10],[90,20],[90,30],[90,40]]
    env.velocities.fill(0)
    observation=env.observations()[0]
    np.testing.assert_allclose(observation[4:9],[.7,0,0,0,1])
    np.testing.assert_allclose(observation[24:29],[0,.7,0,0,1])
    assert not observation[9:24].any()
    assert not observation[29:49].any()
    np.testing.assert_array_equal(env.central_state(),env.observations().reshape(245))


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
    np.testing.assert_array_equal(state[0],observations.reshape(245))
    mappo=make_policy('mappo',config,MAPPOConfig(hidden_size=16),'cpu',9)
    assert mappo.critic.network[0].in_features==245
    assert mappo.policy_actions(batch_obs,masks).shape==(1,5,2)
    assert mappo.values(state).shape==(1,)
    maddpg=make_policy('maddpg',config,MADDPGConfig(hidden_dim=16,batch_size=2,learning_starts=2),'cpu',9)
    assert maddpg.critics[0].net[0].in_features==255
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


def test_approach_rewards_defender_motion_only_with_visible_target():
    config=InterceptionConfig()
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.attacker_alive[:]=False
    env.attacker_alive[0]=True
    env.positions[1,0]=[0,0]
    env.positions[0]=[[0,-50],[20,-50],[100,-90],[-100,-90],[100,-80]]
    env.velocities.fill(0)
    env.velocities[1,0]=[0,-8]
    env.velocities[0,1]=[0,5]
    observations=env.observations()
    start=4+5*config.visible_neighbors
    np.testing.assert_allclose(observations[0,start:start+5],[0,0,0,-1,1])
    assert observations[2,start+4]==0
    _,_,_,_,info=env.step(np.zeros((5,2)))
    assert info['global_reward']==0
    assert 'danger' not in info['reward_components']
    assert info['local_rewards'][0]==pytest.approx(0,abs=1e-12)
    before=np.hypot(20,50)
    after=np.hypot(20,49.5)
    assert info['local_rewards'][1]==pytest.approx(config.approach_weight*(before-after))
    assert info['local_rewards'][2]==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('algorithm',('mappo','maddpg'))
def test_visible_attacker_reaches_actor_input(algorithm):
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import make_policy

    env=InterceptionEnv()
    env.reset(seed=7)
    env.attacker_alive[:]=False
    env.attacker_alive[0]=True
    env.positions[0,0]=[0,-50]
    env.positions[1,0]=[10,0]
    observations=env.observations()
    start=4+5*env.config.visible_neighbors
    assert observations[0,start+4]==1
    parameters=MAPPOConfig(hidden_size=16) if algorithm=='mappo' else MADDPGConfig(hidden_dim=16)
    policy=make_policy(algorithm,env.config,parameters,'cpu',7)
    layer=policy.actor.network[0] if algorithm=='mappo' else policy.actors[0].net[0]
    seen=[]
    hook=layer.register_forward_pre_hook(lambda _module,args:seen.append(args[0].detach().cpu().numpy().copy()))
    try:
        policy.policy_actions(observations[None],np.ones((1,5),np.float32),deterministic=True)
    finally:
        hook.remove()
    actual=seen[0][0,0] if algorithm=='mappo' else seen[0][0]
    np.testing.assert_array_equal(actual[start:start+5],observations[0,start:start+5])


@pytest.mark.parametrize('algorithm',('mappo','maddpg'))
@pytest.mark.parametrize('eval_interval',(0,4))
def test_training_writes_compact_episode_rewards(tmp_path,algorithm,eval_interval):
    import json

    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import train

    config=InterceptionConfig(episode_steps=2)
    parameters=(MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4) if algorithm=='mappo'
                else MADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=4,
                                  learning_starts=2,update_interval_steps=2,updates_per_interval=1))
    output=tmp_path/algorithm
    train(config,parameters,2,algorithm,steps=4,num_envs=2,seed=7,device='cpu',
          attack_mode='mixed',output=output,log_every_steps=4,
          eval_interval=eval_interval,eval_episodes=1)
    assert not (output/'reward_steps.jsonl.gz').exists()
    rows=[json.loads(line) for line in (output/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2
    for row in rows:
        assert set(row['reward_components'])=={'capture','boundary','penetration','approach'}
        assert np.isclose(sum(row['reward_components'].values()),row['team_return'],rtol=0,atol=1e-6)
    expected_selection='periodic_evaluation' if eval_interval else 'final_unvalidated'
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    index=json.loads((output/'checkpoints'/'index.json').read_text(encoding='utf-8'))
    assert summary['best_selection']==expected_selection
    assert index['checkpoints']['Best']['selection']==expected_selection
    assert (summary['best_score'] is not None)==bool(eval_interval)
