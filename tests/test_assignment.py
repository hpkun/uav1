"""The new assignment interface and historical interception interface coexist."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from assignment import DistanceAssignment, NearestVisibleAssignment
from env.interception import InterceptionConfig, InterceptionEnv
from experiment.interception import make_policy, read_config


def test_periodic_and_event_reassignment():
    allocator=NearestVisibleAssignment(20)
    defenders=np.array([[0.,0.],[50,0],[60,0],[70,0],[80,0]])
    attackers=np.array([[0.,10.],[0,20.],[80,70],[90,70],[100,70]])
    alive=np.ones(5,bool)
    targets,changed=allocator.update(defenders,attackers,alive,alive,80,0)
    assert changed and targets[0]==0
    attackers[0,1]=30
    attackers[1,1]=5
    targets,changed=allocator.update(defenders,attackers,alive,alive,80,1)
    assert not changed and targets[0]==0
    targets,changed=allocator.update(defenders,attackers,alive,alive,80,20)
    assert changed and targets[0]==1
    alive[1]=False
    targets,changed=allocator.update(defenders,attackers,np.ones(5,bool),alive,80,21,
                                     [{'type':'capture'}])
    assert changed and targets[0]==0


def test_distance_matching_covers_targets_without_duplicate_assignments():
    defenders=np.array([[0.,0.],[1,0],[10,0],[20,0],[30,0]])
    attackers=np.array([[0.,1.],[2,1],[10,1],[20,1],[30,1]])
    alive=np.ones(5,bool)
    matching=DistanceAssignment(20,'distance_matching')
    nearest=DistanceAssignment(20,'independent_nearest')
    targets,replanned=matching.update(defenders,attackers,alive,alive,50,0)
    assert replanned and targets.tolist()==[0,1,2,3,4]
    assert len(set(targets))==5
    assert nearest.update(defenders,attackers,alive,alive,50,0)[0].tolist()==[0,0,2,3,4]


def test_matching_maximizes_feasible_pairs_before_distance():
    defenders=np.array([[0.,0.],[0,2],[90,90],[90,91],[90,92]])
    attackers=np.array([[0.,1.],[2,0],[90,93],[90,94],[90,95]])
    alive=np.ones(5,bool)
    targets,_=DistanceAssignment(20).update(defenders,attackers,alive,alive,3,0)
    assert sorted(targets)==list(range(5))
    assert targets[0]==1 and targets[1]==0


@pytest.mark.parametrize('method',('distance_matching','independent_nearest'))
def test_only_system_first_detection_period_and_resolution_trigger(method):
    allocator=DistanceAssignment(20,method)
    defenders=np.array([[0.,0.],[60,0],[60,1],[60,2],[60,3]])
    attackers=np.array([[0.,10.],[0,100],[200,200],[210,210],[220,220]])
    alive=np.ones(5,bool)
    assert allocator.update(defenders,attackers,alive,alive,20,0)[1]
    assert allocator.detected_targets.tolist()==[True,False,False,False,False]
    attackers[1]=[0,12]
    assert allocator.update(defenders,attackers,alive,alive,20,1)[1]
    assert allocator.detected_targets.tolist()==[True,True,False,False,False]
    targets=allocator.targets.copy()
    defenders[1]=[0,14]  # A second defender sees an already known target.
    _,replanned=allocator.update(defenders,attackers,alive,alive,20,2)
    assert not replanned and np.array_equal(allocator.targets,targets)
    defenders[0]=[60,0]  # The old edge disappears; retain the assignment until a trigger.
    _,replanned=allocator.update(defenders,attackers,alive,alive,20,3)
    assert not replanned and np.array_equal(allocator.targets,targets)
    assert allocator.update(defenders,attackers,alive,alive,20,20)[1]
    attacker_alive=alive.copy(); attacker_alive[0]=False
    assert allocator.update(defenders,attackers,alive,attacker_alive,20,21,
                            [{'type':'penetration'}])[1]
    assert 0 not in allocator.targets


def test_relative_actor_and_global_critic_state():
    config,_,_=read_config('configs/interception_assignment.yaml')
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.positions[0,0]=[0,0]
    env.positions[0,1]=[10,0]
    env.positions[1,0]=[0,30]
    env.positions[1,1:]=[90,90]
    env.velocities.fill(0)
    env.velocities[0,0]=[1,0]
    env.velocities[1,0]=[0,-config.attacker_speed_max]
    env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                          env.attacker_alive,config.sensing_radius,0)
    obs=env.observations()
    state=env.central_state()
    assert env.contract['version']=='interception_2'
    assert obs.shape==(5,54) and state.shape==(75,)
    speed_sum=config.defender_speed_max+config.attacker_speed_max
    np.testing.assert_allclose(obs[0,4:9],[10/config.sensing_radius,0,-1/speed_sum,0,1])
    np.testing.assert_allclose(obs[0,24:29],[0,30/config.sensing_radius,-1/speed_sum,-config.attacker_speed_max/speed_sum,1])
    np.testing.assert_allclose(obs[0,-5:],[0,30/config.sensing_radius,-1/speed_sum,-config.attacker_speed_max/speed_sum,1])
    assert state[25+5:25+7].tolist()==pytest.approx([.9,.9])
    assert state[50]==1
    env.attacker_alive[0]=False
    env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                          env.attacker_alive,config.sensing_radius,1,[{'type':'capture'}])
    assert env.assignment.targets[0]==-1 and not env.observations()[0,-5:].any()


def test_out_of_range_assignment_is_held_without_actor_position_leak():
    config=InterceptionConfig(observation_layout='relative_assignment',
                              assignment_method='distance_matching')
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.positions[0,0]=[0,0]
    env.positions[1,0]=[0,20]
    env.positions[1,1:]=[90,90]
    env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                          env.attacker_alive,config.sensing_radius,0)
    assert env.assignment.targets[0]==0
    env.positions[1,0]=[100,90]
    _,replanned=env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                                      env.attacker_alive,config.sensing_radius,1)
    assert not replanned and env.assignment.targets[0]==0
    assert not env.observations()[0,-5:].any()
    assert env.central_state()[50]==1


def test_action_effect_on_distance_reward_is_delayed():
    config=InterceptionConfig(observation_layout='relative_assignment')
    assert config.approach_weight==0.5
    assert InterceptionConfig().approach_weight==0.05
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.positions[0,0]=[0,0]
    env.positions[1,0]=[0,10]
    env.positions[1,1:]=[90,90]
    env.velocities.fill(0)
    env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                          env.attacker_alive,config.sensing_radius,0)
    actions=np.zeros((5,2));actions[0]=[0,2]
    _,_,_,_,first=env.step(actions)
    assert first['reward_target_ids'][0]==0
    assert first['local_rewards'][0]==pytest.approx(0)
    distance_before=np.linalg.norm(env.positions[0,0]-env.positions[1,0])
    _,_,_,_,second=env.step(np.zeros((5,2)))
    assert second['reward_target_ids'][0]==0
    distance_after=np.linalg.norm(env.positions[0,0]-env.positions[1,0])
    assert second['local_rewards'][0]==pytest.approx(0.5*(distance_before-distance_after))
    assert second['local_rewards'][0]>0


def test_new_assignment_configs_use_the_same_approach_weight():
    for path in ('configs/interception_assignment.yaml','configs/interception_assignment_nearest.yaml',
                 'configs/interception_assignment_speed_parity.yaml'):
        config,_,_=read_config(path)
        assert config.approach_weight==0.5
        assert config.attacker_speed_max==config.defender_speed_max==5.0
        assert config.attacker_acceleration_max==config.defender_acceleration_max==2.0
        assert config.attacker_initial_speed==config.defender_initial_speed==2.0
        assert config.sensing_radius==100.0
        env=InterceptionEnv(config)
        env.reset(seed=7)
        np.testing.assert_allclose(env.velocities[0],np.tile([0,2.0],(5,1)))
        np.testing.assert_allclose(np.linalg.norm(env.velocities[1],axis=1),2.0)
        assert np.all(env.velocities[1,:,1]<0)
    old,_,_=read_config('configs/interception.yaml')
    assert old.approach_weight==0.05
    assert old.attacker_speed_max==8.0
    assert old.attacker_acceleration_max==3.0
    assert old.attacker_initial_speed==old.defender_initial_speed==0.0
    assert old.sensing_radius==80.0


def test_100m_sensing_boundary_controls_observation_and_assignment():
    config,_,_=read_config('configs/interception_assignment.yaml')
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.defender_alive[:]=False
    env.defender_alive[:3]=True
    env.attacker_alive[:]=False
    env.attacker_alive[:2]=True
    env.positions[0,:3]=[[0,0],[60,80],[60,80.01]]
    env.positions[1,:2]=[[80,60],[80,60.01]]
    env.velocities.fill(0)
    obs=env.observations()[0]
    np.testing.assert_allclose(obs[4:9],[.6,.8,0,0,1])
    assert not obs[9:24].any()
    np.testing.assert_allclose(obs[24:29],[.8,.6,0,0,1])
    assert not obs[29:49].any()
    env.defender_alive[1:]=False
    targets,_=env.assignment.update(env.positions[0],env.positions[1],
                                    env.defender_alive,env.attacker_alive,config.sensing_radius,0)
    assert targets[0]==0
    assert env.assignment.detected_targets[:2].tolist()==[True,False]


@pytest.mark.parametrize('mode',('direct_rush','split_attack'))
def test_initial_velocity_follows_attack_mode_and_moves_on_first_step(mode):
    config,_,_=read_config('configs/interception_assignment.yaml')
    env=InterceptionEnv(replace(config,attack_mode=mode))
    observations,info=env.reset(seed=7)
    if mode=='direct_rush':
        targets=np.broadcast_to(env.base,(5,2))
    else:
        targets=np.column_stack((env.split_flank*config.split_waypoint_x,
                                 np.full(5,config.split_waypoint_y)))
    directions=targets-env.positions[1]
    expected=config.attacker_initial_speed*directions/np.linalg.norm(directions,axis=1,keepdims=True)
    np.testing.assert_allclose(env.velocities[1],expected)
    np.testing.assert_allclose(observations[:,2:4],np.tile([0,2.0/config.defender_speed_max],(5,1)))
    np.testing.assert_allclose(info['central_state'][:25].reshape(5,5)[:,2:4],
                               np.tile([0,2.0/config.defender_speed_max],(5,1)))
    before=env.positions.copy()
    velocity=env.velocities.copy()
    env.step(np.zeros((5,2)))
    np.testing.assert_allclose(env.positions,before+velocity*config.dt)


@pytest.mark.parametrize('initial_speed',(-0.1,5.1,float('inf')))
def test_initial_speed_respects_speed_limit(initial_speed):
    with pytest.raises(ValueError,match='Initial speeds'):
        InterceptionConfig(defender_initial_speed=initial_speed)


def test_checkpoint_environment_without_initial_speed_fields_starts_at_rest():
    config,_,_=read_config('configs/interception_assignment.yaml')
    historical=asdict(config)
    historical.pop('defender_initial_speed')
    historical.pop('attacker_initial_speed')
    restored=InterceptionConfig(**historical)
    env=InterceptionEnv(restored)
    env.reset(seed=7)
    assert not env.velocities.any()
    assert env.contract['version']=='interception_2'


@pytest.mark.parametrize('name',('mappo','maddpg'))
def test_new_policy_and_old_contract(name):
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    new=InterceptionConfig(observation_layout='relative_assignment')
    old=InterceptionConfig()
    parameters=MAPPOConfig(hidden_size=16) if name=='mappo' else MADDPGConfig(hidden_dim=16)
    policy=make_policy(name,new,parameters,'cpu',7)
    env=InterceptionEnv(new);obs,info=env.reset(seed=7)
    assert policy.policy_actions(obs[None],info['active_masks'][None]).shape==(1,5,2)
    assert InterceptionEnv(old).contract['version']=='interception_1'
    old_policy=make_policy(name,old,parameters,'cpu',7)
    assert old_policy.policy_actions(InterceptionEnv(old).reset(seed=7)[0][None],
                                     np.ones((1,5),np.float32)).shape==(1,5,2)


def test_dual_critic_is_explicitly_legacy_only():
    from algorithm.dual_critic import DualCriticConfig
    with pytest.raises(ValueError,match='only Interception/1'):
        make_policy('dual_critic',InterceptionConfig(observation_layout='relative_assignment'),
                    DualCriticConfig(),'cpu',7)
    with pytest.raises(ValueError,match='only Interception/1'):
        read_config('configs/interception_assignment.yaml','configs/dual_critic_interception.yaml')


@pytest.mark.parametrize('name',('mappo','maddpg'))
@pytest.mark.parametrize('attack_mode',('direct_rush','split_attack'))
def test_new_interface_short_training_and_checkpoint(tmp_path,name,attack_mode):
    import json
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import load_run,read_update_log,train

    config=InterceptionConfig(observation_layout='relative_assignment',episode_steps=2)
    parameters=(MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4) if name=='mappo'
                else MADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=4,
                                  learning_starts=2,update_interval_steps=2,updates_per_interval=1))
    output=tmp_path/name
    train(config,parameters,2,name,steps=4,num_envs=2,seed=7,device='cpu',
          attack_mode=attack_mode,output=output,eval_interval=0,log_every_steps=4)
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    assert settings['environment_contract']['version']=='interception_2'
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['updates']>0
    assert (output/'updates.jsonl.gz').is_file() and not (output/'updates.jsonl').exists()
    update=read_update_log(output)[0]
    assert {'actor_grad_norm','critic_grad_norm'} <= update.keys()
    if name=='mappo':
        assert {'policy_log_std_mean','policy_std_mean','advantage_raw_mean','advantage_raw_std'} <= update.keys()
        assert update['policy_std_mean']>0
    else:
        assert {'q_mean','q_std','target_q_mean'} <= update.keys()
        assert update['q_std']>=0
    restored,policy,algorithm,_,_=load_run(output,'Best','cpu')
    assert algorithm==name and restored.observation_layout=='relative_assignment'
    assert restored.attack_mode==attack_mode
    env=InterceptionEnv(restored)
    obs,info=env.reset(seed=7)
    assert policy.policy_actions(obs[None],info['active_masks'][None]).shape==(1,5,2)
