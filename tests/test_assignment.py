"""The new assignment interface and historical interception interface coexist."""

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
    env.velocities[1,0]=[0,-8]
    env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                          env.attacker_alive,config.sensing_radius,0)
    obs=env.observations()
    state=env.central_state()
    assert env.contract['version']=='interception_2'
    assert obs.shape==(5,54) and state.shape==(75,)
    np.testing.assert_allclose(obs[0,4:9],[.125,0,-1/13,0,1])
    np.testing.assert_allclose(obs[0,24:29],[0,.375,-1/13,-8/13,1])
    np.testing.assert_allclose(obs[0,-5:],[0,.375,-1/13,-8/13,1])
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
    env.positions[1,0]=[0,90]
    _,replanned=env.assignment.update(env.positions[0],env.positions[1],env.defender_alive,
                                      env.attacker_alive,config.sensing_radius,1)
    assert not replanned and env.assignment.targets[0]==0
    assert not env.observations()[0,-5:].any()
    assert env.central_state()[50]==1


def test_action_effect_on_distance_reward_is_delayed():
    config=InterceptionConfig(observation_layout='relative_assignment')
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
    _,_,_,_,second=env.step(np.zeros((5,2)))
    assert second['reward_target_ids'][0]==0
    assert second['local_rewards'][0]>0


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
def test_new_interface_short_training_and_checkpoint(tmp_path,name):
    import json
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import load_run,train

    config=InterceptionConfig(observation_layout='relative_assignment',episode_steps=2)
    parameters=(MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4) if name=='mappo'
                else MADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=4,
                                  learning_starts=2,update_interval_steps=2,updates_per_interval=1))
    output=tmp_path/name
    train(config,parameters,2,name,steps=4,num_envs=2,seed=7,device='cpu',
          attack_mode='direct_rush',output=output,eval_interval=0,log_every_steps=4)
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    assert settings['environment_contract']['version']=='interception_2'
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['updates']>0
    restored,policy,algorithm,_,_=load_run(output,'Best','cpu')
    assert algorithm==name and restored.observation_layout=='relative_assignment'
    env=InterceptionEnv(restored)
    obs,info=env.reset(seed=7)
    assert policy.policy_actions(obs[None],info['active_masks'][None]).shape==(1,5,2)
