import json
from dataclasses import replace

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from experiment.interception import ReplayBuffer, load_run, make_policy, read_config, train
from experiment.prioritized_replay import PrioritizedReplayBuffer, ReplayConfig


def _row(value):
    return dict(states=np.array([value],np.float32))


def _events(capture=0,first_breach=0,boundary=0):
    return dict(capture=capture,first_breach=first_breach,boundary=boundary)


def test_joint_priority_tree_distribution_duplicates_and_overwrite():
    replay=PrioritizedReplayBuffer(3,7,alpha=1.0,epsilon=0.01)
    replay.add(event_metadata=_events(capture=1,boundary=1),**_row(0))
    replay.add(event_metadata=_events(first_breach=1),**_row(1))
    replay.add(event_metadata=_events(),**_row(2))
    replay.update_priorities([0,1,2,1],[0.99,0.49,3.99,1.99])
    np.testing.assert_allclose(replay.raw_priorities,[1,2,4])
    chosen=replay._prioritized(7000)
    np.testing.assert_allclose(np.bincount(chosen,minlength=3)/len(chosen),[1/7,2/7,4/7],atol=.025)
    critic,actor,indices,weights,stats=replay.sample_update(8,prioritized_fraction=.5,beta=.5)
    np.testing.assert_array_equal(critic['states'][:4],actor['states'][:4])
    assert len(indices)==len(weights)==len(critic['states'])==len(actor['states'])==8
    expected_probability=.5/3+.5*replay.sums[replay.leaves+indices]/replay.sums[1]
    expected=(3*expected_probability)**(-.5)
    np.testing.assert_allclose(weights,expected/expected.max())
    assert stats['per_buffer_capture_fraction']==pytest.approx(1/3)
    assert stats['per_buffer_boundary_fraction']==pytest.approx(1/3)
    assert stats['per_buffer_first_breach_fraction']==pytest.approx(1/3)
    assert stats['per_buffer_ordinary_fraction']==pytest.approx(1/3)
    replay.add(event_metadata=_events(capture=1),**_row(3))
    assert float(replay.rows[0]['states'][0])==3
    assert stats['per_buffer_boundary_fraction']!=replay.buffer_event_counts[2]/len(replay)
    assert replay.buffer_event_counts.tolist()==[1,1,0]


def test_replay_configuration_is_separate_from_legacy_uniform_path():
    config,parameters,rollout=read_config('configs/interception_huang_global_target_velocity.yaml',
                                           'configs/maddpg_per_interception.yaml')
    assert (config.observation_layout,parameters.batch_size,rollout)==(
        'relative_assignment_target_velocity',128,128)
    replay_config=ReplayConfig.from_mapping(dict(mode='critic_per_with_uniform'))
    assert replay_config.beta(0,100)==pytest.approx(.4)
    assert replay_config.beta(100,100)==pytest.approx(1)
    uniform=ReplayBuffer(3,7)
    for i in range(3):uniform.add(**_row(i))
    expected=np.random.default_rng(7).integers(0,3,size=8)
    np.testing.assert_array_equal(uniform.sample(8)['states'][:,0],expected)
    with pytest.raises(ValueError):
        ReplayConfig.from_mapping(dict(mode='critic_per_with_uniform',alpha=0))


def test_actor_uses_its_uniform_batch_while_critic_uses_priority_batch():
    config,_,_=read_config('configs/interception_huang_global_target_velocity.yaml')
    parameters=MADDPGConfig(hidden_dim=16,batch_size=4,buffer_capacity=8,learning_starts=4)
    first=make_policy('maddpg',config,parameters,'cpu',7)
    second=make_policy('maddpg',config,parameters,'cpu',7)
    critic_batch=dict(observations=np.zeros((4,5,36),np.float32),
                      next_observations=np.zeros((4,5,36),np.float32),
                      states=np.zeros((4,75),np.float32),
                      next_states=np.zeros((4,75),np.float32),
                      actions=np.zeros((4,5,2),np.float32),
                      rewards=np.ones((4,5),np.float32),
                      dones=np.zeros(4,np.float32),
                      active_masks=np.ones((4,5),np.float32),
                      next_active_masks=np.ones((4,5),np.float32))
    actor_batch={key:value.copy() for key,value in critic_batch.items()}
    actor_batch['observations'][:]=0.5
    actor_batch['states'][:]=0.5
    metrics,td=first.update(critic_batch,actor_batch=actor_batch,
                            critic_weights=np.ones(4,np.float32),return_td_errors=True)
    other=second.update(critic_batch)
    assert td.shape==(4,) and np.isfinite(td).all()
    assert metrics['critic_loss']==pytest.approx(other['critic_loss'])
    for left,right in zip(first.critics.parameters(),second.critics.parameters()):
        np.testing.assert_allclose(left.detach().numpy(),right.detach().numpy(),atol=1e-7)
    assert any(not np.allclose(left.detach().numpy(),right.detach().numpy())
               for left,right in zip(first.actors.parameters(),second.actors.parameters()))


def test_prioritized_training_keeps_checkpoint_contract(tmp_path):
    config,_,_=read_config('configs/interception_huang_global_target_velocity.yaml')
    parameters=MADDPGConfig(hidden_dim=16,batch_size=4,buffer_capacity=8,
                            learning_starts=4,update_interval_steps=2,updates_per_interval=1)
    output=tmp_path/'per'
    train(replace(config,episode_steps=2),parameters,2,'maddpg',steps=8,num_envs=2,seed=7,
          device='cpu',attack_mode='direct_rush',output=output,eval_interval=0,
          log_every_steps=8,replay_config=ReplayConfig(mode='critic_per_with_uniform'))
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    assert settings['replay']['mode']=='critic_per_with_uniform'
    updates=[json.loads(line) for line in (output/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert updates and all('per_beta' in row and 'per_is_ess' in row for row in updates)
    progress=[json.loads(line) for line in (output/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert 'per_sample_capture_fraction' in progress[-1]['learning']
    _,policy,name,_,_=load_run(output,'Last','cpu')
    assert name=='maddpg' and policy.config==parameters
