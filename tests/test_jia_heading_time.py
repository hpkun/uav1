"""Fixed per-step time cost in the isolated Jia speed-heading environment."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from algorithm.maddpg_config import NormalizedMADDPGConfig
from env import make_environment
from env.jia_heading import wrap_heading
from experiment.interception import ReplayBuffer, evaluate_policy, load_run, make_policy, read_config, train


BASE = 'configs/interception_radial_5v5_jia_heading.yaml'
TIMED = 'configs/interception_radial_5v5_jia_heading_time01.yaml'
ALGORITHM = 'configs/maddpg_jia_heading_interval10.yaml'


def pair(seed=7):
    base, _, _ = read_config(BASE, ALGORITHM)
    timed, _, _ = read_config(TIMED, ALGORITHM)
    assert {key for key in asdict(base) if getattr(base,key) != getattr(timed,key)} == {'time_penalty_per_step'}
    old, new = make_environment(base), make_environment(timed)
    assert old.contract['reward_schema'] != new.contract['reward_schema']
    old.reset(seed=seed)
    new.reset(seed=seed)
    return old, new


@pytest.mark.parametrize('steps', [1,10,80])
def test_time_cost_and_identical_trajectory(steps):
    old, new = pair()
    assert old.team_return == new.team_return == 0
    total = 0.0
    for _ in range(steps):
        active_before = old.defender_alive.copy()
        actions = np.zeros((5,2), np.float32)
        old_obs, old_reward, old_term, old_trunc, old_info = old.step(actions)
        new_obs, new_reward, new_term, new_trunc, new_info = new.step(actions)
        np.testing.assert_array_equal(old_obs, new_obs)
        np.testing.assert_array_equal(old.positions, new.positions)
        np.testing.assert_array_equal(old.velocities, new.velocities)
        np.testing.assert_array_equal(old.headings, new.headings)
        np.testing.assert_array_equal(old.defender_alive, new.defender_alive)
        np.testing.assert_array_equal(old.attacker_alive, new.attacker_alive)
        np.testing.assert_array_equal(old.assignment.targets, new.assignment.targets)
        assert old_info['events'] == new_info['events']
        assert (old_term,old_trunc) == (new_term,new_trunc) == (False,False)
        assert old_info['reward_components'].get('time',0) == 0
        assert new_info['reward_components']['time'] == pytest.approx(-.1)
        assert new_info['team_reward'] == pytest.approx(old_info['team_reward']-.1)
        np.testing.assert_allclose(new_reward[active_before], old_reward[active_before]-.1, atol=2e-5)
        np.testing.assert_allclose(new_reward[~active_before], old_reward[~active_before], atol=2e-5)
        np.testing.assert_allclose(new_info['individual_rewards'],new_reward,atol=2e-5)
        assert new_info['global_reward'] == old_info['global_reward']
        np.testing.assert_array_equal(new_info['local_rewards'],old_info['local_rewards'])
        assert sum(new_info['reward_components'].values()) == pytest.approx(new_info['team_reward'])
        total += new_info['reward_components']['time']
    assert total == pytest.approx(-.1*steps)
    assert new.team_return == pytest.approx(old.team_return-.1*steps)


def test_terminal_capture_and_breach_charge_once():
    for capture in (True,False):
        old,new=pair()
        for env in (old,new):
            if capture:
                env.positions[0] = env.positions[1].copy()
            else:
                env.defender_alive[:] = False
                env.velocities[0] = 0
                env.positions[1] *= 30.1/np.linalg.norm(env.positions[1],axis=1)[:,None]
        old_result=old.step(np.zeros((5,2),np.float32))
        new_result=new.step(np.zeros((5,2),np.float32))
        assert old.finished and new.finished and old.step_count == new.step_count == 1
        assert old_result[4]['events'] == new_result[4]['events']
        assert new_result[4]['reward_components']['time'] == pytest.approx(-.1)
        assert new_result[4]['episode_summary']['team_return'] == pytest.approx(old.team_return-.1)
        if capture:
            assert (new.captured_count,new.penetrated_count) == (5,0)
            np.testing.assert_allclose(new_result[1],old_result[1]-.1,atol=2e-5)
        else:
            assert (new.captured_count,new.penetrated_count) == (0,5)
            np.testing.assert_array_equal(new_result[1],old_result[1])
        with pytest.raises(RuntimeError,match='reset required'):
            new.step(np.zeros((5,2)))


def test_retired_agent_mask_and_replay_reward():
    old,new=pair()
    for env in (old,new):env.positions[0,0]=env.positions[1,0].copy()
    actions=np.zeros((5,2),np.float32)
    old.step(actions)
    new.step(actions)
    assert not new.defender_alive[0]
    previous=new.positions[0,0].copy()
    old_info_before=old._info([],0,np.zeros(5),None)
    new_info_before=new._info([],0,np.zeros(5),None)
    observations_before=new.observations()
    old_result=old.step(actions)
    new_result=new.step(actions)
    np.testing.assert_array_equal(new.positions[0,0],previous)
    assert new_result[4]['active_masks'][0] == 0
    assert new_result[1][0] == pytest.approx(old_result[1][0])
    np.testing.assert_allclose(new_result[1][1:],old_result[1][1:]-.1,atol=2e-5)
    buffer=ReplayBuffer(4,7)
    buffer.add(observations=observations_before,next_observations=new_result[0],
               states=new_info_before['central_state'],next_states=new_result[4]['central_state'],
               actions=actions,rewards=new_result[4]['individual_rewards'],dones=0,
               active_masks=new_info_before['active_masks'],next_active_masks=new_result[4]['active_masks'])
    stored=buffer.sample(1)
    np.testing.assert_allclose(stored['rewards'][0],new_result[4]['individual_rewards'],atol=2e-5)
    assert stored['active_masks'][0,0] == 0
    assert old_info_before['active_masks'][0] == 0


def test_complete_episode_time_ledger():
    old,new=pair(seed=17)
    totals={}
    while not new.finished:
        actions=np.zeros((5,2),np.float32)
        for i in np.flatnonzero(new.defender_alive):
            target=new.assignment.targets[i]
            if target<0:continue
            delta=new.positions[1,target]-new.positions[0,i]
            error=wrap_heading(np.arctan2(delta[1],delta[0])-new.headings[0,i])
            actions[i,0]=-1 if abs(error)>.35 else 1
            actions[i,1]=np.clip(4*error,-1,1)
        old_result=old.step(actions)
        new_result=new.step(actions)
        assert old_result[4]['events']==new_result[4]['events']
        for key,value in new_result[4]['reward_components'].items():
            totals[key]=totals.get(key,0)+value
    assert (new.captured_count,new.penetrated_count)==(5,0)
    assert new.step_count==old.step_count
    assert totals['time']==pytest.approx(-.1*new.step_count)
    assert sum(totals.values())==pytest.approx(new.team_return)
    assert new.team_return==pytest.approx(old.team_return-.1*new.step_count)


def test_timed_train_eval_checkpoint_and_legacy_config(tmp_path,capsys):
    config,_,_=read_config(TIMED,ALGORITHM)
    parameters=NormalizedMADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=8,
                                      learning_starts=2,update_interval_steps=2,updates_per_interval=1,
                                      exploration_std_mps2=0)
    output=tmp_path/'timed'
    train(replace(config,episode_steps=2),parameters,2,'maddpg',steps=4,num_envs=1,
          seed=7,device='cpu',attack_mode='radial_straight',output=output,
          eval_interval=2,eval_episodes=1,log_every_steps=4)
    stdout=capsys.readouterr().out
    assert 'time=' in stdout
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['updates']==2
    for line in (output/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        assert row['reward_components']['time']==pytest.approx(-.1*row['steps'])
        assert sum(row['reward_components'].values())==pytest.approx(row['team_return'])
    for line in (output/'evaluations.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)['modes']['radial_straight']
        assert row['mean_reward_components']['time']==pytest.approx(-.1*row['mean_duration_s']/.1)
    restored,policy,_,_,settings=load_run(output,'Last','cpu')
    assert settings['environment']['time_penalty_per_step']==.1
    rows=evaluate_policy(restored,policy,'radial_straight',1,900001)
    assert rows[0]['reward_components']['time']==pytest.approx(-.1*rows[0]['steps'])
    assert sum(rows[0]['reward_components'].values())==pytest.approx(rows[0]['team_return'])
    old,_,_=read_config(BASE,ALGORITHM)
    assert old.time_penalty_per_step==0
    with pytest.raises(ValueError,match='Incompatible MADDPG'):
        other=make_policy('maddpg',old,parameters,'cpu',7)
        other.load(output/'checkpoints'/'last.pt',load_optimizers=False)
