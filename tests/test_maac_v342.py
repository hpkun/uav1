"""Compatibility only: original v3.42 mechanics, unchanged MAAC configuration."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from algorithm.maac.metrics import evaluation_metrics
from algorithm.maac.protocol import DIMENSIONS, validate_config_pair
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.common.evaluator import aggregate_combat_records
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config, environment_dimensions
from env.models import AircraftState
from env.v338_observation import build_team_observations_v338


def configs(v='342'):
    return load_config(f'configs/combat_environment_v{v}.yaml'), yaml.safe_load(
        Path(f'configs/maac_4v4_v{v}.yaml').read_text(encoding='utf-8'))


def test_config_is_exact_v341_copy_except_identity():
    ec,ac=configs();_,old=configs('341')
    expected=deepcopy(old);expected['environment']['environment_version']='3.42'
    assert ac==expected
    assert Path('configs/maac_4v4_v342.yaml').read_text(encoding='utf-8')==Path(
        'configs/maac_4v4_v341.yaml').read_text(encoding='utf-8').replace("'3.41'","'3.42'")
    assert DIMENSIONS['3.42']==environment_dimensions(ec)==(45,3,4)
    validate_config_pair(ec,ac)
    assert 'research_reward' not in ec and 'candidate_reward' not in ec


@pytest.mark.parametrize('version',['341','342','348','350'])
def test_registered_pairs_still_valid(version):
    ec,ac=configs(version);validate_config_pair(ec,ac)
    assert environment_dimensions(ec)==(45,3,4)


@pytest.mark.parametrize('field,value',[('observation_dim',49),('action_dim',4),('num_agents',5),
                                     ('hidden_dim',256),('attention_heads',2)])
def test_exact_dimensions_rejected(field,value):
    ec,ac=configs();ac['network'][field]=value
    with pytest.raises(RuntimeError,match='exact dimensions'):validate_config_pair(ec,ac)


@pytest.mark.parametrize('version',['341','348','350'])
def test_cross_version_config_rejected(version):
    ec,_=configs();_,ac=configs(version)
    with pytest.raises(RuntimeError,match='identity mismatch'):validate_config_pair(ec,ac)


def test_original_observation_reset_and_step():
    ec,_=configs();env=MultiUAVCombatEnv(ec)
    obs,info=env.reset(342)
    assert type(env).__name__=='CombatEnvironmentV342' and obs.shape==(4,45)
    obs,reward,_,_,info=env.step(np.zeros((4,3)))
    np.testing.assert_array_equal(obs,build_team_observations_v338(env.red,env.blue,ec['observation']))
    np.testing.assert_allclose(reward,sum(info[f'r{i}_rewards'] for i in range(1,5)),rtol=0,atol=1e-12)
    assert not info['pbrs_enabled']


def test_real_window_without_attempt_is_not_fire_rate(tmp_path,capsys):
    ec,ac=configs();env=MultiUAVCombatEnv(ec);env.reset(342)
    env._advance=lambda states,actions:np.zeros(4)
    env.red=[AircraftState(0,20*i,-3000,225,0,0) for i in range(4)]
    env.blue=[AircraftState(1500,20*i,-3000,225,0,np.pi) for i in range(4)]
    env.red_fire_cooldown_steps.fill(10);env.blue_fire_cooldown_steps.fill(10)
    _,reward,_,_,info=env.step(np.zeros((4,3)),np.zeros((4,3)))
    assert info['red_first_fire_window_step']==1 and info['red_first_attempt_step'] is None
    assert info['red_fire_attempts']==info['red_step_fire_attempts']==0
    row=dict(info,episode_return=float(reward.sum()),mean_agent_episode_return=float(reward.mean()))
    metrics=evaluation_metrics(aggregate_combat_records([row]),ec)
    assert metrics['red_fire_window_episode_rate']==1.
    assert metrics['red_attempt_episode_rate']==metrics['fire_rate']==0.
    runner=MAACTrainingRunner.__new__(MAACTrainingRunner)
    runner.env_config,runner.config,runner.records=ec,ac,[row]
    assert runner.recent()['fire_rate']==0.
    runner.trainer=SimpleNamespace(sampled_steps=12);runner.total_steps=24;runner.last_metrics={}
    runner.log_train(runner.recent())
    assert 'fire=0.000' in capsys.readouterr().out


def test_evaluation_fire_alias_uses_attempt_rate():
    ec,_=configs()
    raw=dict(red_attempt_episode_rate=.25,red_fire_window_episode_rate=1.,red_kill_episode_rate=.1)
    original=deepcopy(raw);result=evaluation_metrics(raw,ec)
    assert raw==original and result['fire_rate']==.25
