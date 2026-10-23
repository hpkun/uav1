"""45D observation ablation: exact v3.46 dynamics, events and rewards."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v346 import CombatEnvironmentV346
from env.combat_v347 import CombatEnvironmentV347
from env.config import load_config, validate_config, environment_dimensions
from env.v338_observation import build_team_observations_v338
from env.v343_config import validate_shaping_gamma
from tools.audit_v339_attack_geometry import POLICIES, red_actions, states
from tests.test_combat_v344 import frozen


def config(version):
    return load_config(f'configs/combat_environment_v{version}.yaml')


def assert_observation(env, obs):
    assert obs.shape == (4, 45)
    np.testing.assert_array_equal(obs, build_team_observations_v338(
        env.red, env.blue, env.config['observation']))
    assert np.isfinite(obs).all()
    assert not obs[env.red_alive_mask == 0].any()


def compare_transition(a, b, x, y):
    assert_observation(b, y[0])
    np.testing.assert_array_equal(states(a), states(b))
    np.testing.assert_array_equal(x[1], y[1])
    assert x[2:4] == y[2:4]
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert set(x[4]) == set(y[4])
    for key in x[4]:
        if key != 'environment_version':
            np.testing.assert_equal(x[4][key], y[4][key], err_msg=key)
    for k in range(1, 5):
        np.testing.assert_array_equal(x[4][f'r{k}_rewards'], y[4][f'r{k}_rewards'])
    np.testing.assert_array_equal(y[1], sum(y[4][f'r{k}_rewards'] for k in range(1, 5)))


def test_registration_config_identity_and_no_new_information():
    a, b = config('346'), config('347')
    original = deepcopy(b)
    assert validate_config(b) is b and b == original
    assert environment_dimensions(b) == (45, 3, 4)
    b['environment_version'] = a['environment_version']
    assert a == b
    e = MultiUAVCombatEnv(config('347'))
    assert type(e) is CombatEnvironmentV347 and isinstance(e, CombatEnvironmentV346)
    assert set(CombatEnvironmentV347.__dict__) - {
        '__module__', '__doc__', 'environment_version', '_observations'} == set()
    assert_observation(e, e.reset(7)[0])
    e.red[0].alive = False
    e.blue[1].alive = False
    assert_observation(e, e._observations())
    for algo in ('mappo', 'ippo', 'maac'):
        old = yaml.safe_load(Path(f'configs/{algo}_4v4_v346.yaml').read_text())
        new = yaml.safe_load(Path(f'configs/{algo}_4v4_v347.yaml').read_text())
        validate_shaping_gamma(config('347'), new)
        assert new['network']['observation_dim'] == 45
        assert new['network']['action_dim'] == 3
        new['network']['observation_dim'] = 49
        new['environment'] = old['environment']
        assert old == new
        new['training']['gamma'] = .98
        with pytest.raises(ValueError, match='gamma'):
            validate_shaping_gamma(config('347'), new)


@pytest.mark.parametrize('section,key,value', [
    ('weapon', 'range_max', 2100), ('fire_control', 'dwell_seconds', .6),
    ('simulation', 'dt', .2), ('reward', 'kill_reward', 11),
    ('reward', 'offense_reward_5deg', .11), ('research_reward', 'gamma', .98),
    ('observation', 'extra_information', True)])
def test_strict_inherited_validation(section, key, value):
    c = config('347'); c[section][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('seed', [0, 7, 19])
def test_complete_fixed_action_trajectory_reward_and_events(policy, seed):
    a, b = (MultiUAVCombatEnv(config(v)) for v in ('346', '347'))
    a.reset(seed); assert_observation(b, b.reset(seed)[0])
    while True:
        action = red_actions(a, policy)[0]
        np.testing.assert_array_equal(action, red_actions(b, policy)[0])
        x, y = a.step(action), b.step(action)
        compare_transition(a, b, x, y)
        if x[2] or x[3]:
            break


@pytest.mark.parametrize('sharing', [False, True])
def test_fifth_step_dwell_death_and_credit(sharing):
    a, b = frozen('346', sharing), frozen('347', sharing)
    for _ in range(5):
        x = a.step(np.zeros((4, 3)), np.zeros((4, 3)))
        y = b.step(np.zeros((4, 3)), np.zeros((4, 3)))
        compare_transition(a, b, x, y)
    assert x[2] and not x[3]
    assert y[4]['red_dwell_completed'] == 4


def test_truncation_terminal_potential():
    a, b = frozen('346', True), frozen('347', True)
    a.max_steps = b.max_steps = 3
    for _ in range(3):
        x = a.step(np.zeros((4, 3)), np.zeros((4, 3)))
        y = b.step(np.zeros((4, 3)), np.zeros((4, 3)))
        compare_transition(a, b, x, y)
    assert y[3] and not y[4]['phi_next_for_shaping'].any()
