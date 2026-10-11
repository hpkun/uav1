"""Nearest pursuit wrapper correctness, deterministic RNG and frozen history."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from env.models import AircraftState
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from env.v310_policy import PaperNearestTargetBluePolicy
from env.v35_policy import PaperTieredBluePolicy
from env.combat_v310 import CombatEnvironmentV310
from env.combat_v39 import CombatEnvironmentV39
from env.sensor import sensor_geometry
from tools._capture_v33_reference import canonical, states
from tools._prepare_v310 import capture39
from tools.audit_combat_v310 import expected_targets, trial
from tools.audit_combat_v39 import check_decomposition

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return load_config(ROOT / 'configs/combat_environment_v310.yaml')


def state(x=0, y=0, z=-3000, psi=0, theta=0, alive=True, v=225):
    return AircraftState(x, y, z, v, theta, psi, alive)


def policy():
    c = cfg()
    return PaperNearestTargetBluePolicy(c['blue_policy'], c['action'])


def test_global_nearest_outside_fov():
    p = policy()
    own = state()
    red = [state(8000), state(-3000), state(5000)]
    assert not sensor_geometry([own], [red[1]], cfg()['sensor'])[0].any()
    actions = p.team_actions([own], red)
    assert p.last_targets == [1]
    np.testing.assert_array_equal(actions[0], NearestTargetPursuitPolicy.action(p, own, red))


def test_immediate_switch_and_target_death():
    p = policy()
    own = state()
    red = [state(3000), state(4000)]
    p.team_actions([own], red)
    assert p.last_targets == [0] and p.target_switches == 0
    red[0].x, red[1].x = 4500, 2500
    p.team_actions([own], red)
    assert p.last_targets == [1] and p.target_switches == 1
    red[1].alive = False
    p.team_actions([own], red)
    assert p.last_targets == [0] and p.target_switches == 2


@pytest.mark.parametrize('order', [(1000, -1000), (-1000, 1000)])
def test_exact_tie_smallest_index(order):
    p = policy()
    p.team_actions([state()], [state(order[0]), state(order[1])])
    assert p.last_targets == [0]


def test_true_3d_selection_not_horizontal_only():
    p = policy()
    p.team_actions([state(z=-500)], [state(1000, z=-5500), state(2000, z=-500)])
    assert p.last_targets == [1]


def test_outnumbered_does_not_escape():
    p = policy()
    own = state(psi=.4)
    targets = [state(1000 + i * 500, y=100) for i in range(5)]
    action = p.team_actions([own], targets)[0]
    assert p.last_targets == [0]
    np.testing.assert_array_equal(action, NearestTargetPursuitPolicy.action(p, own, targets))


def test_dispersion_and_all_invisible_no_random_target_or_center_action():
    p = policy()
    own = state()
    targets = [state(-1000), state(-4500, 500), state(-4500, -2000), state(-3500, 3000)]
    assert not sensor_geometry([own], targets, cfg()['sensor'])[0].any()
    for _ in range(10):
        action = p.team_actions([own], targets)
        assert p.last_targets == [0] and action.any()
    assert p.target_switches == 0 and p.pursuit_agent_steps == 10


@pytest.mark.parametrize('missing', ['all_dead_red', 'no_red', 'dead_blue'])
def test_no_target_or_dead_blue_zero_action(missing):
    p = policy()
    own, targets = state(), [state(1000)]
    if missing == 'all_dead_red':
        targets[0].alive = False
    elif missing == 'no_red':
        targets = []
    else:
        own.alive = False
    np.testing.assert_array_equal(p.team_actions([own], targets), np.zeros((1, 3)))
    assert p.last_targets == [None] and not p.pursuit_agent_steps


@pytest.mark.parametrize('height,pitch_sign', [(3000, 0), (4000, 1), (2000, -1)])
@pytest.mark.parametrize('y,heading_sign', [(1000, 1), (-1000, -1)])
def test_3d_heading_and_ned_pitch(height, pitch_sign, y, heading_sign):
    p = policy()
    action = p.team_actions([state()], [state(1000, y, -height)])[0]
    assert np.sign(action[0]) == heading_sign and np.sign(action[1]) == pitch_sign
    assert action[2] == .5  # original speed_delta_max=50, desired 250, own 225


def test_100_random_legal_scenes_exact_base_outputs_and_targets():
    rng = np.random.default_rng(310)
    c = cfg()
    p = policy()
    base = NearestTargetPursuitPolicy(c['blue_policy'], c['action'])
    def random_state():
        radius, heading = rng.uniform(0, 4999), rng.uniform(-np.pi, np.pi)
        return state(radius*np.cos(heading), radius*np.sin(heading), -rng.uniform(1, 5999),
            rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi/3, np.pi/3),
            bool(rng.random() > .2), rng.uniform(150, 300))
    for _ in range(100):
        blue = [random_state() for _ in range(5)]
        red = [random_state() for _ in range(5)]
        np.testing.assert_array_equal(p.team_actions(blue, red), base.team_actions(blue, red))
        assert p.last_targets == expected_targets(blue, red)
    # No controller math copied: inherited methods are the exact base functions.
    for method in ('nearest_target_index', 'action', 'action_toward'):
        assert getattr(PaperNearestTargetBluePolicy, method) is getattr(NearestTargetPursuitPolicy, method)


def test_policy_reset_and_calls_do_not_create_or_consume_rng():
    p = policy()
    with patch.object(np.random, 'default_rng', side_effect=AssertionError('policy must not create RNG')), \
         patch.object(np.random, 'SeedSequence', side_effect=AssertionError('no random stream')), \
         patch.object(np.random, 'random', side_effect=AssertionError('no sampling')), \
         patch.object(np.random, 'choice', side_effect=AssertionError('no random target')):
        p.reset(1)
        p.team_actions([state()], [state(1000)])
        p.reset(99)
    assert p.last_targets == [] and p.target_switches == p.pursuit_agent_steps == 0
    assert not hasattr(p, 'rng')
    # Diagnostics can change without changing control.
    p.last_targets, p.target_switches = [123], 999
    a = p.team_actions([state()], [state(1000)])
    p.reset(1)
    np.testing.assert_array_equal(a, p.team_actions([state()], [state(1000)]))


@pytest.mark.parametrize('seed', [1, 31, 10000000])
def test_repeat_reset_actions_initialization_and_rng_before_weapon(seed):
    e = MultiUAVCombatEnv(cfg())
    old = MultiUAVCombatEnv(ROOT / 'configs/combat_environment_v39.yaml')
    e.reset(seed)
    old.reset(seed)
    np.testing.assert_array_equal([s.as_array() for s in e.red+e.blue], [s.as_array() for s in old.red+old.blue])
    assert e.rng.bit_generator.state == old.rng.bit_generator.state
    initial_rng = deepcopy(e.rng.bit_generator.state)
    def execute():
        result = []
        for _ in range(5):
            obs, reward, term, trunc, info = e.step(np.zeros((5, 3)))
            assert info['red_fire_attempts'] == info['blue_fire_attempts'] == 0
            assert e.rng.bit_generator.state == initial_rng
            result.append(canonical((states(e), obs, reward, info)))
        return result
    first = execute()
    e.reset(seed)
    assert e.fixed_policy.last_targets == [] and not e.fixed_policy.target_switches
    assert execute() == first
    assert not hasattr(e.fixed_policy, 'rng')


def test_weapon_rng_unchanged_with_external_blue_actions():
    old = MultiUAVCombatEnv(ROOT / 'configs/combat_environment_v39.yaml')
    new = MultiUAVCombatEnv(cfg())
    for e in (old, new):
        e.reset(31)
        e.red = [state(alive=i == 0) for i in range(5)]
        e.blue = [state(1000, psi=np.pi, alive=i == 0) for i in range(5)]
    results = []
    for e in (old, new):
        with patch.object(e, '_advance', return_value=np.zeros(5)):
            results.append(e.step(np.zeros((5, 3)), np.zeros((5, 3))))
    a, b = results
    assert a[4]['red_fire_attempts'] == b[4]['red_fire_attempts'] == 1
    assert a[4]['blue_fire_attempts'] == b[4]['blue_fire_attempts'] == 1
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:4] == b[2:4]
    assert canonical({k:v for k,v in a[4].items() if k != 'environment_version'}) == canonical({k:v for k,v in b[4].items() if k != 'environment_version'})
    assert new.fixed_policy.last_targets == []  # external override never calls policy


@pytest.mark.parametrize('cause', ['horizontal', 'ground', 'ceiling', 'both'])
def test_blue_boundary_return_exactly_inherited(cause):
    old = MultiUAVCombatEnv(ROOT / 'configs/combat_environment_v39.yaml')
    new = MultiUAVCombatEnv(cfg())
    for e in (old, new):
        e.reset(31)
        e.blue[0] = state(5001 if cause in ('horizontal', 'both') else 2000, psi=.3,
            z=0 if cause == 'ground' else (-6001 if cause in ('ceiling', 'both') else -3000), theta=.2)
        e._resolve_noncombat_losses()
    assert canonical(states(old)) == canonical(states(new))
    assert old.blue_returns == new.blue_returns and old.combat_counts == new.combat_counts
    assert new.blue[0].alive and new.blue_returns['boundary'] == 1


def test_minimal_class_frozen_config_and_legacy_policy():
    old = load_config(ROOT / 'configs/combat_environment_v39.yaml')
    new = cfg()
    assert new == dict(old, environment_version='3.10', blue_policy=dict(desired_speed=250.))
    assert environment_dimensions(new) == (66, 3, 5)
    assert isinstance(MultiUAVCombatEnv(new).fixed_policy, PaperNearestTargetBluePolicy)
    assert type(MultiUAVCombatEnv(old).fixed_policy) is PaperTieredBluePolicy
    assert set(CombatEnvironmentV310.__dict__) - {'__module__', '__doc__'} == {'environment_version', '__init__'}
    for method in ('reset', 'step', 'initial_states', 'dense_combat_reward', '_advance', '_entry_attempts',
        '_resolve_combat', '_resolve_noncombat_losses', '_observations', 'build_weapon', 'attempt_hit_for_pair'):
        assert getattr(CombatEnvironmentV310, method) is getattr(CombatEnvironmentV39, method)
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        assert (ROOT / f'configs/{name}_5v5_v310.yaml').read_bytes() == (ROOT / f'configs/{name}_5v5_v39.yaml').read_bytes()


@pytest.mark.parametrize('key,value', [('close_distance', 1800.), ('dispersion_threshold', 1500.),
    ('random_target', True), ('target_lock', True), ('guard_radius', 4500.), ('desired_speed', 251.),
    ('desired_speed', True), ('desired_speed', float('nan')), ('undefined_behavior', 1.)])
def test_blue_schema_rejects_extra_behaviors_and_speed_changes(key, value):
    c = cfg()
    c['blue_policy'][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_no_tiered_diagnostics_and_evaluator_reward_fields():
    from algorithm.common.evaluator import aggregate_combat_records
    e = MultiUAVCombatEnv(cfg())
    e.reset(1)
    _, reward, _, _, info = e.step(np.zeros((5, 3)))
    check_decomposition(reward, info, reward)
    assert info['blue_paper_nearest_agent_steps'] == 5 and len(info['blue_last_target_indices']) == 5
    assert not any(word in key for key in info for word in ('escape', 'guerrilla', 'centripetal'))
    r = aggregate_combat_records([dict(info, episode_return=float(reward.sum()), mean_agent_episode_return=float(reward.mean()))])
    assert r['average_blue_paper_nearest_agent_steps'] == 5 and r['average_blue_target_switches'] == 0
    assert 'average_episode_paper_reward_total' in r and 'average_episode_extended_reward_total' in r
    assert not any(word in key for key in r for word in ('escape', 'guerrilla', 'centripetal'))


@pytest.mark.parametrize('mode,seed', [('ZERO', 1), ('ZERO', 31), ('ZERO', 10000000),
    ('PURSUIT', 1), ('PURSUIT', 31), ('PURSUIT', 10000000)])
def test_each_agent_step_nearest_selection_sanity(mode, seed):
    summary, decisions = trial(mode, seed)
    assert summary['target_selection_violations'] == 0
    assert len(decisions) == summary['episode_length']


def test_complete_pre_extension_v39_reference():
    assert capture39() == json.loads((ROOT / 'tests/v310_v39_reference.json').read_text())
