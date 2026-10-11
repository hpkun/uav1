"""Paper-range rewards, unchanged hardware, exact diagnostics and history."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config, validate_config, environment_dimensions
from env.models import AircraftState
from env.sensor import sensor_geometry
from env.combat_v38 import CombatEnvironmentV38
from env.combat_v39 import CombatEnvironmentV39
from env.v39_reward import tactical_rewards
from tools._prepare_v39 import capture38
from tools.audit_combat_v39 import check_decomposition, initial_visibility

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return load_config(ROOT / 'configs/combat_environment_v39.yaml')


def state(x=0, y=0, z=-3000, psi=0, theta=0, alive=True):
    return AircraftState(x, y, z, 225, theta, psi, alive)


def pair(d=3500, ata=0, ha=0, aa=0):
    return state(psi=-np.deg2rad(ata), theta=-np.deg2rad(ha)), state(d, psi=-np.deg2rad(aa))


def rewards(red, blue):
    c = cfg()
    return tactical_rewards(red, blue, c['sensor'], c['reward'])


@pytest.mark.parametrize('d,ata,ha,expected', [
    (9000, 0, 0, .001), (5000, 20, 20, .001), (3999, 0, 0, 0),
    (5000, 31, 0, 0), (5000, 0, 31, 0), (4000, 30, 30, .001),
    (10000, 0, 0, .001), (10001, 0, 0, 0)])
def test_r3_exact_conditions(d, ata, ha, expected):
    a, b = pair(d, ata, ha)
    assert rewards([a], [b])['guide'][0] == expected


@pytest.mark.parametrize('reason', ['fov', 'dead_red', 'dead_blue'])
def test_r3_visibility_gate_and_no_multi_target_sum(reason):
    a, b = pair(9000)
    assert rewards([a], [b, deepcopy(b)])['guide'][0] == .001
    if reason == 'fov':
        a.psi = np.pi
    elif reason == 'dead_red':
        a.alive = False
    else:
        b.alive = False
    assert rewards([a], [b])['guide'][0] == 0


@pytest.mark.parametrize('d,ata,ha,aa,value', [
    (3500, 25, 25, 20, .01), (3500, 10, 10, 20, .02), (3500, 4, 4, 20, .1),
    (4001, 0, 0, 20, 0), (3500, 0, 0, 31, 0),
    (4000, 30, 30, 30, .01), (4000, 15, 15, 30, .02), (4000, 5, 5, 30, .1),
    (3500, 31, 0, 0, 0), (3500, 0, 31, 0, 0)])
def test_r4_offense(d, ata, ha, aa, value):
    a, b = pair(d, ata, ha, aa)
    assert rewards([a], [b])['tactical_offense'][0] == value


@pytest.mark.parametrize('d,angle,aa,value', [
    (3500, 25, 20, -.015), (3500, 10, 20, -.025), (3500, 4, 20, -.15),
    (4000, 5, 30, -.15), (4001, 4, 20, 0), (3500, 4, 31, 0),
    (3500, 31, 20, 0)])
def test_reverse_threat(d, angle, aa, value):
    b, a = pair(d, angle, angle, aa)
    b.x = -d
    a.x = 0
    assert not sensor_geometry([a], [b], cfg()['sensor'])[0].any()
    assert rewards([a], [b])['tactical_defense'][0] == value


@pytest.mark.parametrize('d', [2001, 3500, 4000])
def test_tactical_range_is_not_weapon_range(d):
    e = MultiUAVCombatEnv(cfg())
    a, b = pair(d)
    assert not e.weapon.qualifies(a, b, 6)
    assert rewards([a], [b])['tactical_offense'][0] == .1
    assert rewards([a], [b])['guide'][0] == (.001 if d == 4000 else 0)


def test_strongest_target_and_simultaneous_offense_defense():
    a = state()
    forward, behind = state(3500), state(-3500)
    result = rewards([a], [forward, deepcopy(forward), behind, deepcopy(behind)])
    assert result['tactical_offense'][0] == .1
    assert result['tactical_defense'][0] == -.15
    assert result['tactical'][0] == pytest.approx(-.05)
    forward.alive = behind.alive = False
    assert not rewards([a], [forward, behind])['tactical'].any()


@pytest.mark.parametrize('d,heading,visible', [(9999, 0, True), (10000, 0, True),
    (10001, 0, False), (9000, 91, False)])
def test_sensor_distance_and_unchanged_fov(d, heading, visible):
    a, b = pair(d, heading)
    assert bool(sensor_geometry([a], [b], cfg()['sensor'])[0][0, 0]) == visible
    assert cfg()['sensor'] == dict(range_max=10000., off_boresight_angle_max=np.pi / 2)


def test_100_seed_visibility_and_identical_v38_spawn_rng():
    audit = initial_visibility()
    assert len(audit['rows']) == 100
    for row in audit['rows']:
        seed = row['seed']
        old = MultiUAVCombatEnv(ROOT / 'configs/combat_environment_v38.yaml')
        new = MultiUAVCombatEnv(cfg())
        old.reset(seed)
        new.reset(seed)
        np.testing.assert_array_equal([s.as_array() for s in old.red + old.blue],
            [s.as_array() for s in new.red + new.blue])
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert old.fixed_policy.rng.bit_generator.state == new.fixed_policy.rng.bit_generator.state
        assert 0 < row['red_visible_pairs'] <= 25 and 0 < row['blue_visible_pairs'] <= 25
        for side in ('red', 'blue'):
            assert not new.weapon.eligibility(getattr(new, side), getattr(new,
                'blue' if side == 'red' else 'red'), getattr(new, f'{side}_ammo'))[0].any()


def test_step_and_episode_reward_decomposition_safety_reset_evaluation():
    from algorithm.common.evaluator import aggregate_combat_records
    e = MultiUAVCombatEnv(cfg())
    _, initial = e.reset(1)
    assert initial['episode_paper_reward_total'] == initial['episode_extended_reward_total'] == 0
    # Explicit proximity makes the preserved safety extension nonzero.
    e.red = [state(x=i * 50) for i in range(5)]
    e.blue = [state(x=3500 + i * 50) for i in range(5)]
    total = np.zeros(5)
    with patch.object(e, '_advance', return_value=np.zeros(5)):
        for _ in range(20):
            _, r, _, _, info = e.step(np.zeros((5, 3)), np.zeros((5, 3)))
            total += r
            check_decomposition(r, info, total)
    assert info['episode_safety_total'] < 0
    row = dict(info, episode_return=float(total.sum()), mean_agent_episode_return=float(total.mean()))
    result = aggregate_combat_records([row])
    for name in ('paper_reward', 'safety', 'extended_reward'):
        assert result[f'average_episode_{name}_total'] == info[f'episode_{name}_total']
    _, reset = e.reset(1)
    for name in ('paper_reward', 'safety', 'extended_reward'):
        assert reset[f'episode_{name}_total'] == 0


@pytest.mark.parametrize('loss', ['boundary', 'ground', 'combat'])
def test_event_credit_and_decomposition(loss):
    e = MultiUAVCombatEnv(cfg())
    e.reset(1)
    e.red = [state(alive=i == 0) for i in range(5)]
    e.blue = [state(-1000, alive=i == 0) for i in range(5)]
    if loss == 'boundary':
        e.red[0].x = 5001
    elif loss == 'ground':
        e.red[0].z = 1
    with patch.object(e, '_advance', return_value=np.zeros(5)), \
         patch.object(type(e.weapon), 'attempt_hit', return_value=True):
        _, r, _, _, info = e.step(np.zeros((5, 3)), np.zeros((5, 3)))
    check_decomposition(r, info, r)
    assert info['r1_rewards'][0] == (0 if loss == 'boundary' else -10)
    assert info['r2_rewards'][0] == (-10 if loss == 'boundary' else 0)


def test_inherited_mechanisms_and_configs_are_frozen():
    old = load_config(ROOT / 'configs/combat_environment_v38.yaml')
    new = cfg()
    assert environment_dimensions(new) == (66, 3, 5)
    for key in old.keys() - {'environment_version', 'sensor', 'reward'}:
        assert old[key] == new[key]
    assert new['reward'] == dict(old['reward'], guide_distance_min=4000., tactical_distance_max=4000.)
    assert old['sensor']['range_max'] == 4000 and new['weapon']['range_max'] == 2000
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        assert (ROOT / f'configs/{name}_5v5_v39.yaml').read_bytes() == (ROOT / f'configs/{name}_5v5_v38.yaml').read_bytes()
    for method in ('initial_states', '_advance', '_entry_attempts', '_resolve_combat',
        '_resolve_noncombat_losses', '_outcome', 'build_weapon', 'attempt_hit_for_pair',
        'reward_aliases', 'event_reward_extension', '_observations'):
        assert getattr(CombatEnvironmentV39, method) is getattr(CombatEnvironmentV38, method)


@pytest.mark.parametrize('section,key,value', [('sensor', 'range_max', 10001),
    ('sensor', 'off_boresight_angle_max', 1.), ('reward', 'guide_distance_min', 2000),
    ('reward', 'tactical_distance_max', 2000), ('reward', 'guide_reward', .002),
    ('reward', 'tactical_aspect_max', np.pi / 2), ('reward', 'safe_distance', 100),
    ('reward', 'safety_penalty_scale', .3), ('weapon', 'range_max', 4000),
    ('scenario', 'center_radius', 5000), ('arena', 'radius', 6000)])
def test_strict_rejections(section, key, value):
    c = cfg()
    c[section][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_complete_pre_extension_v38_trajectory_reference():
    assert capture38() == json.loads((ROOT / 'tests/v39_v38_reference.json').read_text())


def test_v38_legacy_reward_distances_not_rewritten():
    from env.v36_reward import tactical_rewards as old_reward
    old = load_config(ROOT / 'configs/combat_environment_v38.yaml')
    for distance, guide, offense in ((3000, .001, 0), (3500, .001, 0), (4000, .001, 0), (5000, 0, 0)):
        a, b = pair(distance)
        r = old_reward([a], [b], old['sensor'], old['weapon'], old['reward'])
        assert r['guide'][0] == guide and r['tactical_offense'][0] == offense


def test_v39_rollout_logging_without_finished_episodes():
    from types import SimpleNamespace
    from algorithm.mappo.runner import MAPPOTrainingRunner
    runner = object.__new__(MAPPOTrainingRunner)
    runner.env_config = cfg()
    runner.trainer = SimpleNamespace(sampled_steps=1, ppo_update_count=1)
    runner.last_metrics = {}
    e = MultiUAVCombatEnv(cfg())
    e.reset(1)
    _, reward, _, _, info = e.step(np.zeros((5, 3)))
    runner._begin_rollout_metrics()
    runner._accumulate_rollout_metrics(SimpleNamespace(rewards=reward[None], infos=[info]), [])
    r = runner._finalize_rollout_metrics()
    assert r['episode_paper_reward_total'] is None and r['episode_extended_reward_total'] is None
    assert r['mean_paper_reward'] == pytest.approx(info['paper_rewards'].mean())
    assert r['mean_safety_reward'] == pytest.approx(info['safety_rewards'].mean())
    assert r['mean_extended_reward'] == pytest.approx(reward.mean())
    assert r['mean_extended_reward'] == pytest.approx(r['mean_paper_reward'] + r['mean_safety_reward'])
