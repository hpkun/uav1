"""Pitch-independent height LOS geometry and v3.12 historical isolation."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v312 import CombatEnvironmentV312
from env.combat_v313 import CombatEnvironmentV313
from env.config import load_config, validate_config, environment_dimensions
from env.v36_geometry import combat_geometry
from env.v313_geometry import combat_geometry_v313
from env.v311_weapon import MADSACFourKmWeapon
from env.v313_weapon import MADSACPaperHAWeapon
from env.v313_reward import tactical_rewards
from tests.test_combat_v312 import state
from tools._prepare_v313 import capture312

CONFIG = 'configs/combat_environment_v313.yaml'
PITCHES = np.deg2rad([-60, -30, 0, 30, 60])


def env(version='313'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(1)
    return e


@pytest.mark.parametrize('horizontal,delta_alt,expected', [
    (1000, 0, 0), (1000, 1000, np.pi/4), (1000, -1000, np.pi/4),
    (0, 1000, np.pi/2), (0, -1000, np.pi/2), (0, 0, 0)])
def test_A_to_F_height_los_pitch_independent(horizontal, delta_alt, expected):
    for pitch in PITCHES:
        a, b = state(theta=pitch, psi=.23), state(x=horizontal, altitude=3000+delta_alt, psi=-.31)
        g = combat_geometry_v313([a], [b])
        assert set(g) == {'distance', 'ata', 'ha', 'aa', 'los_elevation'}
        assert all(np.isfinite(value).all() for value in g.values())
        assert g['ha'][0, 0] == pytest.approx(expected, abs=1e-14)
        assert 0 <= g['ha'][0, 0] <= np.pi/2
        if horizontal == 0:
            assert g['ata'][0, 0] == 0
            assert g['aa'][0, 0] == pytest.approx(.54)
        assert np.sign(g['los_elevation'][0, 0]) == np.sign(delta_alt)


def test_1000_random_pairs_five_pitches_and_old_ata_aa_distance_exact():
    rng = np.random.default_rng(313)
    errors = {k: 0. for k in ('distance', 'ata', 'aa', 'ha')}
    for _ in range(1000):
        xyz = rng.uniform(-6000, 6000, (2, 3))
        psi = rng.uniform(-np.pi, np.pi, 2)
        a = state(x=xyz[0, 0], y=xyz[0, 1], altitude=xyz[0, 2], psi=psi[0])
        b = state(x=xyz[1, 0], y=xyz[1, 1], altitude=xyz[1, 2], psi=psi[1])
        reference = combat_geometry_v313([a], [b])
        for pitch in PITCHES:
            a.theta = pitch
            current = combat_geometry_v313([a], [b])
            old = combat_geometry([a], [b])
            for key in errors:
                errors[key] = max(errors[key], float(np.max(np.abs(current[key]-reference[key]))))
                np.testing.assert_allclose(current[key], reference[key], rtol=0, atol=1e-12)
            for key in ('distance', 'ata', 'aa', 'los_elevation'):
                np.testing.assert_array_equal(current[key], old[key])
    assert max(errors.values()) == 0.


def test_old_and_new_qualification_difference_same_height_pitch60():
    old, new = env('312'), env()
    a, b = state(theta=np.pi/3), state(x=3000, psi=np.pi)
    assert combat_geometry([a], [b])['ha'][0, 0] == pytest.approx(np.pi/3)
    assert combat_geometry_v313([a], [b])['ha'][0, 0] == 0.
    assert not old.weapon.qualifies(a, b, 6)
    assert new.weapon.qualifies(a, b, 6)


@pytest.mark.parametrize('excess,qualified', [(0., True), (1e-5, False)])
def test_weapon_height30_boundary(excess, qualified):
    e = env()
    a = state(theta=np.pi/6)
    b = state(x=3000, altitude=3000+3000*np.tan(np.pi/6)+excess)
    assert e.weapon.qualifies(a, b, 6) == qualified


@pytest.mark.parametrize('change', ['dead_attacker', 'dead_target', 'no_ammo', 'range', 'ata', 'ha'])
def test_weapon_frozen_rejections(change):
    e = env()
    a, b, ammo = state(), state(x=3000, psi=np.pi), 6
    if change == 'dead_attacker': a.alive = False
    if change == 'dead_target': b.alive = False
    if change == 'no_ammo': ammo = 0
    if change == 'range': b.x = 4000.01
    if change == 'ata': a.psi = np.pi/6+1e-5
    if change == 'ha': b.z -= 3000*np.tan(np.pi/6)+1e-5
    assert not e.weapon.qualifies(a, b, ammo)


def test_sampler_receives_new_ha_and_equation8_is_inherited_shared_gaussian():
    e = env()
    a, b = state(theta=np.pi/3), state(x=3000)
    with patch.object(MADSACPaperHAWeapon, 'attempt_hit', return_value=True) as hit:
        assert e.attempt_hit_for_pair(a, b)
        hit.assert_called_once_with(e.rng, 3000., 0., 0.)
    assert MADSACPaperHAWeapon.attempt_hit is MADSACFourKmWeapon.attempt_hit
    assert MADSACPaperHAWeapon.expected_hit_probability is MADSACFourKmWeapon.expected_hit_probability
    class Gaussian:
        calls = 0
        def normal(self, mean, std):
            assert (mean, std) == (0., 1.)
            self.calls += 1
            return -.2
    rng = Gaussian()
    distance, ata, ha = 3500., .2, .3
    threshold = np.pi*np.exp(-distance/e.weapon.effective_hit_distance)
    expected = ata + e.weapon.ata_noise_scale*(-.2) <= threshold and ha + e.weapon.ha_noise_scale*(-.2) <= threshold
    assert e.weapon.attempt_hit(rng, distance, ata, ha) == expected and rng.calls == 1


@pytest.mark.parametrize('distance', [3000., 5000.])
def test_reward_red_pitch_and_blue_reverse_pitch_independent(distance):
    e = env()
    red, blue = [state()], [state(x=distance)]
    expected = tactical_rewards(red, blue, e.config['reward'])
    for red_pitch in PITCHES:
        for blue_pitch in PITCHES:
            red[0].theta, blue[0].theta = red_pitch, blue_pitch
            current = tactical_rewards(red, blue, e.config['reward'])
            for key in current:
                np.testing.assert_array_equal(current[key], expected[key])
    assert expected['guide'][0] == (0.001 if distance >= 4000 else 0.)
    assert expected['tactical_offense'][0] == (.1 if distance <= 4000 else 0.)
    # Reverse threat is tested in the opposite same-heading configuration.
    red, blue = [state(x=distance)], [state()]
    expected = tactical_rewards(red, blue, e.config['reward'])
    assert expected['tactical_defense'][0] == (-.15 if distance <= 4000 else 0.)
    for pitch in PITCHES:
        blue[0].theta = pitch
        np.testing.assert_array_equal(tactical_rewards(red, blue, e.config['reward'])['tactical_defense'],
                                      expected['tactical_defense'])


@pytest.mark.parametrize('degrees,norm', [(0, 0), (30, 1/3), (45, .5), (90, 1)])
def test_observation_ha_normalization_and_common_geometry(degrees, norm):
    e, old = env(), env('312')
    e.red[0] = state()
    e.blue[0] = state(x=0 if degrees == 90 else 1000,
        altitude=4000 if degrees == 90 else 3000+1000*np.tan(np.deg2rad(degrees)))
    for pitch in PITCHES:
        e.red[0].theta = pitch
        old.red, old.blue = deepcopy(e.red), deepcopy(e.blue)
        obs, legacy = e._observations(), old._observations()
        g = combat_geometry_v313(e.red, e.blue)
        assert obs.shape == (5, 56)
        assert obs[0, 35] == pytest.approx(norm)
        np.testing.assert_allclose(obs[0, [31, 33, 34, 35]]*[10000, np.pi, np.pi, np.pi/2],
            [g['distance'][0, 0], g['aa'][0, 0], g['ata'][0, 0], g['ha'][0, 0]], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(obs[:, :31], legacy[:, :31])
        for j in range(5):
            np.testing.assert_array_equal(obs[:, 31+5*j:35+5*j], legacy[:, 31+5*j:35+5*j])
        assert obs[0, 6] == pitch/(np.pi/3)
        old_ha = combat_geometry(old.red, old.blue)['ha'][0, 0]
        assert legacy[0, 35] == old_ha/np.pi


def test_all_blue_observable_fixed_order_dead_slots_zero_and_no_old_geometry_or_radar():
    e = env()
    e.red[0] = state()
    e.blue[0] = state(x=-11000, v=250.)
    e.blue[2].alive = False
    e.red[1].alive = False
    with (patch('env.v36_geometry.combat_geometry', side_effect=AssertionError('old geometry')),
          patch('env.sensor.sensor_geometry', side_effect=AssertionError('radar'))):
        obs = e._observations()
        e.weapon.eligibility(e.red, e.blue, e.red_ammo)
        e.dense_combat_reward()
        e.attempt_hit_for_pair(e.red[0], e.blue[0])
    assert obs[0, 31] == 1.1 and obs[0, 32] == 1/3
    assert not obs[0, 41:46].any() and not obs[0, 7:13].any() and not obs[1].any()
    before = obs.copy()
    e.red_ammo.fill(0)
    e.blue_ammo.fill(0)
    e.config['sensor'] = dict(range_max=0., off_boresight_angle_max=0.)
    np.testing.assert_array_equal(e._observations(), before)


def test_class_only_overrides_five_hooks_frozen_initialization_and_parameters():
    assert set(CombatEnvironmentV313.__dict__)-{'__module__', '__doc__'} == {
        'environment_version', 'build_weapon', 'attempt_hit_for_pair', 'dense_combat_reward', '_observations'}
    for method in ('__init__', 'step', 'reset', 'initial_states', '_advance', '_entry_attempts',
                   '_resolve_combat', '_resolve_noncombat_losses', '_outcome'):
        assert getattr(CombatEnvironmentV313, method) is getattr(CombatEnvironmentV312, method)
    old, new = env('312'), env()
    cfg = deepcopy(new.config)
    cfg['environment_version'] = '3.12'
    assert cfg == old.config
    for side in ('red', 'blue'):
        for a, b in zip(getattr(old, side), getattr(new, side)):
            np.testing.assert_array_equal(a.as_array(), b.as_array())
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    assert environment_dimensions(new.config) == (56, 3, 5)
    for section, key, value in [('weapon', 'range_max', 2000.), ('reward', 'guide_reward', .1),
                                ('blue_policy', 'desired_speed', 260.), ('observation', 'altitude_scale', 10000.)]:
        cfg = deepcopy(new.config)
        cfg[section][key] = value
        with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('name', ['mappo', 'rmappo', 'ea_mappo', 'stea_mappo'])
def test_algorithm_parameters_byte_identical_to_v312(name):
    assert Path(f'configs/{name}_5v5_v313.yaml').read_bytes() == Path(f'configs/{name}_5v5_v312.yaml').read_bytes()


def test_v312_full_trajectory_reference_unchanged():
    assert capture312() == json.loads(Path('tests/v313_v312_reference.json').read_text())


def test_same_state_fork_only_after_ha_qualification_change():
    from tools.audit_combat_v313 import run_audit
    audit = run_audit()
    assert audit['pre_first_qualification_motion_and_rng_identical']
    assert audit['qualification_snapshot_v312']['red_eligible'] is False
    assert audit['qualification_snapshot_v313']['red_eligible'] is True
