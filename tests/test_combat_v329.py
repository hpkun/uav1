"""v3.29 controller, frozen mechanics and environment protocol contracts."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v328 import CombatEnvironmentV328
from env.combat_v329 import CombatEnvironmentV329
from env.v329_control import action_to_control_v329, target_state_v329
from env.v310_policy import PaperNearestTargetBluePolicy
from env.math_utils import wrap_angle
from env.config import validate_config, environment_dimensions
from env.models import AircraftState
from tools.audit_combat_v317_v318 import FixedNormal


def env(version='329'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def state(v=225, theta=0, psi=0, alive=True):
    return AircraftState(0, 0, -3000, v, theta, psi, alive)


def control(e, s, a):
    return action_to_control_v329(s, a, e.config['action'], e.spec)


@pytest.mark.parametrize('theta', [-1., -.2, 0., .2, 1.])
def test_zero_action_general_state(theta):
    e = env(); s = state(theta=theta, psi=2.4)
    np.testing.assert_allclose(target_state_v329(s, [0, 0, 0], e.config['action'], e.spec),
                               [s.psi, s.theta, s.v], atol=1e-14)
    c = control(e, s, [0, 0, 0])
    assert c.nx == pytest.approx(math.sin(theta))
    assert c.nz == pytest.approx(math.cos(theta))
    assert c.phi == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(e.dynamics.derivatives(s, c)[3:], 0, atol=1e-14)


@pytest.mark.parametrize('dimension', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_unsaturated_channel_signs(dimension, sign):
    e = env(); s = state(); a = np.zeros(3); a[dimension] = .01 * sign
    rates = e.dynamics.derivatives(s, control(e, s, a))
    assert rates[{0: 5, 1: 4, 2: 3}[dimension]] * sign > 0


@pytest.mark.parametrize('psi, increment', [(math.pi-.01, .1), (-math.pi+.01, -.1)])
def test_heading_wrap_shortest_path(psi, increment):
    e = env(); s = state(psi=psi); action = [increment/math.pi, 0, 0]
    assert e.dynamics.derivatives(s, control(e, s, action))[5] == pytest.approx(increment/2)


@pytest.mark.parametrize('theta,v,sign', [(1., 295, 1), (-1., 155, -1)])
def test_targets_clipped_before_tracking(theta, v, sign):
    e = env(); s = state(v, theta); a = [0, sign, sign]
    _, pitch, speed = target_state_v329(s, a, e.config['action'], e.spec)
    assert pitch == sign * math.pi/3
    assert speed == (300 if sign > 0 else 150)
    c = control(e, s, a)
    assert e.dynamics.derivatives(s, c)[3] == pytest.approx((speed-v)/2)


def test_negative_vertical_load_clamped_and_zero_load_roll():
    e = env(); s = state()
    for heading in (-1, 0, 1):
        c = control(e, s, [heading, -1, 0])
        assert c.nz * math.cos(c.phi) >= -1e-12
        assert -math.pi/2 <= c.phi <= math.pi/2
        if heading == 0:
            assert c.nz == c.phi == 0


@pytest.mark.parametrize('psi', [-3., -.3, 0., .3, 3.])
@pytest.mark.parametrize('sign', [-1, 1])
def test_exact_pi_tie_preserves_command_sign(psi, sign):
    e = env(); s = state(psi=psi)
    rate = e.dynamics.derivatives(s, control(e, s, [sign, 0, 0]))[5]
    assert rate*sign > 0


def test_saturated_heading_can_reduce_vertical_tracking():
    e = env(); s = state()
    c = control(e, s, [1, .025, 0])
    # Radial projection preserves A/B ratio, not independent rate tracking.
    assert c.nz == pytest.approx(10)
    assert e.dynamics.derivatives(s, c)[4] < 0


@pytest.mark.parametrize('v', [150, 225, 300])
@pytest.mark.parametrize('a_psi', [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
@pytest.mark.parametrize('a_theta', [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
def test_heading_pitch_grid(v, a_psi, a_theta):
    e = env(); s = state(v); c = control(e, s, [a_psi, a_theta, 0])
    a_raw = 1 + v * (a_theta*math.pi/3/2) / 9.81
    b_raw = v * (a_psi*math.pi/2) / 9.81
    a = max(a_raw, 0); length = math.hypot(a, b_raw)
    scale = min(1., 10/length) if length else 1.
    np.testing.assert_allclose([c.nz*math.cos(c.phi), c.nz*math.sin(c.phi)],
                               [a*scale, b_raw*scale], atol=1e-12)
    assert np.isfinite([c.nx, c.nz, c.phi]).all()
    assert 0 <= c.nz <= 10+1e-12 and abs(c.phi) <= math.pi/2
    rates = e.dynamics.derivatives(s, c)
    if a_psi:
        assert rates[5]*a_psi > 0
    if a_theta and a_raw >= 0 and length <= 10:
        assert rates[4]*a_theta > 0


def test_legal_state_action_random_domain():
    e = env(); rng = np.random.default_rng(81)
    for _ in range(500):
        s = state(rng.uniform(150, 300), rng.uniform(-math.pi/3, math.pi/3), rng.uniform(-math.pi, math.pi))
        a = rng.uniform(-1, 1, 3); c = control(e, s, a)
        assert np.isfinite([c.nx, c.nz, c.phi]).all()
        assert 0 <= c.nz <= 10+1e-12 and abs(c.phi) <= math.pi/2
        assert c.nz*math.cos(c.phi) >= -1e-12


def test_same_controller_red_blue_advance():
    e = env(); s = state(225, .2, -.3)
    e.red = [s.copy()] + [state(alive=False) for _ in range(4)]; e.blue = deepcopy(e.red)
    actions = np.tile([.3, -.04, .6], (5, 1))
    with patch('env.combat_v329.action_to_control_v329', wraps=action_to_control_v329) as mapper:
        red_phi = e._advance(e.red, actions); blue_phi = e._advance(e.blue, actions)
    assert mapper.call_count == 2
    assert mapper.call_args_list[0].args[0] == mapper.call_args_list[1].args[0]
    np.testing.assert_array_equal(red_phi, blue_phi)
    assert e.red == e.blue
    assert e.controller_diagnostics['red'] == e.controller_diagnostics['blue']


@pytest.mark.parametrize('dy,dz', [(200, -100), (-200, 100)])
def test_blue_paper_encoder_ned_and_response(dy, dz):
    e = env(); own = state(200); target = AircraftState(1000, dy, -3000+dz, 225, 0, 0)
    action = e.fixed_policy.action(own, [target])
    expected = np.clip([math.atan2(dy, 1000)/math.pi,
                        math.atan2(-dz, math.hypot(1000, dy))/(math.pi/3), .5], -1, 1)
    np.testing.assert_allclose(action, expected, atol=1e-7)
    rates = e.dynamics.derivatives(own, control(e, own, action))
    assert rates[5]*dy > 0 and rates[4]*(-dz) > 0
    assert rates[3] > 0


def test_blue_nearest_switch_and_off_target_fire_suppression():
    e = env(); e.blue = [state()] + [state(alive=False) for _ in range(4)]
    e.red = [AircraftState(0, 400, -3000, 225, 0, 0), AircraftState(700, 0, -3000, 225, 0, 0)] + [state(alive=False) for _ in range(3)]
    e.rng = FixedNormal(4)
    with patch.object(e, '_advance', return_value=np.zeros(5)):
        first = e.step(np.zeros((5, 3)))
        assert first[4]['blue_pursuit_target_indices'][0] == 0
        assert e.weapon.qualifies(e.blue[0], e.red[1], 6)
        assert first[4]['blue_step_fire_attempts'] == 0 and e.blue_ammo[0] == 6
        e.red[1].x = 200
        second = e.step(np.zeros((5, 3)))
    assert second[4]['blue_pursuit_target_indices'][0] == 1 and e.red[0].alive
    assert second[4]['blue_step_fire_attempts'] == 1
    assert not hasattr(e, 'locked_target_indices')
    assert type(e.fixed_policy) is PaperNearestTargetBluePolicy


def test_v328_frozen_methods_configs_and_reset():
    old, new = env('328'), env()
    assert CombatEnvironmentV329.__bases__ == (CombatEnvironmentV328,)
    for name in ('_observations', 'dense_combat_reward', 'build_weapon', 'attempt_hit_for_pair',
                 '_resolve_combat', '_resolve_noncombat_losses', '_outcome', 'event_reward_extension'):
        assert getattr(type(old), name) is getattr(type(new), name)
    cfg = deepcopy(new.config); cfg['environment_version'] = '3.28'; assert cfg == old.config
    assert environment_dimensions(new.config) == (56, 3, 5)
    assert old.red == new.red and old.blue == new.blue
    assert type(old.integrator) is type(new.integrator) and type(old.dynamics) is type(new.dynamics)
    np.testing.assert_array_equal(old._observations(), new._observations())
    for name in ('mappo_5v5', 'madsac_5v5'):
        assert Path(f'configs/{name}_v329.yaml').read_bytes().replace(b'3.29', b'3.28') == Path(f'configs/{name}_v328.yaml').read_bytes()


def test_frozen_physics_when_using_baseline_action_interface():
    old, new = env('328'), env()
    # Reinsert only the old action interface to isolate every other transition.
    new._advance = MethodType(CombatEnvironmentV328._advance, new)
    new.fixed_policy = deepcopy(old.fixed_policy)
    rng = np.random.default_rng(101)
    for _ in range(50):
        action = rng.uniform(-1, 1, (5, 3))
        a = old.step(action); b = new.step(action)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        assert a[2:4] == b[2:4] and old.red == new.red and old.blue == new.blue
        for key, value in a[4].items():
            if key == 'environment_version': continue
            np.testing.assert_equal(value, b[4][key])
        assert old.rng.bit_generator.state == new.rng.bit_generator.state


@pytest.mark.parametrize('side', ['red', 'blue'])
def test_firing_ammo_cooldown_rng_hits_and_kills_frozen(side):
    old, new = env('328'), env()
    for e in (old, new):
        e.red = [state()] + [state(alive=False) for _ in range(4)]
        e.blue = [AircraftState(500, 0, -3000, 225, 0, 0), AircraftState(700, 0, -3000, 225, 0, 0)] + [state(alive=False) for _ in range(3)]
        if side == 'blue':
            e.blue, e.red = e.red, e.blue
            e.blue_step_target_indices[0] = 0
        e.rng = FixedNormal(4)
    for step in range(61):
        if step == 1:
            for e in (old, new):
                targets = e.blue if side == 'red' else e.red
                targets[1].x = 300
                if side == 'blue': e.blue_step_target_indices[0] = 1
        attempts = [e._entry_attempts(getattr(e, side), e.blue if side == 'red' else e.red, None, side) for e in (old, new)]
        assert attempts[0] == attempts[1] and old.rng.calls == new.rng.calls
        np.testing.assert_array_equal(getattr(old, side+'_ammo'), getattr(new, side+'_ammo'))
        np.testing.assert_array_equal(getattr(old, side+'_fire_cooldown_steps'), getattr(new, side+'_fire_cooldown_steps'))
    for e in (old, new):
        getattr(e, side+'_ammo')[0] = 1; getattr(e, side+'_fire_cooldown_steps')[0] = 0; e.rng = FixedNormal(0)
    attempts = [e._entry_attempts(getattr(e, side), e.blue if side == 'red' else e.red, None, side) for e in (old, new)]
    assert attempts[0] == attempts[1] and attempts[0][0][2]
    results = [e._resolve_combat(a if side=='red' else [], a if side=='blue' else []) for e,a in zip((old,new), attempts)]
    assert results[0] == results[1] and old.combat_counts == new.combat_counts


@pytest.mark.parametrize('section,key,value', [('weapon','range_max',800), ('reward','guide_reward',.002), ('blue_policy','desired_speed',250), ('scenario','center_radius',4800)])
def test_strict_config_freeze(section, key, value):
    cfg = deepcopy(env().config); cfg[section][key] = value
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('source,target', [('328','329'), ('329','328')])
def test_checkpoint_resume_eval_replay_isolation(source, target, tmp_path):
    import torch
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
    from algorithm.madsac.protocol import validate_madsac_config, validate_madsac_checkpoint
    from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
    from tools.record_combat_episode import load_recording_policy
    from tools.combat_visualization import RecordingCombatEnv
    ec = env(target).config
    assert isinstance(RecordingCombatEnv(ec), type(env(target)))
    for algorithm, name in [('MAPPO','mappo'), ('madsac','madsac')]:
        cfg = yaml.safe_load(Path(f'configs/{name}_5v5_v{target}.yaml').read_text())
        extra = dict(environment_version=f'3.{source[-2:]}', training_seed=0)
        checkpoint = dict(algorithm=algorithm, extra=extra, implementation_version=MADSAC_IMPL_VERSION)
        if name == 'mappo':
            for validator in (validate_checkpoint_for_evaluation, validate_checkpoint_for_resume):
                with pytest.raises(RuntimeError, match='environment_version'): validator(checkpoint, ec, cfg)
        else:
            validate_madsac_config(ec, cfg)
            with pytest.raises(RuntimeError, match='environment_version'): validate_madsac_checkpoint(checkpoint, ec, cfg)
        p = tmp_path/f'{name}.pt'; torch.save(checkpoint, p)
        with pytest.raises(RuntimeError, match='environment_version'): load_recording_policy(p, ec, cfg, 'cuda')


def test_diagnostics_do_not_change_transition_or_rng():
    from algorithm.common.controller_diagnostics import controller_metric_values
    a, b = env(), env()
    # B bypasses the read-only firing geometry wrapper only.
    b._entry_attempts = MethodType(CombatEnvironmentV328._entry_attempts, b)
    for _ in range(10):
        x, y = a.step(np.full((5,3), .05)), b.step(np.full((5,3), .05))
        np.testing.assert_array_equal(x[0],y[0]); np.testing.assert_array_equal(x[1],y[1])
        assert a.red == b.red and a.blue == b.blue and a.rng.bit_generator.state == b.rng.bit_generator.state
    values = controller_metric_values([x[4]])
    assert np.isfinite(np.concatenate(list(values.values()))).all()
    assert 'red_mean_nz' in values and 'blue_mean_abs_high_level_action_psi' in values
    a.reset(81)
    assert not a.controller_diagnostics and not a.firing_geometry_diagnostics
