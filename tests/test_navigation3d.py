"""Environment, actual-step PBRS, terminal masks and deterministic trajectories."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from env.navigation3d import Navigation3DEnv, load_config, validate_config, observation_dim
from env.navigation3d.geometry import direction
from env.v311_kinematics import advance_aircraft
from algorithm.navigation3d.evaluation import waypoint_controller, run_episode

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT/'configs/navigation3d_v1.yaml')


def environment(**updates):
    c = config(); c.update(updates)
    env = Navigation3DEnv(c); env.reset(42)
    return env


def test_observation_layout_and_finiteness():
    env = environment()
    obs = env.observe()
    assert obs.shape == (4, 50) and observation_dim(env.config) == 50
    assert np.isfinite(obs).all() and np.abs(obs).max() <= 1
    s = env.states[0]
    assert obs[0, 4:8] == pytest.approx([np.sin(s.psi), np.cos(s.psi), np.sin(s.theta), np.cos(s.theta)])
    assert obs[:, 18] == pytest.approx(1)
    assert obs[:, [29, 39, 49]] == pytest.approx(np.ones((4, 3)))
    assert obs[0, 20:23] == pytest.approx((env.positions[1]-env.positions[0])/6000)


def test_inactive_agent_and_invisible_peer_masks():
    env = environment(active_agents=[True, False, True, True])
    assert not env.observe()[1].any()
    assert not env.observe()[0, 20:30].any()
    assert env.valid_mask.tolist() == [True, False, True, True]
    env.config['visibility_radius_m'] = 1
    assert not env.observe()[:, 20:].any()
    start = env.states[1].as_array()
    _, reward, *_ = env.step(np.ones((4, 3)))
    assert reward[1] == 0 and np.array_equal(start, env.states[1].as_array())


def test_existing_increment_kinematics_exact():
    env = environment()
    action = np.array([[.2, -.1, .3]]*4)
    expected = [advance_aircraft(s, a, env.config['dt'], env.config['motion'], env.spec).as_array() for s, a in zip(env.states, action)]
    env.step(action)
    assert np.array_equal(np.stack([s.as_array() for s in env.states]), np.stack(expected))


@pytest.mark.parametrize('action', [np.zeros((3, 3)), np.full((4, 3), np.nan), np.full((4, 3), 1.01)])
def test_invalid_actions_fail_before_mutation(action):
    env = environment(); previous = env.positions.copy()
    with pytest.raises(ValueError):
        env.step(action)
    assert np.array_equal(previous, env.positions) and env.steps == 0


def test_real_step_potential_is_counted_exactly_once():
    env = environment(); before = env.potential_components()
    _, reward, _, _, info = env.step(np.zeros((4, 3)))
    for key in before:
        expected = .99*info['potential_after'][key]-before[key]
        assert info['reward_components'][key] == pytest.approx(expected)
    assert reward == pytest.approx(sum(info['reward_components'].values()), abs=2e-8)
    assert info['episode_return'] == pytest.approx(float(reward.sum()), abs=2e-8)


def test_switching_waypoint_does_not_reset_potential():
    env = environment(); before = env.potential_components()
    env.waypoint_index = 1
    after = env.potential_components()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    env.waypoint_index = 0
    for s, p in zip(env.states, env.goals):
        s.x, s.y, s.z = p; s.theta = 0; s.psi = 0
    _, _, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert info['waypoint_switched'] and not terminated and not truncated
    for k in before:
        assert info['reward_components'][k] == pytest.approx(.99*info['potential_after'][k]-info['potential_before'][k])


@pytest.mark.parametrize('ending', ['success', 'time_limit', 'boundary', 'collision'])
def test_terminal_potential_and_valid_masks(ending):
    env = environment(max_steps=1 if ending == 'time_limit' else 400)
    if ending == 'success':
        env.waypoint_index = 1
        for s, p in zip(env.states, env.goals):
            s.x, s.y, s.z = p; s.theta = 0; s.psi = 0
    elif ending == 'boundary':
        env.states[0].x = 3999; env.states[0].psi = 0
    elif ending == 'collision':
        env.states[0].x, env.states[0].y, env.states[0].z = env.positions[1]
    before = env.potential_components()
    obs, reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert terminated or truncated
    assert info['termination_reason'] == ending
    assert not obs.any() and not env.valid_mask.any()
    assert all(not v.any() for v in info['potential_after'].values())
    for k in before:
        assert info['reward_components'][k] == pytest.approx(-before[k])
    assert reward == pytest.approx(sum(info['reward_components'].values()), abs=2e-6)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((4, 3)))
    env.reset(42)
    assert env.valid_mask.all() and env.steps == 0


def test_swept_collision_detects_between_frames():
    env = environment()
    a, b = env.states[:2]
    a.x, b.x, a.y, b.y, a.z, b.z = -25., 25., 0., 0., -800., -800.
    a.psi, b.psi, a.theta, b.theta, a.v, b.v = 0., np.pi, 0., 0., 250., 250.
    _, _, done, _, info = env.step(np.zeros((4, 3)))
    assert done and info['collision'][:2].all()
    assert np.linalg.norm(env.positions[0]-env.positions[1]) > 20


def test_determinism_and_readonly_diagnostics():
    a, b = environment(), environment()
    rng = np.random.default_rng(99)
    for _ in range(30):
        action = rng.uniform(-.2, .2, (4, 3))
        a.observe(); a.potential_components(); a._info(False, 'running')
        x, y = a.step(action), b.step(action)
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
        assert np.array_equal(a.positions, b.positions)
        if x[2] or x[3]:
            break
    assert not np.array_equal(a.positions, environment(initial_jitter_m=0).positions)


@pytest.mark.parametrize('controller', ['waypoint', 'orbit', 'zero', 'retreat'])
def test_no_positive_shaping_harvest_and_accounting(controller):
    env = environment()
    initial = sum(env.potential_components().values())
    discounted = np.zeros(4); undiscounted = np.zeros(4)
    while True:
        if controller == 'waypoint':
            action = waypoint_controller(env)
        elif controller == 'orbit':
            action = np.tile([1., -.2, 0.], (4, 1))
        elif controller == 'retreat':
            action = np.tile([1., 0., 0.], (4, 1))
        else:
            action = np.zeros((4, 3))
        _, _, term, trunc, info = env.step(action)
        shaping = sum(info['reward_components'][k] for k in ('distance', 'alignment', 'formation'))
        discounted += .99**(env.steps-1)*shaping
        undiscounted += shaping
        if term or trunc:
            break
    assert discounted == pytest.approx(-initial, abs=1e-12)
    assert (undiscounted <= 1e-12).all()
    assert np.allclose(sum(info['component_totals'].values()).sum(), info['episode_return'])
    if controller == 'waypoint':
        assert info['success']
        assert info['component_totals']['completion'] == pytest.approx([10]*4)
    else:
        assert not info['component_totals']['completion'].any()
    for k in info['component_totals']:
        assert info['component_totals'][k] == pytest.approx(info['component_positive'][k]+info['component_negative'][k])


def test_stationary_and_closed_loop_potential_cannot_pay_forever():
    gamma = config()['reward']['gamma']
    phi = np.linspace(0, .2, 51)
    assert ((gamma-1)*phi <= 0).all()
    # Every cyclic nonnegative potential sequence has nonpositive raw cycle sum.
    rng = np.random.default_rng(0)
    cycle = rng.uniform(0, .2, 400)
    assert np.sum(gamma*np.roll(cycle, -1)-cycle) == pytest.approx((gamma-1)*sum(cycle))


@pytest.mark.parametrize('key,value', [('environment_id', '3.41'), ('max_steps', 0), ('num_agents', 0),
    ('dt', float('nan')), ('active_agents', [False]*4), ('initial_center_ned', [0, 0, 0])])
def test_config_rejects_invalid_identity_or_numeric_boundary(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_geometry_and_potential_are_continuous_near_waypoint_direction():
    env = environment()
    before = sum(env.potential_components().values())
    env.states[0].psi += 1e-7
    after = sum(env.potential_components().values())
    assert np.max(np.abs(after-before)) < 1e-7
