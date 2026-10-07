"""Strict V2.3 environment-contract tests."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pytest

from env.config import ENVIRONMENT_VERSION, load_config
from env.dynamics import PointMassDynamics
from env.control import action_to_control, action_to_target
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.observation import build_team_observations
from env.reward import paper_state_reward_components
from env.scenario import random_combat_states
from env.weapon import FireState, WeaponEnvelope
from env.integrator import RK4Integrator
from env.models import AircraftSpec, AircraftState

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/combat_environment.yaml")


def state(x=0.0, y=0.0, altitude=3000.0, psi=0.0, theta=0.0,
          speed=225.0, alive=True):
    return AircraftState(x, y, -altitude, speed, theta, psi, alive)


def one_alive(primary):
    return [primary] + [state(alive=False) for _ in range(3)]


def test_version_and_frozen_bottom_level_contract():
    cfg = config()
    assert cfg["environment_version"] == ENVIRONMENT_VERSION == "2.3"
    assert cfg["simulation"] == {"dt": 0.1, "max_steps": 1000}
    assert cfg["aircraft"] == {
        "v_min": 150.0, "v_max": 300.0,
        "theta_min": -np.pi / 3.0, "theta_max": np.pi / 3.0,
    }
    assert MultiUAVCombatEnv.observation_dim == 52
    assert MultiUAVCombatEnv.action_dim == 3


def test_action_order_and_relative_zero_target():
    cfg = config()["action"]["command"]
    own = state(psi=0.4, theta=-0.2, speed=231.0)
    zero = action_to_target(own, np.zeros(3), cfg)
    assert (zero.heading, zero.pitch, zero.speed) == pytest.approx(
        (own.psi, own.theta, own.v)
    )
    target = action_to_target(own, np.array([0.5, -0.25, 0.4]), cfg)
    assert target.heading == pytest.approx(own.psi + np.pi / 2)
    assert target.pitch == pytest.approx(own.theta - np.pi / 12)
    assert target.speed == pytest.approx(own.v + 20.0)


@pytest.mark.parametrize("heading_deg", [-180, -90, -30, 0, 30, 90, 180])
@pytest.mark.parametrize("pitch_deg", [-60, -30, -10, 0, 10, 30, 60])
@pytest.mark.parametrize("speed_delta", [-50, -25, 0, 25, 50])
def test_controller_grid_is_finite_and_respects_load_caps(
    heading_deg, pitch_deg, speed_delta
):
    cfg = config()["action"]
    own = state()
    action = np.asarray([
        heading_deg / 180.0, pitch_deg / 60.0, speed_delta / 50.0
    ])
    control = action_to_control(own, action, cfg)
    assert np.all(np.isfinite([control.nx, control.nz, control.phi]))
    assert 0.0 <= control.nz <= 8.0 + 1e-12
    assert abs(control.phi) <= np.pi / 2.0 + 1e-12


def test_zero_action_is_trim_and_sustained_commands_stay_in_state_envelope():
    cfg = config()
    own = state()
    trim = action_to_control(own, np.zeros(3), cfg["action"])
    assert (trim.nx, trim.nz, trim.phi) == pytest.approx((0.0, 1.0, 0.0))
    dynamics = PointMassDynamics()
    integrator = RK4Integrator(0.1)
    spec = AircraftSpec(**cfg["aircraft"])
    previous_errors = []
    for _ in range(300):
        control = action_to_control(own, np.array([0.5, 0.3, 0.5]), cfg["action"])
        assert control.nz <= 8.0 + 1e-12
        own = integrator.step(own, control, dynamics, spec)
        previous_errors.append(abs(own.theta))
        assert np.all(np.isfinite(own.as_array()))
        assert spec.v_min <= own.v <= spec.v_max
        assert spec.theta_min <= own.theta <= spec.theta_max
    assert max(previous_errors) <= np.pi / 3.0 + 1e-12


def test_canonical_geometry_signs_and_reverse_relation():
    red = state(psi=0.0)
    blue = state(x=1000.0, y=1000.0, altitude=4000.0, psi=np.pi / 2)
    g = engagement_geometry(red, blue)
    assert g.line_of_sight == pytest.approx(np.pi / 4)
    assert g.ata == pytest.approx(np.pi / 4)
    assert g.aa == pytest.approx(np.pi / 4)
    assert g.ha == pytest.approx(np.arctan2(1000.0, np.sqrt(2e6)))
    expected_forward = np.asarray([1.0, 0.0, 0.0])
    expected_los = np.asarray([1000.0, 1000.0, -1000.0]) / np.sqrt(3e6)
    assert g.off_boresight == pytest.approx(
        np.arccos(np.dot(expected_forward, expected_los))
    )
    reverse = engagement_geometry(blue, red)
    assert reverse.distance == pytest.approx(g.distance)
    assert reverse.ha == pytest.approx(-g.ha)


def weapon():
    return WeaponEnvelope(**config()["weapon"])


@pytest.mark.parametrize("distance,expected", [
    (0.0, True), (4000.0, True), (4000.0001, False),
])
def test_weapon_range_boundaries(distance, expected):
    g = engagement_geometry(state(), state(x=distance, psi=0.0))
    assert weapon().in_fire_window(g) is expected


def test_head_on_fire_gate_ignores_target_aspect_but_uses_true_3d_boresight():
    g = engagement_geometry(state(psi=0.0), state(x=1000.0, psi=np.pi))
    assert abs(g.aa) == pytest.approx(np.pi)
    assert g.off_boresight == pytest.approx(0.0)
    assert weapon().in_fire_window(g)


@pytest.mark.parametrize(
    "attacker_pitch_deg,target_altitude,expected",
    [
        (60.0, 3000.0, False),
        (-60.0, 3000.0, False),
        (30.0, 3000.0 + 1000.0 * np.tan(np.pi / 6.0), True),
        (-30.0, 3000.0 - 1000.0 * np.tan(np.pi / 6.0), True),
    ],
)
def test_fire_gate_uses_aircraft_pitch_in_true_3d_cone(
    attacker_pitch_deg, target_altitude, expected
):
    attacker = state(theta=np.deg2rad(attacker_pitch_deg))
    target = state(x=1000.0, altitude=target_altitude)
    geometry = engagement_geometry(attacker, target)
    assert weapon().in_fire_window(geometry) is expected


@pytest.mark.parametrize(
    "pitch_deg,los_elevation_deg,expected_off_boresight_deg",
    [(0.0, 0.0, 0.0), (30.0, 30.0, 0.0), (-30.0, -30.0, 0.0),
     (60.0, 0.0, 60.0), (-60.0, 0.0, 60.0)],
)
def test_velocity_frame_boresight_analytic_cases(
    pitch_deg, los_elevation_deg, expected_off_boresight_deg
):
    distance = 3000.0
    elevation = np.deg2rad(los_elevation_deg)
    geometry = engagement_geometry(
        state(theta=np.deg2rad(pitch_deg)),
        state(x=distance * np.cos(elevation),
              altitude=3000.0 + distance * np.sin(elevation)),
    )
    assert np.rad2deg(geometry.off_boresight) == pytest.approx(
        expected_off_boresight_deg, abs=1e-6
    )
    if expected_off_boresight_deg == 0.0:
        assert geometry.boresight_azimuth_error == pytest.approx(0.0, abs=1e-12)
        assert geometry.boresight_elevation_error == pytest.approx(0.0, abs=1e-12)
    assert np.cos(geometry.off_boresight) == pytest.approx(
        np.cos(geometry.boresight_elevation_error)
        * np.cos(geometry.boresight_azimuth_error), abs=1e-12
    )


def test_true_pointing_hit_probability_is_rotation_consistent():
    model = weapon()
    rates = []
    for pitch_deg in (0.0, 30.0, -30.0):
        distance = 3000.0
        pitch = np.deg2rad(pitch_deg)
        geometry = engagement_geometry(
            state(theta=pitch),
            state(x=distance * np.cos(pitch),
                  altitude=3000.0 + distance * np.sin(pitch)),
        )
        rng = np.random.default_rng(314159)
        rates.append(np.mean([model.attempt_hit(geometry, rng) for _ in range(100_000)]))
    assert max(rates) - min(rates) < 1e-12


def test_old_separable_ata_ha_window_cannot_authorize_sideways_3d_shot():
    attacker = state(theta=np.deg2rad(60.0))
    target = state(x=3000.0, altitude=1500.0)
    geometry = engagement_geometry(attacker, target)
    assert abs(geometry.ata) <= np.deg2rad(30.0)
    assert abs(geometry.ha) <= np.deg2rad(30.0)
    assert geometry.off_boresight > np.deg2rad(80.0)
    assert not weapon().in_fire_window(geometry)


def test_hit_threshold_monotonic_and_4km_ideal_probability():
    model = weapon()
    assert model.hit_threshold(0.0) > model.hit_threshold(2000.0) > model.hit_threshold(4000.0)
    assert model.hit_threshold(4000.0) == pytest.approx(np.pi / 6.0)
    g = engagement_geometry(state(), state(x=4000.0))
    rng = np.random.default_rng(2023)
    rate = np.mean([model.attempt_hit(g, rng) for _ in range(100_000)])
    assert rate == pytest.approx(0.16, abs=0.01)


def test_weapon_rng_reproducibility_and_angle_sign_symmetry():
    model = weapon()
    positive = engagement_geometry(state(), state(x=3000.0, y=300.0))
    negative = engagement_geometry(state(), state(x=3000.0, y=-300.0))
    a = np.random.default_rng(71)
    b = np.random.default_rng(71)
    assert [model.attempt_hit(positive, a) for _ in range(100)] == [
        model.attempt_hit(positive, b) for _ in range(100)
    ]
    p_rate = np.mean([
        model.attempt_hit(positive, np.random.default_rng(seed))
        for seed in range(4000)
    ])
    n_rate = np.mean([
        model.attempt_hit(negative, np.random.default_rng(seed))
        for seed in range(4000)
    ])
    assert p_rate == pytest.approx(n_rate, abs=0.03)


def test_entry_trigger_is_one_attempt_per_continuous_window_and_rearms():
    env = MultiUAVCombatEnv(config())
    env.reset(1)
    attackers = one_alive(state())
    targets = one_alive(state(x=1000.0))
    fire_states = [FireState() for _ in range(4)]
    counts = [len(env._entry_attempts(attackers, targets, fire_states, "red"))]
    counts += [len(env._entry_attempts(attackers, targets, fire_states, "red")) for _ in range(99)]
    assert sum(counts) == 1
    targets[0].x = 5000.0
    assert env._entry_attempts(attackers, targets, fire_states, "red") == []
    targets[0].x = 1000.0
    assert len(env._entry_attempts(attackers, targets, fire_states, "red")) == 1


def test_random_diameter_scenario_contract_over_1000_resets():
    cfg = config()
    pair_distances, radii, altitudes, speeds = [], [], [], []
    radial_angles = []
    for seed in range(1000):
        red, blue, angle = random_combat_states(
            np.random.default_rng(seed), **cfg["scenario"]
        )
        radial_angles.append(angle)
        all_states = red + blue
        radii.extend(np.hypot(s.x, s.y) for s in all_states)
        altitudes.extend(s.altitude for s in all_states)
        speeds.extend(s.v for s in all_states)
        pair_distances.extend(
            engagement_geometry(r, b).distance for r in red for b in blue
        )
        assert max(np.hypot(s.x, s.y) for s in all_states) < 5000.0
        assert min(engagement_geometry(r, b).distance for r in red for b in blue) > 4000.0
    assert np.ptp(radial_angles) > 6.0
    assert np.mean(radii) == pytest.approx(np.sqrt(4000.0 ** 2 + 250.0 ** 2), abs=50)
    assert np.mean(altitudes) == pytest.approx(3000.0, abs=3.0)
    assert np.min(altitudes) >= 2900.0 and np.max(altitudes) <= 3100.0
    assert np.mean(speeds) == pytest.approx(225.0, abs=0.3)
    assert np.percentile(pair_distances, 50) > 7900.0


def _reward(red, blue):
    return paper_state_reward_components(one_alive(red), one_alive(blue), config()["reward"])


def test_reward_standard_states_r3_and_r4_tiers():
    far = _reward(state(psi=0.0), state(x=5000.0, psi=0.0))
    assert far["r3"][0] == pytest.approx(0.001)
    assert far["r4"][0] == 0.0
    expected = [(0.0, 0.1), (10.0, 0.02), (25.0, 0.01), (35.0, 0.0)]
    for angle_deg, value in expected:
        angle = np.deg2rad(angle_deg)
        blue = state(
            x=3000.0 * np.cos(angle), y=3000.0 * np.sin(angle),
            altitude=3000.0 + 3000.0 * np.tan(angle), psi=angle,
        )
        result = _reward(state(), blue)
        assert result["r4"][0] == pytest.approx(value)


def test_reward_threat_tiers_and_r41_precedence():
    for angle_deg, value in [(0.0, -0.15), (10.0, -0.025), (25.0, -0.015)]:
        angle = np.deg2rad(angle_deg)
        red = state(psi=np.pi + angle)
        blue = state(x=3000.0, psi=np.pi - angle)
        result = _reward(red, blue)
        assert result["r4"][0] == pytest.approx(value)
    red = state(psi=np.pi)
    blue = state(x=3000.0, psi=0.0)
    assert _reward(red, blue)["r4"][0] == 0.0


def test_observation_exact_layout_and_dead_slots():
    cfg = config()["observation"]
    team = [state(x=1000, y=-500, psi=0.2, theta=0.1, speed=240)]
    team += [state(x=1200), state(alive=False), state(y=1000)]
    opponents = [state(x=2000), state(alive=False), state(y=2000), state(x=-1000)]
    obs = build_team_observations(team, opponents, cfg, np.array([0.3, 0, 0, 0]))
    assert obs.shape == (4, 52)
    assert obs[0, :7] == pytest.approx([
        0.2, -0.1, 0.3, 0.2, 0.3 / (np.pi / 2), 0.2 / np.pi, 0.1 / (np.pi / 3)
    ])
    assert np.all(obs[0, 14:21] == 0.0)
    assert np.all(obs[0, 34:40] == 0.0)
    assert np.all(obs[1] != np.nan)


def test_boundary_ground_no_ceiling_and_timeout_semantics():
    cfg = config()
    env = MultiUAVCombatEnv(cfg)
    env.reset(2)
    env.red = one_alive(state(x=4999.0, psi=0.0))
    env.blue = one_alive(state(x=0.0, y=1000.0, psi=np.pi / 2))
    _, reward, terminated, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert terminated and reward[0] == pytest.approx(-10.0)
    assert info["r1_rewards"][0] == 0.0 and info["r2_rewards"][0] == -10.0

    env.reset(3)
    env.red = one_alive(state(altitude=1.0, theta=-np.pi / 3))
    env.blue = one_alive(state(x=1000.0))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert reward[0] == pytest.approx(-10.0)
    assert info["red_ground_losses"] == 1

    env.reset(4)
    env.red[0] = state(altitude=20_000.0)
    env.step(np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32))
    assert env.red[0].alive

    short = copy.deepcopy(cfg)
    short["simulation"]["max_steps"] = 1
    env = MultiUAVCombatEnv(short)
    env.reset(5)
    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert not terminated and truncated
    assert info["termination_reason"] == "red_failure_timeout"
    assert not info["red_success"] and not info["red_win"] and not info["draw"]


def test_blue_exit_is_not_a_red_kill_and_simultaneous_shared_kill_rewards():
    cfg = config()
    env = MultiUAVCombatEnv(cfg)
    env.reset(6)
    env.red = one_alive(state())
    env.blue = one_alive(state(x=4999.0, psi=0.0))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["blue_boundary_exits"] == 1
    assert info["red_attack_kills"] == 0
    assert reward.sum() == pytest.approx(0.0)

    deterministic = copy.deepcopy(cfg)
    deterministic["weapon"]["attack_noise_scale"] = 0.0
    deterministic["weapon"]["height_noise_scale"] = 0.0
    env = MultiUAVCombatEnv(deterministic)
    env.reset(7)
    env.red = [state(y=-10.0), state(y=10.0), state(alive=False), state(alive=False)]
    env.blue = one_alive(state(x=1000.0, psi=0.0))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["red_attack_kills"] == 1
    assert info["r1_rewards"][:2] == pytest.approx([5.0, 5.0])
    assert reward[:2].sum() >= 10.0


def test_noncombat_deaths_are_exclusive_and_have_exact_reward_attribution():
    cfg = config()

    # A Blue ground impact is neither an attack kill nor a Red reward.
    env = MultiUAVCombatEnv(cfg)
    env.reset(61)
    env.red = one_alive(state(x=-4000.0))
    env.blue = one_alive(state(x=4000.0, altitude=1.0, theta=-np.pi / 3))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["blue_ground_losses"] == 1
    assert info["blue_boundary_exits"] == 0
    assert info["red_attack_kills"] == 0
    assert reward.sum() == pytest.approx(0.0)

    # A Red ground impact receives one R1 loss penalty and no R2 penalty.
    env.reset(62)
    env.red = one_alive(state(x=-4000.0, altitude=1.0, theta=-np.pi / 3))
    env.blue = one_alive(state(x=4000.0))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["red_ground_losses"] == 1
    assert info["red_boundary_exits"] == 0
    assert info["blue_attack_kills"] == 0
    assert info["r1_rewards"][0] == pytest.approx(-10.0)
    assert info["r2_rewards"][0] == pytest.approx(0.0)
    assert reward[0] == pytest.approx(-10.0)

    # A Red boundary exit receives only R2, never the R1 loss penalty too.
    env.reset(63)
    env.red = one_alive(state(x=4999.0, psi=0.0))
    env.blue = one_alive(state(x=-4000.0))
    _, reward, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["red_boundary_exits"] == 1
    assert info["red_ground_losses"] == 0
    assert info["blue_attack_kills"] == 0
    assert info["r1_rewards"][0] == pytest.approx(0.0)
    assert info["r2_rewards"][0] == pytest.approx(-10.0)
    assert reward[0] == pytest.approx(-10.0)
