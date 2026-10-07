"""Persistent-wave v2 ground guard without changing Direct or v1 semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.config import load_config
from env.control import action_to_control
from env.control import action_to_target
from env.factory import make_combat_environment
from env.fixed_policy import (
    GroundAwareNearestTargetPursuitPolicy,
    NearestTargetPursuitPolicy,
)
from env.integrator import RK4Integrator
from env.math_utils import wrap_angle
from env.models import AircraftState
from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.common.vector_env import ParallelVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def config(name: str) -> dict:
    return load_config(ROOT / "configs" / name)


def state(*, x=0.0, y=0.0, altitude=3000.0, speed=225.0,
          theta=0.0, psi=0.0, alive=True) -> AircraftState:
    return AircraftState(x, y, -altitude, speed, theta, psi, alive)


def team(primary: AircraftState) -> list[AircraftState]:
    return [primary] + [state(alive=False) for _ in range(3)]


def policies():
    cfg = config("persistent_wave_v2_environment.yaml")
    nominal = NearestTargetPursuitPolicy(cfg["blue_policy"], cfg["action"])
    guarded = GroundAwareNearestTargetPursuitPolicy(
        cfg["blue_policy"], cfg["action"], cfg["aircraft"]
    )
    return cfg, nominal, guarded


def test_direct_and_v1_keep_original_nearest_target_policy_and_command():
    direct = make_combat_environment(config("combat_environment.yaml"))
    v1 = make_combat_environment(config("persistent_wave_environment.yaml"))
    own = state(theta=-0.2, psi=0.3, speed=210.0)
    target = state(x=1200.0, y=-300.0, altitude=400.0)
    expected = direct.fixed_policy.action(own, team(target))
    dx, dy = target.x - own.x, target.y - own.y
    analytic = np.clip(np.asarray([
        wrap_angle(np.arctan2(dy, dx) - own.psi) / np.pi,
        (np.arctan2(own.z - target.z, np.hypot(dx, dy)) - own.theta)
        / (np.pi / 3.0),
        (250.0 - own.v) / 50.0,
    ], dtype=np.float32), -1.0, 1.0)

    assert type(direct.fixed_policy) is NearestTargetPursuitPolicy
    assert type(v1.fixed_policy) is NearestTargetPursuitPolicy
    assert np.array_equal(expected, analytic)
    assert np.array_equal(v1.fixed_policy.action(own, team(target)), expected)


def test_v2_high_altitude_nominal_pursuit_is_bit_identical_to_v1():
    _, nominal, guarded = policies()
    own = state(altitude=3000.0, theta=-0.1, psi=0.4)
    target = state(x=1500.0, y=400.0, altitude=2400.0)
    assert np.array_equal(
        guarded.team_actions(team(own), team(target))[0],
        nominal.team_actions(team(own), team(target))[0],
    )
    assert not guarded.last_override_mask[0]


def test_v2_low_descending_pursuit_commands_max_available_pull_up():
    _, nominal, guarded = policies()
    own = state(altitude=300.0, theta=np.deg2rad(-20.0), speed=225.0)
    target = state(x=1000.0, altitude=100.0, theta=np.deg2rad(-20.0))
    baseline = nominal.team_actions(team(own), team(target))[0]
    action = guarded.team_actions(team(own), team(target))[0]

    baseline_target = action_to_target(
        own, baseline, config("persistent_wave_v2_environment.yaml")["action"]["command"]
    )
    assert baseline_target.pitch < 0.0
    assert action[1] == pytest.approx(1.0)
    assert guarded.last_override_mask[0]


def test_v2_non_descending_low_flight_does_not_trigger():
    _, nominal, guarded = policies()
    own = state(altitude=100.0, theta=np.deg2rad(10.0))
    target = state(x=1000.0, altitude=300.0)
    assert np.array_equal(
        guarded.team_actions(team(own), team(target))[0],
        nominal.team_actions(team(own), team(target))[0],
    )
    assert not guarded.last_override_mask[0]


def test_v2_guard_is_stateless_and_returns_to_los_after_danger_clears():
    _, nominal, guarded = policies()
    danger = state(altitude=250.0, theta=np.deg2rad(-20.0))
    low_target = state(x=1000.0, altitude=100.0)
    guarded.team_actions(team(danger), team(low_target))
    assert guarded.last_override_mask[0]

    safe = state(altitude=2000.0, theta=np.deg2rad(5.0))
    target = state(x=1000.0, altitude=1800.0)
    recovered = guarded.team_actions(team(safe), team(target))[0]
    expected = nominal.team_actions(team(safe), team(target))[0]
    assert np.array_equal(recovered, expected)
    assert not guarded.last_override_mask[0]


def test_v2_low_target_pull_up_survives_closed_loop_without_nonfinite_state():
    cfg, _, guarded = policies()
    env = make_combat_environment(cfg)
    blue = state(altitude=500.0, theta=np.deg2rad(-20.0), speed=225.0)
    red = state(x=1000.0, altitude=150.0, theta=np.deg2rad(-20.0))
    integrator = RK4Integrator(0.1)
    minimum_altitude = blue.altitude
    for step_index in range(100):
        red.theta = np.deg2rad(-20.0 if step_index < 20 else 30.0)
        action = guarded.team_actions(team(blue), team(red))[0]
        control = action_to_control(blue, action, cfg["action"])
        blue = integrator.step(blue, control, env.dynamics, env.spec)
        minimum_altitude = min(minimum_altitude, blue.altitude)
        assert np.all(np.isfinite(blue.as_array()))
        assert blue.altitude > 0.0
    assert minimum_altitude > 0.0
    assert guarded.override_steps > 0


def test_v2_fixed_seed_is_deterministic():
    first = make_combat_environment(config("persistent_wave_v2_environment.yaml"))
    second = make_combat_environment(config("persistent_wave_v2_environment.yaml"))
    obs_first, _ = first.reset(90210)
    obs_second, _ = second.reset(90210)
    assert np.array_equal(obs_first, obs_second)
    for _ in range(50):
        actions = np.zeros((4, 3), dtype=np.float32)
        result_first = first.step(actions)
        result_second = second.step(actions)
        assert np.array_equal(result_first[0], result_second[0])
        assert np.array_equal(result_first[1], result_second[1])


@pytest.mark.parametrize("source", ["direct_v2_3", "persistent_wave_v1"])
def test_v2_checkpoint_resume_rejects_other_variants(source):
    checkpoint = {"extra": {
        "environment_version": "2.3", "environment_variant": source,
    }}
    with pytest.raises(RuntimeError, match="environment_variant mismatch"):
        validate_checkpoint_environment(
            checkpoint, {"environment_variant": "persistent_wave_v2"}
        )


def test_v2_worker_constructs_ground_aware_policy():
    cfg = config("persistent_wave_v2_environment.yaml")
    env = make_combat_environment(cfg)
    assert isinstance(env.fixed_policy, GroundAwareNearestTargetPursuitPolicy)
    with ParallelVectorEnv(1, cfg, base_seed=91_000_000) as vector:
        vector.reset()
        assert vector.worker_environment_variants == ["persistent_wave_v2"]
        assert vector.worker_fixed_policy_classes == [
            "GroundAwareNearestTargetPursuitPolicy"
        ]
