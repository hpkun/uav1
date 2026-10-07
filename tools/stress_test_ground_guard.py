"""Closed-loop synthetic stress for persistent-wave v2 Blue ground avoidance."""
from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from env.control import action_to_control
from env.factory import make_combat_environment
from env.fixed_policy import (
    GroundAwareNearestTargetPursuitPolicy,
)
from env.models import AircraftState


ROOT = PROJECT_ROOT


def plain(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def state(x: float, y: float, altitude: float, speed: float, theta: float,
          psi: float = 0.0, alive: bool = True) -> AircraftState:
    return AircraftState(x, y, -altitude, speed, theta, psi, alive)


def team(primary: AircraftState) -> list[AircraftState]:
    dead = [state(0.0, 0.0, 1000.0, 225.0, 0.0, alive=False) for _ in range(3)]
    return [primary, *dead]


def step_aircraft(env, aircraft: AircraftState, action: np.ndarray) -> AircraftState:
    control = action_to_control(aircraft, action, env.config["action"])
    return env.integrator.step(aircraft, control, env.dynamics, env.spec)


def target_command(policy, target: AircraftState, behavior: str,
                   step_index: int) -> np.ndarray:
    desired_pitch = {
        "horizontal": 0.0,
        "descending": np.deg2rad(-20.0),
        "pull_up": np.deg2rad(-20.0 if step_index < 20 else 30.0),
        "lateral": np.deg2rad(-10.0),
    }[behavior]
    desired_heading = np.deg2rad(45.0) if behavior == "lateral" else 0.0
    return policy.action_toward(target, desired_heading, desired_pitch, 225.0)


def oracle_survives(env, initial: AircraftState, initial_target: AircraftState,
                    behavior: str, steps: int) -> bool:
    own = initial.copy()
    target = initial_target.copy()
    policy = env.fixed_policy
    for step_index in range(steps):
        dx, dy = target.x - own.x, target.y - own.y
        action = policy.action_toward(
            own, float(np.arctan2(dy, dx)), env.spec.theta_max,
            float(policy.config["desired_speed"]),
        )
        own = step_aircraft(env, own, action)
        target = step_aircraft(
            env, target, target_command(policy, target, behavior, step_index)
        )
        if own.altitude <= 0.0:
            return False
    return True


def run_case(env, initial: AircraftState, target: AircraftState,
             behavior: str, steps: int) -> dict[str, Any]:
    policy = env.fixed_policy
    assert isinstance(policy, GroundAwareNearestTargetPursuitPolicy)
    policy.reset_diagnostics()
    blue = initial.copy()
    red = target.copy()
    minimum_altitude = blue.altitude
    nonfinite = False
    boundary = False
    ground = False
    pitch_commands = []
    for step_index in range(steps):
        action = policy.team_actions(team(blue), team(red))[0]
        pitch_commands.append(float(action[1]))
        blue = step_aircraft(env, blue, action)
        red = step_aircraft(
            env, red, target_command(policy, red, behavior, step_index)
        )
        values = blue.as_array()
        nonfinite = nonfinite or not np.all(np.isfinite(values))
        minimum_altitude = min(minimum_altitude, blue.altitude)
        boundary = boundary or np.hypot(blue.x, blue.y) > env.arena_radius
        if blue.altitude <= 0.0:
            ground = True
            break
    return {
        "ground_loss": ground,
        "minimum_altitude_m": float(minimum_altitude),
        "nonfinite": nonfinite,
        "boundary_exit": boundary,
        "override_steps": policy.override_steps,
        "decision_steps": policy.total_decision_steps,
        "activations": policy.activation_count,
        "maximum_duration_steps": policy.maximum_activation_duration_steps,
        "pitch_command_min": min(pitch_commands),
        "pitch_command_max": max(pitch_commands),
    }


def run_stress(config: dict[str, Any], steps: int = 100) -> dict[str, Any]:
    env = make_combat_environment(config)
    cases = []
    values = product(
        (100, 200, 300, 500, 750, 1000),
        (150, 225, 300),
        (0, -10, -20, -30, -45),
        ("horizontal", "descending", "pull_up", "lateral"),
        (-500, 0, 500),
    )
    for altitude, speed, pitch_deg, behavior, lateral in values:
        blue = state(0.0, float(lateral), float(altitude), float(speed),
                     np.deg2rad(pitch_deg))
        red = state(1000.0, 0.0, max(100.0, altitude * 0.4), 225.0,
                    np.deg2rad(-20.0))
        result = run_case(env, blue, red, behavior, steps)
        result.update({
            "altitude_m": altitude, "speed_mps": speed,
            "pitch_deg": pitch_deg, "behavior": behavior,
            "lateral_offset_m": lateral,
            "immediate_pull_up_oracle_survives": oracle_survives(
                env, blue, red, behavior, steps
            ),
        })
        cases.append(result)
    avoidable = [
        row for row in cases
        if row["ground_loss"] and row["immediate_pull_up_oracle_survives"]
    ]
    return {
        "case_count": len(cases),
        "ground_loss_count": sum(row["ground_loss"] for row in cases),
        "unavoidable_ground_loss_count": sum(
            row["ground_loss"] and not row["immediate_pull_up_oracle_survives"]
            for row in cases
        ),
        "avoidable_ground_loss_count": len(avoidable),
        "override_case_rate": float(np.mean([
            row["override_steps"] > 0 for row in cases
        ])),
        "override_step_ratio": (
            sum(row["override_steps"] for row in cases)
            / max(sum(row["decision_steps"] for row in cases), 1)
        ),
        "maximum_activation_duration_steps": max(
            row["maximum_duration_steps"] for row in cases
        ),
        "nonfinite_count": sum(row["nonfinite"] for row in cases),
        "boundary_exit_count": sum(row["boundary_exit"] for row in cases),
        "minimum_altitude_m": min(row["minimum_altitude_m"] for row in cases),
        "avoidable_examples": avoidable[:20],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/persistent_wave_v2_environment.yaml"
    )
    parser.add_argument("--guard-time-constants", type=float)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.guard_time_constants is not None:
        config = copy.deepcopy(config)
        config["blue_policy"]["ground_avoidance"][
            "guard_time_constants"
        ] = args.guard_time_constants
    report = run_stress(config, args.steps)
    report["guard_time_constants"] = config["blue_policy"]["ground_avoidance"][
        "guard_time_constants"
    ]
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = plain(report)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"cases", "avoidable_examples"}}, indent=2))


if __name__ == "__main__":
    main()
