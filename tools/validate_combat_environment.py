"""End-to-end V2.3 controller, weapon, initialization and baseline validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.config import ENVIRONMENT_VERSION, aircraft_spec, load_config
from env.control import action_to_control
from env.combat_env import MultiUAVCombatEnv
from env.dynamics import PointMassDynamics
from env.geometry import engagement_geometry
from env.integrator import RK4Integrator
from env.models import AircraftState
from env.scenario import random_combat_states
from env.weapon import WeaponEnvelope


def state(x=0.0, y=0.0, altitude=3000.0, psi=0.0, theta=0.0, speed=225.0):
    return AircraftState(x, y, -altitude, speed, theta, psi)


def summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def controller_validation(config: dict) -> dict:
    cases = [(0, 0, 0)]
    cases += [(heading, 0, 0) for heading in (-180, -90, -30, 30, 90, 180)]
    cases += [(0, pitch, 0) for pitch in (-60, -30, -10, 10, 30, 60)]
    cases += [(0, 0, speed) for speed in (-50, -25, 25, 50)]
    cases += [(90, 30, 50), (-90, -30, -50), (180, 60, 25), (-180, -60, -25)]
    dynamics, integrator, spec = PointMassDynamics(), RK4Integrator(0.1), aircraft_spec(config)
    rows = []
    for heading, pitch, speed in cases:
        own = state()
        max_nz = max_phi = 0.0
        finite = envelope = True
        pitch_error_signs = []
        for _ in range(300):
            action = np.asarray([heading / 180.0, pitch / 60.0, speed / 50.0])
            control = action_to_control(own, action, config["action"])
            max_nz = max(max_nz, control.nz)
            max_phi = max(max_phi, abs(control.phi))
            own = integrator.step(own, control, dynamics, spec)
            finite &= bool(np.all(np.isfinite(own.as_array())))
            envelope &= bool(
                spec.v_min <= own.v <= spec.v_max
                and spec.theta_min <= own.theta <= spec.theta_max
            )
            error = np.deg2rad(pitch) - own.theta
            pitch_error_signs.append(np.sign(error) if abs(error) > 1e-4 else 0.0)
        nonzero = np.asarray([x for x in pitch_error_signs[100:] if x != 0.0])
        sign_changes = int(np.count_nonzero(np.diff(nonzero) != 0.0)) if len(nonzero) else 0
        rows.append({
            "heading_deg": heading, "pitch_deg": pitch, "speed_delta": speed,
            "max_nz": max_nz, "max_abs_phi_deg": float(np.rad2deg(max_phi)),
            "final_speed": own.v, "final_pitch_deg": float(np.rad2deg(own.theta)),
            "finite": finite, "in_state_envelope": envelope,
            "post_transient_pitch_error_sign_changes": sign_changes,
        })
    passed = all(
        row["finite"] and row["in_state_envelope"]
        and row["max_nz"] <= 8.0 + 1e-9
        and row["max_abs_phi_deg"] <= 90.0 + 1e-9
        and row["post_transient_pitch_error_sign_changes"] <= 2
        for row in rows
    )
    return {"cases": rows, "passed": passed}


def weapon_validation(config: dict, trials: int = 100_000) -> dict:
    model = WeaponEnvelope(**config["weapon"])
    rates = {}
    for distance in (0.0, 1000.0, 2000.0, 3000.0, 4000.0):
        geometry = engagement_geometry(state(), state(x=distance))
        rng = np.random.default_rng(2023 + int(distance))
        rates[str(int(distance))] = float(np.mean([
            model.attempt_hit(geometry, rng) for _ in range(trials)
        ]))
    positive = engagement_geometry(state(), state(x=3000.0, y=300.0))
    negative = engagement_geometry(state(), state(x=3000.0, y=-300.0))
    rng_positive, rng_negative = np.random.default_rng(91), np.random.default_rng(92)
    positive_rate = float(np.mean([
        model.attempt_hit(positive, rng_positive) for _ in range(trials)
    ]))
    negative_rate = float(np.mean([
        model.attempt_hit(negative, rng_negative) for _ in range(trials)
    ]))
    passed = bool(
        all(rates[str(a)] >= rates[str(b)] for a, b in zip(
            (0, 1000, 2000, 3000), (1000, 2000, 3000, 4000)
        ))
        and abs(rates["4000"] - 0.16) <= 0.015
        and abs(positive_rate - negative_rate) <= 0.015
    )
    return {
        "trials_per_case": trials,
        "hit_rate_by_distance_m": rates,
        "ideal_4km_hit_rate": rates["4000"],
        "positive_ata_rate": positive_rate,
        "negative_ata_rate": negative_rate,
        "sign_symmetry_gap": abs(positive_rate - negative_rate),
        "passed": passed,
    }


def initialization_validation(config: dict, resets: int = 1000) -> dict:
    values = {name: [] for name in (
        "cross_pair_distance", "abs_ata_deg", "abs_aa_deg", "abs_ha_deg",
        "altitude", "speed", "radius",
    )}
    initial_fire_pairs = 0
    radial_angles = []
    weapon = WeaponEnvelope(**config["weapon"])
    for seed in range(resets):
        red, blue, radial = random_combat_states(
            np.random.default_rng(seed), **config["scenario"]
        )
        radial_angles.append(radial)
        for aircraft in red + blue:
            values["altitude"].append(aircraft.altitude)
            values["speed"].append(aircraft.v)
            values["radius"].append(float(np.hypot(aircraft.x, aircraft.y)))
        for own in red:
            for target in blue:
                geometry = engagement_geometry(own, target)
                values["cross_pair_distance"].append(geometry.distance)
                values["abs_ata_deg"].append(abs(np.rad2deg(geometry.ata)))
                values["abs_aa_deg"].append(abs(np.rad2deg(geometry.aa)))
                values["abs_ha_deg"].append(abs(np.rad2deg(geometry.ha)))
                initial_fire_pairs += int(weapon.in_fire_window(geometry))
    report = {name: summary(data) for name, data in values.items()}
    report.update({
        "resets": resets,
        "radial_coverage_deg": float(np.rad2deg(np.ptp(radial_angles))),
        "nominal_center_separation_m": 2.0 * float(config["scenario"]["center_radius"]),
        "initial_fire_pairs": initial_fire_pairs,
    })
    report["passed"] = bool(
        report["radial_coverage_deg"] > 340.0
        and report["radius"]["max"] < float(config["arena"]["radius"])
        and report["cross_pair_distance"]["min"] > 4000.0
        and initial_fire_pairs == 0
    )
    return report


def rule_based_validation(config: dict, episodes: int = 100) -> dict:
    records = []
    for seed in range(20_000, 20_000 + episodes):
        env = MultiUAVCombatEnv(config)
        _, _ = env.reset(seed)
        returns = np.zeros(4, dtype=float)
        while True:
            actions = env.fixed_policy.team_actions(env.red, env.blue)
            _, reward, terminated, truncated, info = env.step(actions)
            returns += reward
            if terminated or truncated:
                records.append({"return": float(returns.sum()), **info})
                break
    rate = lambda key: float(np.mean([row[key] is not None for row in records]))
    mean = lambda key: float(np.mean([row[key] for row in records]))
    episode_rate = lambda key: float(np.mean([row[key] > 0 for row in records]))
    return {
        "episodes": episodes,
        "team_return": summary([row["return"] for row in records]),
        "red_win_rate": mean("red_success"),
        "red_loss_rate": mean("blue_win"),
        "timeout_rate": float(np.mean([
            row["termination_reason"] == "red_failure_timeout" for row in records
        ])),
        "failure_rate": float(np.mean([not row["red_success"] for row in records])),
        "average_episode_length": mean("episode_length"),
        "red_fire_window_episode_rate": rate("red_first_fire_window_step"),
        "red_attempt_episode_rate": rate("red_first_attempt_step"),
        "red_hit_episode_rate": rate("red_first_hit_step"),
        "red_kill_episode_rate": rate("red_first_kill_step"),
        "blue_fire_window_episode_rate": rate("blue_first_fire_window_step"),
        "blue_attempt_episode_rate": rate("blue_first_attempt_step"),
        "blue_hit_episode_rate": rate("blue_first_hit_step"),
        "blue_kill_episode_rate": rate("blue_first_kill_step"),
        "mean_red_kills": mean("red_attack_kills"),
        "mean_blue_kills": mean("blue_attack_kills"),
        "mean_red_exits": mean("red_boundary_exits"),
        "mean_blue_exits": mean("blue_boundary_exits"),
        "red_exit_episode_rate": episode_rate("red_boundary_exits"),
        "blue_exit_episode_rate": episode_rate("blue_boundary_exits"),
        "mean_red_ground_losses": mean("red_ground_losses"),
        "mean_blue_ground_losses": mean("blue_ground_losses"),
        "red_ground_loss_episode_rate": episode_rate("red_ground_losses"),
        "blue_ground_loss_episode_rate": episode_rate("blue_ground_losses"),
        **{f"mean_episode_{name}_total": mean(f"episode_{name}_total")
           for name in ("r1", "r2", "r3", "r4")},
        "passed": bool(
            any(row["red_first_fire_window_step"] is not None for row in records)
            and any(row["red_first_attempt_step"] is not None for row in records)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="outputs/paper_environment_v2_2_validation.json"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--weapon-trials", type=int, default=100_000)
    args = parser.parse_args()
    root = PROJECT_ROOT
    config = load_config(root / "configs/combat_environment.yaml")
    result = {
        "environment_version": ENVIRONMENT_VERSION,
        "controller": controller_validation(config),
        "weapon": weapon_validation(config, args.weapon_trials),
        "initialization": initialization_validation(config),
        "rule_based": rule_based_validation(config, args.episodes),
    }
    result["passed"] = all(section["passed"] for section in result.values()
                           if isinstance(section, dict))
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
