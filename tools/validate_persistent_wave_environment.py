"""Pure-environment stress audit for persistent-wave replacement geometry."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.config import load_config
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv


ROOT = PROJECT_ROOT
CATEGORIES = ("center", "boundary", "spread", "altitude", "heading", "speed")


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "min", "p1", "p10", "p50", "p90", "p99", "max")}
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "min": float(data.min()),
        "p1": float(np.percentile(data, 1)),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def red_layout(category: str, survivors: int, rng: np.random.Generator) -> list[AircraftState]:
    if category == "center":
        xy = [(-450, -250), (-150, 250), (150, -250), (450, 250)]
    elif category in ("boundary", "edge"):
        bearing = float(rng.uniform(-np.pi, np.pi))
        radial = np.array([np.cos(bearing), np.sin(bearing)])
        lateral = np.array([-radial[1], radial[0]])
        xy = [tuple(4100.0 * radial + offset * lateral) for offset in (-300, -100, 100, 300)]
    elif category in ("spread", "dispersed"):
        rotation = float(rng.uniform(-np.pi, np.pi))
        xy = [(3000.0 * np.cos(rotation + k * np.pi / 2),
               3000.0 * np.sin(rotation + k * np.pi / 2)) for k in range(4)]
    else:
        xy = [(-2200, -1500), (-800, 2100), (1800, -1700), (2400, 1200)]
    states = []
    for index, (x, y) in enumerate(xy):
        altitude = 1800.0 + 700.0 * index if category in ("altitude", "altitude_heading") else 3000.0 + rng.uniform(-300, 300)
        speed = [155.0, 190.0, 245.0, 295.0][index] if category == "speed" else float(rng.uniform(175, 275))
        if category == "altitude_heading":
            theta = float(rng.uniform(-0.25, 0.25))
            heading = float(rng.uniform(-np.pi, np.pi))
        else:
            heading = float(-np.pi + index * np.pi / 2) if category == "heading" else float(rng.uniform(-np.pi, np.pi))
            theta = float(rng.uniform(-0.25, 0.25))
        states.append(AircraftState(
            x=float(x), y=float(y), z=-float(altitude),
            v=speed, theta=theta,
            psi=heading, alive=index < survivors,
        ))
    return states


def _run_stress_batch(
    config: dict[str, Any], start: int, stop: int
) -> dict[str, Any]:
    env = PersistentWaveCombatEnv(config)
    selected_indices: list[int] = []
    selected_angles_degrees: list[float] = []
    minimum_distances: list[float] = []
    first_red_window_steps: list[float] = []
    first_blue_window_steps: list[float] = []
    immediate_red_pairs = immediate_blue_pairs = failures = invalid_spawns = 0
    immediate_red_cases = immediate_blue_cases = 0
    failure_examples: list[dict[str, Any]] = []
    strata: dict[str, dict[str, int]] = {}

    for case_index in range(start, stop):
        category = CATEGORIES[case_index % len(CATEGORIES)]
        survivors = (case_index // len(CATEGORIES)) % 4 + 1
        key = f"{category}/red_{survivors}"
        stratum = strata.setdefault(key, {"cases": 0, "successes": 0, "failures": 0})
        stratum["cases"] += 1
        seed = 70_000_000 + case_index
        env.reset(seed)
        env.red = red_layout(category, survivors, env.rng)
        try:
            selected_angle = env._spawn_next_wave()
        except RuntimeError as error:
            failures += 1
            stratum["failures"] += 1
            if len(failure_examples) < 20:
                failure_examples.append({
                    "case": case_index, "seed": seed, "category": category,
                    "red_survivors": survivors, "error": str(error),
                })
            continue
        stratum["successes"] += 1
        selected_indices.append(int(env.last_spawn_candidate_index))
        selected_angles_degrees.append(float(np.rad2deg(selected_angle)))
        alive_red = [state for state in env.red if state.alive]
        pair_distances = [
            engagement_geometry(red, blue).distance
            for red in alive_red for blue in env.blue
        ]
        minimum_distances.append(float(min(pair_distances)))
        red_pairs = env._window_pair_count(alive_red, env.blue)
        blue_pairs = env._window_pair_count(env.blue, alive_red)
        invalid_spawns += int(not env._blue_wave_inside_arena(env.blue))
        immediate_red_pairs += red_pairs
        immediate_blue_pairs += blue_pairs
        immediate_red_cases += int(red_pairs > 0)
        immediate_blue_cases += int(blue_pairs > 0)

        red_first = blue_first = None
        zero_actions = np.zeros((4, 3), dtype=np.float32)
        for step in range(1, 11):
            _, _, terminated, truncated, info = env.step(zero_actions)
            if red_first is None and info["red_fire_window_pairs"]:
                red_first = step
            if blue_first is None and info["blue_fire_window_pairs"]:
                blue_first = step
            if terminated or truncated:
                break
        if red_first is not None:
            first_red_window_steps.append(float(red_first))
        if blue_first is not None:
            first_blue_window_steps.append(float(blue_first))

    return {
        "selected_indices": selected_indices,
        "selected_angles_degrees": selected_angles_degrees,
        "minimum_distances": minimum_distances,
        "first_red_window_steps": first_red_window_steps,
        "first_blue_window_steps": first_blue_window_steps,
        "immediate_red_pairs": immediate_red_pairs,
        "immediate_blue_pairs": immediate_blue_pairs,
        "failures": failures,
        "invalid_spawns": invalid_spawns,
        "immediate_red_cases": immediate_red_cases,
        "immediate_blue_cases": immediate_blue_cases,
        "failure_examples": failure_examples,
        "strata": strata,
    }


def run_stress(
    config: dict[str, Any], cases: int, workers: int = 1
) -> dict[str, Any]:
    workers = max(1, min(int(workers), cases))
    boundaries = np.linspace(0, cases, workers + 1, dtype=int)
    if workers == 1:
        batches = [_run_stress_batch(config, 0, cases)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = list(executor.map(
                _run_stress_batch,
                [config] * workers,
                boundaries[:-1].tolist(), boundaries[1:].tolist(),
            ))
    merged_lists = {
        key: [value for batch in batches for value in batch[key]]
        for key in (
            "selected_indices", "selected_angles_degrees", "minimum_distances", "first_red_window_steps",
            "first_blue_window_steps", "failure_examples",
        )
    }
    totals = {
        key: sum(batch[key] for batch in batches)
        for key in (
            "immediate_red_pairs", "immediate_blue_pairs", "failures",
            "invalid_spawns", "immediate_red_cases", "immediate_blue_cases",
        )
    }
    strata: dict[str, dict[str, int]] = {}
    for batch in batches:
        for key, row in batch["strata"].items():
            target = strata.setdefault(
                key, {"cases": 0, "successes": 0, "failures": 0}
            )
            for field in target:
                target[field] += row[field]
    selected_indices = merged_lists["selected_indices"]
    selected_angles_degrees = merged_lists["selected_angles_degrees"]
    minimum_distances = merged_lists["minimum_distances"]
    first_red_window_steps = merged_lists["first_red_window_steps"]
    first_blue_window_steps = merged_lists["first_blue_window_steps"]
    failures = totals["failures"]
    successes = cases - failures
    dt = float(config["simulation"]["dt"])
    index_counts = {
        str(index): selected_indices.count(index)
        for index in range(int(config["persistent_waves"]["spawn_direction_count"]))
    }
    result = {
        "environment_variant": config["environment_variant"],
        "cases": cases,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / cases,
        "failure_rate": failures / cases,
        "workers": workers,
        "selected_candidate_index_counts": index_counts,
        "selected_angle_degrees": percentile_summary(selected_angles_degrees),
        "invalid_spawns": totals["invalid_spawns"],
        "minimum_3d_red_blue_distance_m": percentile_summary(minimum_distances),
        "immediate_red_fire_window_pairs": totals["immediate_red_pairs"],
        "immediate_blue_fire_window_pairs": totals["immediate_blue_pairs"],
        "immediate_red_fire_window_rate": totals["immediate_red_cases"] / max(successes, 1),
        "immediate_blue_fire_window_rate": totals["immediate_blue_cases"] / max(successes, 1),
        "first_red_fire_window_within_1s_rate": len(first_red_window_steps) / max(successes, 1),
        "first_blue_fire_window_within_1s_rate": len(first_blue_window_steps) / max(successes, 1),
        "first_red_fire_window_step": percentile_summary(first_red_window_steps),
        "first_blue_fire_window_step": percentile_summary(first_blue_window_steps),
        "mean_first_red_fire_window_time_s": (
            float(np.mean(first_red_window_steps)) * dt
            if first_red_window_steps else None
        ),
        "mean_first_blue_fire_window_time_s": (
            float(np.mean(first_blue_window_steps)) * dt
            if first_blue_window_steps else None
        ),
        "strata": strata,
        "failure_examples": merged_lists["failure_examples"][:20],
        "minimum_distance_below_rate": {
            str(threshold): float(np.mean(np.asarray(minimum_distances) < threshold))
            for threshold in (500, 1000, 1500, 2000, 2500)
        },
    }
    result["passed"] = bool(
        failures == 0
        and totals["invalid_spawns"] == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--config", default="configs/persistent_wave_environment.yaml"
    )
    parser.add_argument(
        "--output", default="outputs/persistent_wave_spawn_stress.json"
    )
    args = parser.parse_args()
    if args.cases <= 0:
        raise ValueError("cases must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    result = run_stress(load_config(config_path), args.cases, args.workers)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
