"""Behavior-only audit for deterministic persistent-wave best checkpoints.

The scan batches policy inference but steps ordinary environment instances.  It
stores compact per-episode/event diagnostics, then replays representative seeds
to export full step trajectories and simple diagnostic figures.  No environment
semantics are changed by this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import yaml

from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.mappo.factory import build_mappo_trainer
from env.control import action_to_control
from env.factory import make_combat_environment


ROOT = PROJECT_ROOT
HISTORY_STEPS = 50  # five seconds at dt=0.1
GUARD_CANDIDATES = (1, 2, 3)


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def state_row(state, action: np.ndarray | None = None,
              target_index: int | None = None) -> dict[str, Any]:
    radial = float(np.hypot(state.x, state.y))
    velocity = state.velocity_vector()
    radial_velocity = (
        float((state.x * velocity[0] + state.y * velocity[1]) / radial)
        if radial > 1e-9 else 0.0
    )
    return {
        "alive": bool(state.alive), "x": float(state.x), "y": float(state.y),
        "altitude": float(state.altitude), "v": float(state.v),
        "theta": float(state.theta), "psi": float(state.psi),
        "vertical_velocity": float(state.v * np.sin(state.theta)),
        "radial_velocity": radial_velocity,
        "action_heading": None if action is None else float(action[0]),
        "action_pitch": None if action is None else float(action[1]),
        "action_speed": None if action is None else float(action[2]),
        "target_index": target_index,
    }


def classify_death(state, arena_radius: float) -> str:
    if np.hypot(state.x, state.y) > arena_radius:
        return "boundary_loss"
    if state.altitude <= 0.0:
        return "ground_loss"
    return "attack_kill"


def history_features(rows: list[dict[str, Any]], final_state,
                     target_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    own = rows[-HISTORY_STEPS:]
    theta = np.asarray([r["theta"] for r in own], dtype=float)
    altitude = np.asarray([r["altitude"] for r in own], dtype=float)
    radial_v = np.asarray([r["radial_velocity"] for r in own], dtype=float)
    pitch = np.asarray([
        r["action_pitch"] for r in own if r["action_pitch"] is not None
    ], dtype=float)
    targets = [r["target_index"] for r in own if r["target_index"] is not None]
    result = {
        "window_steps": len(own), "window_seconds": len(own) * 0.1,
        "altitude_start_m": float(altitude[0]),
        "altitude_final_m": float(final_state.altitude),
        "altitude_min_m": float(min(altitude.min(), final_state.altitude)),
        "theta_mean_deg": float(np.rad2deg(theta.mean())),
        "theta_min_deg": float(np.rad2deg(theta.min())),
        "descending_fraction": float(np.mean(theta < np.deg2rad(-5.0))),
        "steep_dive_fraction": float(np.mean(theta < np.deg2rad(-15.0))),
        "low_altitude_fraction_below_500m": float(np.mean(altitude < 500.0)),
        "radial_velocity_mean_mps": float(radial_v.mean()),
        "outward_fraction": float(np.mean(radial_v > 0.0)),
        "pitch_action_mean": float(pitch.mean()) if pitch.size else None,
        "pitch_action_min": float(pitch.min()) if pitch.size else None,
        "dominant_target_fraction": (
            Counter(targets).most_common(1)[0][1] / len(targets) if targets else None
        ),
        "dominant_target_index": (
            Counter(targets).most_common(1)[0][0] if targets else None
        ),
    }
    if target_rows:
        target_alt = np.asarray([r["altitude"] for r in target_rows], dtype=float)
        target_theta = np.asarray([r["theta"] for r in target_rows], dtype=float)
        result.update({
            "target_altitude_mean_m": float(target_alt.mean()),
            "target_altitude_min_m": float(target_alt.min()),
            "target_low_fraction_below_750m": float(np.mean(target_alt < 750.0)),
            "target_descending_fraction": float(
                np.mean(target_theta < np.deg2rad(-5.0))
            ),
        })
        result["low_diving_target_pattern"] = bool(
            (result["dominant_target_fraction"] or 0.0) >= 0.8
            and result["target_low_fraction_below_750m"] >= 0.5
            and result["target_descending_fraction"] >= 0.5
        )
    return result


def new_scan_state(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "red_history": [deque(maxlen=HISTORY_STEPS) for _ in range(4)],
        "blue_history": [deque(maxlen=HISTORY_STEPS) for _ in range(4)],
        "blue_target_history": [deque(maxlen=HISTORY_STEPS) for _ in range(4)],
        "last_blue_target": [None] * 4,
        "previous_blue_target": [None] * 4,
        "target_switches": 0, "target_transitions": 0,
        "target_aba_oscillations": 0,
        "ground_events": [], "boundary_events": [], "spawn_events": [],
        "pending_spawns": [], "max_red_displacement_m": 0.0,
        "max_blue_displacement_m": 0.0, "invalid_state_count": 0,
        "no_red_kill_steps": 0, "no_red_kill_shaping": 0.0,
        "max_no_red_kill_steps": 0, "max_no_red_kill_shaping_abs": 0.0,
        "blue_decision_steps": 0,
        "guard_candidate_steps": {value: 0 for value in GUARD_CANDIDATES},
        "guard_first_trigger_steps": {
            value: [None] * 4 for value in GUARD_CANDIDATES
        },
    }


def guard_candidate(state, commanded_pitch: float, config: dict[str, Any],
                    multiplier: int) -> tuple[bool, float | None]:
    downward_speed = max(
        -state.v * np.sin(state.theta),
        -state.v * np.sin(commanded_pitch),
        0.0,
    )
    if downward_speed <= 1e-6:
        return False, None
    time_to_ground = state.altitude / downward_speed
    guard_time = (
        multiplier
        * float(config["action"]["controller"]["pitch_time_constant"])
    )
    return bool(time_to_ground <= guard_time), float(time_to_ground)


def update_first_attempts(audit: dict[str, Any], step: int,
                          red_attempts: int, blue_attempts: int) -> None:
    for event in audit["pending_spawns"]:
        if event["red_first_attempt_step"] is None and red_attempts:
            event["red_first_attempt_step"] = step
        if event["blue_first_attempt_step"] is None and blue_attempts:
            event["blue_first_attempt_step"] = step
    audit["pending_spawns"] = [
        event for event in audit["pending_spawns"]
        if event["red_first_attempt_step"] is None
        or event["blue_first_attempt_step"] is None
    ]


def predicted_states(env, states, actions: np.ndarray):
    """Reproduce _advance without mutation, including states replaced at spawn."""
    result = []
    for state, action in zip(states, actions):
        if not state.alive:
            result.append(state.copy())
            continue
        control = action_to_control(state, action, env.config["action"])
        result.append(env.integrator.step(state, control, env.dynamics, env.spec))
    return result


def scan_checkpoint(actor, config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    envs = [make_combat_environment(config) for _ in seeds]
    observations = []
    audits = [new_scan_state(seed) for seed in seeds]
    active = np.ones(len(seeds), dtype=bool)
    final_rows: list[dict[str, Any] | None] = [None] * len(seeds)
    for env, seed in zip(envs, seeds):
        observation, _ = env.reset(seed)
        observations.append(observation)

    while np.any(active):
        indices = np.flatnonzero(active)
        batch_observation = np.stack([observations[i] for i in indices])
        batch_masks = np.stack([envs[i].red_alive_mask for i in indices])
        batch_actions = actor.act(
            batch_observation, batch_masks, deterministic=True
        )
        for batch_index, env_index in enumerate(indices):
            env, audit = envs[env_index], audits[env_index]
            wave_before = int(env.wave_index)
            red_actions = batch_actions[batch_index]
            blue_actions = env.fixed_policy.team_actions(env.blue, env.red)
            pre_red = [state.copy() for state in env.red]
            pre_blue = [state.copy() for state in env.blue]
            predicted_blue = predicted_states(env, pre_blue, blue_actions)
            blue_targets = [
                env.fixed_policy.nearest_target_index(state, env.red)
                if state.alive else None for state in env.blue
            ]
            for i, state in enumerate(pre_red):
                if state.alive:
                    audit["red_history"][i].append(
                        state_row(state, red_actions[i], None)
                    )
            for i, state in enumerate(pre_blue):
                if not state.alive:
                    continue
                target = blue_targets[i]
                row = state_row(state, blue_actions[i], target)
                audit["blue_history"][i].append(row)
                audit["blue_decision_steps"] += 1
                target_state = pre_red[target]
                commanded_pitch = float(np.arctan2(
                    state.z - target_state.z,
                    np.hypot(target_state.x - state.x, target_state.y - state.y),
                ))
                for multiplier in GUARD_CANDIDATES:
                    triggered, _ = guard_candidate(
                        state, commanded_pitch, config, multiplier
                    )
                    if triggered:
                        audit["guard_candidate_steps"][multiplier] += 1
                        if audit["guard_first_trigger_steps"][multiplier][i] is None:
                            audit["guard_first_trigger_steps"][multiplier][i] = env.steps
                if target is not None and pre_red[target].alive:
                    audit["blue_target_history"][i].append(
                        state_row(pre_red[target], red_actions[target], None)
                    )
                last, previous = (
                    audit["last_blue_target"][i],
                    audit["previous_blue_target"][i],
                )
                if last is not None and target is not None:
                    audit["target_transitions"] += 1
                    if target != last:
                        audit["target_switches"] += 1
                    if previous == target and last != target:
                        audit["target_aba_oscillations"] += 1
                audit["previous_blue_target"][i] = last
                audit["last_blue_target"][i] = target

            observation, reward, terminated, truncated, info = env.step(
                red_actions, blue_actions
            )
            observations[env_index] = observation
            step = int(env.steps)
            update_first_attempts(
                audit, step, int(info["red_step_fire_attempts"]),
                int(info["blue_step_fire_attempts"]),
            )

            shaping = float(np.sum(info["r3_rewards"] + info["r4_rewards"]))
            audit["no_red_kill_steps"] += 1
            audit["no_red_kill_shaping"] += shaping
            audit["max_no_red_kill_steps"] = max(
                audit["max_no_red_kill_steps"], audit["no_red_kill_steps"]
            )
            audit["max_no_red_kill_shaping_abs"] = max(
                audit["max_no_red_kill_shaping_abs"],
                abs(audit["no_red_kill_shaping"]),
            )
            if int(info["red_step_attack_kills"]):
                audit["no_red_kill_steps"] = 0
                audit["no_red_kill_shaping"] = 0.0

            spawned = bool(info.get("spawned_next_wave", False))
            for side, before, after, histories in (
                ("red", pre_red, env.red, audit["red_history"]),
                ("blue", pre_blue, env.blue, audit["blue_history"]),
            ):
                if side == "blue" and spawned:
                    # env.blue now holds fresh aircraft; use the independently
                    # reproduced post-motion old states to classify noncombat deaths.
                    newly_dead = [i for i, state in enumerate(before) if state.alive]
                    after_for_death = predicted_blue
                else:
                    newly_dead = [
                        i for i, (old, new) in enumerate(zip(before, after))
                        if old.alive and not new.alive
                    ]
                    after_for_death = after
                for aircraft in newly_dead:
                    death_type = classify_death(
                        after_for_death[aircraft], env.arena_radius
                    )
                    if death_type not in ("ground_loss", "boundary_loss"):
                        continue
                    target_rows = None
                    if side == "blue":
                        target_rows = list(audit["blue_target_history"][aircraft])
                    event = {
                        "seed": seeds[env_index], "step": step,
                        "wave_index": wave_before,
                        "side": side, "aircraft": aircraft,
                        **history_features(
                            list(histories[aircraft]), after_for_death[aircraft],
                            target_rows,
                        ),
                    }
                    if side == "blue" and death_type == "ground_loss":
                        for multiplier in GUARD_CANDIDATES:
                            first = audit["guard_first_trigger_steps"][multiplier][aircraft]
                            event[f"guard_{multiplier}tau_triggered"] = first is not None
                            event[f"guard_{multiplier}tau_lead_steps"] = (
                                None if first is None else step - first
                            )
                    audit["ground_events" if death_type == "ground_loss"
                          else "boundary_events"].append(event)

            if spawned:
                red_alive = [state for state in env.red if state.alive]
                xy = np.asarray([[s.x, s.y] for s in red_alive], dtype=float)
                centroid = xy.mean(axis=0)
                spread = float(np.max(np.linalg.norm(xy - centroid, axis=1)))
                event = {
                    "seed": seeds[env_index], "clearing_step": step,
                    "new_wave_index": int(info["wave_index"]),
                    "candidate_index": int(info["wave_spawn_candidate_index"]),
                    "spawn_angle_deg": float(np.rad2deg(info["wave_spawn_radial_angle"])),
                    "minimum_spawn_distance_m": float(info["minimum_spawn_distance"]),
                    "red_centroid_x_m": float(centroid[0]),
                    "red_centroid_y_m": float(centroid[1]),
                    "red_centroid_radius_m": float(np.linalg.norm(centroid)),
                    "red_spread_m": spread,
                    "red_immediate_fire_window_pairs": int(
                        env._window_pair_count(env.red, env.blue)
                    ),
                    "blue_immediate_fire_window_pairs": int(
                        env._window_pair_count(env.blue, env.red)
                    ),
                    "fresh_blue_attempts_on_clearing_step": 0,
                    "fresh_blue_hits_on_clearing_step": 0,
                    "fresh_blue_kills_on_clearing_step": 0,
                    "red_first_attempt_step": None,
                    "blue_first_attempt_step": None,
                }
                audit["spawn_events"].append(event)
                audit["pending_spawns"].append(event)
                audit["blue_history"] = [
                    deque(maxlen=HISTORY_STEPS) for _ in range(4)
                ]
                audit["blue_target_history"] = [
                    deque(maxlen=HISTORY_STEPS) for _ in range(4)
                ]
                audit["last_blue_target"] = [None] * 4
                audit["previous_blue_target"] = [None] * 4
                audit["guard_first_trigger_steps"] = {
                    value: [None] * 4 for value in GUARD_CANDIDATES
                }

            for side, before, after in (
                ("red", pre_red, env.red), ("blue", pre_blue, env.blue)
            ):
                if side == "blue" and spawned:
                    continue
                for old, new in zip(before, after):
                    if old.alive and new.alive:
                        displacement = float(np.linalg.norm(np.asarray([
                            new.x - old.x, new.y - old.y, new.z - old.z,
                        ])))
                        key = f"max_{side}_displacement_m"
                        audit[key] = max(audit[key], displacement)
                        values = new.as_array()
                        if (not np.all(np.isfinite(values))
                                or not config["aircraft"]["v_min"] <= new.v <= config["aircraft"]["v_max"]
                                or not config["aircraft"]["theta_min"] <= new.theta <= config["aircraft"]["theta_max"]):
                            audit["invalid_state_count"] += 1

            if terminated or truncated:
                for event in audit["spawn_events"]:
                    red_step = event["red_first_attempt_step"]
                    blue_step = event["blue_first_attempt_step"]
                    event["red_time_to_first_attempt_s"] = (
                        None if red_step is None else 0.1 * (red_step - event["clearing_step"])
                    )
                    event["blue_time_to_first_attempt_s"] = (
                        None if blue_step is None else 0.1 * (blue_step - event["clearing_step"])
                    )
                final_rows[env_index] = {
                    "seed": seeds[env_index],
                    "success": bool(info["red_success"]),
                    "termination_reason": info["termination_reason"],
                    "episode_length": int(info["episode_length"]),
                    "waves_cleared": int(info["waves_cleared"]),
                    **{key: plain(info[key]) for key in (
                        "red_losses", "blue_losses", "red_attack_kills",
                        "blue_attack_kills", "red_boundary_exits",
                        "blue_boundary_exits", "red_ground_losses",
                        "blue_ground_losses", "episode_r1_total",
                        "episode_r2_total", "episode_r3_total", "episode_r4_total",
                        "red_fire_window_steps", "blue_fire_window_steps",
                        "red_fire_attempts", "blue_fire_attempts",
                        "red_weapon_hits", "blue_weapon_hits",
                    )},
                    "target_switches": audit["target_switches"],
                    "target_transitions": audit["target_transitions"],
                    "target_aba_oscillations": audit["target_aba_oscillations"],
                    "max_no_red_kill_steps": audit["max_no_red_kill_steps"],
                    "max_no_red_kill_shaping_abs": audit["max_no_red_kill_shaping_abs"],
                    "max_red_displacement_m": audit["max_red_displacement_m"],
                    "max_blue_displacement_m": audit["max_blue_displacement_m"],
                    "invalid_state_count": audit["invalid_state_count"],
                    "ground_events": audit["ground_events"],
                    "boundary_events": audit["boundary_events"],
                    "spawn_events": audit["spawn_events"],
                    "blue_decision_steps": audit["blue_decision_steps"],
                    "guard_candidate_steps": audit["guard_candidate_steps"],
                    **({key: plain(info[key]) for key in (
                        "blue_ground_guard_decision_steps",
                        "blue_ground_guard_override_steps",
                        "blue_ground_guard_activations",
                        "blue_ground_guard_activation_ratio",
                        "blue_ground_guard_max_duration_steps",
                    )} if "blue_ground_guard_decision_steps" in info else {}),
                }
                active[env_index] = False
    return [row for row in final_rows if row is not None]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grounds = [event for row in rows for event in row["ground_events"]]
    blue_grounds = [e for e in grounds if e["side"] == "blue"]
    boundaries = [event for row in rows for event in row["boundary_events"]]
    red_boundaries = [e for e in boundaries if e["side"] == "red"]
    spawns = [event for row in rows for event in row["spawn_events"]]
    index_counts = Counter(e["candidate_index"] for e in spawns)
    first = Counter()
    for event in spawns:
        red = event["red_time_to_first_attempt_s"]
        blue = event["blue_time_to_first_attempt_s"]
        if red is None and blue is None:
            first["neither"] += 1
        elif blue is None or (red is not None and red < blue):
            first["red"] += 1
        elif red is None or blue < red:
            first["blue"] += 1
        else:
            first["tie"] += 1
    red_times = [e["red_time_to_first_attempt_s"] for e in spawns
                 if e["red_time_to_first_attempt_s"] is not None]
    blue_times = [e["blue_time_to_first_attempt_s"] for e in spawns
                  if e["blue_time_to_first_attempt_s"] is not None]
    transitions = sum(r["target_transitions"] for r in rows)
    return {
        "episodes_audited": len(rows),
        "success_rate": mean(rows, "success"),
        "clear_wave_1_probability": float(np.mean([r["waves_cleared"] >= 1 for r in rows])),
        "clear_wave_2_probability": float(np.mean([r["waves_cleared"] >= 2 for r in rows])),
        "clear_wave_3_probability": float(np.mean([r["waves_cleared"] >= 3 for r in rows])),
        "average_waves_cleared": mean(rows, "waves_cleared"),
        **{f"average_{key}": mean(rows, key) for key in (
            "red_attack_kills", "blue_attack_kills", "red_ground_losses",
            "blue_ground_losses", "red_boundary_exits", "blue_boundary_exits",
            "episode_r1_total", "episode_r2_total", "episode_r3_total",
            "episode_r4_total",
        )},
        "red_ground_loss_episode_rate": float(np.mean([r["red_ground_losses"] > 0 for r in rows])),
        "blue_ground_loss_episode_rate": float(np.mean([r["blue_ground_losses"] > 0 for r in rows])),
        "red_boundary_episode_rate": float(np.mean([r["red_boundary_exits"] > 0 for r in rows])),
        "blue_boundary_episode_rate": float(np.mean([r["blue_boundary_exits"] > 0 for r in rows])),
        "blue_ground_event_count": len(blue_grounds),
        **{
            f"guard_{multiplier}tau_historical_ground_event_coverage": (
                float(np.mean([
                    event.get(f"guard_{multiplier}tau_triggered", False)
                    for event in blue_grounds
                ])) if blue_grounds else None
            )
            for multiplier in GUARD_CANDIDATES
        },
        **{
            f"guard_{multiplier}tau_candidate_step_ratio": (
                sum(row["guard_candidate_steps"][str(multiplier)]
                    if str(multiplier) in row["guard_candidate_steps"]
                    else row["guard_candidate_steps"][multiplier]
                    for row in rows)
                / max(sum(row["blue_decision_steps"] for row in rows), 1)
            )
            for multiplier in GUARD_CANDIDATES
        },
        "blue_ground_low_diving_target_pattern_rate": (
            float(np.mean([e.get("low_diving_target_pattern", False) for e in blue_grounds]))
            if blue_grounds else None
        ),
        "blue_ground_dominant_target_fraction_mean": (
            float(np.mean([e["dominant_target_fraction"] for e in blue_grounds
                           if e["dominant_target_fraction"] is not None]))
            if blue_grounds else None
        ),
        "red_ground_commanded_descent_rate": (
            float(np.mean([
                (e["pitch_action_mean"] or 0.0) < -0.05
                and e["descending_fraction"] >= 0.5
                for e in grounds if e["side"] == "red"
            ])) if any(e["side"] == "red" for e in grounds) else None
        ),
        "red_boundary_sustained_outward_rate": (
            float(np.mean([
                e["outward_fraction"] >= 0.8 and e["radial_velocity_mean_mps"] > 0
                for e in red_boundaries
            ])) if red_boundaries else None
        ),
        "blue_target_switches_per_1000_transitions": (
            1000.0 * sum(r["target_switches"] for r in rows) / max(transitions, 1)
        ),
        "blue_target_aba_per_1000_transitions": (
            1000.0 * sum(r["target_aba_oscillations"] for r in rows) / max(transitions, 1)
        ),
        "spawn_count": len(spawns),
        "spawn_candidate_index_counts": dict(sorted(index_counts.items())),
        "spawn_unique_candidate_count": len(index_counts),
        "spawn_minimum_distance_mean_m": (
            float(np.mean([e["minimum_spawn_distance_m"] for e in spawns])) if spawns else None
        ),
        "spawn_minimum_distance_min_m": (
            float(np.min([e["minimum_spawn_distance_m"] for e in spawns])) if spawns else None
        ),
        "spawn_red_immediate_fire_window_rate": (
            float(np.mean([e["red_immediate_fire_window_pairs"] > 0 for e in spawns])) if spawns else None
        ),
        "spawn_blue_immediate_fire_window_rate": (
            float(np.mean([e["blue_immediate_fire_window_pairs"] > 0 for e in spawns])) if spawns else None
        ),
        "spawn_first_attempt_side_counts": dict(first),
        "spawn_red_time_to_first_attempt_mean_s": float(np.mean(red_times)) if red_times else None,
        "spawn_blue_time_to_first_attempt_mean_s": float(np.mean(blue_times)) if blue_times else None,
        "fresh_blue_same_step_violation_count": sum(
            e["fresh_blue_attempts_on_clearing_step"]
            + e["fresh_blue_hits_on_clearing_step"]
            + e["fresh_blue_kills_on_clearing_step"] for e in spawns
        ),
        "first_fresh_blue_attempt_minimum_offset_steps": min(
            [e["blue_first_attempt_step"] - e["clearing_step"] for e in spawns
             if e["blue_first_attempt_step"] is not None], default=None
        ),
        "maximum_red_step_displacement_m": max(r["max_red_displacement_m"] for r in rows),
        "maximum_blue_step_displacement_m": max(r["max_blue_displacement_m"] for r in rows),
        "invalid_physical_state_count": sum(r["invalid_state_count"] for r in rows),
        "maximum_no_red_kill_interval_s": 0.1 * max(r["max_no_red_kill_steps"] for r in rows),
        "maximum_abs_shaping_without_red_kill": max(r["max_no_red_kill_shaping_abs"] for r in rows),
        "average_blue_fire_window_steps": mean(rows, "blue_fire_window_steps"),
        "average_blue_fire_attempts": mean(rows, "blue_fire_attempts"),
        "average_blue_weapon_hits": mean(rows, "blue_weapon_hits"),
        **({
            "ground_guard_activation_episode_rate": float(np.mean([
                row["blue_ground_guard_activations"] > 0 for row in rows
            ])),
            "average_ground_guard_activations": mean(
                rows, "blue_ground_guard_activations"
            ),
            "ground_guard_override_step_ratio": (
                sum(row["blue_ground_guard_override_steps"] for row in rows)
                / max(sum(row["blue_ground_guard_decision_steps"] for row in rows), 1)
            ),
            "ground_guard_mean_activation_duration_steps": (
                sum(row["blue_ground_guard_override_steps"] for row in rows)
                / max(sum(row["blue_ground_guard_activations"] for row in rows), 1)
            ),
            "average_ground_guard_max_duration_steps": mean(
                rows, "blue_ground_guard_max_duration_steps"
            ),
            "maximum_ground_guard_duration_steps": max(
                row["blue_ground_guard_max_duration_steps"] for row in rows
            ),
        } if "blue_ground_guard_activations" in rows[0] else {}),
    }


def representatives(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    predicates = {
        "three_wave_low_loss_success": lambda r: r["success"] and r["red_losses"] <= 1,
        "three_wave_high_loss_success": lambda r: r["success"] and r["red_losses"] >= 2,
        "first_or_second_wave_failure": lambda r: not r["success"] and r["waves_cleared"] <= 1,
        "red_boundary_loss": lambda r: r["red_boundary_exits"] > 0,
        "red_ground_loss": lambda r: r["red_ground_losses"] > 0,
        "blue_ground_loss": lambda r: r["blue_ground_losses"] > 0,
        "ground_guard_active": lambda r: r.get(
            "blue_ground_guard_activations", 0
        ) > 0,
    }
    result = {}
    for name, predicate in predicates.items():
        matches = [row for row in rows if predicate(row)]
        result[name] = matches[0]["seed"] if matches else None
    return result


def capture_trajectory(actor, config: dict[str, Any], seed: int,
                       category: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = make_combat_environment(config)
    observation, _ = env.reset(seed)
    rows: list[dict[str, Any]] = []
    cumulative = {name: 0.0 for name in ("r1", "r2", "r3", "r4")}
    while True:
        wave_before = int(env.wave_index)
        actions = actor.act(observation, env.red_alive_mask, deterministic=True)
        blue_actions = env.fixed_policy.team_actions(env.blue, env.red)
        pre_red = [state.copy() for state in env.red]
        pre_blue = [state.copy() for state in env.blue]
        predicted_blue = predicted_states(env, pre_blue, blue_actions)
        targets = [
            env.fixed_policy.nearest_target_index(state, env.red)
            if state.alive else None for state in env.blue
        ]
        guard_mask = getattr(
            env.fixed_policy, "last_override_mask", np.zeros(4, dtype=bool)
        ).copy()
        step_rows = []
        for side, states, side_actions in (
            ("red", pre_red, actions), ("blue", pre_blue, blue_actions)
        ):
            for aircraft, state in enumerate(states):
                row = {
                    "category": category, "seed": seed, "step": env.steps,
                    "wave_index": wave_before, "side": side,
                    "aircraft": aircraft, **state_row(
                        state, side_actions[aircraft],
                        targets[aircraft] if side == "blue" else None,
                    ),
                    "last_executed_phi": float(
                        env.red_last_executed_phi[aircraft]
                        if side == "red" else env.blue_last_executed_phi[aircraft]
                    ),
                    "event": "", "spawned_next_wave": False,
                    "ground_guard_override": (
                        bool(guard_mask[aircraft]) if side == "blue" else False
                    ),
                    "wave_spawn_candidate_index": None,
                    "minimum_spawn_distance": None,
                }
                rows.append(row); step_rows.append(row)
        observation, reward, terminated, truncated, info = env.step(
            actions, blue_actions
        )
        for name in cumulative:
            cumulative[name] += float(np.sum(info[f"{name}_rewards"]))
        for row in step_rows:
            row.update({
                "transition_step": int(env.steps),
                "red_step_fire_attempts": int(info["red_step_fire_attempts"]),
                "blue_step_fire_attempts": int(info["blue_step_fire_attempts"]),
                "red_step_weapon_hits": int(info["red_step_weapon_hits"]),
                "blue_step_weapon_hits": int(info["blue_step_weapon_hits"]),
                "red_step_attack_kills": int(info["red_step_attack_kills"]),
                "blue_step_attack_kills": int(info["blue_step_attack_kills"]),
                **{f"cumulative_{name}": cumulative[name] for name in cumulative},
                "spawned_next_wave": bool(info.get("spawned_next_wave", False)),
                "wave_spawn_candidate_index": info.get("wave_spawn_candidate_index"),
                "minimum_spawn_distance": info.get("minimum_spawn_distance"),
            })
        spawned = bool(info.get("spawned_next_wave", False))
        for side, before, after in (
            ("red", pre_red, env.red), ("blue", pre_blue, env.blue)
        ):
            for aircraft, old in enumerate(before):
                if not old.alive:
                    continue
                died = spawned if side == "blue" else not after[aircraft].alive
                if died:
                    final_state = predicted_blue[aircraft] if side == "blue" and spawned else after[aircraft]
                    death = classify_death(final_state, env.arena_radius)
                    for row in step_rows:
                        if row["side"] == side and row["aircraft"] == aircraft:
                            row["event"] = death
                            break
        if info.get("wave_cleared_this_step"):
            for row in step_rows:
                row["event"] = (row["event"] + "|wave_clear").strip("|")
        if terminated or truncated:
            summary = {
                "seed": seed, "category": category,
                "termination_reason": info["termination_reason"],
                "waves_cleared": int(info["waves_cleared"]),
                "red_losses": int(info["red_losses"]),
                "blue_losses": int(info["blue_losses"]),
                **{f"episode_{name}_total": float(info[f"episode_{name}_total"])
                   for name in cumulative},
            }
            return rows, summary


def write_trajectories(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_actor(checkpoint: Path, algorithm_config: Path, device: str,
               environment_config: dict[str, Any],
               checkpoint_environment_variant: str | None = None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = dict(environment_config)
    if checkpoint_environment_variant is not None:
        checkpoint_config["environment_variant"] = checkpoint_environment_variant
    validate_checkpoint_environment(state, checkpoint_config)
    config = yaml.safe_load(algorithm_config.read_text(encoding="utf-8"))
    actor = build_mappo_trainer(config, device)
    actor.load(checkpoint)
    return actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("mappo",), default="mappo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algorithm-config", required=True)
    parser.add_argument("--env-config", default="configs/persistent_wave_environment.yaml")
    parser.add_argument("--seed-base", type=int, default=30_000_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-environment-variant",
        help="Explicit source variant for diagnostic cross-environment evaluation",
    )
    parser.add_argument(
        "--ground-guard-time-constants", type=float,
        help="Diagnostic override for the v2 guard multiplier",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    env_path = resolved(args.env_config)
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    if args.ground_guard_time_constants is not None:
        env_config["blue_policy"]["ground_avoidance"][
            "guard_time_constants"
        ] = args.ground_guard_time_constants
    checkpoint = resolved(args.checkpoint)
    actor = load_actor(
        checkpoint, resolved(args.algorithm_config), args.device, env_config,
        args.checkpoint_environment_variant,
    )
    output = resolved(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    rows = scan_checkpoint(actor, env_config, seeds)
    selected = representatives(rows)
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_summaries = []
    for category, seed in selected.items():
        if seed is None:
            continue
        captured, summary = capture_trajectory(actor, env_config, seed, category)
        trajectory_rows.extend(captured); trajectory_summaries.append(summary)
    write_trajectories(trajectory_rows, output / "representative_trajectories.csv")
    report = {
        "algorithm": "MAPPO", "checkpoint": str(checkpoint),
        "environment_config": str(env_path), "seed_base": args.seed_base,
        "episodes": args.episodes, "summary": summarize(rows),
        "representative_seeds": selected,
        "representative_summaries": trajectory_summaries,
        "episodes_detail": rows,
    }
    (output / "behavior_audit.json").write_text(
        json.dumps(plain(report), indent=2), encoding="utf-8"
    )
    print(json.dumps(plain({k: v for k, v in report.items()
                            if k != "episodes_detail"}), indent=2))


if __name__ == "__main__":
    main()
