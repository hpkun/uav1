"""Read-only Persistent-Mission V3 task-definition audit.

This tool never trains or updates a model.  It evaluates only the three existing
Plain exact-3M checkpoints on the already-exposed 44M development seeds.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_modular_checkpoint
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from env.factory import make_combat_environment
from env.models import AircraftState
from env.reward import paper_state_reward_components
from env.weapon import FireState


ENV_PATH = ROOT / "configs/persistent_wave_v2_environment.yaml"
PLAIN_ROOT = ROOT / "outputs/diag_mappo_learnability"
OUTPUT_DIR = ROOT / "outputs/persistent_mission_v3_audit"
JSON_PATH = OUTPUT_DIR / "persistent_mission_v3_audit.json"
TEXT_PATH = OUTPUT_DIR / "persistent_mission_v3_audit.txt"
TRAINING_SEEDS = (5301, 5302, 5303)
EVALUATION_SEEDS = tuple(range(44_000_000, 44_000_050))


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def describe(values):
    values = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not values.size:
        return None
    return {
        "N": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "median": float(np.median(values)), "P25": float(np.quantile(values, 0.25)),
        "P75": float(np.quantile(values, 0.75)), "min": float(values.min()), "max": float(values.max()),
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return {"N": len(pairs), "pearson": None, "spearman": None, "kendall_tau_b": None}
    a = np.asarray([row[0] for row in pairs]); b = np.asarray([row[1] for row in pairs])
    concordant = discordant = ties_a = ties_b = 0
    for index in range(len(a)):
        for other in range(index + 1, len(a)):
            da, db = np.sign(a[index] - a[other]), np.sign(b[index] - b[other])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return {
        "N": len(pairs), "pearson": pearson(a, b),
        "spearman": pearson(average_ranks(a), average_ranks(b)),
        "kendall_tau_b": None if denominator == 0 else (concordant - discordant) / denominator,
    }


def safe_states():
    red = [
        AircraftState(-500.0, 0.0, -3000.0, 200.0, 0.0, 0.0),
        AircraftState(-1000.0, 1000.0, -3200.0, 200.0, 0.0, 0.5),
        AircraftState(-1000.0, -1000.0, -2800.0, 200.0, 0.0, -0.5),
        AircraftState(-1500.0, 0.0, -3400.0, 200.0, 0.0, 1.0),
    ]
    blue = [
        AircraftState(500.0, 0.0, -3000.0, 200.0, 0.0, 0.0),
        AircraftState(1800.0, 1200.0, -3400.0, 200.0, 0.0, 2.5),
        AircraftState(1800.0, -1200.0, -2600.0, 200.0, 0.0, -2.5),
        AircraftState(2400.0, 0.0, -3600.0, 200.0, 0.0, math.pi),
    ]
    return red, blue


def configure_alias_environment(environment, seed=87_000_001):
    environment.reset(seed)
    red, blue = safe_states()
    environment.red = [state.copy() for state in red]
    environment.blue = [state.copy() for state in blue]
    environment.red_last_executed_phi.fill(0.0)
    environment.blue_last_executed_phi.fill(0.0)
    environment.red_fire_states = [FireState(False) for _ in range(4)]
    environment.blue_fire_states = [FireState(False) for _ in range(4)]
    return environment


def alias_audits(environment_config):
    zero = np.zeros((4, 3), dtype=np.float32)

    armed = configure_alias_environment(make_combat_environment(environment_config))
    armed.red_fire_states[0].armed = True
    unarmed = copy.deepcopy(armed)
    unarmed.red_fire_states[0].armed = False
    fire_observation_equal = np.array_equal(armed._observations(), unarmed._observations())
    armed_rng = copy.deepcopy(armed.rng.bit_generator.state)
    unarmed.rng.bit_generator.state = copy.deepcopy(armed_rng)
    _, reward_armed, _, _, info_armed = armed.step(zero, zero)
    _, reward_unarmed, _, _, info_unarmed = unarmed.step(zero, zero)
    fire_transition_difference = any((
        info_armed["red_step_fire_attempts"] != info_unarmed["red_step_fire_attempts"],
        info_armed["red_step_weapon_hits"] != info_unarmed["red_step_weapon_hits"],
        not np.array_equal(info_armed["blue_alive_mask"], info_unarmed["blue_alive_mask"]),
        not np.array_equal(reward_armed, reward_unarmed),
    ))

    middle = configure_alias_environment(make_combat_environment(environment_config), 87_000_002)
    for state in middle.blue:
        state.alive = False
    middle.wave_index = 1; middle.waves_cleared = 0
    final = copy.deepcopy(middle); final.wave_index = final.total_waves; final.waves_cleared = final.total_waves - 1
    wave_observation_equal = np.array_equal(middle._observations(), final._observations())
    middle.rng.bit_generator.state = copy.deepcopy(final.rng.bit_generator.state)
    _, middle_reward, middle_terminated, middle_truncated, middle_info = middle.step(zero, zero)
    _, final_reward, final_terminated, final_truncated, final_info = final.step(zero, zero)
    wave_transition_difference = (
        middle_info["spawned_next_wave"] and not middle_terminated and not middle_truncated
        and not final_info["spawned_next_wave"] and final_terminated and not final_truncated
    )

    early = configure_alias_environment(make_combat_environment(environment_config), 87_000_003)
    late = copy.deepcopy(early)
    early.steps = 100; late.steps = late.max_steps - 1
    horizon_observation_equal = np.array_equal(early._observations(), late._observations())
    early.rng.bit_generator.state = copy.deepcopy(late.rng.bit_generator.state)
    _, _, early_terminated, early_truncated, _ = early.step(zero, zero)
    _, _, late_terminated, late_truncated, _ = late.step(zero, zero)
    horizon_transition_difference = not early_terminated and not early_truncated and late_truncated

    semantic = configure_alias_environment(make_combat_environment(environment_config), 87_000_004)
    semantic.red_fire_states[0].armed = False
    r4 = paper_state_reward_components(semantic.red, semantic.blue, semantic.config["reward"])["r4"]
    target_available = semantic._select_target(semantic.red[0], semantic.blue) is not None
    unarmed_positive = bool(r4[0] > 0 and target_available and not semantic.red_fire_states[0].armed)

    fire_confirmed = bool(fire_observation_equal and fire_transition_difference)
    wave_confirmed = bool(wave_observation_equal and wave_transition_difference and np.array_equal(middle_reward, final_reward))
    horizon_confirmed = bool(horizon_observation_equal and horizon_transition_difference)
    return {
        "fire_state": {
            "label": "FIRE_STATE_OBSERVATION_ALIAS_CONFIRMED" if fire_confirmed else "FIRE_STATE_OBSERVATION_ALIAS_NOT_CONFIRMED",
            "observation_exact_equal": fire_observation_equal,
            "armed_attempts": int(info_armed["red_step_fire_attempts"]),
            "unarmed_attempts": int(info_unarmed["red_step_fire_attempts"]),
            "armed_hits": int(info_armed["red_step_weapon_hits"]),
            "unarmed_hits": int(info_unarmed["red_step_weapon_hits"]),
            "reward_exact_equal": bool(np.array_equal(reward_armed, reward_unarmed)),
            "next_blue_alive_equal": bool(np.array_equal(info_armed["blue_alive_mask"], info_unarmed["blue_alive_mask"])),
        },
        "wave_index": {
            "label": "WAVE_INDEX_OBSERVATION_ALIAS_CONFIRMED" if wave_confirmed else "WAVE_INDEX_OBSERVATION_ALIAS_NOT_CONFIRMED",
            "observation_exact_equal": wave_observation_equal,
            "clearing_reward_exact_equal": bool(np.array_equal(middle_reward, final_reward)),
            "middle_spawned_next_wave": bool(middle_info["spawned_next_wave"]),
            "middle_terminal": bool(middle_terminated or middle_truncated),
            "final_spawned_next_wave": bool(final_info["spawned_next_wave"]),
            "final_terminal": bool(final_terminated or final_truncated),
        },
        "remaining_horizon": {
            "label": "REMAINING_HORIZON_OBSERVATION_ALIAS_CONFIRMED" if horizon_confirmed else "REMAINING_HORIZON_OBSERVATION_ALIAS_NOT_CONFIRMED",
            "observation_exact_equal": horizon_observation_equal,
            "early_terminal": bool(early_terminated or early_truncated),
            "late_truncated": bool(late_truncated),
        },
        "unarmed_positive_r4": {
            "label": "UNARMED_POSITIVE_ATTACK_GEOMETRY_REWARD_CONFIRMED" if unarmed_positive else "UNARMED_POSITIVE_ATTACK_GEOMETRY_REWARD_NOT_CONFIRMED",
            "r4_agent_0": float(r4[0]), "target_in_weapon_window": bool(target_available),
            "armed": bool(semantic.red_fire_states[0].armed),
        },
        "observation_summary": (
            "ACTOR_OBSERVATION_NOT_MARKOV_FOR_PERSISTENT_MISSION"
            if any((fire_confirmed, wave_confirmed, horizon_confirmed))
            else "OBSERVATION_MISMATCH_NOT_CONFIRMED"
        ),
        "explicit_wave_clear_reward": "EXPLICIT_WAVE_CLEAR_REWARD_ABSENT" if np.array_equal(middle_reward, final_reward) else "INCONCLUSIVE",
        "explicit_survivor_at_clear_reward": "EXPLICIT_SURVIVOR_AT_CLEAR_REWARD_ABSENT",
    }


def load_plain_trainer(training_seed, environment_config):
    checkpoint_path = PLAIN_ROOT / f"l3_seed{training_seed}" / "checkpoint_3000000.pt"
    state = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    algorithm_config = state.get("extra", {}).get("algorithm_config")
    if not isinstance(algorithm_config, dict):
        algorithm_config = yaml.safe_load((checkpoint_path.parent / "algorithm_config.yaml").read_text(encoding="utf-8"))
    validate_modular_checkpoint(state, environment_config, algorithm_config)
    architecture = state.get("extra", {}).get("network_architecture", {})
    hidden_dim = int(architecture.get("hidden_dim", algorithm_config["network"]["actor_hidden_layers"][0]))
    trainer = build_modular_mappo_trainer(algorithm_config, "cuda", hidden_dim, 3_000_000)
    trainer.load(checkpoint_path, strict_protocol=True, restore_rng=False)
    trainer.actor.eval(); trainer.critic.eval()
    return trainer, checkpoint_path


def evaluate_episode(trainer, environment_config, evaluation_seed):
    environment = make_combat_environment(environment_config)
    observation, _ = environment.reset(evaluation_seed)
    alive = environment.red_alive_mask.copy()
    actor_hidden, _ = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, dtype=np.float32)
    agent_returns = np.zeros(4, dtype=np.float64)
    step_local_variances = []
    weapon = {"living_samples": 0, "unarmed_samples": 0, "positive_r4_samples": 0,
              "unarmed_positive_r4_samples": 0, "positive_r4_magnitude": 0.0,
              "unarmed_positive_r4_magnitude": 0.0, "unarmed_positive_run_lengths": []}
    active_unarmed_positive = np.zeros(4, dtype=np.int64)
    transitions = []
    active_transitions = []
    previous_boundary = 0
    final_info = None
    while True:
        current_alive = environment.red_alive_mask.astype(bool)
        current_armed = np.asarray([state.armed for state in environment.red_fire_states], dtype=bool)
        wave = np.asarray([environment.wave_index], dtype=np.int64)
        total = np.asarray([environment.total_waves], dtype=np.int64)
        context = mission_context_numpy(trainer, wave, total, environment.blue_alive_mask[None], np.asarray([environment.steps]), environment.max_steps)
        actions, actor_hidden = trainer.act(observation[None], alive[None], True, False, context, actor_hidden, episode_mask)
        observation, reward, terminated, truncated, info = environment.step(actions[0])
        agent_returns += reward
        step_local_variances.append(float(np.var(reward)))
        r4 = np.asarray(info["r4_rewards"], dtype=np.float64)
        positive = current_alive & (r4 > 0)
        unarmed = current_alive & ~current_armed
        unarmed_positive = positive & ~current_armed
        weapon["living_samples"] += int(current_alive.sum())
        weapon["unarmed_samples"] += int(unarmed.sum())
        weapon["positive_r4_samples"] += int(positive.sum())
        weapon["unarmed_positive_r4_samples"] += int(unarmed_positive.sum())
        weapon["positive_r4_magnitude"] += float(r4[positive].sum())
        weapon["unarmed_positive_r4_magnitude"] += float(r4[unarmed_positive].sum())
        for index in range(4):
            if unarmed_positive[index]:
                active_unarmed_positive[index] += 1
            elif active_unarmed_positive[index]:
                weapon["unarmed_positive_run_lengths"].append(int(active_unarmed_positive[index]))
                active_unarmed_positive[index] = 0

        boundary_now = int(info.get("red_boundary_exits", 0))
        boundary_delta = boundary_now - previous_boundary
        previous_boundary = boundary_now
        for transition in active_transitions:
            elapsed = int(environment.steps - transition["transition_step"])
            if boundary_delta > 0:
                for horizon, steps in (("2s", 20), ("5s", 50), ("10s", 100)):
                    if elapsed <= steps:
                        transition[f"boundary_exit_within_{horizon}"] = True
                        transition[f"boundary_exit_count_within_{horizon}"] += boundary_delta
        active_transitions = [row for row in active_transitions if environment.steps - row["transition_step"] <= 100]
        if info.get("spawned_next_wave", False):
            radii = [float(np.hypot(state.x, state.y)) for state in environment.red if state.alive]
            margins = [environment.arena_radius - radius for radius in radii]
            transition = {
                "source_wave": int(info["wave_index"]) - 1,
                "target_wave": int(info["wave_index"]),
                "transition_step": int(environment.steps),
                "red_survivors": len(radii), "red_radial_distances": radii,
                "boundary_margins": margins, "entry_red_max_radius": max(radii),
                "entry_min_boundary_margin": min(margins),
                "entry_mean_boundary_margin": float(np.mean(margins)),
                "entry_max_boundary_margin": max(margins),
                "spawn_angle": float(info["wave_spawn_radial_angle"]),
                "spawn_candidate_index": int(info["wave_spawn_candidate_index"]),
                "minimum_spawn_distance": float(info["minimum_spawn_distance"]),
                "boundary_exit_within_2s": False, "boundary_exit_within_5s": False,
                "boundary_exit_within_10s": False, "boundary_exit_count_within_2s": 0,
                "boundary_exit_count_within_5s": 0, "boundary_exit_count_within_10s": 0,
            }
            transitions.append(transition); active_transitions.append(transition)

        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        episode_mask[:] = 1.0
        final_info = info
        if terminated or truncated:
            break
    for index in range(4):
        if active_unarmed_positive[index]:
            weapon["unarmed_positive_run_lengths"].append(int(active_unarmed_positive[index]))
    wave_records = [dict(row) for row in final_info.get("per_wave_metrics", [])]
    by_wave = {int(row["wave_index"]): row for row in wave_records}
    for transition in transitions:
        following = by_wave.get(transition["target_wave"])
        transition["next_wave_clear"] = bool(following and following.get("wave_cleared", False))
    return {
        "training_seed": None, "evaluation_seed": int(evaluation_seed),
        "waves_cleared": int(environment.waves_cleared),
        "return": float(agent_returns.sum()), "agent_returns": agent_returns.tolist(),
        "individual_return_variance": float(np.var(agent_returns)),
        "individual_return_disparity": float(agent_returns.max() - agent_returns.min()),
        "mean_step_local_reward_variance": float(np.mean(step_local_variances)),
        "red_losses": int(final_info["red_losses"]), "blue_losses": int(final_info["blue_losses"]),
        "boundary_exits": int(final_info["red_boundary_exits"]), "ground_losses": int(final_info["red_ground_losses"]),
        "episode_length": int(final_info["episode_length"]),
        "r1_total": float(final_info["episode_r1_total"]), "r2_total": float(final_info["episode_r2_total"]),
        "r3_total": float(final_info["episode_r3_total"]), "r4_total": float(final_info["episode_r4_total"]),
        "wave_records": wave_records, "weapon_frequency": weapon, "spawn_transitions": transitions,
    }


def reward_analysis(episodes):
    returns = [row["return"] for row in episodes]; waves = [row["waves_cleared"] for row in episodes]
    grouped = {str(wave): describe([row["return"] for row in episodes if row["waves_cleared"] == wave]) for wave in range(4)}
    comparisons = {}
    for low, high in ((0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)):
        lower = [row["return"] for row in episodes if row["waves_cleared"] == low]
        higher = [row["return"] for row in episodes if row["waves_cleared"] == high]
        count = len(lower) * len(higher)
        inversions = sum(high_value <= low_value for high_value in higher for low_value in lower)
        comparisons[f"{low}_vs_{high}"] = {"pairs": count, "inversions": inversions, "inversion_rate": None if not count else inversions / count}
    all_pairs = sum(item["pairs"] for item in comparisons.values())
    all_inversions = sum(item["inversions"] for item in comparisons.values())
    adjacent = [comparisons[key] for key in ("0_vs_1", "1_vs_2", "2_vs_3")]
    populated_adjacent = [item for item in adjacent if item["pairs"]]
    correlation = correlations(returns, waves)
    if len(populated_adjacent) < 2:
        label = "REWARD_MISSION_RANKING_MISMATCH_INCONCLUSIVE"
    elif all(item["inversions"] > 0 for item in populated_adjacent) and correlation["spearman"] != 1.0:
        label = "REWARD_MISSION_RANKING_MISMATCH_SUPPORTED"
    else:
        label = "REWARD_MISSION_RANKING_MISMATCH_NOT_SUPPORTED"
    return {"correlation": correlation, "return_by_waves": grouped, "pairwise": comparisons,
            "pooled_pairwise_inversion_rate": None if not all_pairs else all_inversions / all_pairs,
            "label": label}


def component_analysis(episodes):
    result = {}
    total_absolute = sum(abs(row[f"r{index}_total"]) for row in episodes for index in range(1, 5))
    for index in range(1, 5):
        key = f"r{index}_total"; values = [row[key] for row in episodes]
        survivors_1, survivors_2 = [], []
        for row in episodes:
            by_wave = {int(record["wave_index"]): record for record in row["wave_records"]}
            survivors_1.append(by_wave.get(1, {}).get("red_survivors_end") if by_wave.get(1, {}).get("wave_cleared") else None)
            survivors_2.append(by_wave.get(2, {}).get("red_survivors_end") if by_wave.get(2, {}).get("wave_cleared") else None)
        result[f"R{index}"] = {
            "mean_absolute_contribution": float(np.mean(np.abs(values))),
            "median_absolute_contribution": float(np.median(np.abs(values))),
            "signed_mean": float(np.mean(values)),
            "fraction_of_absolute_components": None if total_absolute == 0 else float(np.sum(np.abs(values)) / total_absolute),
            "correlation_with_waves": correlations(values, [row["waves_cleared"] for row in episodes]),
            "correlation_with_red_survivors_after_wave1": correlations(values, survivors_1),
            "correlation_with_red_survivors_after_wave2": correlations(values, survivors_2),
        }
    return result


def survivor_strata(episodes):
    output = {"wave1": {}, "wave2": {}}
    for wave in (1, 2):
        rows = []
        for episode in episodes:
            record = next((item for item in episode["wave_records"] if int(item["wave_index"]) == wave), None)
            if record and record.get("wave_cleared"):
                rows.append((episode, record))
        for survivors in range(1, 5):
            selected = [(episode, record) for episode, record in rows if int(record["red_survivors_end"]) == survivors]
            item = {
                "N": len(selected),
                "current_wave_team_return": describe([record["team_return"] for _, record in selected]),
            }
            if wave == 1:
                item["future_wave2_clear_probability"] = None if not selected else float(np.mean([episode["waves_cleared"] >= 2 for episode, _ in selected]))
                item["future_wave3_clear_probability"] = None if not selected else float(np.mean([episode["waves_cleared"] >= 3 for episode, _ in selected]))
            else:
                item["future_wave3_clear_probability"] = None if not selected else float(np.mean([episode["waves_cleared"] >= 3 for episode, _ in selected]))
            output[f"wave{wave}"][str(survivors)] = item
    return output


def weapon_frequency(episodes):
    totals = defaultdict(float); runs = []
    for episode in episodes:
        for key, value in episode["weapon_frequency"].items():
            if key == "unarmed_positive_run_lengths": runs.extend(value)
            else: totals[key] += value
    living = totals["living_samples"]; positive = totals["positive_r4_samples"]
    return {
        **{key: int(value) if key.endswith("samples") else float(value) for key, value in totals.items()},
        "unarmed_fraction_of_living_samples": None if not living else totals["unarmed_samples"] / living,
        "unarmed_positive_r4_fraction_of_living_samples": None if not living else totals["unarmed_positive_r4_samples"] / living,
        "unarmed_fraction_of_positive_r4_samples": None if not positive else totals["unarmed_positive_r4_samples"] / positive,
        "unarmed_fraction_of_positive_r4_magnitude": None if not totals["positive_r4_magnitude"] else totals["unarmed_positive_r4_magnitude"] / totals["positive_r4_magnitude"],
        "unarmed_positive_r4_run_length_steps": describe(runs),
        "miss_then_unarmed_in_window_duration": "unavailable: per-agent attempt/hit identity is not exposed by existing info",
    }


def gae_analysis(environment_config, episodes):
    algorithm = yaml.safe_load((PLAIN_ROOT / "l3_seed5301/algorithm_config.yaml").read_text(encoding="utf-8"))
    gamma = float(algorithm["training"]["gamma"]); gae_lambda = float(algorithm["training"]["gae_lambda"])
    dt = float(environment_config["simulation"]["dt"])
    durations = {wave: [record["duration_steps"] for episode in episodes for record in episode["wave_records"] if int(record["wave_index"]) == wave] for wave in (1, 2, 3)}
    duration_summary = {str(wave): describe(values) for wave, values in durations.items()}
    def horizon(lam):
        rho = gamma * lam
        entries = {}
        for target in (0.5, 0.1, 0.01):
            steps = math.log(target) / math.log(rho)
            entries[str(target)] = {"steps": steps, "seconds": steps * dt}
        return rho, entries
    rho, current = horizon(gae_lambda); candidate_rho, candidate = horizon(0.995)
    residual = {}
    for wave, summary in duration_summary.items():
        residual[wave] = None if summary is None else {key: rho ** summary[key] for key in ("P25", "median", "P75")}
    medians = [value["median"] for value in duration_summary.values() if value]
    label = "CURRENT_GAE_CREDIT_HORIZON_SHORT_RELATIVE_TO_WAVE_DURATION" if medians and all(rho ** duration < 0.01 for duration in medians) else "CURRENT_GAE_CREDIT_HORIZON_NOT_SUPPORTED"
    return {"gamma": gamma, "lambda": gae_lambda, "dt": dt, "rho": rho,
            "trace_horizons": current, "candidate_lambda_0.995": {"rho": candidate_rho, "trace_horizons": candidate},
            "wave_duration_steps": duration_summary, "remaining_trace_weight": residual, "label": label,
            "interpretation_limit": "This supports only a temporal-scale mismatch; it does not prove lambda=0.995 is better."}


def spawn_analysis(environment_config, episodes):
    arena = float(environment_config["arena"]["radius"]); spawn = float(environment_config["persistent_waves"]["spawn_radius"])
    offset = max(abs(float(value)) for value in environment_config["scenario"]["formation_offsets"])
    maximum_radius = math.hypot(spawn, offset); margin = arena - maximum_radius
    desired_speed = float(environment_config["blue_policy"]["desired_speed"])
    transitions = [item for episode in episodes for item in episode["spawn_transitions"]]
    sectors = Counter(item["spawn_candidate_index"] for item in transitions)
    early = [item for item in transitions if item["boundary_exit_within_10s"]]
    no_early = [item for item in transitions if not item["boundary_exit_within_10s"]]
    margin_relation = {
        "early_exit_margin": describe([item["entry_min_boundary_margin"] for item in early]),
        "no_early_exit_margin": describe([item["entry_min_boundary_margin"] for item in no_early]),
        "margin_vs_next_wave_clear": correlations([item["entry_min_boundary_margin"] for item in transitions], [float(item["next_wave_clear"]) for item in transitions]),
        "max_radius_vs_next_wave_clear": correlations([item["entry_red_max_radius"] for item in transitions], [float(item["next_wave_clear"]) for item in transitions]),
    }
    rates = {horizon: None if not transitions else float(np.mean([item[f"boundary_exit_within_{horizon}"] for item in transitions])) for horizon in ("2s", "5s", "10s")}
    if not transitions:
        label = "SPAWN_BOUNDARY_GEOMETRY_INCONCLUSIVE"
    elif not early:
        label = "SPAWN_BOUNDARY_GEOMETRY_NOT_PRIMARY"
    elif no_early and np.mean([item["entry_min_boundary_margin"] for item in early]) < np.mean([item["entry_min_boundary_margin"] for item in no_early]):
        label = "SPAWN_BOUNDARY_GEOMETRY_SENSITIVITY_SUPPORTED"
    else:
        label = "SPAWN_BOUNDARY_GEOMETRY_INCONCLUSIVE"
    return {
        "configuration": {"arena_radius": arena, "spawn_radius": spawn, "maximum_formation_offset": offset,
                          "fresh_blue_maximum_initial_radius": maximum_radius, "minimum_static_boundary_margin": margin,
                          "desired_speed": desired_speed, "straight_line_time_to_boundary_seconds": margin / desired_speed,
                          "heading_time_constant": float(environment_config["action"]["controller"]["heading_time_constant"]),
                          "pitch_time_constant": float(environment_config["action"]["controller"]["pitch_time_constant"])},
        "transition_count": len(transitions), "early_boundary_exit_rates": rates,
        "entry_red_max_radius": describe([item["entry_red_max_radius"] for item in transitions]),
        "entry_min_boundary_margin": describe([item["entry_min_boundary_margin"] for item in transitions]),
        "minimum_spawn_distance": describe([item["minimum_spawn_distance"] for item in transitions]),
        "spawn_sector_counts": dict(sorted(sectors.items())), "unique_spawn_sectors": len(sectors),
        "maximum_sector_fraction": None if not transitions else max(sectors.values()) / len(transitions),
        "margin_relationship": margin_relation, "label": label,
    }


def team_local_analysis(episodes):
    reward_matrix = [
        {"event": "Red attack kill (R1)", "agents": "credited attacker(s), split among simultaneous credited attackers", "team_shared": False},
        {"event": "Red weapon/ground loss (R1)", "agents": "lost Red agent only", "team_shared": False},
        {"event": "Red boundary exit (R2)", "agents": "exiting Red agent only", "team_shared": False},
        {"event": "Approach geometry (R3)", "agents": "living Red agent with qualifying nearest-target geometry", "team_shared": False},
        {"event": "Advantage/threat geometry (R4)", "agents": "living Red agent with qualifying nearest-target geometry", "team_shared": False},
        {"event": "Intermediate wave clear", "agents": "no explicit reward", "team_shared": False},
        {"event": "Survivors at wave clear", "agents": "no explicit reward", "team_shared": False},
    ]
    disparities = [row["individual_return_disparity"] for row in episodes]
    local_variances = [row["individual_return_variance"] for row in episodes]
    step_variances = [row["mean_step_local_reward_variance"] for row in episodes]
    empirical_nonidentical = sum(value > 0 for value in local_variances)
    return {"reward_allocation_matrix": reward_matrix,
            "team_total_return": describe([row["return"] for row in episodes]),
            "individual_return_variance": describe(local_variances),
            "individual_return_disparity": describe(disparities),
            "mean_step_local_reward_variance": describe(step_variances),
            "episodes_with_nonidentical_individual_returns": empirical_nonidentical,
            "label": "MISSION_TEAM_OBJECTIVE_LOCAL_REWARD_MISMATCH_SUPPORTED" if empirical_nonidentical and all(not row["team_shared"] for row in reward_matrix) else "MISSION_TEAM_OBJECTIVE_LOCAL_REWARD_MISMATCH_NOT_SUPPORTED"}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint diagnostic evaluation")
    if any(45_000_000 <= seed <= 45_000_199 for seed in EVALUATION_SEEDS):
        raise RuntimeError("45M future-final seeds are forbidden")
    if JSON_PATH.exists() or TEXT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit report in {OUTPUT_DIR}")
    environment_config = yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))
    aliases = alias_audits(environment_config)
    episodes = []
    checkpoint_paths = {}
    for training_seed in TRAINING_SEEDS:
        trainer, checkpoint_path = load_plain_trainer(training_seed, environment_config)
        checkpoint_paths[str(training_seed)] = str(checkpoint_path.relative_to(ROOT))
        for index, evaluation_seed in enumerate(EVALUATION_SEEDS, 1):
            episode = evaluate_episode(trainer, environment_config, evaluation_seed)
            episode["training_seed"] = training_seed
            episodes.append(episode)
            if index % 10 == 0:
                print(f"[AUDIT] training_seed={training_seed} episodes={index}/50", flush=True)
        del trainer
        torch.cuda.empty_cache()

    per_seed = {}
    for training_seed in TRAINING_SEEDS:
        selected = [row for row in episodes if row["training_seed"] == training_seed]
        with (PLAIN_ROOT / f"l3_seed{training_seed}" / "evaluation_history.csv").open(newline="", encoding="utf-8") as stream:
            existing = next(row for row in csv.DictReader(stream) if int(row["sampled_steps"]) == 3_000_000)
        reproduced = {"average_return": float(np.mean([row["return"] for row in selected])),
                      "average_waves_cleared": float(np.mean([row["waves_cleared"] for row in selected]))}
        existing_values = {"average_return": float(existing["average_return"]),
                           "average_waves_cleared": float(existing["average_waves_cleared"])}
        consistency = all(abs(reproduced[key] - existing_values[key]) <= 1e-6 for key in reproduced)
        if not consistency:
            raise RuntimeError(f"diagnostic evaluation does not reproduce existing exact-3M aggregate for seed {training_seed}")
        per_seed[str(training_seed)] = {
            "reward_mission": reward_analysis(selected), "reward_components": component_analysis(selected),
            "weapon_frequency": weapon_frequency(selected), "team_local": team_local_analysis(selected),
            "existing_aggregate_consistency": {"matched": consistency, "existing": existing_values, "reproduced": reproduced},
        }
    pooled_reward = reward_analysis(episodes)
    pooled_components = component_analysis(episodes)
    strata = survivor_strata(episodes)
    weapon = weapon_frequency(episodes)
    gae = gae_analysis(environment_config, episodes)
    spawn = spawn_analysis(environment_config, episodes)
    team_local = team_local_analysis(episodes)

    populated_wave1 = [item for item in strata["wave1"].values() if item["N"]]
    survivor_future_values = [item["future_wave2_clear_probability"] for item in populated_wave1 if item["future_wave2_clear_probability"] is not None]
    reward_means = [item["current_wave_team_return"]["mean"] for item in populated_wave1 if item["current_wave_team_return"]]
    boundary_value_underrepresented = (
        len(survivor_future_values) >= 2 and max(survivor_future_values) > min(survivor_future_values)
        and len(reward_means) >= 2 and not all(later > earlier for earlier, later in zip(reward_means, reward_means[1:]))
    )
    boundary_label = "BOUNDARY_STATE_VALUE_UNDERREPRESENTED_BY_REWARD" if boundary_value_underrepresented else "BOUNDARY_STATE_REWARD_MISMATCH_NOT_ESTABLISHED"
    if weapon["unarmed_positive_r4_samples"] <= 0 or weapon["unarmed_positive_r4_magnitude"] <= 0:
        weapon_empirical_label = "WEAPON_REWARD_SEMANTIC_MISMATCH_NOT_SUPPORTED"
    elif weapon["unarmed_positive_r4_fraction_of_living_samples"] < 0.01:
        # The semantic mismatch is real, but it occurs in fewer than one in a
        # hundred living step-agent samples.  Keep this distinct from its high
        # conditional share among the much rarer positive-R4 samples.
        weapon_empirical_label = "WEAPON_REWARD_SEMANTIC_ONLY_LOW_FREQUENCY"
    else:
        weapon_empirical_label = "WEAPON_REWARD_SEMANTIC_MISMATCH_EMPIRICALLY_RELEVANT"

    diagnostic_matrix = {
        "A": aliases["fire_state"]["label"], "B": aliases["wave_index"]["label"],
        "C": aliases["remaining_horizon"]["label"], "D": aliases["observation_summary"],
        "E": pooled_reward["label"], "F": boundary_label,
        "G": aliases["unarmed_positive_r4"]["label"], "H": weapon_empirical_label,
        "I": team_local["label"], "J": gae["label"], "K": spawn["label"],
    }
    formulation_evidence = sum((
        diagnostic_matrix["D"] == "ACTOR_OBSERVATION_NOT_MARKOV_FOR_PERSISTENT_MISSION",
        diagnostic_matrix["E"] == "REWARD_MISSION_RANKING_MISMATCH_SUPPORTED",
        diagnostic_matrix["F"] == "BOUNDARY_STATE_VALUE_UNDERREPRESENTED_BY_REWARD",
        diagnostic_matrix["H"] == "WEAPON_REWARD_SEMANTIC_MISMATCH_EMPIRICALLY_RELEVANT",
        diagnostic_matrix["I"] == "MISSION_TEAM_OBJECTIVE_LOCAL_REWARD_MISMATCH_SUPPORTED",
        diagnostic_matrix["J"] == "CURRENT_GAE_CREDIT_HORIZON_SHORT_RELATIVE_TO_WAVE_DURATION",
        diagnostic_matrix["K"] in {"SPAWN_BOUNDARY_GEOMETRY_ARTIFACT_SUPPORTED", "SPAWN_BOUNDARY_GEOMETRY_SENSITIVITY_SUPPORTED"},
    ))
    overall = "CASE_B_TASK_FORMULATION_IS_MAJOR_CO_LIMITATION" if formulation_evidence >= 3 else "CASE_D_INCONCLUSIVE"
    report = {
        "audit": "Persistent-Mission V3 task-definition audit", "read_only": True,
        "training_performed": False, "evaluation_seed_range": [44_000_000, 44_000_049],
        "future_final_45m_touched": False, "checkpoints": checkpoint_paths,
        "environment": {"path": str(ENV_PATH.relative_to(ROOT)), "variant": environment_config["environment_variant"],
                        "observation_dim": 52, "action_dim": 3},
        "episode_count": len(episodes), "aliasing": aliases, "per_training_seed": per_seed,
        "pooled_reward_mission_alignment": pooled_reward, "pooled_reward_components": pooled_components,
        "survivor_strata": strata, "boundary_state_reward_label": boundary_label,
        "weapon_ready_r4": weapon, "weapon_empirical_label": weapon_empirical_label,
        "team_vs_local_reward": team_local, "gae_temporal_horizon": gae,
        "spawn_boundary_geometry": spawn, "diagnostic_matrix": diagnostic_matrix,
        "overall_case": overall, "episode_records": episodes,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "Persistent-Mission V3 task-definition audit", "",
        f"Episodes: {len(episodes)} (3 Plain exact-3M checkpoints x 50 exposed 44M seeds)",
        f"Return/waves Pearson: {pooled_reward['correlation']['pearson']}",
        f"Return/waves Spearman: {pooled_reward['correlation']['spearman']}",
        f"Pairwise inversion rate: {pooled_reward['pooled_pairwise_inversion_rate']}",
        f"Unarmed fraction of positive-R4 samples: {weapon['unarmed_fraction_of_positive_r4_samples']}",
        f"Unarmed fraction of positive-R4 magnitude: {weapon['unarmed_fraction_of_positive_r4_magnitude']}",
        f"Spawn transitions: {spawn['transition_count']}", "", "Diagnostic matrix:",
    ]
    lines.extend(f"{key}. {value}" for key, value in diagnostic_matrix.items())
    lines.extend(("", f"Overall: {overall}", "", "No training; no environment/reward/algorithm modification; no 45M seeds."))
    TEXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"JSON: {JSON_PATH}", flush=True)
    print(f"TXT: {TEXT_PATH}", flush=True)


if __name__ == "__main__":
    main()
