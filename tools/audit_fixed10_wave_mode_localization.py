"""Fixed10 mixed deployment-mode (DDS/SSD) localization audit.

This is a read-only checkpoint diagnostic.  It reuses the completed DDD/SSS
deployment audit and runs only the two mixed modes.  There is intentionally no
training, backward, optimizer, resume, or checkpoint-writing path here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.factory import make_combat_environment
from tools.audit_comprehensive_persistent_wave import entry_features
from tools.audit_fixed10_deployment_modes import (
    CHECKPOINTS,
    ENVIRONMENT_SEEDS,
    POLICY_STREAM_IDS,
    RUN_DIR,
    aggregate,
    csv_write,
    isolated_policy_rng,
    policy_rng_seed,
    read_csv,
    validate_cuda_device_scope,
    validate_environment_seeds,
    validate_protocol,
)
from tools.fixed10_wave_state_drift_common import json_dump, load_checkpoint, sha256, state_dict_sha256

TOOL_VERSION = 1
TRAINING_SEED = 5303
SOURCE_AUDIT = ROOT / "outputs" / "fixed10_deployment_mode_audit"
DEFAULT_FULL_OUTPUT = ROOT / "outputs" / "fixed10_wave_mode_localization"
DEFAULT_SMOKE_OUTPUT = ROOT / "outputs" / "fixed10_wave_mode_localization_smoke"
MIXED_MODES = ("DDS", "SSD")
ALL_MODES = ("DDD", "DDS", "SSD", "SSS")
MODE_DETERMINISTIC = {
    "DDD": {1: True, 2: True, 3: True},
    "DDS": {1: True, 2: True, 3: False},
    "SSD": {1: False, 2: False, 3: True},
    "SSS": {1: False, 2: False, 3: False},
}
ENTRY_FIELDS = (
    "W3_entry_global_step", "W3_entry_remaining_steps", "W3_entry_remaining_horizon_fraction",
    "W3_entry_red_survivors", "W3_entry_red_mean_altitude",
    "W3_entry_red_min_boundary_distance", "W3_entry_nearest_blue_distance",
    "W3_entry_red_pairwise_dispersion", "W3_entry_red_speed_mean",
    "W3_entry_red_heading_mean", "W3_entry_red_pitch_mean", "W3_entry_red_armed_fraction",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed10 DDD/DDS/SSD/SSS wave-mode localization")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-existing", action="store_true")
    return parser


def deterministic_for_wave(mode: str, wave_index: int) -> bool:
    if mode not in MODE_DETERMINISTIC:
        raise ValueError(f"unknown deployment mode: {mode}")
    if int(wave_index) not in (1, 2, 3):
        raise ValueError(f"invalid environment wave_index: {wave_index}")
    return MODE_DETERMINISTIC[mode][int(wave_index)]


def should_capture_w3_entry(action_wave: int, info: dict[str, Any], env: Any) -> bool:
    return bool(
        int(action_wave) == 2
        and info.get("spawned_next_wave")
        and int(info.get("wave_index", -1)) == 3
        and int(env.wave_index) == 3
        and int(env.steps) == int(env._wave_start_step)
    )


def _as_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid historical integer field {key!r}") from exc


def validate_source_audit(source_dir: Path = SOURCE_AUDIT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    source_dir = Path(source_dir)
    status = json.loads((source_dir / "run_status.json").read_text(encoding="utf-8"))
    analysis = json.loads((source_dir / "analysis.json").read_text(encoding="utf-8"))
    inventory = json.loads((source_dir / "inventory.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise RuntimeError("source deployment audit run_status is not COMPLETE")
    if analysis.get("status") != "COMPLETE" or not analysis.get("full_audit_complete"):
        raise RuntimeError("source deployment analysis is not a completed FULL audit")
    if int(analysis.get("stochastic_episode_count", -1)) != 96:
        raise RuntimeError("source deployment audit does not contain 96 SSS episodes")
    for guard_name in ("checkpoint_mutation_guard", "in_memory_actor_mutation_guard"):
        guard = analysis.get(guard_name, {})
        if guard.get("status") != "PASS" or guard.get("before") != guard.get("after"):
            raise RuntimeError(f"source deployment audit {guard_name} did not pass")
    if inventory.get("45m_accessed") is not False:
        raise RuntimeError("source deployment audit 45M access record is not false")
    for role, spec in CHECKPOINTS.items():
        recorded = inventory.get("checkpoints", {}).get(role, {}).get("checkpoint_sha256")
        current = sha256(spec["path"])
        if recorded != current:
            raise RuntimeError(f"source/current checkpoint SHA mismatch for {role}")

    ddd_raw = read_csv(source_dir / "deterministic_reused_episodes.csv")
    sss_raw = read_csv(source_dir / "stochastic_episodes.csv")
    if len(ddd_raw) != 32 or len(sss_raw) != 96:
        raise RuntimeError("source DDD/SSS row count mismatch")
    ddd: list[dict[str, Any]] = []
    sss: list[dict[str, Any]] = []
    expected_ddd = {(role, seed) for role in CHECKPOINTS for seed in ENVIRONMENT_SEEDS}
    expected_sss = {(role, seed, stream) for role in CHECKPOINTS for seed in ENVIRONMENT_SEEDS for stream in POLICY_STREAM_IDS}
    ddd_keys = set()
    sss_keys = set()
    for row in ddd_raw:
        role, env_seed = row.get("checkpoint_role"), _as_int(row, "environment_seed")
        key = (role, env_seed)
        if key in ddd_keys:
            raise RuntimeError(f"duplicate DDD source key: {key}")
        ddd_keys.add(key)
        if row.get("deployment_mode") not in ("deterministic", "DDD"):
            raise RuntimeError("DDD source row has wrong deployment mode")
        converted = dict(row)
        converted.update(_numeric_episode_fields(row))
        converted["checkpoint_role"] = role
        converted["deployment_mode"] = "DDD"
        converted["policy_rng_stream_id"] = None
        converted["policy_rng_seed"] = None
        ddd.append(converted)
    for row in sss_raw:
        role = row.get("checkpoint_role")
        env_seed, stream = _as_int(row, "environment_seed"), _as_int(row, "policy_rng_stream_id")
        key = (role, env_seed, stream)
        if key in sss_keys:
            raise RuntimeError(f"duplicate SSS source key: {key}")
        sss_keys.add(key)
        if row.get("deployment_mode") not in ("stochastic", "SSS"):
            raise RuntimeError("SSS source row has wrong deployment mode")
        expected_policy_seed = policy_rng_seed(env_seed, stream)
        if _as_int(row, "policy_rng_seed") != expected_policy_seed:
            raise RuntimeError(f"SSS policy RNG mapping mismatch: {key}")
        converted = dict(row)
        converted.update(_numeric_episode_fields(row))
        converted.update({"checkpoint_role": role, "deployment_mode": "SSS",
                          "policy_rng_stream_id": stream, "policy_rng_seed": expected_policy_seed})
        sss.append(converted)
    if ddd_keys != expected_ddd or sss_keys != expected_sss:
        raise RuntimeError("source DDD/SSS key coverage mismatch")
    return ddd, sss, {"run_status": status, "analysis": analysis, "inventory": inventory}


def _numeric_episode_fields(row: dict[str, Any]) -> dict[str, Any]:
    waves = _as_int(row, "waves_cleared")
    values = {
        "environment_seed": _as_int(row, "environment_seed"), "waves_cleared": waves,
        "episode_length": _as_int(row, "episode_length"), "red_losses": _as_int(row, "red_losses"),
        "red_boundary_exits": _as_int(row, "red_boundary_exits"),
        "red_ground_losses": _as_int(row, "red_ground_losses"),
        "reached_w2": _as_int(row, "reached_w2"), "reached_w3": _as_int(row, "reached_w3"),
        "w3_cleared": _as_int(row, "w3_cleared"),
    }
    if not 0 <= waves <= 3:
        raise RuntimeError("historical waves_cleared outside 0..3")
    if values["reached_w2"] != int(waves >= 1) or values["reached_w3"] != int(waves >= 2):
        raise RuntimeError("historical reach flags contradict waves_cleared")
    ret = row.get("team_episode_return")
    values["team_episode_return"] = None if ret in (None, "") else float(ret)
    return values


def _entry_row(env: Any, role: str, mode: str, env_seed: int, stream: int, policy_seed_value: int,
               checkpoint_step: int) -> dict[str, Any]:
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    base = entry_features(env, 3, mode, TRAINING_SEED, role, checkpoint_step, env_seed)
    live = [state for state in env.red if state.alive]
    after_cpu = torch.random.get_rng_state()
    after_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    if not torch.equal(before_cpu, after_cpu) or (before_cuda is not None and not torch.equal(before_cuda, after_cuda)):
        raise RuntimeError("W3 entry extraction changed Torch RNG state")
    return {
        "checkpoint_role": role, "deployment_mode": mode, "environment_seed": env_seed,
        "policy_rng_stream_id": stream, "policy_rng_seed": policy_seed_value,
        "W3_entry_reached": 1, "W3_entry_not_reached_reason": None,
        "W3_entry_global_step": int(env.steps),
        "W3_entry_remaining_steps": int(env.max_steps - env.steps),
        "W3_entry_remaining_horizon_fraction": float(base["remaining_horizon"]),
        "W3_entry_red_survivors": int(base["red_survivors"]),
        "W3_entry_red_mean_altitude": float(base["red_altitude_mean"]),
        "W3_entry_red_min_boundary_distance": float(base["distance_to_boundary_min"]),
        "W3_entry_nearest_blue_distance": float(base["nearest_blue_distance"]),
        "W3_entry_red_pairwise_dispersion": float(base["red_pairwise_dispersion"]),
        "W3_entry_red_speed_mean": float(base["red_speed_mean"]),
        "W3_entry_red_heading_mean": float(base["red_heading_mean"]),
        "W3_entry_red_pitch_mean": float(base["red_pitch_mean"]),
        "W3_entry_red_armed_fraction": float(base["red_armed_fraction"]),
        "W3_entry_live_red_state_json": json.dumps([
            {"x": float(s.x), "y": float(s.y), "z": float(s.z), "speed": float(s.v),
             "heading": float(s.psi), "pitch": float(s.theta),
             "armed": bool(env.red_fire_states[index].armed)}
            for index, s in enumerate(env.red) if s.alive
        ], sort_keys=True),
        "captured_post_spawn_before_first_w3_action": True,
    }


def _missing_entry(role: str, mode: str, env_seed: int, stream: int, policy_seed_value: int) -> dict[str, Any]:
    row = {"checkpoint_role": role, "deployment_mode": mode, "environment_seed": env_seed,
           "policy_rng_stream_id": stream, "policy_rng_seed": policy_seed_value,
           "W3_entry_reached": 0, "W3_entry_not_reached_reason": "NOT_REACHED",
           "W3_entry_live_red_state_json": None, "captured_post_spawn_before_first_w3_action": False}
    row.update({key: None for key in ENTRY_FIELDS})
    return row


def run_mixed_episode(trainer: Any, env_config: dict, role: str, mode: str, environment_seed: int,
                      stream_id: int, policy_seed_value: int, checkpoint_meta: dict, device: str) -> tuple[dict, dict]:
    if mode not in MIXED_MODES:
        raise ValueError("run_mixed_episode only accepts DDS or SSD")
    env = make_combat_environment(deepcopy(env_config))
    observation, _ = env.reset(int(environment_seed))
    alive = env.red_alive_mask.copy()
    agent_returns = np.zeros(4, dtype=np.float64)
    entry = None
    sampled_decisions = 0
    deterministic_decisions = 0
    with isolated_policy_rng(policy_seed_value, device), torch.no_grad():
        while True:
            action_wave = int(env.wave_index)
            deterministic = deterministic_for_wave(mode, action_wave)
            actions, _ = trainer.act(observation[None], alive[None], deterministic=deterministic)
            sampled_decisions += int(not deterministic)
            deterministic_decisions += int(deterministic)
            observation, reward, terminated, truncated, info = env.step(actions[0])
            agent_returns += reward
            alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
            if should_capture_w3_entry(action_wave, info, env):
                if entry is not None:
                    raise RuntimeError("W3 entry captured more than once")
                entry = _entry_row(env, role, mode, environment_seed, stream_id, policy_seed_value,
                                   int(checkpoint_meta["sampled_steps"]))
            elif info.get("spawned_next_wave") and int(info.get("wave_index", -1)) == 3:
                raise RuntimeError("illegal W2->W3 transition/capture boundary")
            if terminated or truncated:
                team_return, _ = episode_return_metrics(agent_returns)
                waves = int(info["waves_cleared"])
                if bool(entry is not None) != bool(waves >= 2):
                    raise RuntimeError("W3 entry presence contradicts waves_cleared")
                entry = entry or _missing_entry(role, mode, environment_seed, stream_id, policy_seed_value)
                episode = {
                    "training_seed": TRAINING_SEED, "checkpoint_id": CHECKPOINTS[role]["id"],
                    "checkpoint_role": role, "checkpoint_step": int(checkpoint_meta["sampled_steps"]),
                    "checkpoint_sha256": checkpoint_meta["checkpoint_sha256"], "deployment_mode": mode,
                    "environment_seed": int(environment_seed), "policy_rng_stream_id": int(stream_id),
                    "policy_rng_seed": int(policy_seed_value), "data_source": "NEW_DIAGNOSTIC",
                    "waves_cleared": waves, "reached_w2": int(waves >= 1), "reached_w3": int(waves >= 2),
                    "w3_cleared": int(waves >= 3), "episode_length": int(info["episode_length"]),
                    "team_episode_return": float(team_return), "red_losses": int(info["red_losses"]),
                    "red_boundary_exits": int(info["red_boundary_exits"]),
                    "red_ground_losses": int(info["red_ground_losses"]),
                    "termination_reason": str(info["termination_reason"]),
                    "sampled_actor_decisions": sampled_decisions,
                    "deterministic_actor_decisions": deterministic_decisions,
                    **{key: entry.get(key) for key in ("W3_entry_reached",) + ENTRY_FIELDS},
                }
                return episode, entry


def validate_mixed_coverage(rows: list[dict], full: bool) -> None:
    expected = 192 if full else 4
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} mixed episodes, got {len(rows)}")
    keys = {(row["deployment_mode"], row["checkpoint_role"], int(row["environment_seed"]),
             int(row["policy_rng_stream_id"])) for row in rows}
    if len(keys) != expected:
        raise RuntimeError("duplicate mixed-mode episode key")
    seeds = ENVIRONMENT_SEEDS if full else ENVIRONMENT_SEEDS[:1]
    streams = POLICY_STREAM_IDS if full else POLICY_STREAM_IDS[:1]
    expected_keys = {(mode, role, seed, stream) for mode in MIXED_MODES for role in CHECKPOINTS
                     for seed in seeds for stream in streams}
    if keys != expected_keys:
        raise RuntimeError("mixed-mode episode coverage mismatch")


def validate_prefixes(mixed: list[dict], ddd: list[dict], sss: list[dict]) -> dict[str, Any]:
    ddd_by = {(row["checkpoint_role"], row["environment_seed"]): row for row in ddd}
    sss_by = {(row["checkpoint_role"], row["environment_seed"], row["policy_rng_stream_id"]): row for row in sss}
    mismatches = []
    checked = {"DDD_DDS": 0, "SSD_SSS": 0}
    for row in mixed:
        if row["deployment_mode"] == "DDS":
            ref = ddd_by[(row["checkpoint_role"], row["environment_seed"])]
            pair = "DDD_DDS"
        else:
            ref = sss_by[(row["checkpoint_role"], row["environment_seed"], row["policy_rng_stream_id"])]
            pair = "SSD_SSS"
        checked[pair] += 1
        for key in ("reached_w2", "reached_w3"):
            if int(row[key]) != int(ref[key]):
                mismatches.append({"comparison": pair, "checkpoint_role": row["checkpoint_role"],
                                   "environment_seed": row["environment_seed"],
                                   "policy_rng_stream_id": row["policy_rng_stream_id"], "field": key,
                                   "reference": int(ref[key]), "mixed": int(row[key])})
    if mismatches:
        raise RuntimeError(f"prefix reproduction failed closed: {mismatches[:3]}")
    return {
        "status": "PASS", "checks": checked, "mismatch_count": 0,
        "verified_fields": ["reached_w2", "reached_w3"],
        "verification_level": "OBSERVABLE_PREFIX_OUTCOMES_ONLY",
        "full_state_identity_claimed": False,
        "limitation": "historical DDD/SSS CSVs do not contain complete prefix trajectories or W3 entry vectors",
    }


def build_action_pairs(mixed: list[dict], ddd: list[dict], sss: list[dict]) -> list[dict[str, Any]]:
    ddd_by = {(r["checkpoint_role"], r["environment_seed"]): r for r in ddd}
    sss_by = {(r["checkpoint_role"], r["environment_seed"], r["policy_rng_stream_id"]): r for r in sss}
    output = []
    for row in mixed:
        if row["deployment_mode"] == "DDS":
            reference_mode, reference = "DDD", ddd_by[(row["checkpoint_role"], row["environment_seed"])]
            comparison = "DDD_vs_DDS"
            reference_reused = True
        else:
            reference_mode, reference = "SSS", sss_by[(row["checkpoint_role"], row["environment_seed"], row["policy_rng_stream_id"])]
            comparison = "SSD_vs_SSS"
            reference_reused = False
        a = int(reference["w3_cleared"]); b = int(row["w3_cleared"])
        label = "BOTH_SUCCESS" if a and b else f"{reference_mode}_ONLY" if a else f"{row['deployment_mode']}_ONLY" if b else "BOTH_FAIL"
        output.append({
            "comparison": comparison, "checkpoint_role": row["checkpoint_role"],
            "environment_seed": row["environment_seed"], "policy_rng_stream_id": row["policy_rng_stream_id"],
            "reference_mode": reference_mode, "mixed_mode": row["deployment_mode"],
            "reference_w3_cleared": a, "mixed_w3_cleared": b, "w3_paired_outcome": label,
            "reference_waves_cleared": int(reference["waves_cleared"]),
            "mixed_waves_cleared": int(row["waves_cleared"]),
            "reference_is_reused_per_stream_not_independent": reference_reused,
        })
    return output


def _summary_by_mode_role(ddd: list[dict], mixed: list[dict], sss: list[dict]) -> dict[str, Any]:
    all_rows = {"DDD": ddd, "DDS": [r for r in mixed if r["deployment_mode"] == "DDS"],
                "SSD": [r for r in mixed if r["deployment_mode"] == "SSD"], "SSS": sss}
    return {mode: {role: aggregate([r for r in rows if r["checkpoint_role"] == role])
                   for role in CHECKPOINTS} for mode, rows in all_rows.items()}


def _stream_rows(mixed: list[dict], sss: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for mode, source in (("DDS", mixed), ("SSD", mixed), ("SSS", sss)):
        for role in CHECKPOINTS:
            for stream in POLICY_STREAM_IDS:
                group = [r for r in source if r["deployment_mode"] == mode and r["checkpoint_role"] == role
                         and r["policy_rng_stream_id"] == stream]
                if group:
                    rows.append({"deployment_mode": mode, "checkpoint_role": role,
                                 "policy_rng_stream_id": stream, **aggregate(group),
                                 "independent_unit": "environment_seed", "training_seed_replication_n": 1})
    return rows


def _pair_summary(pairs: list[dict], summary: dict) -> dict[str, Any]:
    result = {}
    for comparison, a_mode, b_mode in (("DDD_vs_DDS", "DDD", "DDS"), ("SSD_vs_SSS", "SSD", "SSS")):
        result[comparison] = {}
        for role in CHECKPOINTS:
            group = [r for r in pairs if r["comparison"] == comparison and r["checkpoint_role"] == role]
            labels = [r["w3_paired_outcome"] for r in group]
            a, b = summary[a_mode][role], summary[b_mode][role]
            result[comparison][role] = {
                "pair_rows": len(group), "independent_environment_scenarios": len({r["environment_seed"] for r in group}),
                "policy_streams_per_scenario": len({r["policy_rng_stream_id"] for r in group}),
                "w3_pair_counts": {label: labels.count(label) for label in sorted(set(labels))},
                f"{a_mode}_W3": a["W3"], f"{b_mode}_W3": b["W3"],
                f"{b_mode}_minus_{a_mode}_W3": b["W3"] - a["W3"],
                f"{a_mode}_Q3": a["Q3"], f"{b_mode}_Q3": b["Q3"],
                f"{b_mode}_minus_{a_mode}_Q3": None if a["Q3"] is None or b["Q3"] is None else b["Q3"] - a["Q3"],
                f"{b_mode}_minus_{a_mode}_AverageWaves": b["AverageWaves"] - a["AverageWaves"],
                "DDD_reference_rows_are_not_independent_stream_repeats": comparison == "DDD_vs_DDS",
            }
    return result


def _entry_summary(entries: list[dict]) -> dict[str, Any]:
    result = {"conditioning": "ACTUAL_W3_ENTRY_ONLY", "selection_bias_warning": True, "groups": {}}
    for mode in MIXED_MODES:
        result["groups"][mode] = {}
        for role in CHECKPOINTS:
            all_group = [r for r in entries if r["deployment_mode"] == mode and r["checkpoint_role"] == role]
            reached = [r for r in all_group if int(r["W3_entry_reached"]) == 1]
            item = {"episodes": len(all_group), "W3_entry_count": len(reached),
                    "W3_entry_rate": len(reached) / len(all_group) if all_group else None}
            for field in ENTRY_FIELDS:
                values = [float(r[field]) for r in reached if r.get(field) is not None]
                item[f"mean_{field}"] = float(statistics.mean(values)) if values else None
            result["groups"][mode][role] = item
    return result


def _descriptive_prefix_comparison(summary: dict, entry_summary: dict) -> dict[str, Any]:
    result = {"causal_claim": False, "comparisons": {}, "DDS_vs_SSD_conditional_entry_state": {}}
    for name, a_mode, b_mode in (("DDD_vs_SSD", "DDD", "SSD"), ("DDS_vs_SSS", "DDS", "SSS")):
        result["comparisons"][name] = {}
        for role in CHECKPOINTS:
            a, b = summary[a_mode][role], summary[b_mode][role]
            result["comparisons"][name][role] = {
                "delta_W3_reach": b["W2"] - a["W2"], "delta_W3": b["W3"] - a["W3"],
                "delta_Q3": None if a["Q3"] is None or b["Q3"] is None else b["Q3"] - a["Q3"],
                "delta_AverageWaves": b["AverageWaves"] - a["AverageWaves"],
                "entry_state_comparison_available": a_mode in MIXED_MODES and b_mode in MIXED_MODES,
                "interpretation": "DESCRIPTIVE_ONLY; W3 entry and RNG-consumption paths are not controlled",
            }
    # DDS and SSD have different prefixes, but their W3 action has not happened at
    # the exact post-spawn entry capture point.  This is therefore the available
    # descriptive prefix-state contrast.  It remains selected on reaching W3.
    for role in CHECKPOINTS:
        dds = entry_summary["groups"]["DDS"][role]
        ssd = entry_summary["groups"]["SSD"][role]
        deltas = {}
        for field in ENTRY_FIELDS:
            a, b = dds[f"mean_{field}"], ssd[f"mean_{field}"]
            deltas[f"SSD_minus_DDS_mean_{field}"] = None if a is None or b is None else b - a
        result["DDS_vs_SSD_conditional_entry_state"][role] = {
            "DDS_entry_count": dds["W3_entry_count"], "SSD_entry_count": ssd["W3_entry_count"],
            "SSD_minus_DDS_entry_rate": None if dds["W3_entry_rate"] is None or ssd["W3_entry_rate"] is None
            else ssd["W3_entry_rate"] - dds["W3_entry_rate"],
            **deltas,
            "interpretation": "DESCRIPTIVE_ONLY; conditional on actual W3 entry and subject to selection bias",
        }
    return result


def _research_questions(summary: dict, prefix_comparison: dict, label: str, full: bool) -> dict[str, Any]:
    def difference(a: str, b: str, role: str, metric: str) -> float | None:
        x, y = summary[a][role][metric], summary[b][role][metric]
        return None if x is None or y is None else y - x

    if not full:
        conclusion = "SMOKE_ONLY_NO_FORMAL_RESEARCH_CONCLUSION"
    else:
        conclusion = label
    return {
        "interpretation_status": conclusion,
        "1_peak_DDS_minus_DDD_W3": difference("DDD", "DDS", "Peak", "W3"),
        "2_final_DDS_minus_DDD_W3": difference("DDD", "DDS", "Final", "W3"),
        "3_peak_SSS_minus_SSD_W3": difference("SSD", "SSS", "Peak", "W3"),
        "4_final_SSS_minus_SSD_W3": difference("SSD", "SSS", "Final", "W3"),
        "5_peak_deterministic_W3_advantage_retained_in_DDS": None if not full else {
            "DDD_Q3": summary["DDD"]["Peak"]["Q3"], "DDS_Q3": summary["DDS"]["Peak"]["Q3"],
            "DDS_minus_DDD_Q3": difference("DDD", "DDS", "Peak", "Q3"),
        },
        "6_random_prefix_entry_state": prefix_comparison["DDS_vs_SSD_conditional_entry_state"],
        "7_prefix_and_W3_action_effects": conclusion,
        "8_peak_and_final_both_reported": full,
        "9_algorithm_design_support": "NO_FORMAL_RECOMMENDATION_FROM_SMOKE" if not full else (
            "COMBINATION" if label == "BOTH_EFFECTS_OBSERVED" else
            "ACTION_DISTRIBUTION" if label == "W3_ACTION_MODE_EFFECT_OBSERVED" else
            "ENTRY_STATE_QUALITY" if label == "PREFIX_MODE_EFFECT_OBSERVED" else
            "MIXED_OR_INSUFFICIENT"
        ),
        "causal_proof": False,
    }


def _label(summary: dict, full: bool) -> str:
    if not full:
        return "MIXED_OR_INSUFFICIENT_EVIDENCE"
    action = any(abs(summary[b][role]["W3"] - summary[a][role]["W3"]) >= 0.05
                 for a, b in (("DDD", "DDS"), ("SSD", "SSS")) for role in CHECKPOINTS)
    prefix = any(abs(summary[b][role]["W2"] - summary[a][role]["W2"]) >= 0.05
                 for a, b in (("DDD", "SSD"), ("DDS", "SSS")) for role in CHECKPOINTS)
    if action and prefix:
        return "BOTH_EFFECTS_OBSERVED"
    if action:
        return "W3_ACTION_MODE_EFFECT_OBSERVED"
    if prefix:
        return "PREFIX_MODE_EFFECT_OBSERVED"
    return "MIXED_OR_INSUFFICIENT_EVIDENCE"


def run_audit(device: str, output_dir: Path, smoke: bool) -> dict[str, Any]:
    validate_cuda_device_scope(device)
    validate_environment_seeds(ENVIRONMENT_SEEDS)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if smoke and output_dir.resolve() == DEFAULT_FULL_OUTPUT.resolve():
        raise RuntimeError("SMOKE cannot write to FULL output directory")
    output_dir.mkdir(parents=True)
    json_dump(output_dir / "run_status.json", {"status": "IN_PROGRESS", "smoke": smoke})
    checkpoint_before = {role: sha256(spec["path"]) for role, spec in CHECKPOINTS.items()}
    actor_before: dict[str, str] = {}
    actor_after: dict[str, str] = {}
    try:
        env_config, algorithm_config, checkpoint_meta = validate_protocol()
        protocol = checkpoint_meta.pop("protocol")
        ddd_all, sss_all, source = validate_source_audit()
        env_seeds = ENVIRONMENT_SEEDS[:1] if smoke else ENVIRONMENT_SEEDS
        streams = POLICY_STREAM_IDS[:1] if smoke else POLICY_STREAM_IDS
        ddd = [r for r in ddd_all if r["environment_seed"] in env_seeds]
        sss = [r for r in sss_all if r["environment_seed"] in env_seeds and r["policy_rng_stream_id"] in streams]
        mixed: list[dict] = []
        entries: list[dict] = []
        total = len(MIXED_MODES) * len(CHECKPOINTS) * len(env_seeds) * len(streams)
        completed = 0
        for role, spec in CHECKPOINTS.items():
            checkpoint = load_checkpoint(spec["path"])
            trainer = build_modular_mappo_trainer(algorithm_config, device=device, hidden_dim=256,
                                                  total_sampled_steps=1_805_280)
            trainer.actor.load_state_dict(checkpoint["actor"], strict=True)
            trainer.actor.eval(); trainer.critic.eval()
            actor_before[role] = state_dict_sha256(trainer.actor.state_dict())
            for mode in MIXED_MODES:
                for env_seed in env_seeds:
                    for stream in streams:
                        seed = policy_rng_seed(env_seed, stream)
                        try:
                            episode, entry = run_mixed_episode(trainer, env_config, role, mode, env_seed,
                                                               stream, seed, checkpoint_meta[role], device)
                        except Exception:
                            print(f"[AUDIT][FAILED] mode={mode} role={role} completed={completed}/{total} "
                                  f"seed={env_seed} stream={stream}", flush=True)
                            raise
                        mixed.append(episode); entries.append(entry); completed += 1
                        print(f"[AUDIT] mode={mode} role={role} completed={completed}/{total} "
                              f"seed={env_seed} stream={stream}", flush=True)
            actor_after[role] = state_dict_sha256(trainer.actor.state_dict())
            if actor_before[role] != actor_after[role]:
                raise RuntimeError(f"{role} Actor mutated")
            del trainer
            torch.cuda.empty_cache()

        full = not smoke
        validate_mixed_coverage(mixed, full)
        prefix = validate_prefixes(mixed, ddd, sss)
        pairs = build_action_pairs(mixed, ddd, sss)
        summary = _summary_by_mode_role(ddd, mixed, sss)
        stream_rows = _stream_rows(mixed, sss)
        pair_summary = _pair_summary(pairs, summary)
        entry_summary = _entry_summary(entries)
        prefix_comparison = _descriptive_prefix_comparison(summary, entry_summary)
        checkpoint_after = {role: sha256(spec["path"]) for role, spec in CHECKPOINTS.items()}
        if checkpoint_before != checkpoint_after:
            raise RuntimeError("checkpoint file mutation guard failed")
        if actor_before != actor_after:
            raise RuntimeError("in-memory Actor mutation guard failed")
        label = _label(summary, full)
        research_questions = _research_questions(summary, prefix_comparison, label, full)
        tool_path = Path(__file__)
        inventory = {
            "tool_version": TOOL_VERSION, "tool_source_sha256": sha256(tool_path),
            "mode": "FULL" if full else "SMOKE", "device": device,
            "cuda_device": torch.cuda.get_device_name(0), "visible_cuda_device_count": torch.cuda.device_count(),
            "training_seed": TRAINING_SEED, "checkpoints": checkpoint_meta,
            "canonical_protocol_hashes": protocol, "environment_seeds": list(env_seeds),
            "policy_rng_mapping": [{"environment_seed": seed, "stream_id": stream,
                                    "policy_seed": policy_rng_seed(seed, stream)}
                                   for seed in env_seeds for stream in streams],
            "deployment_modes": MODE_DETERMINISTIC,
            "historical_sources": {
                "DDD": str((SOURCE_AUDIT / "deterministic_reused_episodes.csv").relative_to(ROOT)),
                "SSS": str((SOURCE_AUDIT / "stochastic_episodes.csv").relative_to(ROOT)),
                "source_status": source["run_status"]["status"],
                "source_checkpoint_sha_verified": True,
            },
            "new_episode_count": len(mixed), "actual_W3_entry_count": sum(int(r["W3_entry_reached"]) for r in entries),
            "formal_training_performed": False, "backward_calls": 0, "optimizer_steps": 0,
            "checkpoint_files_protected": True, "in_memory_actor_parameters_protected": True,
            "45m_accessed": False,
        }
        analysis = {
            "status": "COMPLETE" if full else "SMOKE_COMPLETE_NOT_FULL",
            "full_audit_complete": full, "new_mixed_episode_count": len(mixed),
            "descriptive_label": label, "prefix_validation": prefix,
            "research_questions": research_questions,
            "checkpoint_mutation_guard": {"status": "PASS", "before": checkpoint_before, "after": checkpoint_after},
            "in_memory_actor_mutation_guard": {"status": "PASS", "before": actor_before, "after": actor_after},
            "training_performed": False, "backward_calls": 0, "optimizer_steps": 0, "45m_accessed": False,
            "training_seed_replication_count": 1,
            "independent_environment_scenarios": len(env_seeds),
            "policy_streams_are_within_environment_repeats": True,
        }
        limitations = {
            "single_training_seed": True, "peak_selection_bias": True,
            "historical_prefix_state_limitation": prefix["limitation"],
            "conditional_entry_means_have_selection_bias": True,
            "policy_streams_are_not_independent_training_seeds": True,
            "DDD_repeated_reference_is_not_an_independent_observation": True,
            "no_root_cause_proof": True,
        }
        csv_write(output_dir / "mixed_mode_episodes.csv", mixed)
        csv_write(output_dir / "w3_entry_states.csv", entries)
        json_dump(output_dir / "four_mode_summary.json", summary)
        if stream_rows:
            csv_write(output_dir / "stream_summary.csv", stream_rows)
        csv_write(output_dir / "w3_action_mode_pairs.csv", pairs)
        json_dump(output_dir / "w3_action_mode_pair_summary.json", pair_summary)
        json_dump(output_dir / "prefix_mode_comparison.json", {"prefix_validation": prefix, **prefix_comparison})
        json_dump(output_dir / "entry_quality_summary.json", entry_summary)
        json_dump(output_dir / "inventory.json", inventory)
        json_dump(output_dir / "analysis.json", analysis)
        json_dump(output_dir / "limitations.json", limitations)
        (output_dir / "decision_support.txt").write_text(
            f"STATUS={analysis['status']}\nLABEL={label}\nNEW_EPISODES={len(mixed)}\n"
            f"PEAK_DDS_MINUS_DDD_W3={research_questions['1_peak_DDS_minus_DDD_W3']}\n"
            f"FINAL_DDS_MINUS_DDD_W3={research_questions['2_final_DDS_minus_DDD_W3']}\n"
            f"PEAK_SSS_MINUS_SSD_W3={research_questions['3_peak_SSS_minus_SSD_W3']}\n"
            f"FINAL_SSS_MINUS_SSD_W3={research_questions['4_final_SSS_minus_SSD_W3']}\n"
            f"DESIGN_SUPPORT={research_questions['9_algorithm_design_support']}\n"
            "CAUSAL_PROOF=NO\nTRAINING=NO\n45M_ACCESSED=NO\n", encoding="utf-8")
        json_dump(output_dir / "run_status.json", {"status": analysis["status"], "smoke": smoke})
        print_report(output_dir)
        return analysis
    except Exception as exc:
        json_dump(output_dir / "run_status.json", {"status": "FAILED", "smoke": smoke,
                                                    "error_type": type(exc).__name__, "error": str(exc)})
        raise


def print_report(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "four_mode_summary.json").read_text(encoding="utf-8"))
    print("=" * 96)
    print("FIXED10 WAVE MODE LOCALIZATION (DDD/DDS/SSD/SSS)")
    print(f"status={analysis['status']} label={analysis['descriptive_label']}")
    for role in ("Peak", "Final"):
        for mode in ALL_MODES:
            row = summary[mode][role]
            print(f"{role:5s} {mode}: W1={row['W1']:.4f} W2={row['W2']:.4f} W3={row['W3']:.4f} "
                  f"AW={row['AverageWaves']:.4f} Q3={row['Q3']}")
    print(f"output={output_dir}")
    print("=" * 96)


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir or (DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_FULL_OUTPUT)
    if args.print_existing:
        print_report(output)
        return
    run_audit(args.device, output, args.smoke)


if __name__ == "__main__":
    main()
