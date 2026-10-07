"""Collect the bounded 5-policy x 16-seed Fixed10 natural-entry diagnostic."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.persistent_env import PersistentWaveCombatEnv
from tools.audit_comprehensive_persistent_wave import entry_features
try:
    from tools.fixed10_wave_state_drift_common import (
        AUDIT_DIR, DIAGNOSTIC_SEEDS, ROOT, assert_safe_diagnostic_seeds,
        checkpoint_mutation_guard, csv_write, json_dump, load_checkpoint, run_dir,
        stable_snapshot_identity,
    )
except ModuleNotFoundError:  # direct ``python tools/...py`` execution
    from fixed10_wave_state_drift_common import (
        AUDIT_DIR, DIAGNOSTIC_SEEDS, ROOT, assert_safe_diagnostic_seeds,
        checkpoint_mutation_guard, csv_write, json_dump, load_checkpoint, run_dir,
        stable_snapshot_identity,
    )

FEATURES = (
    "remaining_horizon", "red_survivors", "red_altitude_mean", "red_altitude_min", "red_speed_mean",
    "distance_to_boundary_mean", "distance_to_boundary_min", "red_radial_velocity_mean", "red_pairwise_dispersion",
    "nearest_blue_distance", "mean_nearest_blue_distance", "relative_bearing_abs_mean",
    "AA_abs_mean", "ATA_abs_mean",
)

# Exploratory descriptive thresholds declared before collection/analysis.
NATURAL_SHIFT_THRESHOLDS = {
    "reach_probability_absolute": 0.20,
    "red_survivors_mean_absolute": 0.50,
    "remaining_horizon_mean_absolute": 0.05,
    "distance_to_boundary_min_mean_absolute_metres": 500.0,
    "nearest_blue_distance_mean_absolute_metres": 500.0,
    "red_altitude_mean_absolute_metres": 300.0,
    "red_speed_mean_absolute_metres_per_second": 20.0,
}
MIN_CONDITIONAL_ENTRIES = 4


def collection_specs() -> list[dict[str, Any]]:
    return [
        {"id": "Treatment10_seed5303_peak", "arm": "10", "role": "peak", "path": run_dir("10", 5303) / "best_eval.pt", "step": 1_701_888},
        {"id": "Treatment10_seed5303_1800192", "arm": "10", "role": "checkpoint_1800192", "path": run_dir("10", 5303) / "checkpoint_1800192.pt", "step": 1_800_192},
        {"id": "Treatment10_seed5303_final", "arm": "10", "role": "final", "path": run_dir("10", 5303) / "final.pt", "step": 1_805_280},
        {"id": "Control05_seed5303_1800192", "arm": "05", "role": "checkpoint_1800192", "path": run_dir("05", 5303) / "checkpoint_1800192.pt", "step": 1_800_192},
        {"id": "Control05_seed5303_final", "arm": "05", "role": "final", "path": run_dir("05", 5303) / "final.pt", "step": 1_805_280},
    ]


def summarize(episodes: list[dict], states: list[dict]) -> dict:
    policies = {}
    for spec in collection_specs():
        eps = [row for row in episodes if row["policy_id"] == spec["id"]]
        item: dict[str, Any] = {"episodes": len(eps)}
        for wave in (2, 3):
            entries = [row for row in states if row["policy_id"] == spec["id"] and row["entry_wave"] == wave]
            item[f"reach_w{wave}_count"] = len(entries)
            item[f"reach_w{wave}_probability"] = len(entries) / len(eps) if eps else None
            strata = {}
            for survivors in (1, 2, 3, 4):
                group = [row for row in entries if row["red_survivors"] == survivors]
                values = {key: (float(np.mean([row[key] for row in group])) if group else None) for key in FEATURES if key != "red_survivors"}
                strata[str(survivors)] = {"n": len(group), "descriptive_means": values,
                                          "inference": "DESCRIPTIVE_ONLY" if len(group) else "NO_SAMPLES"}
            item[f"w{wave}_conditional_entry"] = {
                "n": len(entries), "survivor_strata": strata,
                "overall_descriptive_means": {
                    key: (float(np.mean([row[key] for row in entries])) if entries else None)
                    for key in FEATURES
                },
                "selection_bias_warning": "conditioning on reaching W3 introduces selection bias" if wave == 3 else None,
            }
        policies[spec["id"]] = item
    peak = policies["Treatment10_seed5303_peak"]
    final = policies["Treatment10_seed5303_final"]
    shifts = {}
    any_sufficient = False
    observed = False
    feature_thresholds = {
        "red_survivors": NATURAL_SHIFT_THRESHOLDS["red_survivors_mean_absolute"],
        "remaining_horizon": NATURAL_SHIFT_THRESHOLDS["remaining_horizon_mean_absolute"],
        "distance_to_boundary_min": NATURAL_SHIFT_THRESHOLDS["distance_to_boundary_min_mean_absolute_metres"],
        "nearest_blue_distance": NATURAL_SHIFT_THRESHOLDS["nearest_blue_distance_mean_absolute_metres"],
        "red_altitude_mean": NATURAL_SHIFT_THRESHOLDS["red_altitude_mean_absolute_metres"],
        "red_speed_mean": NATURAL_SHIFT_THRESHOLDS["red_speed_mean_absolute_metres_per_second"],
    }
    for wave in (2, 3):
        peak_entry = peak[f"w{wave}_conditional_entry"]
        final_entry = final[f"w{wave}_conditional_entry"]
        sufficient = peak_entry["n"] >= MIN_CONDITIONAL_ENTRIES and final_entry["n"] >= MIN_CONDITIONAL_ENTRIES
        any_sufficient |= sufficient
        reach_delta = final[f"reach_w{wave}_probability"] - peak[f"reach_w{wave}_probability"]
        deltas = {}
        for feature, threshold in feature_thresholds.items():
            a = peak_entry["overall_descriptive_means"][feature]
            b = final_entry["overall_descriptive_means"][feature]
            value = None if a is None or b is None else float(b - a)
            deltas[feature] = {"final_minus_peak": value, "exploratory_threshold": threshold,
                               "exceeds_threshold": bool(sufficient and value is not None and abs(value) >= threshold)}
        wave_observed = sufficient and (
            abs(reach_delta) >= NATURAL_SHIFT_THRESHOLDS["reach_probability_absolute"]
            or any(item["exceeds_threshold"] for item in deltas.values())
        )
        observed |= wave_observed
        shifts[f"W{wave}"] = {"coverage": "SUFFICIENT" if sufficient else "INSUFFICIENT_COVERAGE",
                               "reach_probability_final_minus_peak": reach_delta,
                               "feature_deltas": deltas, "shift_observed": wave_observed}
    shift_status = "OBSERVED" if observed else "SMALL_OR_MIXED" if any_sufficient else "INSUFFICIENT_COVERAGE"
    return {
        "status": "COMPLETE",
        "formal_evaluation": False,
        "training": False,
        "training_seed_replication_count": 1,
        "diagnostic_environment_repetitions_per_policy": len(DIAGNOSTIC_SEEDS),
        "episode_count": len(episodes),
        "policies": policies,
        "natural_entry_distribution_shift": shift_status,
        "peak_vs_final_shifts": shifts,
        "exploratory_descriptive_thresholds": NATURAL_SHIFT_THRESHOLDS,
        "minimum_conditional_entries_per_policy": MIN_CONDITIONAL_ENTRIES,
        "thresholds_are_not_significance_tests": True,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint diagnostic rollouts")
    assert_safe_diagnostic_seeds(DIAGNOSTIC_SEEDS)
    if not AUDIT_DIR.is_dir():
        raise FileNotFoundError("run the fixed-bank audit first")
    targets = [AUDIT_DIR / name for name in ("natural_entry_episodes.csv", "natural_entry_states.csv", "entry_snapshot_bank.pt")]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite existing natural-entry diagnostic data")
    status = json.loads((AUDIT_DIR / "natural_entry_distribution_summary.json").read_text(encoding="utf-8"))
    if status.get("status") != "NOT_RUN":
        raise RuntimeError("natural-entry stage is not in NOT_RUN state")
    specs = collection_specs()
    paths = [spec["path"] for spec in specs]
    before = checkpoint_mutation_guard(paths)
    episodes: list[dict] = []
    states: list[dict] = []
    snapshots: list[dict] = []
    for spec in specs:
        run = run_dir(spec["arm"], 5303)
        config = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
        env_config = yaml.safe_load((run / "runtime_env_config.yaml").read_text(encoding="utf-8"))
        checkpoint = load_checkpoint(spec["path"])
        actual_step = int(checkpoint["sampled_steps"])
        if actual_step != spec["step"]:
            raise RuntimeError(f"{spec['id']} checkpoint step mismatch")
        trainer = build_modular_mappo_trainer(config, device="cuda:0", hidden_dim=256, total_sampled_steps=1_805_280)
        trainer.actor.load_state_dict(checkpoint["actor"], strict=True)
        trainer.actor.eval()
        for episode_seed in DIAGNOSTIC_SEEDS:
            env = PersistentWaveCombatEnv(deepcopy(env_config))
            observation, info = env.reset(int(episode_seed))
            alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
            done = False
            entry_rows = []
            while not done:
                with torch.no_grad():
                    actions, _ = trainer.act(observation[None], alive[None], deterministic=True)
                observation, reward, terminated, truncated, info = env.step(actions[0])
                alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
                done = bool(terminated or truncated)
                if info.get("spawned_next_wave"):
                    wave = int(info["wave_index"])
                    if wave not in (2, 3) or env.steps != env._wave_start_step:
                        raise RuntimeError("entry was not captured at the exact legal post-spawn moment")
                    snapshot = env.export_curriculum_state()
                    verifier = PersistentWaveCombatEnv(deepcopy(env_config))
                    verifier.reset(88_399_999)
                    restored = verifier.restore_curriculum_state(snapshot)
                    if not stable_snapshot_identity(snapshot, restored, verifier):
                        raise RuntimeError("snapshot restore roundtrip mismatch")
                    if not np.array_equal(observation, restored["observation"]):
                        raise RuntimeError("returned post-spawn observation differs from exported snapshot")
                    row = entry_features(env, wave, spec["id"], 5303, spec["role"], actual_step, int(episode_seed))
                    live_red = [state for state in env.red if state.alive]
                    row["red_altitude_min"] = float(min(state.altitude for state in live_red))
                    row["distance_to_boundary_mean"] = float(np.mean([
                        env.arena_radius - np.hypot(state.x, state.y) for state in live_red
                    ]))
                    row.update({"policy_id": spec["id"], "entry_collected_before_next_actor_action": True,
                                "snapshot_restore_identity": True})
                    states.append(row)
                    entry_rows.append(row)
                    snapshots.append({"policy_id": spec["id"], "checkpoint_role": spec["role"],
                                      "checkpoint_step": actual_step, "training_seed": 5303,
                                      "episode_seed": int(episode_seed), "entry_wave": wave,
                                      "features": deepcopy(row), "snapshot": snapshot})
            for row in entry_rows:
                row["next_wave_cleared"] = int(int(info.get("waves_cleared", 0)) >= int(row["entry_wave"]))
            episodes.append({
                "policy_id": spec["id"], "checkpoint_role": spec["role"], "checkpoint_step": actual_step,
                "training_seed": 5303, "diagnostic_episode_seed": int(episode_seed),
                "waves_cleared": int(info.get("waves_cleared", 0)), "episode_length": int(info.get("episode_length", env.steps)),
                "red_losses": int(info.get("red_losses", 0)), "red_boundary_exits": int(info.get("red_boundary_exits", 0)),
                "red_ground_losses": int(info.get("red_ground_losses", 0)),
                "reached_w2": int(any(row["entry_wave"] == 2 for row in entry_rows)),
                "reached_w3": int(any(row["entry_wave"] == 3 for row in entry_rows)),
            })
        del trainer
        torch.cuda.empty_cache()
    if len(episodes) != 80:
        raise RuntimeError(f"expected exactly 80 diagnostic episodes, got {len(episodes)}")
    csv_write(AUDIT_DIR / "natural_entry_episodes.csv", episodes)
    csv_write(AUDIT_DIR / "natural_entry_states.csv", states)
    torch.save(snapshots, AUDIT_DIR / "entry_snapshot_bank.pt")
    summary = summarize(episodes, states)
    json_dump(AUDIT_DIR / "natural_entry_distribution_summary.json", summary)
    manifest_rows = [{key: value for key, value in item.items() if key != "snapshot"} for item in snapshots]
    manifest = {"status": "COMPLETE", "snapshot_count": len(snapshots),
                "snapshot_count_w2": sum(item["entry_wave"] == 2 for item in snapshots),
                "snapshot_count_w3": sum(item["entry_wave"] == 3 for item in snapshots),
                "items": manifest_rows}
    json_dump(AUDIT_DIR / "entry_snapshot_manifest.json", manifest)
    after = checkpoint_mutation_guard(paths)
    if before != after:
        raise RuntimeError("input checkpoint mutation guard failed")
    analysis_path = AUDIT_DIR / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["stages"]["natural_entry"] = "COMPLETE"
    analysis["natural_entry_distribution_shift"] = summary["natural_entry_distribution_shift"]
    analysis["natural_entry_mutation_guard"] = {"status": "PASS", "before": before, "after": after}
    json_dump(analysis_path, analysis)
    print(json.dumps({"status": "COMPLETE", "episodes": len(episodes), "snapshots": len(snapshots),
                      "w2": manifest["snapshot_count_w2"], "w3": manifest["snapshot_count_w3"]}, indent=2))


if __name__ == "__main__":
    main()
