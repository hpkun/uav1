#!/usr/bin/env python3
"""Evaluation-only MARC sensitivity to active Blue force size by wave."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment

RUN = ROOT / "outputs/dev_marc_mappo_v1_1m_seed5301"
CHECKPOINT = RUN / "final.pt"
OUTPUT = ROOT / "outputs/marc_blue_wave_size_sensitivity"
SEEDS = tuple(range(44_000_000, 44_000_050))
VARIANTS = {
    "444": ROOT / "configs/persistent_wave_v2_environment.yaml",
    "443": ROOT / "configs/persistent_wave_v2_blue443_environment.yaml",
    "433": ROOT / "configs/persistent_wave_v2_blue433_environment.yaml",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blue_counts(config: dict) -> list[int]:
    waves = config["persistent_waves"]
    return list(map(int, waves.get("blue_units_per_wave",
                                   [config["scenario"]["team_size"]] * int(waves["total_waves"]))))


def validate_configs(configs: dict[str, dict]) -> None:
    base = deepcopy(configs["444"])
    if "blue_units_per_wave" in base["persistent_waves"]:
        raise RuntimeError("legacy 444 config must retain implicit default force size")
    expected = {"444": [4, 4, 4], "443": [4, 4, 3], "433": [4, 3, 3]}
    for name, config in configs.items():
        identity = (config.get("environment_variant"), config["persistent_waves"]["total_waves"],
                    config["simulation"]["max_steps"], config["scenario"]["team_size"])
        if identity != ("persistent_wave_v2", 3, 3000, 4):
            raise RuntimeError(f"{name} environment identity mismatch: {identity}")
        if blue_counts(config) != expected[name]:
            raise RuntimeError(f"{name} Blue force-size mismatch")
        if name != "444":
            candidate = deepcopy(config)
            candidate["persistent_waves"].pop("blue_units_per_wave")
            if candidate != base:
                raise RuntimeError(f"{name} differs from 444 outside blue_units_per_wave")


def validate_source(state: dict, algorithm_config: dict, training_env: dict) -> dict:
    extra = state.get("extra", {})
    checks = {
        "algorithm": state.get("algorithm") == "modular_mappo",
        "development_method": extra.get("development_method") == "marc_mappo_v1",
        "training_seed": int(extra.get("training_seed", -1)) == 5301,
        "sampled_steps": int(state.get("sampled_steps", -1)) == 1_000_008,
        "observation_dim": int(extra.get("observation_dim", -1)) == 52,
        "action_dim": int(extra.get("action_dim", -1)) == 3,
        "num_agents": int(extra.get("num_agents", -1)) == 4,
        "marc_version": int(state.get("development_feature_versions", {}).get(
            "milestone_aware_retention_credit", -1)) == 1,
        "environment_variant": extra.get("environment_variant") == "persistent_wave_v2",
        "training_blue_force": blue_counts(training_env) == [4, 4, 4],
        "enabled_modules": state.get("enabled_modules") == [
            "actor_lr_decay", "milestone_aware_retention_credit"],
        "run_local_config_method": algorithm_config.get("development_method") == "marc_mappo_v1",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"source checkpoint identity failure: {failed}")
    return checks


def red_state_fingerprint(env) -> str:
    values = np.asarray([[state.x, state.y, state.z, state.v, state.theta, state.psi]
                         for state in env.red], dtype="<f8")
    alive = np.asarray([state.alive for state in env.red], dtype=np.uint8)
    return hashlib.sha256(values.tobytes() + alive.tobytes()).hexdigest()


def evaluate_episode(trainer, env_config: dict, seed: int, variant: str) -> dict:
    env = make_combat_environment(env_config)
    observation, _ = env.reset(seed)
    alive = env.red_alive_mask.copy()
    actor_hidden, critic_hidden = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, np.float32)
    wave = 1
    total = 3
    returns = np.zeros(4, dtype=np.float64)
    entries: dict[int, dict] = {}
    while True:
        context = mission_context_numpy(
            trainer, np.asarray([wave]), np.asarray([total]), env.blue_alive_mask[None],
            np.asarray([env.steps]), env.max_steps)
        actions, actor_hidden = trainer.act(
            observation[None], alive[None], True, False, context, actor_hidden,
            episode_mask, wave_indices=np.asarray([wave]))
        _, critic_hidden = trainer.values_step(
            observation[None], alive[None], context, critic_hidden, episode_mask)
        observation, reward, terminated, truncated, info = env.step(actions[0])
        returns += reward
        alive = np.asarray(info["red_alive_mask"], np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        critic_hidden = trainer.recurrent.apply_alive(critic_hidden, alive[None])
        episode_mask[:] = 1
        wave = int(info["wave_index"])
        if info.get("spawned_next_wave", False):
            entries[wave] = {
                "red_survivors": int(info["red_survivors"]),
                "red_state_fingerprint": red_state_fingerprint(env),
                "spawn_candidate_index": int(info["wave_spawn_candidate_index"]),
            }
        if terminated or truncated:
            team_return, mean_agent_return = episode_return_metrics(returns)
            records = {int(row["wave_index"]): row for row in info.get("per_wave_metrics", [])}
            return {
                "variant": variant,
                "blue_units_per_wave": "-".join(map(str, blue_counts(env_config))),
                "seed": seed,
                "W1": int(info["waves_cleared"] >= 1),
                "W2": int(info["waves_cleared"] >= 2),
                "W3": int(info["waves_cleared"] >= 3),
                "waves_cleared": int(info["waves_cleared"]),
                "red_success": int(bool(info["red_success"])),
                "return": team_return,
                "mean_agent_return": mean_agent_return,
                "red_loss": int(info["red_losses"]),
                "blue_loss": int(info["blue_losses"]),
                "boundary": int(info["red_boundary_exits"]),
                "ground": int(info["red_ground_losses"]),
                "timeout": int(info["termination_reason"] == "red_failure_timeout"),
                "episode_length": int(info["episode_length"]),
                "red_entering_W2": entries.get(2, {}).get("red_survivors"),
                "red_entering_W3": entries.get(3, {}).get("red_survivors"),
                "W2_entry_red_state_sha256": entries.get(2, {}).get("red_state_fingerprint"),
                "W3_entry_red_state_sha256": entries.get(3, {}).get("red_state_fingerprint"),
                "W2_spawn_candidate_index": entries.get(2, {}).get("spawn_candidate_index"),
                "W3_spawn_candidate_index": entries.get(3, {}).get("spawn_candidate_index"),
                "W1_blue_survivors_start": records.get(1, {}).get("blue_survivors_start"),
                "W2_blue_survivors_start": records.get(2, {}).get("blue_survivors_start"),
                "W3_blue_survivors_start": records.get(3, {}).get("blue_survivors_start"),
            }


def safe_ratio(numerator: float, denominator: float):
    return None if denominator == 0 else numerator / denominator


def summarize(variant: str, counts: list[int], records: list[dict]) -> dict:
    mean = lambda key: float(np.mean([row[key] for row in records]))
    w1, w2, w3 = mean("W1"), mean("W2"), mean("W3")
    enter2 = [row["red_entering_W2"] for row in records if row["red_entering_W2"] is not None]
    enter3 = [row["red_entering_W3"] for row in records if row["red_entering_W3"] is not None]
    return {
        "variant": variant, "blue_units_per_wave": counts,
        "W1": w1, "W2": w2, "W3": w3, "AW": mean("waves_cleared"),
        "Return": mean("return"), "RedLoss": mean("red_loss"), "BlueLoss": mean("blue_loss"),
        "Boundary": mean("boundary"), "Ground": mean("ground"), "Timeout": mean("timeout"),
        "EpisodeLength": mean("episode_length"), "Q2": safe_ratio(w2, w1),
        "Q3": safe_ratio(w3, w2),
        "mean_red_entering_W2": float(np.mean(enter2)) if enter2 else None,
        "mean_red_entering_W3": float(np.mean(enter3)) if enter3 else None,
        "W3_entry_count": len(enter3), "W3_success_count": int(sum(row["W3"] for row in records)),
    }


def load_reference_row() -> dict:
    with (RUN / "evaluation_history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(float(rows[-1]["sampled_steps"])) != 1_000_008:
        raise RuntimeError("final evaluation_history endpoint is missing")
    return rows[-1]


def parity_check(summary: dict, reference: dict, records: list[dict]) -> dict:
    record_mean = lambda key: float(np.mean([row[key] for row in records]))
    actual = {
        "clear_wave_1_probability": summary["W1"],
        "clear_wave_2_probability": summary["W2"],
        "clear_wave_3_probability": summary["W3"],
        "average_waves_cleared": summary["AW"], "average_return": summary["Return"],
        "average_red_loss": summary["RedLoss"], "average_red_boundary_exits": summary["Boundary"],
        "average_red_ground_losses": summary["Ground"],
        "average_red_survivors_after_wave_1_conditional_on_clear": float(np.mean([
            row["red_entering_W2"] for row in records if row["red_entering_W2"] is not None])),
        "average_red_survivors_after_wave_2_conditional_on_clear": float(np.mean([
            row["red_entering_W3"] for row in records if row["red_entering_W3"] is not None])),
    }
    del record_mean
    comparisons = {}
    for key, value in actual.items():
        expected = float(reference[key])
        comparisons[key] = {"actual": value, "expected": expected,
                            "absolute_error": abs(value - expected),
                            "pass": bool(np.isclose(value, expected, atol=1e-10, rtol=0.0))}
    return {"status": "PASS" if all(row["pass"] for row in comparisons.values()) else "FAIL",
            "comparisons": comparisons}


def paired_analysis(by_variant: dict[str, list[dict]]) -> dict:
    maps = {name: {row["seed"]: row for row in rows} for name, rows in by_variant.items()}
    entry444 = {seed for seed, row in maps["444"].items() if row["red_entering_W3"] is not None}
    entry443 = {seed for seed, row in maps["443"].items() if row["red_entering_W3"] is not None}
    survivors_match = all(maps["444"][seed]["red_entering_W3"] == maps["443"][seed]["red_entering_W3"]
                          for seed in entry444 & entry443)
    states_match = all(maps["444"][seed]["W3_entry_red_state_sha256"] == maps["443"][seed]["W3_entry_red_state_sha256"]
                       for seed in entry444 & entry443)
    candidates_match = all(maps["444"][seed]["W3_spawn_candidate_index"] == maps["443"][seed]["W3_spawn_candidate_index"]
                           for seed in entry444 & entry443)
    success444 = sum(maps["444"][seed]["W3"] for seed in entry444)
    success443 = sum(maps["443"][seed]["W3"] for seed in entry444 if seed in maps["443"])
    rescue = sum(not maps["444"][seed]["W3"] and maps["443"][seed]["W3"] for seed in entry444)
    regression = sum(maps["444"][seed]["W3"] and not maps["443"][seed]["W3"] for seed in entry444)
    q444 = safe_ratio(success444, len(entry444)); q443 = safe_ratio(success443, len(entry444))
    delta = None if q444 is None or q443 is None else q443 - q444
    label = ("W3_FORCE_IMBALANCE_MATERIAL" if delta is not None and delta >= .15 else
             "W3_FORCE_IMBALANCE_MODERATE" if delta is not None and delta >= .05 else
             "W3_FORCE_IMBALANCE_WEAK")
    return {
        "entry_seed_sets_identical": entry444 == entry443,
        "entry_seeds_444": sorted(entry444), "entry_seeds_443": sorted(entry443),
        "entry_red_survivors_match": survivors_match,
        "entry_red_state_fingerprints_match": states_match,
        "spawn_candidate_indices_match": candidates_match,
        "paired_W3_entry_count": len(entry444),
        "444_W3_success_count": int(success444), "443_W3_success_count": int(success443),
        "w3_rescue_count": int(rescue), "w3_regression_count": int(regression),
        "paired_Q3_444": q444, "paired_Q3_443": q443, "delta_Q3": delta,
        "interpretation_label": label,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; no CPU fallback")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {OUTPUT}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    configs = {name: load_config(path) for name, path in VARIANTS.items()}
    validate_configs(configs)
    algorithm_config = load_config(RUN / "algorithm_config.yaml")
    training_env = load_config(RUN / "env_config.yaml")
    state = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    identity_checks = validate_source(state, algorithm_config, training_env)
    trainer = build_modular_mappo_trainer(algorithm_config, "cuda",
                                           total_sampled_steps=algorithm_config["training"]["total_sampled_steps"])
    trainer.load(CHECKPOINT, strict_protocol=True, restore_rng=False)

    by_variant: dict[str, list[dict]] = {}
    by_variant["444"] = [evaluate_episode(trainer, configs["444"], seed, "444") for seed in SEEDS]
    summaries = {"444": summarize("444", blue_counts(configs["444"]), by_variant["444"])}
    parity = parity_check(summaries["444"], load_reference_row(), by_variant["444"])
    if parity["status"] != "PASS":
        print(json.dumps({"status": "BLOCKED_444_PARITY_FAILURE", "parity": parity}, indent=2))
        print("BLOCKED_444_PARITY_FAILURE")
        raise SystemExit(2)
    for variant in ("443", "433"):
        by_variant[variant] = [evaluate_episode(trainer, configs[variant], seed, variant) for seed in SEEDS]
        summaries[variant] = summarize(variant, blue_counts(configs[variant]), by_variant[variant])
    paired = paired_analysis(by_variant)
    if not (paired["entry_seed_sets_identical"] and paired["entry_red_survivors_match"]
            and paired["entry_red_state_fingerprints_match"] and paired["spawn_candidate_indices_match"]):
        raise RuntimeError("444/443 pre-W3 matched-comparison invariant failed")

    protocol = {
        "task": "persistent-wave Blue force-size sensitivity evaluation",
        "evaluation_only": True, "training_performed": False,
        "cross_environment_force_size_evaluation": True,
        "source_checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": file_sha256(CHECKPOINT),
        "source_identity_checks": identity_checks,
        "device": "cuda", "cuda_device": torch.cuda.get_device_name(0),
        "deterministic": True, "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
        "episodes_per_variant": len(SEEDS), "variants": {k: blue_counts(v) for k, v in configs.items()},
        "legacy_444_parity": parity,
    }
    result = {"status": "IMPLEMENTATION_AND_EVALUATION_COMPLETE",
              "protocol": protocol, "summary": summaries, "paired_444_vs_443": paired}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary_rows = []
    for row in summaries.values():
        copy = dict(row); copy["blue_units_per_wave"] = "-".join(map(str, copy["blue_units_per_wave"]))
        summary_rows.append(copy)
    write_csv(OUTPUT / "summary.csv", summary_rows)
    write_csv(OUTPUT / "episodes.csv", [row for variant in ("444", "443", "433") for row in by_variant[variant]])
    print(json.dumps(result, indent=2))
    print("IMPLEMENTATION_AND_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
