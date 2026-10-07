"""Strictly offline analyzer for the matched PWTR 300k ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
BRANCHES = ("plain", "stratified", "current_extra", "uniform_recent", "priority_recent", "full")
TARGET = 1_805_280
METRICS = {
    "W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability",
    "W3": "clear_wave_3_probability", "AverageWaves": "average_waves_cleared",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "BlueLoss": "average_blue_loss", "Boundary": "average_red_boundary_exits",
    "Ground": "average_red_ground_losses", "EpisodeLength": "average_episode_length",
    "W2EntrySurvivors": "average_red_survivors_after_wave_1_conditional_on_clear",
    "W3EntrySurvivors": "average_red_survivors_after_wave_2_conditional_on_clear",
}
EXPECTED_IDENTITY = {
    "plain": ("pwtr_plain_matched_control", ["actor_lr_decay"], None),
    "stratified": ("pwtr_stratified", ["actor_lr_decay", "persistent_wave_trajectory_replay"], (True, False, "recent_uniform", False, False, False, False)),
    "current_extra": ("pwtr_current_extra", ["actor_lr_decay", "persistent_wave_trajectory_replay"], (True, True, "current", False, False, True, True)),
    "uniform_recent": ("pwtr_uniform_recent", ["actor_lr_decay", "persistent_wave_trajectory_replay"], (True, True, "recent_uniform", False, False, True, True)),
    "priority_recent": ("pwtr_priority_recent", ["actor_lr_decay", "persistent_wave_trajectory_replay"], (True, True, "recent_priority", True, False, True, True)),
    "full": ("pwtr_full", ["actor_lr_decay", "persistent_wave_trajectory_replay"], (True, True, "recent_priority", True, True, True, True)),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {value}")
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_branch_identity(run: Path, branch: str, config: dict) -> None:
    method, modules, expected_mode = EXPECTED_IDENTITY[branch]
    if config.get("development_method") != method or sorted(config.get("enabled_modules", [])) != sorted(modules):
        raise RuntimeError(f"{run}: branch identity mismatch")
    algorithm = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
    if algorithm.get("development_method") != method:
        raise RuntimeError(f"{run}: algorithm snapshot method mismatch")
    module = algorithm.get("modules", {}).get("persistent_wave_trajectory_replay", {})
    if expected_mode is None:
        if module.get("enabled", False): raise RuntimeError(f"{run}: Plain unexpectedly enables PWTR")
        return
    actual = (bool(module.get("fresh_wave_stratification")), bool(module.get("replay_enabled")),
              module.get("replay_source"), bool(module.get("priority_enabled")),
              bool(module.get("bridge_enabled")), bool(module.get("actor_replay")),
              bool(module.get("critic_replay")))
    fixed = tuple(int(module.get(key, -1)) for key in (
        "sequence_length", "bridge_half_length", "min_segment_length",
        "partition_capacity", "actor_max_age_updates"))
    if actual != expected_mode or fixed != (128, 64, 32, 32, 2):
        raise RuntimeError(f"{run}: frozen PWTR mode mismatch")


def endpoint(run: Path, branch: str):
    required = ("run_summary.json", "run_config.json", "evaluation_history.csv", "optimization_metrics.jsonl",
                "training_metrics.jsonl", "train.log", "algorithm_config.yaml", "latest.pt", "final.pt")
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise RuntimeError(f"{run}: missing {missing}")
    summary = read_json(run / "run_summary.json")
    if int(summary.get("sampled_steps", summary.get("final_sampled_steps", -1))) != TARGET:
        raise RuntimeError(f"{run}: not exact target {TARGET}")
    with (run / "evaluation_history.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    exact = [row for row in rows if int(float(row["sampled_steps"])) == TARGET]
    if len(exact) != 1:
        raise RuntimeError(f"{run}: exact endpoint count={len(exact)}")
    row = exact[0]
    if int(float(row["evaluation_episodes"])) != 50 or int(float(row["evaluation_seed_base"])) != 44_000_000 or int(float(row["evaluation_seed_end"])) != 44_000_049:
        raise RuntimeError(f"{run}: evaluation protocol mismatch")
    values = {name: finite(row[column], f"{run}/{name}") for name, column in METRICS.items() if row.get(column, "") != ""}
    values["Q2"] = None if values["W1"] == 0 else values["W2"] / values["W1"]
    values["Q3"] = None if values["W2"] == 0 else values["W3"] / values["W2"]
    config = read_json(run / "run_config.json")
    if int(config.get("training_seed", config.get("seed", -1))) not in SEEDS:
        raise RuntimeError(f"{run}: seed mismatch")
    validate_branch_identity(run, branch, config)
    for name in ("latest.pt", "final.pt"):
        checkpoint = torch.load(run / name, map_location="cpu", weights_only=False)
        expected_method, expected_modules, _ = EXPECTED_IDENTITY[branch]
        if (checkpoint.get("algorithm") != "modular_mappo"
                or int(checkpoint.get("sampled_steps", -1)) != TARGET
                or sorted(checkpoint.get("enabled_modules", [])) != sorted(expected_modules)
                or checkpoint.get("extra", {}).get("development_method") != expected_method):
            raise RuntimeError(f"{run}/{name}: checkpoint identity/endpoint mismatch")
    opt = []
    for line in (run / "optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            for key, value in item.items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise RuntimeError(f"{run}: non-finite optimization metric {key}")
            opt.append(item)
    if not opt:
        raise RuntimeError(f"{run}: empty optimization metrics")
    log_text = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    if any(marker in log_text for marker in ("Traceback", "CUDA out of memory", "protocol mismatch")):
        raise RuntimeError(f"{run}: failure marker in train.log")
    source = config.get("branch_provenance", config.get("branch_source_provenance", {}))
    return values, config, opt, source


def delta(left, right):
    return {key: (None if left.get(key) is None or right.get(key) is None else left[key] - right[key]) for key in METRICS | {"Q2": "", "Q3": ""}}


def descriptive(rows):
    keys = ("AverageWaves", "W3", "Q2", "Q3")
    means = {}
    for key in keys:
        available = [row[key] for row in rows if row.get(key) is not None]
        means[key] = mean(available) if available else None
    if all(value is not None and value > 0 for value in means.values()): return "POSITIVE", means
    if all(value is not None and value <= 0 for value in means.values()): return "NEGATIVE", means
    return "MIXED", means


def classify_full_gate(rows):
    gate_a = sum(row["AverageWaves"] > 0 for row in rows) >= 2 and mean(row["AverageWaves"] for row in rows) > 0
    gate_b = sum(row["W3"] > 0 for row in rows) >= 2 and mean(row["W3"] for row in rows) > 0
    gate_c = not any(row["AverageWaves"] <= -.50 or row["W1"] <= -.25 for row in rows) and mean(row["W1"] for row in rows) >= -.05
    decision = "SAFETY_FAIL" if not gate_c else "NOT_SUPPORTED" if not gate_a else "LATER_WAVE_MECHANISM_INCONCLUSIVE" if not gate_b else "PROMISING"
    return decision, gate_a, gate_b, gate_c


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/pwtr_300k_analysis")
    args = parser.parse_args(); output = ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    data = {}; opt_data = {}; configs = {}; sources = {}
    for seed in SEEDS:
        for branch in BRANCHES:
            run = ROOT / f"outputs/dev_pwtr_{branch}_seed{seed}_300k"
            data[seed, branch], configs[seed, branch], opt_data[seed, branch], sources[seed, branch] = endpoint(run, branch)
    # Provenance checks are deliberately based only on saved run artifacts.
    for seed in SEEDS:
        source_hashes = {sources[seed, branch].get("parent_checkpoint_sha256") for branch in BRANCHES}
        if None in source_hashes:
            raise RuntimeError(f"seed{seed}: missing parent checkpoint SHA")
        if len(source_hashes) != 1:
            raise RuntimeError(f"seed{seed}: source provenance differs across branches")
    runtime_hashes = {configs[key].get("runtime_source_manifest_sha256") for key in configs}
    if None in runtime_hashes or len(runtime_hashes) != 1:
        raise RuntimeError("runtime source manifest inconsistent across 18 runs")
    endpoint_rows = [{"seed": seed, "branch": branch, "sampled_steps": TARGET, **data[seed, branch]} for seed in SEEDS for branch in BRANCHES]
    paired = []
    for seed in SEEDS:
        for branch in BRANCHES[1:]: paired.append({"seed": seed, "branch": branch, **{f"delta_{k}": v for k, v in delta(data[seed, branch], data[seed, "plain"]).items()}})
    chain = (("stratified", "plain", "stratification"), ("current_extra", "stratified", "extra_current_update"), ("uniform_recent", "current_extra", "historical_reuse"), ("priority_recent", "uniform_recent", "priority"), ("full", "priority_recent", "bridge"))
    incremental = []
    for high, low, effect in chain:
        for seed in SEEDS: incremental.append({"seed": seed, "effect": effect, "comparison": f"{high}-{low}", **{f"delta_{k}": v for k, v in delta(data[seed, high], data[seed, low]).items()}})
    full = [delta(data[s, "full"], data[s, "plain"]) for s in SEEDS]
    decision, gate_a, gate_b, gate_c = classify_full_gate(full)
    reuse = [delta(data[s, "uniform_recent"], data[s, "current_extra"]) for s in SEEDS]
    reuse_wins = sum(row["AverageWaves"] > 0 for row in reuse); reuse_mean = mean(row["AverageWaves"] for row in reuse)
    reuse_label = "SUPPORTED" if reuse_wins >= 2 and reuse_mean > 0 else "NOT_SUPPORTED" if reuse_wins <= 1 and reuse_mean <= 0 else "MIXED"
    priority_label, priority_means = descriptive([delta(data[s, "priority_recent"], data[s, "uniform_recent"]) for s in SEEDS])
    bridge_label, bridge_means = descriptive([delta(data[s, "full"], data[s, "priority_recent"]) for s in SEEDS])
    mechanism = []
    pwtr_keys = sorted({key for rows in opt_data.values() for row in rows for key in row if key.startswith("pwtr_")})
    for seed in SEEDS:
        for branch in BRANCHES:
            rows = opt_data[seed, branch]
            numeric = {key: [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))] for key in pwtr_keys}
            mechanism.append({"seed": seed, "branch": branch,
                "final_cumulative_replay_actor_optimizer_steps": float(rows[-1].get("pwtr_replay_actor_optimizer_steps", 0.0)),
                "final_cumulative_replay_critic_optimizer_steps": float(rows[-1].get("pwtr_replay_critic_optimizer_steps", 0.0)),
                "mean_replay_batches_per_update": mean(numeric.get("pwtr_replay_batches", [0.0])) if numeric.get("pwtr_replay_batches") else 0.0,
                "mean_replay_budget_fill_fraction": mean(numeric.get("pwtr_replay_budget_fill_fraction", [1.0])) if numeric.get("pwtr_replay_budget_fill_fraction") else 1.0,
                **{key: mean(values) for key, values in numeric.items() if values}})
    report = {"status": decision, "training_seed_replication_n": 3, "exact_endpoint": TARGET,
              "gate_A": gate_a, "gate_B": gate_b, "safety_gate_C": gate_c,
              "historical_reuse_effect": reuse_label, "historical_reuse_aw_wins": reuse_wins, "historical_reuse_mean_delta_aw": reuse_mean,
              "priority_effect": priority_label, "priority_effect_means": priority_means,
              "bridge_effect": bridge_label, "bridge_effect_means": bridge_means,
              "runtime_source_manifest_sha256": next(iter(runtime_hashes)), "evaluation_range": [44_000_000, 44_000_049],
              "reserved_45m_used": False,
              "recommendation": "extend a matched branch to 600k or perform fresh multiseed validation" if decision == "PROMISING" else "inspect the frozen ablation chain before any new experiment"}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output / "endpoint_table.csv", endpoint_rows); write_csv(output / "paired_vs_plain.csv", paired)
    write_csv(output / "incremental_ablation.csv", incremental); write_csv(output / "replay_mechanism_summary.csv", mechanism)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
