#!/usr/bin/env python3
"""Read-only analysis for the corrected full-10-epoch Actor clip screen."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
SOURCE_STEP, TARGET = 1_505_280, 1_805_280
OUT = ROOT / "outputs/actor_grad_clip_fixed10_300k_analysis"
CURRENT = {
    "Control05": lambda seed: ROOT / f"outputs/dev_actor_clip05_fixed10_seed{seed}_300k",
    "Treatment10": lambda seed: ROOT / f"outputs/dev_actor_clip10_fixed10_seed{seed}_300k",
}
CONTEXT = {
    "HistoricalPlain": lambda seed: ROOT / f"outputs/dev_pwtr_plain_seed{seed}_300k",
    "Old1EpochControl05": lambda seed: ROOT / f"outputs/dev_actor_clip05_seed{seed}_300k",
}
FIELDS = {
    "AW": "average_waves_cleared", "W1": "clear_wave_1_probability",
    "W2": "clear_wave_2_probability", "W3": "clear_wave_3_probability",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "EpisodeLength": "average_episode_length",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, data):
    data = list(data)
    fields = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def extract(row):
    result = {key: float(row[column]) for key, column in FIELDS.items()}
    result["Q2"] = None if result["W1"] == 0 else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] == 0 else result["W3"] / result["W2"]
    return result


def delta(high, low, key):
    return None if high[key] is None or low[key] is None else high[key] - low[key]


def trainer_sha(run_config):
    manifest = {row["path"]: row["sha256"] for row in run_config["runtime_source_manifest_files"]}
    return manifest["algorithm/modular_mappo/trainer.py"]


def validate_current_run(method, seed, path, source_sha):
    required = ("run_config.json", "run_summary.json", "branch_from.json", "evaluation_history.csv", "optimization_metrics.jsonl", "latest.pt", "final.pt")
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        raise RuntimeError(f"incomplete run: {path}")
    run = json.loads((path / "run_config.json").read_text())
    summary = json.loads((path / "run_summary.json").read_text())
    branch = json.loads((path / "branch_from.json").read_text())
    expected_method = "actor_grad_clip_05_control" if method == "Control05" else "actor_grad_clip_10"
    if run["development_method"] != expected_method or int(run["seed"]) != seed or int(run["total_sampled_steps"]) != TARGET:
        raise RuntimeError(f"run identity mismatch: {path}")
    if branch["parent_checkpoint_sha256"] != source_sha or int(branch["source_training_seed"]) != seed:
        raise RuntimeError(f"parent mismatch: {path}")
    if int(summary["sampled_steps"]) != TARGET:
        raise RuntimeError(f"run summary endpoint mismatch: {path}")
    history = csv_rows(path / "evaluation_history.csv")
    endpoint = next((row for row in history if int(row["sampled_steps"]) == TARGET), None)
    if endpoint is None or (int(endpoint["evaluation_episodes"]), int(endpoint["evaluation_seed_base"]), int(endpoint["evaluation_seed_end"])) != (50, 44_000_000, 44_000_049):
        raise RuntimeError(f"fixed evaluation endpoint mismatch: {path}")
    if any(int(row["evaluation_seed_base"]) >= 45_000_000 for row in history):
        raise RuntimeError(f"45M seed found: {path}")
    optimization = jsonl(path / "optimization_metrics.jsonl")
    if len(optimization) != 49:
        raise RuntimeError(f"expected 49 optimization rows: {path} has {len(optimization)}")
    for index, row in enumerate(optimization):
        per_epoch = int(row["ppo_minibatches_per_epoch"])
        executed = int(row["ppo_minibatches_executed"])
        if int(row["ppo_epochs_executed"]) != 10 or executed != 10 * per_epoch:
            raise RuntimeError(f"epoch execution mismatch: {path} row {index}")
        if int(row["actor_optimizer_steps_this_update"]) != executed or int(row["critic_optimizer_steps_this_update"]) != executed:
            raise RuntimeError(f"optimizer count mismatch: {path} row {index}")
    actor_added = sum(int(row["actor_optimizer_steps_this_update"]) for row in optimization)
    critic_added = sum(int(row["critic_optimizer_steps_this_update"]) for row in optimization)
    if actor_added != 5860 or critic_added != 5860:
        raise RuntimeError(f"continuation optimizer total mismatch: {path}: {actor_added}/{critic_added}")
    final_state = torch.load(path / "final.pt", map_location="cuda", weights_only=False)
    latest_state = torch.load(path / "latest.pt", map_location="cuda", weights_only=False)
    if int(final_state["sampled_steps"]) != TARGET or int(latest_state["sampled_steps"]) != TARGET:
        raise RuntimeError(f"checkpoint endpoint mismatch: {path}")
    source_state = torch.load(ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt", map_location="cuda", weights_only=False)
    if int(final_state["ppo_updates"]) - int(source_state["ppo_updates"]) != 49:
        raise RuntimeError(f"checkpoint PPO update delta mismatch: {path}")
    if int(final_state["actor_updates"]) - int(source_state["actor_updates"]) != 5860:
        raise RuntimeError(f"checkpoint Actor update delta mismatch: {path}")
    if int(final_state["critic_updates"]) - int(source_state["critic_updates"]) != 5860:
        raise RuntimeError(f"checkpoint Critic update delta mismatch: {path}")
    del final_state, latest_state, source_state
    return run, summary, branch, history, optimization, endpoint, actor_added, critic_added


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for checkpoint audit")
    endpoint, paired, learning, clipping, protocol, epoch_audit, context_rows = {}, [], [], [], {}, {}, []
    runtime_shas = {}
    for seed in SEEDS:
        endpoint[seed] = {}
        source = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        source_sha = sha256(source)
        source_state = torch.load(source, map_location="cuda", weights_only=False)
        if int(source_state["sampled_steps"]) != SOURCE_STEP:
            raise RuntimeError(f"source step mismatch: {seed}")
        del source_state
        histories = {}
        for method, path_fn in CURRENT.items():
            path = path_fn(seed)
            run, summary, branch, history, optimization, final_row, actor_added, critic_added = validate_current_run(method, seed, path, source_sha)
            endpoint[seed][method] = extract(final_row)
            histories[method] = history
            runtime_shas[(method, seed)] = trainer_sha(run)
            protocol[f"{method}_seed{seed}"] = {
                "status": "PASS", "sampled_steps": int(summary["sampled_steps"]),
                "parent_checkpoint_sha256": branch["parent_checkpoint_sha256"],
                "runtime_trainer_sha256": runtime_shas[(method, seed)],
                "optimization_rows": len(optimization), "uses_45m": False,
            }
            epoch_audit[f"{method}_seed{seed}"] = {
                "ppo_updates_added": 49, "actor_updates_added": actor_added,
                "critic_updates_added": critic_added, "configured_epochs": 10,
                "all_rows_executed_ten_epochs": True,
            }
            for row in optimization:
                clipping.append({"method": method, "seed": seed, **{key: row.get(key) for key in (
                    "sampled_steps", "actor_grad_clip_preclip_norm", "actor_grad_clip_postclip_norm",
                    "actor_grad_clip_exact_scale", "actor_grad_clip_pressure_fraction",
                    "actor_grad_clip_critic_preclip_norm", "actor_grad_clip_critic_postclip_norm",
                    "actor_grad_clip_critic_exact_scale", "actor_grad_clip_critic_pressure_fraction",
                    "approx_kl", "clip_fraction", "entropy", "ratio_underflow_fraction", "max_abs_log_ratio",
                    "ppo_epochs_executed", "ppo_minibatches_executed")}})
        if runtime_shas[("Control05", seed)] != runtime_shas[("Treatment10", seed)]:
            raise RuntimeError(f"matched runtime trainer SHA mismatch: {seed}")
        common_steps = sorted(set(int(row["sampled_steps"]) for row in histories["Control05"]) & set(int(row["sampled_steps"]) for row in histories["Treatment10"]))
        for step in common_steps:
            current = {method: extract(next(row for row in histories[method] if int(row["sampled_steps"]) == step)) for method in CURRENT}
            learning.append({"seed": seed, "sampled_steps": step, **{f"{method}_{key}": value for method, values in current.items() for key, value in values.items()}, **{f"delta10minus05_{key}": delta(current["Treatment10"], current["Control05"], key) for key in (*FIELDS, "Q2", "Q3")}})
        for key in (*FIELDS, "Q2", "Q3"):
            paired.append({"seed": seed, "metric": key, "control": endpoint[seed]["Control05"][key], "treatment": endpoint[seed]["Treatment10"][key], "delta": delta(endpoint[seed]["Treatment10"], endpoint[seed]["Control05"], key)})
        for context_name, path_fn in CONTEXT.items():
            path = path_fn(seed)
            history = csv_rows(path / "evaluation_history.csv")
            final_row = next(row for row in history if int(row["sampled_steps"]) == TARGET)
            metrics = extract(final_row)
            context_rows.append({"seed": seed, "comparison": context_name,
                "runtime_source_note": "different runtime source SHA",
                "old_protocol_note": "ACTUAL_EFFECTIVE_PPO_EPOCHS=1; INVALID_FOR_10_EPOCH_PROTOCOL" if context_name == "Old1EpochControl05" else "historical Plain context only",
                **{f"control05_minus_context_{key}": delta(endpoint[seed]["Control05"], metrics, key) for key in ("AW", "W1", "W2", "W3")},
                **{f"treatment10_minus_context_{key}": delta(endpoint[seed]["Treatment10"], metrics, key) for key in ("AW", "W1", "W2", "W3")}})
    metrics = {}
    if len(set(runtime_shas.values())) != 1:
        raise RuntimeError("six fixed10 runs do not share one trainer SHA")
    for key in ("AW", "W1", "W2", "W3", "Q2", "Q3"):
        per_seed = {str(seed): delta(endpoint[seed]["Treatment10"], endpoint[seed]["Control05"], key) for seed in SEEDS}
        values = [value for value in per_seed.values() if value is not None]
        metrics[key] = {"per_seed": per_seed, "mean": statistics.mean(values) if values else None, "wins": sum(value > 0 for value in values), "n_defined": len(values)}
    safety = all(metrics["AW"]["per_seed"][str(seed)] > -.50 and metrics["W1"]["per_seed"][str(seed)] > -.25 for seed in SEEDS) and metrics["W1"]["mean"] >= -.05
    efficacy = metrics["AW"]["wins"] >= 2 and metrics["AW"]["mean"] > 0 and metrics["W3"]["wins"] >= 2 and metrics["W3"]["mean"] > 0
    label = "SAFETY_FAIL" if not safety else "PROMISING" if efficacy else "NOT_SUPPORTED"
    report = {
        "status": "ACTOR_GRAD_CLIP_FIXED10_ANALYSIS_COMPLETE",
        "protocol_status": "PASS", "replication_unit": "training_seed", "n": 3,
        "endpoint": endpoint, "primary_treatment10_minus_control05": metrics,
        "gates": {"safety": safety, "primary_efficacy": efficacy},
        "ACTOR_GRAD_CLIP_FIXED10_SCREEN": label,
        "context_comparisons": context_rows,
        "historical_plain_runtime_source_sha_note": "different runtime source SHA; context only",
        "old_control_interpretation": "1-epoch exploratory result only; never a primary control",
        "evaluation_rerun": False, "uses_45m": False,
    }
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2))
    (OUT / "protocol_validation.json").write_text(json.dumps(protocol, indent=2))
    (OUT / "epoch_execution_audit.json").write_text(json.dumps(epoch_audit, indent=2))
    write_csv(OUT / "paired_endpoint.csv", paired)
    write_csv(OUT / "learning_dynamics.csv", learning)
    write_csv(OUT / "clipping_dynamics.csv", clipping)
    (OUT / "decision_support.txt").write_text(f"ACTOR_GRAD_CLIP_FIXED10_SCREEN={label}\nSafety={safety}\nPrimaryEfficacy={efficacy}\n")
    print(json.dumps({"status": report["status"], "label": label, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
