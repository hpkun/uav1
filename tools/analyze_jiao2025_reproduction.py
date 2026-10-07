"""Post-training Jiao 2025 comparison with strict leakage protection.

No training entry point exists here. The 38M development range is touched only
after all four seed-2023 1.5M runs pass protocol and history validation.  The
originally reserved 37M range is rejected because a pre-screening smoke run
already consumed two seeds from it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import is_formal_v2_checkpoint

ENV_CONFIG = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
OUTPUT = ROOT / "outputs" / "jiao2025_reproduction_analysis"
CONTAMINATED_SEED_BASE = 37_000_000
CONTAMINATED_EPISODES = 100
FRESH_SEED_BASE = 38_000_000
FRESH_EPISODES = 100
VALIDATION_SEED_BASE = 10_000_000
VALIDATION_EPISODES = 20
FORMAL_HOLDOUT = set(range(20_000_000, 20_000_200))
RUNS = {
    "All-Off": ROOT / "outputs" / "pw_alloff_matched_1p5m_seed2023",
    "WB-MAPPO": ROOT / "outputs" / "pw_m5_wave_balance_1p5m_seed2023",
    "Jiao-Core": ROOT / "outputs" / "jiao2025_core_1p5m_seed2023",
    "Jiao-Full": ROOT / "outputs" / "jiao2025_full_1p5m_seed2023",
}
EXPECTED_MODULES = {
    "All-Off": set(), "WB-MAPPO": {"wave_balancing"},
    "Jiao-Core": {"wave_context", "recurrent_memory", "popart"},
    "Jiao-Full": {"wave_context", "recurrent_memory", "popart", "multi_wave_reward"},
}
PRIMARY_METHODS = ("All-Off", "Jiao-Core", "WB-MAPPO")
SUPPLEMENTARY_METHODS = ("Jiao-Core", "Jiao-Full")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _range(base: int, count: int) -> set[int]:
    return set(range(int(base), int(base) + int(count)))


def validate_jiao_evaluation_seeds(seeds) -> list[int]:
    values = [int(seed) for seed in seeds]
    expected = list(range(FRESH_SEED_BASE, FRESH_SEED_BASE + FRESH_EPISODES))
    if values != expected or len(values) != len(set(values)):
        raise ValueError("fresh comparison must use exactly 38,000,000-38,000,099")
    if set(values) & FORMAL_HOLDOUT:
        raise ValueError("20M formal-holdout seeds are forbidden")
    contaminated = _range(CONTAMINATED_SEED_BASE, CONTAMINATED_EPISODES)
    if set(values) & contaminated or any(35_000_000 <= seed < 37_000_000 for seed in values):
        raise ValueError("old 35M/36M and contaminated 37M development seeds are forbidden")
    return values


def validation_seed_set(config: dict[str, Any]) -> set[int]:
    return _range(config["implementation"]["evaluation_seed_base"], config["training"]["evaluation_episodes"])


def validate_validation_fresh_disjoint(config: dict[str, Any]) -> dict[str, Any]:
    validation = validation_seed_set(config);fresh = _range(FRESH_SEED_BASE, FRESH_EPISODES)
    if validation & fresh:
        raise RuntimeError("training validation overlaps the reserved 38M fresh range")
    if validation & FORMAL_HOLDOUT or fresh & FORMAL_HOLDOUT:
        raise RuntimeError("Jiao protocol overlaps the 20M formal holdout")
    return {"validation_start": min(validation), "validation_end": max(validation),
            "validation_count": len(validation), "fresh_start": min(fresh),
            "fresh_end": max(fresh), "fresh_count": len(fresh), "overlap_count": 0}


def validate_jiao_config(name: str, config: dict[str, Any]) -> None:
    enabled = {key for key, value in config["modules"].items() if value.get("enabled", False)}
    if enabled != EXPECTED_MODULES[name]:
        raise RuntimeError(f"{name} module mismatch: {sorted(enabled)}")
    training = config["training"]
    if (int(training["evaluation_episodes"]) != VALIDATION_EPISODES or
            int(config["implementation"]["evaluation_seed_base"]) != VALIDATION_SEED_BASE or
            int(training["evaluation_interval_sampled_steps"]) != 100_000):
        raise RuntimeError(f"{name} checkpoint-selection validation protocol mismatch")
    validate_validation_fresh_disjoint(config)
    if name.startswith("Jiao"):
        expected = {"actor_learning_rate": .0005, "critic_learning_rate": .0005,
                    "ppo_epochs": 10, "clip_ratio": .1, "entropy_coefficient": .01,
                    "gae_lambda": .95, "gamma": .99, "rollout_steps": 150,
                    "minibatch_size": 512, "num_train_envs": 24,
                    "total_sampled_steps": 1_500_000}
        for key, value in expected.items():
            if not np.isclose(float(training[key]), float(value)):
                raise RuntimeError(f"{name} {key} mismatch")
        modules = config["modules"]
        context = modules["wave_context"];recurrent = modules["recurrent_memory"]
        if context.get("encoding") != "scalar_round" or context.get("context_target") != "actor_critic":
            raise RuntimeError(f"{name} scalar actor/critic F is not enabled")
        if recurrent.get("mode") != "actor_critic_gru" or int(recurrent.get("hidden_dim", 0)) != 128 or int(recurrent.get("sequence_length", 0)) != 32:
            raise RuntimeError(f"{name} recurrent protocol mismatch")
        expected_reward = "jiao_r2_replacement" if name == "Jiao-Full" else "none"
        if modules["multi_wave_reward"].get("mode") != expected_reward:
            raise RuntimeError(f"{name} reward mode mismatch")
        for forbidden in ("wave_balancing", "warm_start", "curriculum", "policy_anchor"):
            if modules[forbidden].get("enabled", False):
                raise RuntimeError(f"{name} unexpectedly enables {forbidden}")


def validate_evaluation_history(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "evaluation_history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"training evaluation history missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty evaluation history: {path}")
    expected_episodes = int(config["training"]["evaluation_episodes"])
    expected_base = int(config["implementation"]["evaluation_seed_base"])
    require_explicit_seeds = run_dir.name.startswith("jiao2025_")
    for row in rows:
        if int(float(row.get("evaluation_episodes", -1))) != expected_episodes:
            raise RuntimeError(f"{run_dir.name} evaluation episode count mismatch")
        if require_explicit_seeds:
            if "evaluation_seed_base" not in row or "evaluation_seed_end" not in row:
                raise RuntimeError(f"{run_dir.name} lacks explicit evaluation seed provenance")
            if (int(float(row["evaluation_seed_base"])) != expected_base or
                    int(float(row["evaluation_seed_end"])) != expected_base + expected_episodes - 1):
                raise RuntimeError(f"{run_dir.name} evaluation seed provenance mismatch")
        for key, value in row.items():
            if "seed" in key.lower() and value not in (None, ""):
                seed = int(float(value))
                if (CONTAMINATED_SEED_BASE <= seed < CONTAMINATED_SEED_BASE + CONTAMINATED_EPISODES or
                        FRESH_SEED_BASE <= seed < FRESH_SEED_BASE + FRESH_EPISODES):
                    raise RuntimeError(f"forbidden comparison-seed leakage found in {path}: {key}={seed}")
    protocol = validate_validation_fresh_disjoint(config)
    return {"rows": len(rows), "first_sampled_steps": int(rows[0]["sampled_steps"]),
            "last_sampled_steps": int(rows[-1]["sampled_steps"]), **protocol}


def validate_run(name: str, run_dir: Path) -> dict[str, Any]:
    required = ("run_config.json", "run_summary.json", "algorithm_config.yaml",
                "evaluation_history.csv", "best_eval.pt", "latest.pt")
    missing = [filename for filename in required if not (run_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} incomplete run: {missing}")
    run = _json(run_dir / "run_config.json");summary = _json(run_dir / "run_summary.json")
    config = yaml.safe_load((run_dir / "algorithm_config.yaml").read_text(encoding="utf-8"))
    validate_jiao_config(name, config)
    if run.get("algorithm") != "modular_mappo" or run.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError(f"{name} run protocol mismatch")
    if int(run.get("seed", -1)) != 2023 or int(summary.get("sampled_steps", -1)) != 1_500_000:
        raise RuntimeError(f"{name} is not the complete seed2023 1.5M run")
    if run.get("algorithm_config_sha256") != config_sha256(config):
        raise RuntimeError(f"{name} algorithm config snapshot hash mismatch")
    history = validate_evaluation_history(run_dir, config)
    return {"complete": True, "sampled_steps": 1_500_000, "training_seed": 2023,
            "checkpoint_selection_key": "persistent_v2:(W3,average_waves,raw_return,-red_loss)",
            "evaluation_history": history, "run_config": str((run_dir / "run_config.json").resolve())}


def load_checkpoint(name: str, path: Path, device: str):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Jiao reproduction analysis is CUDA-only")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not is_formal_v2_checkpoint(state):
        raise RuntimeError(f"{path} is not a self-describing Modular MAPPO v2 checkpoint")
    extra = state["extra"];config = extra.get("algorithm_config")
    if config is None:
        raise RuntimeError(f"{path} lacks algorithm_config")
    validate_jiao_config(name, config)
    if extra.get("environment_variant") != "persistent_wave_v2" or int(extra.get("training_seed", -1)) != 2023:
        raise RuntimeError(f"{path} environment/training-seed mismatch")
    if extra.get("algorithm_config_sha256") != config_sha256(config):
        raise RuntimeError(f"{path} self-described algorithm hash mismatch")
    trainer = build_modular_mappo_trainer(config, device=device,
        hidden_dim=int(extra["network_architecture"]["hidden_dim"]))
    trainer.load(path)
    metadata = {"checkpoint": str(path.resolve()), "checkpoint_sha256": file_sha256(path),
                "sampled_steps": int(state["sampled_steps"]), "training_seed": 2023,
                "algorithm_config_sha256": config_sha256(config),
                "enabled_modules": sorted(EXPECTED_MODULES[name])}
    return trainer, metadata


def run_episode(trainer, env_config: dict[str, Any], seed: int, include_trace=False) -> dict[str, Any]:
    source = evaluate_modular_episode(trainer, env_config, seed, include_trace=include_trace)
    record = {key: (value.item() if isinstance(value, np.generic) else value)
              for key, value in source.items()
              if isinstance(value, (bool, str, int, float, np.generic)) or value is None}
    record.update({"seed": int(seed), "episode_return": float(source["episode_return"]),
                   "timeout": int(source.get("termination_reason") == "red_failure_timeout"),
                   "waves_cleared": int(source.get("waves_cleared", 0)),
                   "episode_kill_loss_ratio": float(source.get("blue_losses", 0)) / max(float(source.get("red_losses", 0)), 1.0)})
    per_wave = [dict(row) for row in source.get("per_wave_metrics", [])]
    for index in (1, 2, 3):
        row = next((value for value in per_wave if int(value.get("wave_index", 0)) == index), None)
        record[f"clear_wave_{index}"] = int(record["waves_cleared"] >= index)
        record[f"reached_wave_{index}"] = int(index == 1 or record["waves_cleared"] >= index - 1)
        record[f"time_spent_wave_{index}"] = None if row is None else int(row["duration_steps"])
    if include_trace:
        record["action_trace"] = source["action_trace"];record["wave_trace"] = source["wave_trace"]
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda key: float(np.mean([float(row.get(key, 0) or 0) for row in records]))
    result = {"evaluation_episodes": float(len(records)), "average_return": mean("episode_return"),
              **{f"clear_wave_{index}_probability": mean(f"clear_wave_{index}") for index in (1, 2, 3)},
              "average_waves_cleared": mean("waves_cleared"), "average_red_loss": mean("red_losses"),
              "average_blue_loss": mean("blue_losses"), "average_red_boundary_exits": mean("red_boundary_exits"),
              "average_blue_boundary_exits": mean("blue_boundary_exits"), "average_red_ground_losses": mean("red_ground_losses"),
              "average_blue_ground_losses": mean("blue_ground_losses"), "timeout_rate": mean("timeout"),
              "average_episode_length": mean("episode_length"),
              "evaluation_boundary_exit_rate": float(np.mean([row.get("red_boundary_exits", 0) > 0 for row in records]))}
    result["kill_loss_ratio"] = float(sum(row.get("blue_losses", 0) for row in records)) / max(float(sum(row.get("red_losses", 0) for row in records)), 1.0)
    for index in (1, 2, 3):
        reached = [row for row in records if row[f"reached_wave_{index}"]]
        result[f"timeout_conditioned_reached_W{index}"] = float(np.mean([row["timeout"] for row in reached])) if reached else 0.0
        durations = [row[f"time_spent_wave_{index}"] for row in reached if row.get(f"time_spent_wave_{index}") is not None]
        result[f"average_duration_W{index}_conditional_on_reach"] = float(np.mean(durations)) if durations else 0.0
    return result


def reward_summary(run_dir: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (run_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = ("raw_environment_reward", "jiao_training_reward", "paper_R2_blue_kill_component",
            "paper_R2_red_loss_component", "paper_R2_wave1", "paper_R2_wave2", "paper_R2_wave3")
    result = {key: float(sum(float(row.get(key, 0.0)) for row in rows)) for key in keys}
    totals = _json(run_dir / "run_summary.json").get("paper_R2_totals", {})
    result.update({"paper_R2_blue_kill_component": float(totals.get("blue_kill_component", result["paper_R2_blue_kill_component"])),
                   "paper_R2_red_loss_component": float(totals.get("red_loss_component", result["paper_R2_red_loss_component"])),
                   **{f"paper_R2_wave{wave}": float(totals.get(f"wave_{wave}", result[f"paper_R2_wave{wave}"])) for wave in (1, 2, 3)}})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser();parser.add_argument("--device", choices=("cuda",), default="cuda");parser.add_argument("--output-dir", type=Path, default=OUTPUT);args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU evaluation")
    # All completion/leakage gates run before output/cache creation and before any 38M episode.
    integrity = {method: validate_run(method, run_dir) for method, run_dir in RUNS.items()}
    seeds = validate_jiao_evaluation_seeds(range(FRESH_SEED_BASE, FRESH_SEED_BASE + FRESH_EPISODES))
    env_config = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    cache = args.output_dir / "evaluation_cache"
    cache_files_before = sorted(path.name for path in cache.glob("*.json")) if cache.is_dir() else []
    args.output_dir.mkdir(parents=True, exist_ok=True);cache.mkdir(exist_ok=True)
    table=[];metadata={};cache_reused=[]
    for method, run_dir in RUNS.items():
        for role, filename in (("best", "best_eval.pt"), ("latest", "latest.pt")):
            checkpoint=run_dir/filename;digest=file_sha256(checkpoint);cache_path=cache/f"{method.lower().replace('-','_')}_{role}.json"
            cached=_json(cache_path) if cache_path.is_file() else None
            if cached and cached.get("checkpoint_sha256")==digest and cached.get("seeds")==seeds:
                payload=cached;cache_reused.append(cache_path.name)
            else:
                trainer,checkpoint_meta=load_checkpoint(method,checkpoint,args.device)
                records=[run_episode(trainer,env_config,seed) for seed in seeds]
                payload={"checkpoint_sha256":digest,"seeds":seeds,"metadata":checkpoint_meta,"summary":summarize(records),"episodes":records}
                cache_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
            metadata[f"{method}_{role}"]=payload["metadata"];table.append({"method":method,"checkpoint_role":role,**payload["summary"]})
            print(f"[EVAL] {method} {role}: W3={payload['summary']['clear_wave_3_probability']:.3f} waves={payload['summary']['average_waves_cleared']:.3f}",flush=True)
    primary_best=[row for row in table if row["method"] in PRIMARY_METHODS and row["checkpoint_role"]=="best"]
    primary_latest=[row for row in table if row["method"] in PRIMARY_METHODS and row["checkpoint_role"]=="latest"]
    supplementary=[row for row in table if row["method"] in SUPPLEMENTARY_METHODS]
    reward_rows=[{"method":method,**reward_summary(RUNS[method])} for method in SUPPLEMENTARY_METHODS]
    write_csv(args.output_dir/"primary_fair_comparison_best.csv",primary_best)
    write_csv(args.output_dir/"primary_fair_comparison_latest.csv",primary_latest)
    write_csv(args.output_dir/"supplementary_jiao_full_comparison.csv",supplementary)
    write_csv(args.output_dir/"jiao_training_signal_summary.csv",reward_rows)
    result={"protocol":{"seeds":seeds,"episodes":FRESH_EPISODES,"device":args.device,"environment":str(ENV_CONFIG),
                        "comparison_status":"development; not 20M formal holdout","cache_files_before":cache_files_before,"cache_reused":cache_reused,
                        "primary_fair_algorithm_comparison":list(PRIMARY_METHODS),"supplementary_paper_stack_transfer":list(SUPPLEMENTARY_METHODS)},
            "run_integrity":integrity,"checkpoints":metadata,"primary_fair_comparison":{"best":primary_best,"latest":primary_latest},
            "supplementary_paper_stack_transfer":supplementary,"jiao_training_signal_summary":reward_rows}
    (args.output_dir/"comparison_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    columns=("method","checkpoint_role","clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability","average_waves_cleared","average_return","average_red_loss","average_blue_loss","kill_loss_ratio","average_red_boundary_exits","average_red_ground_losses","timeout_rate","average_episode_length")
    def markdown(rows):
        values=["| "+" | ".join(columns)+" |","|"+"---|"*len(columns)];values.extend("| "+" | ".join(str(row[key]) for key in columns)+" |" for row in rows);return values
    report=["# Jiao 2025 reproduction comparison","","Fresh development evaluation: seeds 38,000,000-38,000,099. The originally reserved 37M range is excluded after development-smoke contamination. All performance values are frozen-environment raw metrics.","",
            "## PRIMARY FAIR ALGORITHM COMPARISON","","All-Off, Jiao-Core and WB-MAPPO share the same benchmark, raw training reward and 1.5M sampled-step budget.","",*markdown(primary_best),"","Latest-checkpoint stability:","",*markdown(primary_latest),"",
            "## SUPPLEMENTARY PAPER-STACK TRANSFER","","Jiao-Core versus Jiao-Full isolates transfer of paper Eq. (12). Jiao-Full is excluded from the primary fair ranking because its training reward differs.","",*markdown(supplementary),"",
            "Jiao training signals are separate in `jiao_training_signal_summary.csv`; they are never treated as raw environment return.",""]
    (args.output_dir/"comparison_report.md").write_text("\n".join(report),encoding="utf-8")
    print(f"[DONE] {args.output_dir}",flush=True)


if __name__=="__main__":main()
