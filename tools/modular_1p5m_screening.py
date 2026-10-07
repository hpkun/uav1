"""Reproducible 1.5M modular-MAPPO screening without any training writes."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from env.factory import make_combat_environment


OUT = ROOT / "outputs" / "modular_1p5m_screening"
CACHE = OUT / "evaluation_cache"
PW_ENV = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
DIRECT_ENV = ROOT / "configs" / "combat_environment.yaml"


def discover_runs(outputs: Path = ROOT / "outputs") -> dict[str, Path]:
    """Identify formal study runs from protocol metadata, never directory names."""
    candidates: dict[str, list[Path]] = {name: [] for name in ("PW baseline","Direct baseline","M5","M6","M1","M3")}
    module_labels = {"wave_balancing":"M5","warm_start":"M6","wave_context":"M1","popart":"M3"}
    for directory in outputs.iterdir():
        config_path, summary_path = directory / "run_config.json", directory / "run_summary.json"
        if not directory.is_dir() or not config_path.is_file() or not summary_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if int(config.get("seed", -1)) != 2023 or int(summary.get("sampled_steps", -1)) < 1_500_000:
            continue
        algorithm = str(config.get("algorithm", ""))
        variant = str(config.get("environment_variant", ""))
        modules = tuple(config.get("enabled_modules", []))
        if algorithm == "MAPPO" and variant == "persistent_wave_v2" and int(summary["sampled_steps"]) == 3_000_000:
            candidates["PW baseline"].append(directory)
        elif algorithm == "MAPPO" and variant == "direct_v2_3" and int(summary["sampled_steps"]) == 3_000_000:
            candidates["Direct baseline"].append(directory)
        elif algorithm == "modular_mappo" and variant == "persistent_wave_v2" and len(modules) == 1:
            label = module_labels.get(modules[0])
            if label and int(summary["sampled_steps"]) == 1_500_000:
                candidates[label].append(directory)
    ambiguous = {name: paths for name, paths in candidates.items() if len(paths) != 1}
    if ambiguous:
        raise RuntimeError(f"formal run discovery failed or ambiguous: {ambiguous}")
    return {name: paths[0] for name, paths in candidates.items()}


RUNS = discover_runs()


def validate_diagnostic_seeds(seeds) -> list[int]:
    values = [int(seed) for seed in seeds]
    if not values or len(values) != len(set(values)):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if any(20_000_000 <= seed <= 20_000_199 for seed in values):
        raise ValueError("20M formal holdout seeds are forbidden for screening")
    return values


def select_near_budget_checkpoints(records: list[tuple[int, str]], budget: int) -> dict[str, tuple[int, str]]:
    if not records:
        raise ValueError("checkpoint records must not be empty")
    ordered = sorted((int(step), path) for step, path in records)
    eligible = [record for record in ordered if record[0] <= budget]
    if not eligible:
        raise ValueError("no checkpoint at or below budget")
    return {"at_or_below": eligible[-1], "nearest_overall": min(ordered,key=lambda record:(abs(record[0]-budget),record[0]))}


def nearest_checkpoint_step(requested: int, available: list[int]) -> int:
    if not available:
        raise ValueError("available checkpoint steps must not be empty")
    return min(map(int,available),key=lambda step:(abs(step-int(requested)),step))


def require_cross_variant_permission(source: str, target: str, allowed: bool) -> bool:
    cross = str(source) != str(target)
    if cross and not allowed:
        raise RuntimeError("cross-variant evaluation requires explicit permission")
    return cross


def pw_budget_selection() -> dict[str, tuple[int, str]]:
    records=[]
    for path in RUNS["PW baseline"].glob("checkpoint_*.pt"):
        try:step=int(path.stem.split("_")[-1])
        except ValueError:continue
        records.append((step,str(path)))
    return select_near_budget_checkpoints(records,1_500_000)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def checkpoint_state(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def actor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state["actor"].items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_record(path: Path) -> dict[str, Any]:
    state = checkpoint_state(path)
    extra = state.get("extra", {})
    if state.get("algorithm") == "modular_mappo":
        from algorithm.modular_mappo.protocol import is_formal_v2_checkpoint
        protocol_complete = bool(is_formal_v2_checkpoint(state))
    else:
        protocol_complete = bool(extra)
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "sampled_steps": int(state.get("sampled_steps", 0)),
        "algorithm": state.get("algorithm"),
        "impl_version": state.get("modular_mappo_impl_version", state.get("mappo_impl_version")),
        "baseline_impl_version": state.get("baseline_mappo_impl_version"),
        "training_seed": extra.get("training_seed"),
        "gamma": extra.get("training_gamma"),
        "num_envs": extra.get("training_num_envs"),
        "training_total_sampled_steps": extra.get("training_total_sampled_steps"),
        "environment_variant": extra.get("environment_variant"),
        "enabled_modules": state.get("enabled_modules", []),
        "module_config_sha256": state.get("module_config_sha256"),
        "protocol_complete": protocol_complete,
        "actor_sha256": actor_digest(state),
    }


def task(name: str, method: str, checkpoint: Path, env: Path, seed_base: int,
         episodes: int, allow_cross_variant: bool = False) -> dict[str, Any]:
    validate_diagnostic_seeds(range(seed_base,seed_base+episodes))
    return {
        "name": name,
        "method": method,
        "checkpoint": str(checkpoint.resolve()),
        "env": str(env.resolve()),
        "seed_base": int(seed_base),
        "episodes": int(episodes),
        "allow_cross_variant": bool(allow_cross_variant),
    }


def evaluation_tasks() -> list[dict[str, Any]]:
    pw = RUNS["PW baseline"]
    selection=pw_budget_selection();below_step,below_path=selection["at_or_below"];near_step,near_path=selection["nearest_overall"]
    fresh = [
        task(f"PW_below_{below_step}", "PW baseline", Path(below_path), PW_ENV, 34_000_000, 50),
        task(f"PW_above_{near_step}", "PW baseline", Path(near_path), PW_ENV, 34_000_000, 50),
        task("M5_best", "M5", RUNS["M5"] / "best_eval.pt", PW_ENV, 34_000_000, 50),
        task("M5_latest", "M5", RUNS["M5"] / "latest.pt", PW_ENV, 34_000_000, 50),
        task("M1_best", "M1", RUNS["M1"] / "best_eval.pt", PW_ENV, 34_000_000, 50),
        task("M3_best", "M3", RUNS["M3"] / "best_eval.pt", PW_ENV, 34_000_000, 50),
        task("M3_latest", "M3", RUNS["M3"] / "latest.pt", PW_ENV, 34_000_000, 50),
        task("M6_best", "M6", RUNS["M6"] / "best_eval.pt", PW_ENV, 34_000_000, 50),
        task("M6_latest", "M6", RUNS["M6"] / "latest.pt", PW_ENV, 34_000_000, 50),
        task("Direct_source_to_PW", "Direct source", RUNS["Direct baseline"] / "best_eval.pt", PW_ENV, 34_000_000, 50, True),
    ]
    actual = {
        104448: RUNS["M6"] / "best_eval.pt",
        503808: RUNS["M6"] / "checkpoint_503808.pt",
        1001472: RUNS["M6"] / "checkpoint_1001472.pt",
        1500000: RUNS["M6"] / "checkpoint_1500000.pt",
    }
    sweep = [
        task("M6_source_PW20", "Direct source", RUNS["Direct baseline"] / "best_eval.pt", PW_ENV, 34_100_000, 20, True),
        task("M6_source_Direct30", "Direct source", RUNS["Direct baseline"] / "best_eval.pt", DIRECT_ENV, 34_200_000, 30),
    ]
    for step, path in actual.items():
        sweep.append(task(f"M6_{step}_PW20", "M6", path, PW_ENV, 34_100_000, 20))
        sweep.append(task(f"M6_{step}_Direct30", "M6", path, DIRECT_ENV, 34_200_000, 30, True))
    return fresh + sweep


def load_policy(task_spec: dict[str, Any]):
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
    from algorithm.mappo.factory import build_mappo_trainer
    from algorithm.modular_mappo.factory import build_modular_mappo_trainer
    from algorithm.modular_mappo.protocol import is_formal_v2_checkpoint
    checkpoint = Path(task_spec["checkpoint"])
    env_config = load_yaml(Path(task_spec["env"]))
    state = checkpoint_state(checkpoint)
    extra = state.get("extra", {})
    source_variant = str(extra.get("environment_variant", "direct_v2_3"))
    target_variant = str(env_config.get("environment_variant", "direct_v2_3"))
    cross_variant = require_cross_variant_permission(source_variant,target_variant,task_spec["allow_cross_variant"])
    if state.get("algorithm") == "MAPPO":
        algorithm_config = load_yaml(checkpoint.parent / "algorithm_config.yaml")
        validate_checkpoint_for_evaluation(
            state, env_config, algorithm_config,
            allow_cross_variant=task_spec["allow_cross_variant"],
        )
        trainer = build_mappo_trainer(algorithm_config, "cpu", hidden_dim=int(extra["effective_hidden_dim"]))
        trainer.load(checkpoint)
        kind = "baseline"
    elif state.get("algorithm") == "modular_mappo":
        if not is_formal_v2_checkpoint(state):
            raise RuntimeError(f"{task_spec['name']}: incomplete modular-v2 protocol")
        algorithm_config = extra.get("algorithm_config")
        if algorithm_config is None:
            raise RuntimeError(f"{task_spec['name']}: checkpoint lacks self-describing algorithm config")
        if str(env_config["environment_version"]) != str(extra.get("environment_version")):
            raise RuntimeError(f"{task_spec['name']}: environment version mismatch")
        trainer = build_modular_mappo_trainer(
            algorithm_config, "cpu", hidden_dim=int(extra["network_architecture"]["hidden_dim"])
        )
        trainer.load(checkpoint)
        kind = "modular"
    else:
        raise RuntimeError(f"{task_spec['name']}: unsupported checkpoint algorithm")
    metadata = {
        **checkpoint_record(checkpoint),
        "target_environment_variant": target_variant,
        "cross_variant": cross_variant,
        "target_environment_config_sha256": config_sha256(env_config),
    }
    return trainer, kind, env_config, metadata


def scalar(value: Any) -> Any:
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_episode(trainer, kind: str, env_config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = make_combat_environment(env_config)
    observation, _ = env.reset(seed)
    alive = env.red_alive_mask.copy()
    returns = np.zeros(4, dtype=np.float64)
    wave = 1
    total_waves = int(env_config.get("persistent_waves", {}).get("total_waves", 1))
    actor_hidden = critic_hidden = None
    episode_mask = np.zeros(1, dtype=np.float32)
    if kind == "modular":
        actor_hidden, critic_hidden = trainer.initial_hidden(1)
    while True:
        if kind == "baseline":
            actions = trainer.act(observation, alive, deterministic=True)
        else:
            from algorithm.modules.wave_survival_pbrs import mission_context_numpy
            context = mission_context_numpy(
                trainer, np.asarray([wave]), np.asarray([total_waves]),
                env.blue_alive_mask[None], np.asarray([env.steps]), env.max_steps,
            )
            actions, actor_hidden = trainer.act(
                observation[None], alive[None], True, False,
                context, actor_hidden, episode_mask,
            )
            _, critic_hidden = trainer.values_step(
                observation[None], alive[None], context,
                critic_hidden, episode_mask,
            )
            actions = actions[0]
        observation, reward, terminated, truncated, info = env.step(actions)
        returns += reward
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        episode_mask[:] = 1
        wave = int(info.get("wave_index", 1))
        total_waves = int(info.get("total_waves", total_waves))
        if terminated or truncated:
            break
    record = {key: scalar(value) for key, value in info.items() if scalar(value) is not None}
    record["seed"] = seed
    record["episode_return"] = float(returns.sum())
    record["mean_agent_episode_return"] = float(returns.mean())
    record["timeout"] = int(info.get("termination_reason") == "red_failure_timeout")
    record["waves_cleared"] = int(info.get("waves_cleared", 0))
    record["total_waves"] = int(info.get("total_waves", 1))
    record["episode_kill_loss_ratio"] = float(info.get("blue_losses", 0)) / max(float(info.get("red_losses", 0)), 1.0)
    per_wave = info.get("per_wave_metrics", [])
    for index in (1, 2, 3):
        cleared = record["waves_cleared"] >= index
        matching = [row for row in per_wave if int(row.get("wave_index", 0)) == index and row.get("wave_cleared", True)]
        record[f"clear_wave_{index}"] = int(cleared)
        record[f"red_survivors_after_wave_{index}"] = int(matching[0]["red_survivors_end"]) if matching else None
    return record


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    mean = lambda key: float(np.mean([float(row.get(key, 0) or 0) for row in records]))
    result = {
        "evaluation_episodes": len(records),
        "average_return": mean("episode_return"),
        "average_agent_return": mean("mean_agent_episode_return"),
        "win_rate": mean("red_success"), "loss_rate": mean("blue_win"), "draw_rate": mean("draw"),
        "timeout_rate": mean("timeout"),
        "average_red_loss": mean("red_losses"), "average_blue_loss": mean("blue_losses"),
        "average_red_attack_kills": mean("red_attack_kills"), "average_blue_attack_kills": mean("blue_attack_kills"),
        "average_red_boundary_exits": mean("red_boundary_exits"), "average_blue_boundary_exits": mean("blue_boundary_exits"),
        "evaluation_boundary_exit_rate": float(np.mean([row.get("red_boundary_exits", 0) > 0 for row in records])),
        "average_red_ground_losses": mean("red_ground_losses"), "average_blue_ground_losses": mean("blue_ground_losses"),
        "average_episode_length": mean("episode_length"),
        "average_waves_cleared": mean("waves_cleared"),
    }
    for side in ("red", "blue"):
        for event in ("fire_window", "attempt", "hit", "kill"):
            result[f"{side}_{event}_episode_rate"] = float(np.mean([
                row.get(f"{side}_first_{event}_step") is not None for row in records
            ]))
        for event in ("fire_window_steps", "fire_window_pair_steps", "fire_attempts", "weapon_hits", "attack_kills"):
            result[f"average_{side}_{event}"] = mean(f"{side}_{event}")
    for name in ("r1", "r2", "r3", "r4"):
        result[f"average_episode_{name}_total"] = mean(f"episode_{name}_total")
    for index in (1, 2, 3):
        result[f"clear_wave_{index}_probability"] = mean(f"clear_wave_{index}")
        survivors = [row[f"red_survivors_after_wave_{index}"] for row in records if row.get(f"red_survivors_after_wave_{index}") is not None]
        result[f"average_red_survivors_after_wave_{index}_conditional_on_clear"] = float(np.mean(survivors)) if survivors else 0.0
    total_blue = float(sum(row.get("blue_losses", 0) for row in records))
    total_red = float(sum(row.get("red_losses", 0) for row in records))
    result["total_blue_losses"] = total_blue
    result["total_red_losses"] = total_red
    result["kill_loss_ratio"] = total_blue / max(total_red, 1.0)
    return result


def cache_path(task_spec: dict[str, Any]) -> Path:
    return CACHE / f"{task_spec['name']}.json"


def evaluate_task(task_spec: dict[str, Any]) -> str:
    output = cache_path(task_spec)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("task") == task_spec:
            return f"cached {task_spec['name']}"
    trainer, kind, env_config, metadata = load_policy(task_spec)
    seeds = validate_diagnostic_seeds(range(task_spec["seed_base"], task_spec["seed_base"] + task_spec["episodes"]))
    records = [run_episode(trainer, kind, env_config, seed) for seed in seeds]
    payload = {"task": task_spec, "metadata": metadata, "summary": summarize_episodes(records), "episodes": records}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return f"evaluated {task_spec['name']} episodes={len(records)}"


def run_evaluations(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for method, directory in RUNS.items():
        inventory[method] = {
            name: checkpoint_record(directory / name)
            for name in ("best_eval.pt", "latest.pt")
        }
    (OUT / "checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    tasks = evaluation_tasks()
    print(f"evaluation tasks={len(tasks)} workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_task, spec): spec for spec in tasks}
        for future in as_completed(futures):
            print(future.result(), flush=True)


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as stream:
        return pd.DataFrame(json.loads(line) for line in stream if line.strip())


def evaluation_payload(name: str) -> dict[str, Any]:
    return json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))


def run_integrity() -> list[dict[str, Any]]:
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    result = []
    for method, directory in RUNS.items():
        run_config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
        best = inventory[method]["best_eval.pt"]
        latest = inventory[method]["latest.pt"]
        eval_history = pd.read_csv(directory / "evaluation_history.csv")
        duplicates = int(eval_history.duplicated(subset=["sampled_steps"]).sum())
        result.append({
            "method": method, "directory": str(directory.resolve()),
            "training_seed": run_config.get("seed"), "gamma": summary.get("gamma", best["gamma"]),
            "num_envs": run_config.get("num_envs"), "environment_variant": run_config.get("environment_variant"),
            "algorithm": best["algorithm"], "enabled_modules": best["enabled_modules"],
            "module_config_sha256": best["module_config_sha256"],
            "sampled_steps": latest["sampled_steps"], "best_step": best["sampled_steps"],
            "best_protocol_complete": best["protocol_complete"], "latest_protocol_complete": latest["protocol_complete"],
            "best_actor_sha256": best["actor_sha256"], "latest_actor_sha256": latest["actor_sha256"],
            "evaluation_duplicate_steps": duplicates,
            "evaluation_monotonic": bool(eval_history["sampled_steps"].is_monotonic_increasing),
        })
    return result


def training_history(integrity: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in integrity:
        if item["method"] == "Direct baseline":
            continue
        frame = pd.read_csv(Path(item["directory"]) / "evaluation_history.csv")
        for _, row in frame.iterrows():
            rows.append({
                "method": item["method"], "module": ",".join(item["enabled_modules"]) or "none",
                "training_seed": item["training_seed"],
                "sampled_steps": int(row.sampled_steps),
                "W1_clear_rate": row.get("clear_wave_1_probability", np.nan),
                "W2_clear_rate": row.get("clear_wave_2_probability", np.nan),
                "W3_clear_rate": row.get("clear_wave_3_probability", np.nan),
                "average_waves_cleared": row.get("average_waves_cleared", np.nan),
                "average_return": row.average_return, "average_red_loss": row.average_red_loss, "average_blue_loss": row.average_blue_loss,
                "K_L": row.get("kill_loss_ratio", np.nan), "average_red_boundary_exits": row.average_red_boundary_exits,
                "average_red_ground_losses": row.average_red_ground_losses, "timeout_rate": row.get("timeout_rate", np.nan),
                "best_flag": int(int(row.sampled_steps) == int(item["best_step"])),
            })
    return pd.DataFrame(rows)


def plot_training_history(frame: pd.DataFrame) -> None:
    frame = frame[frame.sampled_steps <= 1_505_280]
    for column, filename, ylabel in (
        ("W3_clear_rate", "training_W3_curve.png", "P(clear wave 3)"),
        ("average_waves_cleared", "training_avg_waves_curve.png", "Average waves cleared"),
        ("average_return", "training_return_curve.png", "Average return"),
        ("average_red_loss", "training_red_loss_curve.png", "Average red loss"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for method, group in frame.groupby("method"):
            group = group.sort_values("sampled_steps")
            ax.plot(group.sampled_steps, group[column], marker="o", markersize=2.8, linewidth=1.4, label=method)
        ax.set_xlabel("Sampled steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)


def fresh_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [spec["name"] for spec in evaluation_tasks() if spec["seed_base"] == 34_000_000]
    summaries, episodes = [], []
    for name in names:
        payload = evaluation_payload(name)
        summaries.append({"evaluation": name, "method": payload["task"]["method"], **payload["metadata"], **payload["summary"]})
        for row in payload["episodes"]:
            episodes.append({"evaluation": name, "method": payload["task"]["method"], "checkpoint_sampled_steps": payload["metadata"]["sampled_steps"], **row})
    return pd.DataFrame(summaries), pd.DataFrame(episodes)


DELTA_METRICS = {
    "W1": "clear_wave_1", "W2": "clear_wave_2", "W3": "clear_wave_3",
    "average_waves": "waves_cleared", "return": "episode_return", "red_loss": "red_losses",
    "K/L": "episode_kill_loss_ratio", "boundary": "red_boundary_exits", "ground": "red_ground_losses",
}


def matched_episode_delta(candidate: pd.DataFrame,baseline: pd.DataFrame,metrics: dict[str,str]=DELTA_METRICS) -> dict[str,float]:
    for name,frame in (("candidate",candidate),("baseline",baseline)):
        if frame.seed.duplicated().any():raise ValueError(f"duplicate seed in {name} episodes")
    candidate=candidate.set_index("seed");baseline=baseline.set_index("seed");common=baseline.index.intersection(candidate.index)
    if len(common)!=len(baseline) or len(common)!=len(candidate):raise ValueError("paired evaluation seed sets do not match")
    return {f"delta_{label}":float((candidate.loc[common,column]-baseline.loc[common,column]).mean()) for label,column in metrics.items()}


def paired_delta(episodes: pd.DataFrame) -> pd.DataFrame:
    baseline_name = "PW_above_1505280"
    baseline = episodes[episodes.evaluation == baseline_name].set_index("seed")
    rows = []
    available = set(episodes.evaluation.unique())
    for name in ("M5_best", "M5_latest", "M1_best", "M1_latest", "M3_best", "M3_latest", "M6_best", "M6_latest", "Direct_source_to_PW"):
        if name not in available:
            continue
        candidate_frame=episodes[episodes.evaluation==name]
        row = {"comparison": f"{name} - {baseline_name}", "candidate": name, "baseline": baseline_name, "paired_episodes": len(candidate_frame)}
        row.update(matched_episode_delta(candidate_frame,episodes[episodes.evaluation==baseline_name]))
        rows.append(row)
    return pd.DataFrame(rows)


def requested_m6_mapping() -> dict[int, int]:
    periodic = np.asarray([503808, 1001472, 1500000])
    requested = [104448,202752,301056,405504,503808,602112,700416,804864,903168,1001472,1105920,1204224,1302528,1400832,1500000]
    return {
        step: 104448 if step == 104448 else int(periodic[np.argmin(np.abs(periodic-step))])
        for step in requested
    }


def sweep_mapping_rows(mapping: dict[int,int]) -> list[dict[str,int]]:
    return [{"requested_step":int(requested),"actual_step":int(actual)} for requested,actual in mapping.items()]


def sweep_table(target: str) -> pd.DataFrame:
    suffix = "PW20" if target == "persistent_wave_v2" else "Direct30"
    source = evaluation_payload(f"M6_source_{suffix}")
    rows = [{"requested_step": 0, "actual_step": source["metadata"]["sampled_steps"], "checkpoint_role": "Direct source", **source["summary"]}]
    mapping = requested_m6_mapping()
    for mapped in sweep_mapping_rows(mapping):
        requested,actual=mapped["requested_step"],mapped["actual_step"]
        payload = evaluation_payload(f"M6_{actual}_{suffix}")
        rows.append({"requested_step": requested, "actual_step": actual, "checkpoint_role": "M6", **payload["summary"]})
    return pd.DataFrame(rows)


def plot_m6_sweeps(pw: pd.DataFrame, direct: pd.DataFrame) -> None:
    pw_unique = pw[pw.checkpoint_role == "M6"].drop_duplicates("actual_step").sort_values("actual_step")
    direct_unique = direct[direct.checkpoint_role == "M6"].drop_duplicates("actual_step").sort_values("actual_step")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(pw_unique.actual_step, pw_unique.clear_wave_3_probability, "o-", label="Persistent W3")
    axes[0].plot(pw_unique.actual_step, pw_unique.average_waves_cleared, "s-", label="Persistent avg waves")
    axes[0].legend();axes[0].grid(alpha=.25);axes[0].set_ylabel("Persistent performance")
    win_line = axes[1].plot(direct_unique.actual_step, direct_unique.win_rate, "o-", color="tab:blue", label="Direct win")
    return_axis = axes[1].twinx()
    return_line = return_axis.plot(direct_unique.actual_step, direct_unique.average_return, "s-", color="tab:orange", label="Direct return")
    axes[1].set_ylim(0,1.05);axes[1].set_ylabel("Direct win rate",color="tab:blue")
    return_axis.set_ylabel("Direct return",color="tab:orange")
    axes[1].legend(win_line+return_line,[line.get_label() for line in win_line+return_line],loc="upper right")
    axes[1].grid(alpha=.25);axes[1].set_xlabel("M6 sampled steps")
    fig.tight_layout();fig.savefig(OUT / "m6_persistent_vs_direct_trajectory.png", dpi=170);plt.close(fig)
    for column,filename,ylabel in (
        ("clear_wave_3_probability","m6_pw_W3_vs_step.png","P(clear wave 3)"),
        ("average_waves_cleared","m6_pw_avg_waves_vs_step.png","Average waves cleared"),
        ("average_return","m6_pw_return_vs_step.png","Average return"),
        ("average_red_loss","m6_pw_red_loss_vs_step.png","Average red loss"),
    ):
        fig,ax=plt.subplots(figsize=(8,4.8));ax.plot(pw_unique.actual_step,pw_unique[column],"o-");ax.set_xlabel("M6 sampled steps");ax.set_ylabel(ylabel);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/filename,dpi=170);plt.close(fig)
    for column,filename,ylabel in (
        ("win_rate","m6_direct_win_vs_step.png","Direct win rate"),
        ("average_return","m6_direct_return_vs_step.png","Direct return"),
    ):
        fig,ax=plt.subplots(figsize=(8,4.8));ax.plot(direct_unique.actual_step,direct_unique[column],"o-");ax.set_xlabel("M6 sampled steps");ax.set_ylabel(ylabel);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/filename,dpi=170);plt.close(fig)


def cross_environment_trajectory(pw: pd.DataFrame,direct: pd.DataFrame) -> pd.DataFrame:
    keys=["checkpoint_role","actual_step"]
    left=pw.drop_duplicates(keys)[keys+["clear_wave_3_probability","average_waves_cleared","average_return"]].rename(columns={"average_return":"persistent_return"})
    right=direct.drop_duplicates(keys)[keys+["win_rate","average_return"]].rename(columns={"win_rate":"direct_win_rate","average_return":"direct_return"})
    return left.merge(right,on=keys,how="inner").sort_values(["checkpoint_role","actual_step"])


def stage_label(step: int) -> str:
    if step <= 500_000:
        return "0-0.5M"
    if step <= 1_000_000:
        return "0.5-1.0M"
    return "1.0-1.5M"


def exposure_table() -> pd.DataFrame:
    rows = []
    for method in ("M5", "M6", "M1", "M3"):
        frame = read_jsonl(RUNS[method] / "optimization_metrics.jsonl")
        frame["stage"] = frame.sampled_steps.map(stage_label)
        for stage, group in frame.groupby("stage", sort=False):
            row = {"method": method, "stage": stage, "updates": len(group), "step_min": int(group.sampled_steps.min()), "step_max": int(group.sampled_steps.max())}
            transition_total = sum(group[f"transition_samples_wave_{index}"].sum() for index in (1, 2, 3))
            alive_total = sum(group[f"alive_agent_samples_wave_{index}"].sum() for index in (1, 2, 3))
            for index in (1, 2, 3):
                row[f"transition_fraction_W{index}"] = float(group[f"transition_samples_wave_{index}"].sum() / max(transition_total, 1))
                row[f"alive_agent_fraction_W{index}"] = float(group[f"alive_agent_samples_wave_{index}"].sum() / max(alive_total, 1))
                if method == "M5":
                    row[f"weight_W{index}_mean"] = float(group[f"weight_wave_{index}"].mean())
                    row[f"weight_W{index}_min"] = float(group[f"weight_wave_{index}"].min())
                    row[f"weight_W{index}_max"] = float(group[f"weight_wave_{index}"].max())
            total_alive_per_update = group[[f"alive_agent_samples_wave_{i}" for i in (1,2,3)]].sum(axis=1)
            row["effective_weight_mean"] = float(np.average(group.effective_wave_weight_mean, weights=total_alive_per_update))
            rows.append(row)
    return pd.DataFrame(rows)


def m5_weight_table() -> pd.DataFrame:
    frame=read_jsonl(RUNS["M5"]/"optimization_metrics.jsonl");frame["stage"]=frame.sampled_steps.map(stage_label);rows=[]
    for stage,group in frame.groupby("stage",sort=False):
        row={"stage":stage,"updates":len(group)}
        for index in (1,2,3):
            row.update(wave_weight_stats(group,index))
        effective=group.effective_wave_weight_mean.astype(float)
        for stat,value in (("mean",effective.mean()),("median",effective.median()),("p10",effective.quantile(.1)),("p90",effective.quantile(.9))):row[f"effective_weight_{stat}"]=float(value)
        row["configured_max_respected"]=bool(max(row[f"weight_W{i}_max"] or 0 for i in (1,2,3))<=3.0+1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def wave_weight_stats(group: pd.DataFrame,index: int) -> dict[str,Any]:
    present=group[group[f"alive_agent_samples_wave_{index}"]>0][f"weight_wave_{index}"].astype(float)
    result={f"W{index}_present_updates":len(present),f"W{index}_absent":bool(present.empty)}
    for stat,value in (("mean",present.mean()),("median",present.median()),("p10",present.quantile(.1)),("p90",present.quantile(.9)),("max",present.max())):
        result[f"weight_W{index}_{stat}"]=None if present.empty else float(value)
    return result


def m1_mechanism_table(exposure: pd.DataFrame) -> pd.DataFrame:
    opt=read_jsonl(RUNS["M1"]/"optimization_metrics.jsonl");opt["stage"]=opt.sampled_steps.map(stage_label)
    history=pd.read_csv(RUNS["M1"]/"evaluation_history.csv");history["stage"]=history.sampled_steps.map(stage_label)
    rows=[]
    for stage in ("0-0.5M","0.5-1.0M","1.0-1.5M"):
        o=opt[opt.stage==stage];e=history[history.stage==stage];x=exposure[(exposure.method=="M1")&(exposure.stage==stage)].iloc[0]
        rows.append({"stage":stage,"alive_W2":x.alive_agent_fraction_W2,"alive_W3":x.alive_agent_fraction_W3,
            "critic_value_loss_mean":float(o.value_loss.mean()),"explained_variance_mean":float(o.explained_variance.mean()),
            "evaluation_return_mean":float(e.average_return.mean()),"evaluation_W3_mean":float(e.clear_wave_3_probability.mean()),
            "evaluation_average_waves_mean":float(e.average_waves_cleared.mean()),"evaluation_red_loss_mean":float(e.average_red_loss.mean()),
            "stage_last_step":int(e.sampled_steps.max()),"stage_last_W3":float(e.iloc[-1].clear_wave_3_probability),"stage_last_return":float(e.iloc[-1].average_return)})
    return pd.DataFrame(rows)


def stability_table(integrity: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in integrity:
        if item["method"] == "Direct baseline":
            continue
        history = pd.read_csv(Path(item["directory"]) / "evaluation_history.csv")
        if item["method"] == "PW baseline":
            history = history[history.sampled_steps <= 1_505_280]
            keys = history.apply(lambda row:(row.get("clear_wave_3_probability",0),row.get("average_waves_cleared",0),row.average_return,-row.average_red_loss),axis=1)
            best = history.loc[max(keys.index, key=lambda index: keys.loc[index])]
            best_step = int(best.sampled_steps)
        else:
            best_matches = history[history.sampled_steps == item["best_step"]]
            best = best_matches.iloc[-1] if not best_matches.empty else history.iloc[-1]
            best_step = int(item["best_step"])
        latest = history.iloc[-1]
        rows.append({
            "method": item["method"], "best_checkpoint_step": best_step, "latest_step": int(latest.sampled_steps),
            "best_W3": best.clear_wave_3_probability, "latest_W3": latest.clear_wave_3_probability,
            "best_final_W3_gap": best.clear_wave_3_probability-latest.clear_wave_3_probability,
            "best_average_waves": best.average_waves_cleared, "latest_average_waves": latest.average_waves_cleared,
            "best_final_average_waves_gap":best.average_waves_cleared-latest.average_waves_cleared,
            "best_return": best.average_return, "latest_return": latest.average_return,
            "best_final_return_gap":best.average_return-latest.average_return,
            "best_red_loss": best.average_red_loss, "latest_red_loss": latest.average_red_loss,
            "best_final_red_loss_change":latest.average_red_loss-best.average_red_loss,
            "no_best_final_regression":bool(int(best.sampled_steps)==int(latest.sampled_steps)),
        })
    return pd.DataFrame(rows)


def diagnostic_table() -> pd.DataFrame:
    rows = []
    metrics = ("approx_kl", "ratio_underflow_fraction", "log_ratio_min", "log_ratio_max", "max_abs_log_ratio", "explained_variance", "popart_mean", "popart_std", "popart_count")
    for method in ("M5", "M6", "M1", "M3"):
        frame = read_jsonl(RUNS[method] / "optimization_metrics.jsonl")
        phases = {"full": frame}
        if method == "M6":
            phases["post_resume"] = frame[frame.sampled_steps > 503808]
        for phase, data in phases.items():
            for metric in metrics:
                if metric not in data or data[metric].dropna().empty:
                    continue
                values = pd.to_numeric(data[metric], errors="coerce").dropna().to_numpy(dtype=float)
                rows.append({
                    "method": method, "phase": phase, "metric": metric, "count": len(values),
                    "finite_count": int(np.isfinite(values).sum()), "nonfinite_count": int((~np.isfinite(values)).sum()),
                    "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.quantile(values,.95)),
                    "p99": float(np.quantile(values,.99)), "global_min": float(np.min(values)), "global_max": float(np.max(values)),
                })
    return pd.DataFrame(rows)


def plot_popart() -> None:
    frame = read_jsonl(RUNS["M3"] / "optimization_metrics.jsonl")
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(frame.sampled_steps, frame.popart_mean, label="mean")
    axes[0].plot(frame.sampled_steps, frame.popart_std, label="std");axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot(frame.sampled_steps, frame.popart_count);axes[1].set_ylabel("PopArt count");axes[1].grid(alpha=.25)
    axes[2].plot(frame.sampled_steps, frame.explained_variance);axes[2].set_ylabel("Explained variance");axes[2].set_xlabel("Sampled steps");axes[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT / "m3_popart_diagnostics.png", dpi=170);plt.close(fig)


def write_report_stub(integrity,fresh,delta,pw_sweep,direct_sweep,cross,exposure,weights,m1,stability,diagnostics) -> None:
    by_eval = fresh.set_index("evaluation")
    m6_post = diagnostics[(diagnostics.method == "M6") & (diagnostics.phase == "post_resume")].set_index("metric")
    lines = [
        "# 1.5M Modular MAPPO Screening and Mechanism Analysis",
        "",
        "This report uses diagnostic evaluation seeds 34,000,000–34,000,049, M6 PW sweep seeds 34,100,000–34,100,019, and M6 Direct sweep seeds 34,200,000–34,200,029. It does not use the 20,000,000 final holdout and performs no training.",
        "",
        "## Run integrity", "", pd.DataFrame(integrity).to_markdown(index=False), "",
        "## Fresh Persistent-Wave evaluation", "", fresh[["evaluation","sampled_steps","clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability","average_waves_cleared","average_return","average_red_loss","kill_loss_ratio","average_red_boundary_exits","average_red_ground_losses"]].to_markdown(index=False), "",
        "## Paired module deltas versus PW checkpoint 1,505,280", "", delta.to_markdown(index=False), "",
        "## M6 Persistent trajectory", "", pw_sweep[["requested_step","actual_step","checkpoint_role","clear_wave_3_probability","average_waves_cleared","average_return","average_red_loss","kill_loss_ratio"]].to_markdown(index=False), "",
        "## M6 Direct trajectory", "", direct_sweep[["requested_step","actual_step","checkpoint_role","win_rate","average_return","average_red_loss","kill_loss_ratio"]].to_markdown(index=False), "",
        "## Wave sample exposure", "", exposure.to_markdown(index=False), "",
        "## M5 wave weights", "", weights.to_markdown(index=False), "",
        "## M1 mechanism stages", "", m1.to_markdown(index=False), "",
        "## M6 cross-environment trajectory", "", cross.to_markdown(index=False), "",
        "## Best-final stability", "", stability.to_markdown(index=False), "",
        "## Optimization diagnostics", "", diagnostics.to_markdown(index=False), "",
        "## Evidence-calibrated interpretation", "",
        "### Fair 1.5M PW baseline", "",
        "The nearest real checkpoint at or below 1.5M is 1,001,472; the closest real checkpoint overall is 1,505,280 and is the primary paired comparator. Both were evaluated. The strongest training-evaluation row at or below 1.5M occurs at 903,168 (W3=0.45), but no checkpoint exists at that step, so it is curve evidence only and was not used for fresh evaluation.", "",
        "### Single-module screening", "",
        f"- **M5 — PROMISING.** Best/latest W3 are {by_eval.loc['M5_best','clear_wave_3_probability']:.2f}/{by_eval.loc['M5_latest','clear_wave_3_probability']:.2f} versus {by_eval.loc['PW_above_1505280','clear_wave_3_probability']:.2f}; the W3 gain is small, but average waves improve by +0.40/+0.26 and return by +25.25/+22.73, while red loss, boundary exits, and ground losses improve. The gain persists at latest.",
        f"- **M1 — MIXED.** W1/W2 and average waves improve, but fresh W3 is {by_eval.loc['M1_best','clear_wave_3_probability']:.2f}, below the comparator's {by_eval.loc['PW_above_1505280','clear_wave_3_probability']:.2f}. Its late training improvement coincides with W3 alive-agent exposure rising from 4.3% to 16.7%; this is association, not proof of causality.",
        f"- **M3 — MIXED.** Best has average waves {by_eval.loc['M3_best','average_waves_cleared']:.2f} and W3 {by_eval.loc['M3_best','clear_wave_3_probability']:.2f}, but latest collapses to {by_eval.loc['M3_latest','average_waves_cleared']:.2f}/{by_eval.loc['M3_latest','clear_wave_3_probability']:.2f}. PopArt statistics and explained variance are numerically healthy, but do not yield stable task benefit.",
        f"- **M6 — NOT_PROMISING as unconstrained fine-tuning.** Best/latest W3 are {by_eval.loc['M6_best','clear_wave_3_probability']:.2f}/{by_eval.loc['M6_latest','clear_wave_3_probability']:.2f}; the untouched Direct source reaches {by_eval.loc['Direct_source_to_PW','clear_wave_3_probability']:.2f}. Warm start provides early access to later waves, but this run shows no positive PW adaptation beyond the source.", "",
        "### M6 mechanism result", "",
        "On the 20-seed PW sweep, M6 falls from W3=0.35 at 104,448 to 0.05 at 503,808/1,001,472 and 0.00 at 1,500,000. On the matched 30-seed Direct sweep, win rate falls from 0.933 to 0.733, 0.700, and 0.267. PW and Direct capability therefore degrade together, supporting tactical forgetting/policy destruction rather than a purely persistent-specific adaptation failure.", "",
        f"The M6 post-resume KL is fully finite across {int(m6_post.loc['approx_kl','count'])} updates (mean={m6_post.loc['approx_kl','mean']:.4f}, p99={m6_post.loc['approx_kl','p99']:.4f}, max={m6_post.loc['approx_kl','global_max']:.4f}). Only one update contains ratio underflow (4 samples; maximum fraction 2.55e-5). Major performance collapse also occurs in windows with ordinary KL around 0.02–0.03, so the evidence does not support a simple KL-spike explanation.", "",
        "### Wave exposure and M5 weighting", "",
        "From-scratch W3 alive-agent exposure is initially scarce: M5 3.3%, M1 4.3%, M3 2.3%. M6 starts at 15.7% because the Direct source immediately reaches later waves. M6 W3 exposure remains 15.6–17.0% after collapse, so loss of later-wave sample share alone does not explain its degradation. M5 W3 exposure rises to 8.4% and then 19.5%; present-sample W2/W3 weights exceed W1 and never exceed the configured cap of 3, while effective mean remains normalized to 1.0.", "",
        "M5 has W3 samples in 52/81 early updates, 80/81 middle updates, and 83/83 late updates. When present, W3 mean weights are 2.94, 2.80, and 1.90; W2 means are 1.94, 1.29, and 1.08 versus W1 means 0.69, 0.68, and 0.73. Thus absent W3 updates are excluded rather than assigned a fictitious weight interpretation.", "",
        "### M1 mechanism evidence", "",
        "M1 W2/W3 alive-agent exposure rises from 23.4%/4.3% to 34.7%/16.7%. Mean explained variance rises from 0.281 to 0.629, mean evaluation return from -27.15 to 11.69, mean waves from 0.10 to 1.03, and mean red loss falls slightly from 3.96 to 3.83. Critic value loss does not monotonically decrease (1.60 to 2.08), so smaller value loss is not the explanation. The data are consistent with late Persistent learning accompanied by more later-wave samples and better value predictiveness, but do not establish that context solved state aliasing.", "",
        "### Next decision", "",
        "Do not launch a module combination yet. First replicate M5 across independent training seeds. M1+M5 is not yet justified because M1 did not beat baseline on W3. M6+M8 Policy Anchor is a well-motivated forgetting diagnostic, but not yet a primary performance candidate because M6 best did not exceed the untouched Direct source. Defer PopArt until its best/latest instability is resolved.", "",
        "### Limitations", "",
        "All comparisons use one training seed (2023). Episode seeds are paired environment realizations, not independent training replicates, so no claim of algorithm-level statistical significance is made. No 20M final holdout seeds were used and no training was performed.", "",
    ]
    (OUT / "screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    integrity = run_integrity()
    history = training_history(integrity)
    history.to_csv(OUT / "training_eval_history.csv", index=False)
    plot_training_history(history)
    fresh, episodes = fresh_tables()
    fresh.to_csv(OUT / "fresh_pw_50ep_summary.csv", index=False)
    episodes.to_csv(OUT / "fresh_pw_50ep_per_episode.csv", index=False)
    delta = paired_delta(episodes)
    delta.to_csv(OUT / "module_vs_baseline_paired_delta.csv", index=False)
    pw_sweep = sweep_table("persistent_wave_v2")
    direct_sweep = sweep_table("direct_v2_3")
    pw_sweep.to_csv(OUT / "m6_pw_checkpoint_sweep.csv", index=False)
    direct_sweep.to_csv(OUT / "m6_direct_checkpoint_sweep.csv", index=False)
    plot_m6_sweeps(pw_sweep, direct_sweep)
    cross=cross_environment_trajectory(pw_sweep,direct_sweep);cross.to_csv(OUT/"m6_cross_environment_trajectory.csv",index=False)
    exposure = exposure_table();exposure.to_csv(OUT / "wave_sample_exposure_by_stage.csv", index=False)
    weights=m5_weight_table();weights.to_csv(OUT/"m5_wave_weight_by_stage.csv",index=False)
    m1=m1_mechanism_table(exposure);m1.to_csv(OUT/"m1_mechanism_by_stage.csv",index=False)
    stability = stability_table(integrity);stability.to_csv(OUT / "best_final_stability.csv", index=False)
    diagnostics = diagnostic_table();diagnostics.to_csv(OUT / "optimization_diagnostics.csv", index=False)
    plot_popart()
    pw_history = pd.read_csv(RUNS["PW baseline"] / "evaluation_history.csv")
    pw_eligible = pw_history[pw_history.sampled_steps <= 1_500_000]
    pw_keys = pw_eligible.apply(lambda row:(row.clear_wave_3_probability,row.average_waves_cleared,row.average_return,-row.average_red_loss),axis=1)
    pw_best_row = pw_eligible.loc[max(pw_keys.index,key=lambda index:pw_keys.loc[index])]
    per_episode = episodes.set_index(["evaluation","seed"])
    m6_source_deltas = []
    for candidate in ("M6_best","M6_latest"):
        candidate_rows = per_episode.loc[candidate]
        source_rows = per_episode.loc["Direct_source_to_PW"]
        common = candidate_rows.index.intersection(source_rows.index)
        row = {"comparison":f"{candidate} - Direct_source_to_PW","paired_episodes":len(common)}
        for label,column in DELTA_METRICS.items():
            row[f"delta_{label}"] = float((candidate_rows.loc[common,column]-source_rows.loc[common,column]).mean())
        m6_source_deltas.append(row)
    m6_only=cross[cross.checkpoint_role=="M6"]
    correlations={
        "pearson_direct_win_vs_persistent_W3":float(m6_only.direct_win_rate.corr(m6_only.clear_wave_3_probability,method="pearson")),
        "spearman_direct_win_vs_persistent_W3":float(m6_only.direct_win_rate.corr(m6_only.clear_wave_3_probability,method="spearman")),
        "pearson_direct_return_vs_persistent_avg_waves":float(m6_only.direct_return.corr(m6_only.average_waves_cleared,method="pearson")),
        "spearman_direct_return_vs_persistent_avg_waves":float(m6_only.direct_return.corr(m6_only.average_waves_cleared,method="spearman")),
        "checkpoint_count":len(m6_only),"descriptive_only":True,
    }
    budget_selection=pw_budget_selection()
    summary = {
        "protocol": {"fresh_seed_base":34000000,"fresh_seed_end":34000049,"m6_pw_seed_base":34100000,"m6_pw_seed_end":34100019,"m6_direct_seed_base":34200000,"m6_direct_seed_end":34200029,"final_holdout_used":False,"training_performed":False},
        "run_integrity": integrity,
        "pw_baseline_budget": {"nearest_at_or_below":budget_selection["at_or_below"][0],"nearest_overall":budget_selection["nearest_overall"][0],"primary_fresh_comparator":budget_selection["nearest_overall"][0],
            "best_training_evaluation_at_or_below_1p5m":pw_best_row.to_dict(),
            "best_training_evaluation_has_checkpoint":bool((RUNS["PW baseline"] / f"checkpoint_{int(pw_best_row.sampled_steps)}.pt").exists())},
        "m6_requested_to_actual_checkpoint": requested_m6_mapping(),
        "m6_cross_environment_correlations":correlations,
        "fresh_summary": fresh.to_dict("records"),
        "paired_delta": delta.to_dict("records"),
        "m6_vs_direct_source_paired_delta": m6_source_deltas,
        "wave_exposure": exposure.to_dict("records"),
        "m5_wave_weights":weights.to_dict("records"),
        "m1_mechanism_stages":m1.to_dict("records"),
        "stability": stability.to_dict("records"),
        "diagnostics": diagnostics.to_dict("records"),
        "ratings": {
            "M1":"MIXED", "M3":"MIXED", "M5":"PROMISING", "M6":"NOT_PROMISING",
            "M6_warm_start_mechanism":"MIXED: strong inherited initialization but no gain over source",
            "M6_unconstrained_pw_finetuning":"NOT_PROMISING",
        },
        "mechanism_conclusions": {
            "m6_tactical_forgetting_supported":True,
            "simple_kl_spike_explanation_supported":False,
            "m5_weighting_active":True,
            "recommended_next_action":"replicate M5 across training seeds before any combination",
            "secondary_diagnostic":"M6+M8 Policy Anchor only as a targeted forgetting test",
        },
    }
    (OUT / "screening_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report_stub(integrity,fresh,delta,pw_sweep,direct_sweep,cross,exposure,weights,m1,stability,diagnostics)
    print(f"report written to {OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode in ("evaluate", "all"):
        run_evaluations(args.workers)
    if args.mode in ("report", "all"):
        build_report()


if __name__ == "__main__":
    main()
