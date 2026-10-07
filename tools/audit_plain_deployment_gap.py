"""Read-only deterministic-vs-stochastic deployment audit for Plain MAPPO.

This command can only inspect the three completed Plain L3 runs and the
already-exposed 44M development scenarios.  It has no training/resume path and
does not permit evaluation seeds or checkpoints to be supplied from the CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Iterable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics, persistent_mission_metrics
from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_modular_checkpoint
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from env.factory import make_combat_environment


TRAINING_SEEDS = (5301, 5302, 5303)
EVALUATION_SEEDS = tuple(range(44_000_000, 44_000_050))
POLICY_RNG_SEEDS = (770001, 770002, 770003)
PLAIN_ROOT = ROOT / "outputs/diag_mappo_learnability"
ENV_CONFIG_PATH = ROOT / "configs/persistent_wave_v2_environment.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/plain_deployment_gap_audit"
JSON_NAME = "plain_deployment_gap_audit.json"
TEXT_NAME = "plain_deployment_gap_audit.txt"
REQUIRED_RUN_FILES = (
    "evaluation_history.csv", "optimization_metrics.jsonl",
    "training_metrics.jsonl", "run_summary.json", "best_eval.pt",
    "latest.pt", "final.pt", "checkpoint_3000000.pt",
    "algorithm_config.yaml", "env_config.yaml", "runtime_env_config.yaml",
    "run_config.json",
)
METRIC_NAMES = (
    "W1", "W2", "W3", "Q2", "Q3", "AverageWaves", "Return",
    "red_loss", "blue_loss", "boundary", "ground", "episode_length",
)
REPRODUCTION_METRICS = (
    "W1", "W2", "W3", "AverageWaves", "Return", "boundary", "ground",
)


class DeterministicReproductionMismatch(RuntimeError):
    """Fail-closed mismatch between the audit runner and saved evaluation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Plain MAPPO deployment-mode audit (44M only)."
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_evaluation_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if values != EVALUATION_SEEDS:
        raise RuntimeError("evaluation seeds must be exactly 44000000..44000049")
    if len(values) != 50 or any(seed >= 45_000_000 for seed in values):
        raise RuntimeError("45M future-final seeds are forbidden")
    return values


def prepare_output_paths(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    json_path, text_path = output_dir / JSON_NAME, output_dir / TEXT_NAME
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if not overwrite and (json_path.exists() or text_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing audit report: {output_dir}")
    return json_path, text_path


def read_history(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise RuntimeError(f"empty evaluation history: {path}")
    rows = []
    for row in raw:
        converted = dict(row)
        converted["sampled_steps"] = int(float(row["sampled_steps"]))
        for key, value in row.items():
            if key == "sampled_steps" or value in (None, ""):
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                pass
        rows.append(converted)
    steps = [row["sampled_steps"] for row in rows]
    if len(steps) != len(set(steps)):
        raise RuntimeError(f"duplicate evaluation steps: {path}")
    return rows


def _checkpoint_step(state: dict, path: Path) -> int:
    if "sampled_steps" not in state:
        raise RuntimeError(f"checkpoint lacks sampled_steps: {path}")
    return int(state["sampled_steps"])


def load_checkpoint_metadata(path: Path) -> dict:
    """Load checkpoint metadata without restoring any RNG or model state."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older supported PyTorch versions.
        return torch.load(path, map_location="cpu")


def select_checkpoints(
    run_dir: Path,
    history: list[dict],
    metadata_reader: Callable[[Path], dict] = load_checkpoint_metadata,
) -> dict:
    """Select STRONG/COLLAPSE/FINAL using only frozen automatic rules."""
    run_dir = Path(run_dir)
    by_step = {int(row["sampled_steps"]): row for row in history}
    periodic: dict[int, Path] = {}
    for path in sorted(run_dir.glob("checkpoint_*.pt")):
        state = metadata_reader(path)
        step = _checkpoint_step(state, path)
        if path.stem != f"checkpoint_{step}":
            raise RuntimeError(f"checkpoint filename/metadata step mismatch: {path}")
        if step in periodic:
            raise RuntimeError(f"duplicate checkpoint sampled_steps={step}")
        periodic[step] = path

    final_path = run_dir / "checkpoint_3000000.pt"
    if 3_000_000 not in periodic or periodic[3_000_000] != final_path:
        raise RuntimeError("FINAL checkpoint_3000000.pt is missing")

    eligible = {
        step: path for step, path in periodic.items()
        if step >= 900_000 and step in by_step
    }
    if not eligible:
        raise RuntimeError("no >=900k periodic checkpoint has an exact evaluation row")

    best_path = run_dir / "best_eval.pt"
    best_state = metadata_reader(best_path)
    best_step = _checkpoint_step(best_state, best_path)
    if best_step in by_step:
        strong_path, strong_step = best_path, best_step
        strong_reason = "best_eval.pt with exact evaluation_history row"
    else:
        strong_step = max(
            eligible,
            key=lambda step: (float(by_step[step]["average_waves_cleared"]), step),
        )
        strong_path = eligible[strong_step]
        strong_reason = (
            "best_eval.pt lacked an exact evaluation row; selected maximum "
            "AverageWaves among >=900k exact periodic checkpoints"
        )

    ordered_collapse = sorted(
        eligible,
        key=lambda step: (float(by_step[step]["average_waves_cleared"]), step),
    )
    collapse_step = next(
        (step for step in ordered_collapse if step not in {strong_step, 3_000_000}),
        None,
    )
    if collapse_step is None:
        raise RuntimeError("no distinct COLLAPSE checkpoint remains after deduplication")

    selected = {
        "STRONG": {
            "path": strong_path, "sampled_steps": strong_step,
            "selection_reason": strong_reason, "evaluation_row": by_step[strong_step],
        },
        "COLLAPSE": {
            "path": eligible[collapse_step], "sampled_steps": collapse_step,
            "selection_reason": (
                "minimum deterministic AverageWaves among >=900k exact periodic "
                "checkpoints, excluding STRONG and FINAL"
            ),
            "evaluation_row": by_step[collapse_step],
        },
        "FINAL": {
            "path": final_path, "sampled_steps": 3_000_000,
            "selection_reason": "fixed exact-3M checkpoint_3000000.pt",
            "evaluation_row": by_step[3_000_000],
        },
    }
    # A best_eval.pt and a periodic/final file may represent the same saved
    # sampled step.  The protocol de-duplicates stages by sampled step, not by
    # filename, so such a policy is evaluated only once.
    unique: dict[int, dict] = {}
    for stage, row in selected.items():
        step = int(row["sampled_steps"])
        entry = unique.setdefault(step, {
            "path": row["path"], "sampled_steps": row["sampled_steps"],
            "stage_labels": [], "evaluation_row": row["evaluation_row"],
        })
        if entry["sampled_steps"] != row["sampled_steps"]:
            raise RuntimeError("deduplicated checkpoint has conflicting sampled_steps")
        entry["stage_labels"].append(stage)
        if stage == "FINAL":
            # Preserve the frozen rule that exact-3M deployment comes from the
            # specifically named periodic final checkpoint.
            entry["path"] = row["path"]
    return {"stages": selected, "unique": list(unique.values())}


def enabled_modules(config: dict) -> list[str]:
    return sorted(
        name for name, value in config.get("modules", {}).items()
        if isinstance(value, dict) and bool(value.get("enabled", False))
    )


def validate_plain_configs(env_config: dict, algorithm_config: dict) -> None:
    network = algorithm_config.get("network", {})
    training = algorithm_config.get("training", {})
    implementation = algorithm_config.get("implementation", {})
    if env_config.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError("Plain environment_variant must be persistent_wave_v2")
    if int(env_config.get("persistent_waves", {}).get("total_waves", -1)) != 3:
        raise RuntimeError("Plain audit requires exactly three waves")
    if int(env_config.get("simulation", {}).get("max_steps", -1)) != 3000:
        raise RuntimeError("Plain audit requires max_steps=3000")
    if bool(env_config.get("observation", {}).get("include_own_fire_ready", False)):
        raise RuntimeError("FireReady environment is forbidden in Plain audit")
    expected_network = {"observation_dim": 52, "action_dim": 3, "num_agents": 4}
    if any(int(network.get(key, -1)) != value for key, value in expected_network.items()):
        raise RuntimeError("Plain network must be 52D observation / 3D action / 4 agents")
    if int(training.get("total_sampled_steps", -1)) != 3_000_000:
        raise RuntimeError("Plain algorithm config is not exact-3M")
    if int(training.get("evaluation_episodes", -1)) != 50:
        raise RuntimeError("Plain algorithm config must use 50 evaluation episodes")
    if int(implementation.get("evaluation_seed_base", -1)) != 44_000_000:
        raise RuntimeError("Plain algorithm config must use 44M development evaluation")
    if enabled_modules(algorithm_config) != ["actor_lr_decay"]:
        raise RuntimeError("Plain baseline must enable only actor_lr_decay")


def validate_run_assets(run_dir: Path, training_seed: int) -> tuple[dict, dict, dict]:
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Plain run assets in {run_dir}: {missing}")
    if not list(run_dir.glob("checkpoint_*.pt")):
        raise FileNotFoundError(f"no periodic checkpoints in {run_dir}")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    env_config = yaml.safe_load((run_dir / "env_config.yaml").read_text(encoding="utf-8"))
    runtime_env = yaml.safe_load((run_dir / "runtime_env_config.yaml").read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load((run_dir / "algorithm_config.yaml").read_text(encoding="utf-8"))
    validate_plain_configs(env_config, algorithm_config)
    if runtime_env != env_config:
        raise RuntimeError("Plain declared/runtime environment mismatch")
    if run_config.get("algorithm") != "modular_mappo":
        raise RuntimeError("run algorithm is not modular_mappo")
    if int(run_config.get("seed", -1)) != int(training_seed):
        raise RuntimeError("run training_seed mismatch")
    if run_config.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError("run environment variant mismatch")
    if int(run_config.get("total_sampled_steps", -1)) != 3_000_000:
        raise RuntimeError("run budget is not exact-3M")
    if run_config.get("enabled_modules") != ["actor_lr_decay"]:
        raise RuntimeError("run is not the Plain actor-lr-decay baseline")
    architecture = run_config.get("network_architecture", {})
    if architecture.get("actor_input_dim") != 52 or architecture.get("actor_context_dim") != 0:
        raise RuntimeError("run is not a 52D feed-forward Plain actor")
    if architecture.get("entity_attention_enabled") or architecture.get("actor_gru_hidden_dim"):
        raise RuntimeError("run contains non-Plain actor modules")
    if run_config.get("development_branch") or run_config.get("branch_provenance"):
        raise RuntimeError("run is a development branch rather than Plain baseline")
    if int(summary.get("sampled_steps", -1)) != 3_000_000:
        raise RuntimeError("run_summary is not exact-3M complete")
    if run_config.get("environment_config_sha256") != config_sha256(env_config):
        raise RuntimeError("run environment hash mismatch")
    return env_config, algorithm_config, run_config


def validate_checkpoint_provenance(
    state: dict, path: Path, target_step: int, training_seed: int,
    env_config: dict, algorithm_config: dict,
) -> dict:
    validate_modular_checkpoint(state, env_config, algorithm_config)
    extra = state.get("extra", {})
    if state.get("algorithm") != "modular_mappo":
        raise RuntimeError("checkpoint is not modular_mappo")
    if int(state.get("sampled_steps", -1)) != int(target_step):
        raise RuntimeError(f"checkpoint sampled_steps mismatch: {path}")
    if int(extra.get("training_seed", -1)) != int(training_seed):
        raise RuntimeError(f"checkpoint training_seed mismatch: {path}")
    if extra.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError("checkpoint environment variant mismatch")
    for key, expected in (("observation_dim", 52), ("action_dim", 3), ("num_agents", 4)):
        if int(extra.get(key, -1)) != expected:
            raise RuntimeError(f"checkpoint {key} mismatch")
    if int(extra.get("current_total_waves", 3)) != 3:
        raise RuntimeError("checkpoint was not trained with three waves")
    method = extra.get("development_method")
    if method not in (None, "plain", "plain_mappo", "plain_mappo_baseline"):
        raise RuntimeError(f"checkpoint development method is not Plain: {method!r}")
    if state.get("enabled_modules") != ["actor_lr_decay"]:
        raise RuntimeError("checkpoint enabled modules are not Plain baseline")
    if any(bool(state.get("module_config", {}).get(name, {}).get("enabled", False))
           for name in state.get("module_config", {}) if name != "actor_lr_decay"):
        raise RuntimeError("checkpoint contains a non-Plain development module")
    architecture = extra.get("network_architecture", {})
    if architecture.get("actor_input_dim") != 52 or architecture.get("actor_context_dim") != 0:
        raise RuntimeError("checkpoint actor is not 52D feed-forward Plain")
    return {
        "path": str(path.relative_to(ROOT)), "sampled_steps": int(target_step),
        "training_seed": int(training_seed), "algorithm": "modular_mappo",
        "environment_variant": "persistent_wave_v2", "observation_dim": 52,
        "action_dim": 3, "num_agents": 4, "total_waves": 3,
        "max_steps": 3000, "enabled_modules": ["actor_lr_decay"],
        "development_method": method or "plain_baseline_identity_from_frozen_run_protocol",
        "restore_rng": False,
    }


def assert_environment_rng_isolated(env_config: dict) -> None:
    """Fail if changing Torch RNG changes reset/step of a seeded environment."""
    observations = []
    infos = []
    for torch_seed in (991, 992):
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        env = make_combat_environment(env_config)
        if not isinstance(getattr(env, "rng", None), np.random.Generator):
            raise RuntimeError("environment does not expose an instance-local NumPy Generator")
        obs, _ = env.reset(EVALUATION_SEEDS[0])
        obs, reward, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
        observations.append((obs.copy(), reward.copy(), terminated, truncated))
        infos.append((info["wave_index"], info["red_losses"], info["blue_losses"]))
    left, right = observations
    if not (np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])
            and left[2:] == right[2:] and infos[0] == infos[1]):
        raise RuntimeError("global Torch RNG affects environment transitions")


def set_policy_rng(seed: int, device: str) -> None:
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.manual_seed_all(int(seed))


def _episode_record(trainer, env_config: dict, evaluation_seed: int, deterministic: bool) -> dict:
    env = make_combat_environment(env_config)
    observation, _ = env.reset(int(evaluation_seed))
    alive = env.red_alive_mask.copy()
    actor_hidden, _ = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, dtype=np.float32)
    agent_returns = np.zeros(4, dtype=np.float64)
    total_waves = int(env_config["persistent_waves"]["total_waves"])
    while True:
        context = mission_context_numpy(
            trainer, np.asarray([env.wave_index]), np.asarray([total_waves]),
            env.blue_alive_mask[None], np.asarray([env.steps]), env.max_steps,
        )
        actions, actor_hidden = trainer.act(
            observation[None], alive[None], deterministic, False,
            context, actor_hidden, episode_mask,
        )
        observation, reward, terminated, truncated, info = env.step(actions[0])
        agent_returns += reward
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        episode_mask[:] = 1.0
        if terminated or truncated:
            team_return, mean_agent_return = episode_return_metrics(agent_returns)
            return {
                "episode_return": team_return,
                "mean_agent_episode_return": mean_agent_return,
                **info,
            }


def aggregate_records(records: list[dict]) -> dict:
    mean = lambda key: float(np.mean([record[key] for record in records]))
    result = {
        "Return": mean("episode_return"),
        "red_loss": mean("red_losses"), "blue_loss": mean("blue_losses"),
        "boundary": mean("red_boundary_exits"),
        "ground": mean("red_ground_losses"),
        "episode_length": mean("episode_length"),
        "evaluation_episodes": len(records),
        **persistent_mission_metrics(records),
    }
    result.update({
        "W1": result["clear_wave_1_probability"],
        "W2": result["clear_wave_2_probability"],
        "W3": result["clear_wave_3_probability"],
        "AverageWaves": result["average_waves_cleared"],
    })
    result["Q2"] = None if result["W1"] == 0 else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] == 0 else result["W3"] / result["W2"]
    return {key: result[key] for key in (*METRIC_NAMES, "evaluation_episodes")}


def evaluate_batch(
    trainer, env_config: dict, training_seed: int, stage_labels: list[str],
    deterministic: bool, repeat: int | None = None,
) -> dict:
    mode = "deterministic" if deterministic else "stochastic"
    records = []
    for index, evaluation_seed in enumerate(EVALUATION_SEEDS, 1):
        records.append(_episode_record(trainer, env_config, evaluation_seed, deterministic))
        if index % 10 == 0:
            repeat_text = "" if repeat is None else f" repeat={repeat}"
            print(
                f"[AUDIT] seed={training_seed} stage={'|'.join(stage_labels)} "
                f"mode={mode}{repeat_text} episode={index}/50",
                flush=True,
            )
    return aggregate_records(records)


def _history_metrics(row: dict) -> dict:
    mapping = {
        "W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability",
        "W3": "clear_wave_3_probability", "AverageWaves": "average_waves_cleared",
        "Return": "average_return", "boundary": "average_red_boundary_exits",
        "ground": "average_red_ground_losses",
    }
    return {target: float(row[source]) for target, source in mapping.items()}


def assert_deterministic_reproduction(
    actual: dict, history_row: dict, tolerance: float = 1e-6,
) -> None:
    expected = _history_metrics(history_row)
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in REPRODUCTION_METRICS
        if not math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=tolerance)
    }
    if mismatches:
        raise DeterministicReproductionMismatch(
            "PLAIN_DETERMINISTIC_REPRODUCTION_MISMATCH " + json.dumps(mismatches)
        )


def summarize_repeats(repeats: list[dict]) -> dict:
    result = {}
    for metric in METRIC_NAMES:
        values = [row[metric] for row in repeats if row[metric] is not None]
        result[metric] = None if not values else {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)), "max": float(max(values)),
        }
    return result


def compute_gaps(deterministic: dict, stochastic_summary: dict) -> dict:
    return {
        f"delta_{metric}": (
            None if deterministic[metric] is None or stochastic_summary[metric] is None
            else stochastic_summary[metric]["mean"] - deterministic[metric]
        )
        for metric in ("AverageWaves", "W1", "W2", "W3", "Q2", "Q3",
                       "Return", "boundary", "ground")
    }


def seed_deployment_gap(deterministic_aw: float, stochastic_aws: list[float]) -> bool:
    return (
        statistics.mean(stochastic_aws) - deterministic_aw >= 0.50
        and all(value - deterministic_aw >= 0.30 for value in stochastic_aws)
    )


def overall_label(per_seed_labels: dict[str, bool]) -> str:
    count = sum(bool(value) for value in per_seed_labels.values())
    if count >= 2:
        return "PLAIN_DETERMINISTIC_DEPLOYMENT_GAP_STRONG"
    if count == 1:
        return "PLAIN_DETERMINISTIC_DEPLOYMENT_GAP_WEAK_OR_MIXED"
    return "PLAIN_DETERMINISTIC_DEPLOYMENT_GAP_NOT_SUPPORTED"


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def validate_report_schema(report: dict) -> None:
    required = {
        "provenance", "selected_checkpoints", "selection_reason",
        "deterministic_results", "stochastic_repeat_results",
        "stochastic_summary", "gaps", "per_seed_labels", "overall_label",
        "evaluation_seed_range", "policy_rng_seeds", "45m_untouched",
    }
    missing = required - set(report)
    if missing:
        raise RuntimeError(f"audit report schema missing fields: {sorted(missing)}")
    if report["evaluation_seed_range"] != [44_000_000, 44_000_049]:
        raise RuntimeError("audit report seed range mismatch")
    if report["45m_untouched"] is not True:
        raise RuntimeError("audit report does not preserve 45M")


def render_text(report: dict) -> str:
    lines = [
        "Plain MAPPO deterministic-vs-stochastic deployment audit", "",
        f"Evaluation seeds: {report['evaluation_seed_range'][0]}..{report['evaluation_seed_range'][1]}",
        f"Policy RNG seeds: {report['policy_rng_seeds']}", "",
    ]
    for seed in TRAINING_SEEDS:
        key = str(seed)
        lines.append(f"Seed {seed}: deployment_gap={report['per_seed_labels'][key]}")
        for stage in ("STRONG", "COLLAPSE", "FINAL"):
            selected = report["selected_checkpoints"][key][stage]
            result_key = selected["result_key"]
            det = report["deterministic_results"][key][result_key]
            sto = report["stochastic_summary"][key][result_key]
            gap = report["gaps"][key][result_key]
            lines.append(
                f"  {stage} step={selected['sampled_steps']}: "
                f"det_AW={det['AverageWaves']:.4f}, "
                f"stoch_AW={sto['AverageWaves']['mean']:.4f} "
                f"[{sto['AverageWaves']['min']:.4f},{sto['AverageWaves']['max']:.4f}], "
                f"delta_AW={gap['delta_AverageWaves']:.4f}"
            )
        lines.append("")
    lines.extend((
        f"Overall: {report['overall_label']}",
        "Plain diagnostic results are intended for later comparison with the already completed FireReady deployment audit.",
        "FUTURE_FINAL_45M_UNTOUCHED",
    ))
    return "\n".join(lines) + "\n"


def run_audit(device: str, output_dir: Path, overwrite: bool) -> dict:
    validate_evaluation_seeds(EVALUATION_SEEDS)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    json_path, text_path = prepare_output_paths(output_dir, overwrite)
    canonical_env = yaml.safe_load(ENV_CONFIG_PATH.read_text(encoding="utf-8"))
    assert_environment_rng_isolated(canonical_env)

    report = {
        "provenance": {}, "selected_checkpoints": {}, "selection_reason": {},
        "deterministic_results": {}, "stochastic_repeat_results": {},
        "stochastic_summary": {}, "gaps": {}, "per_seed_labels": {},
        "overall_label": None,
        "evaluation_seed_range": [EVALUATION_SEEDS[0], EVALUATION_SEEDS[-1]],
        "evaluation_episodes_per_batch": 50,
        "policy_rng_seeds": list(POLICY_RNG_SEEDS),
        "45m_untouched": True, "read_only": True, "training_performed": False,
    }

    for training_seed in TRAINING_SEEDS:
        seed_key = str(training_seed)
        run_dir = PLAIN_ROOT / f"l3_seed{training_seed}"
        env_config, algorithm_config, run_config = validate_run_assets(run_dir, training_seed)
        if env_config != canonical_env:
            raise RuntimeError(f"saved Plain environment differs from canonical config: {run_dir}")
        history = read_history(run_dir / "evaluation_history.csv")
        selection = select_checkpoints(run_dir, history)
        report["selected_checkpoints"][seed_key] = {}
        report["selection_reason"][seed_key] = {}
        report["provenance"][seed_key] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "environment_config_sha256": config_sha256(env_config),
            "algorithm_config_sha256": run_config["algorithm_config_sha256"],
            "enabled_modules": ["actor_lr_decay"], "checkpoints": {},
        }
        for stage, row in selection["stages"].items():
            result_key = f"step_{row['sampled_steps']}"
            report["selected_checkpoints"][seed_key][stage] = {
                "path": str(row["path"].relative_to(ROOT)),
                "sampled_steps": row["sampled_steps"], "result_key": result_key,
            }
            report["selection_reason"][seed_key][stage] = row["selection_reason"]

        report["deterministic_results"][seed_key] = {}
        report["stochastic_repeat_results"][seed_key] = {}
        report["stochastic_summary"][seed_key] = {}
        report["gaps"][seed_key] = {}
        for selected in selection["unique"]:
            path, step = selected["path"], selected["sampled_steps"]
            state = load_checkpoint_metadata(path)
            provenance = validate_checkpoint_provenance(
                state, path, step, training_seed, env_config, algorithm_config,
            )
            report["provenance"][seed_key]["checkpoints"][f"step_{step}"] = provenance
            architecture = state["extra"]["network_architecture"]
            trainer = build_modular_mappo_trainer(
                algorithm_config, device=device,
                hidden_dim=int(architecture["hidden_dim"]),
                total_sampled_steps=3_000_000,
            )
            trainer.load(path, strict_protocol=True, restore_rng=False)
            if trainer.rng_restore_metadata.get("rng_state_restored"):
                raise RuntimeError("checkpoint load unexpectedly restored RNG state")
            trainer.actor.eval(); trainer.critic.eval()

            result_key = f"step_{step}"
            deterministic = evaluate_batch(
                trainer, env_config, training_seed, selected["stage_labels"], True,
            )
            assert_deterministic_reproduction(deterministic, selected["evaluation_row"])
            repeats = []
            for repeat, policy_seed in enumerate(POLICY_RNG_SEEDS, 1):
                set_policy_rng(policy_seed, device)
                repeats.append(evaluate_batch(
                    trainer, env_config, training_seed, selected["stage_labels"],
                    False, repeat,
                ))
            summary = summarize_repeats(repeats)
            report["deterministic_results"][seed_key][result_key] = deterministic
            report["stochastic_repeat_results"][seed_key][result_key] = repeats
            report["stochastic_summary"][seed_key][result_key] = summary
            report["gaps"][seed_key][result_key] = compute_gaps(deterministic, summary)
            del trainer, state
            if device == "cuda":
                torch.cuda.empty_cache()

        collapse = report["selected_checkpoints"][seed_key]["COLLAPSE"]["result_key"]
        det_aw = report["deterministic_results"][seed_key][collapse]["AverageWaves"]
        sto_aws = [
            row["AverageWaves"]
            for row in report["stochastic_repeat_results"][seed_key][collapse]
        ]
        report["per_seed_labels"][seed_key] = seed_deployment_gap(det_aw, sto_aws)

    report["overall_label"] = overall_label(report["per_seed_labels"])
    validate_report_schema(report)
    report = _json_ready(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(render_text(report), encoding="utf-8")
    print(render_text(report), end="", flush=True)
    print(f"JSON: {json_path}\nTXT: {text_path}", flush=True)
    return report


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_audit(args.device, args.output_dir, args.overwrite)
    except DeterministicReproductionMismatch:
        print("PLAIN_DETERMINISTIC_REPRODUCTION_MISMATCH", flush=True)
        raise
    except (FileNotFoundError, RuntimeError):
        print("PLAIN_DEPLOYMENT_AUDIT_INCONCLUSIVE", flush=True)
        raise


if __name__ == "__main__":
    main()
