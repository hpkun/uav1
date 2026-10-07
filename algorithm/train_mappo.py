"""Configuration-driven formal and smoke MAPPO runner."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import csv
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch

from algorithm.common.checkpoint import validate_checkpoint_for_resume
from algorithm.common.protocol import config_sha256
from algorithm.mappo.runner import MAPPOTrainingRunner


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


class TeeOutput:
    """Write entry-point progress to both the terminal and train.log."""

    def __init__(self, terminal, log_stream) -> None:
        self.terminal = terminal
        self.log_stream = log_stream

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_stream.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_stream.flush()


def default_output_dir(seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = PROJECT_ROOT / "outputs" / f"run_{timestamp}_seed_{seed}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def ensure_fresh_output_directory(output_dir: Path) -> None:
    """Create or accept an empty directory, rejecting any existing contents."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"fresh output_dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"fresh training output_dir is non-empty: {output_dir}; "
                "choose a new run directory"
            )
    else:
        output_dir.mkdir(parents=True)


def resolve_run_paths(
    output_arg: str | None,
    resume_arg: str | None,
    seed: int,
) -> tuple[Path, Path | None]:
    """Resolve the final run directory and optional resume checkpoint."""
    if resume_arg is None:
        output_dir = (
            default_output_dir(seed) if output_arg is None else resolved(output_arg)
        )
        ensure_fresh_output_directory(output_dir)
        return output_dir, None
    checkpoint = resolved(resume_arg).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    checkpoint_parent = checkpoint.parent
    output_dir = checkpoint_parent if output_arg is None else resolved(output_arg).resolve()
    if output_dir != checkpoint_parent:
        raise RuntimeError(
            "resume output_dir must equal checkpoint.parent: "
            f"expected {checkpoint_parent}, got {output_dir}"
        )
    return output_dir, checkpoint


def validate_resume_config_snapshots(
    output_dir: Path,
    env_config: dict,
    algorithm_config: dict,
) -> list[tuple[Path, dict]]:
    """Require exact immutable environment and algorithm snapshots."""
    missing: list[tuple[Path, dict]] = []
    for name, current in (
        ("env_config.yaml", env_config),
        ("algorithm_config.yaml", algorithm_config),
    ):
        path = output_dir / name
        if not path.exists():
            raise RuntimeError(f"resume run is missing required {name} snapshot")
        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
        if stored != current:
            raise RuntimeError(f"resume {name} mismatch with stored run snapshot")
    return missing


def write_yaml_snapshot(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def load_run_config(run_dir: Path) -> dict | None:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid resume run_config.json: {path}")
    return value


def resolve_runtime_settings(
    algorithm_config: dict,
    *,
    seed: int | None,
    num_envs: int | None,
    total_sampled_steps: int | None,
    device: str | None,
    smoke: bool | None,
    run_config: dict | None = None,
    checkpoint_state: dict | None = None,
) -> dict:
    """Resolve fresh CLI precedence or strict resume inheritance."""
    training = algorithm_config["training"]
    if run_config is None and checkpoint_state is None:
        effective_smoke = bool(smoke)
        return {
            "seed": int(training["seed"] if seed is None else seed),
            "num_envs": int(
                training["num_train_envs"] if num_envs is None else num_envs
            ),
            "total_sampled_steps": int(
                total_sampled_steps
                if total_sampled_steps is not None
                else (192 if effective_smoke else training["total_sampled_steps"])
            ),
            "device": str(training["device"] if device is None else device),
            "smoke": effective_smoke,
            "legacy_resume": False,
            "original": None,
            "extended_training_target": False,
        }

    state = checkpoint_state or {}
    checkpoint_steps = int(state.get("sampled_steps", 0))
    if run_config is None:
        missing = [
            name
            for name, value in (
                ("--seed", seed),
                ("--num-envs", num_envs),
                ("--total-sampled-steps", total_sampled_steps),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "legacy resume without run_config.json requires explicit "
                + ", ".join(missing)
            )
        effective = {
            "seed": int(seed),
            "num_envs": int(num_envs),
            "total_sampled_steps": int(total_sampled_steps),
            "device": str(training["device"] if device is None else device),
            "smoke": bool(smoke),
            "legacy_resume": True,
            "original": None,
            "extended_training_target": False,
        }
    else:
        required = ("seed", "num_envs", "total_sampled_steps", "smoke", "device")
        absent = [field for field in required if field not in run_config]
        if absent:
            raise RuntimeError(
                "resume run_config.json missing required fields: "
                + ", ".join(absent)
            )
        original = {
            "seed": int(run_config["seed"]),
            "num_envs": int(run_config["num_envs"]),
            "total_sampled_steps": int(run_config["total_sampled_steps"]),
            "smoke": bool(run_config["smoke"]),
            "device": str(run_config["device"]),
        }
        if seed is not None and int(seed) != original["seed"]:
            raise RuntimeError("resume seed mismatch with original run_config.json")
        if num_envs is not None and int(num_envs) != original["num_envs"]:
            raise RuntimeError("resume num_envs mismatch with original run_config.json")
        if smoke is not None and bool(smoke) != original["smoke"]:
            raise RuntimeError("resume smoke mode mismatch with original run_config.json")
        checkpoint_target = state.get("extra", {}).get(
            "training_total_sampled_steps", original["total_sampled_steps"]
        )
        current_target = max(original["total_sampled_steps"], int(checkpoint_target))
        requested_target = (
            current_target
            if total_sampled_steps is None
            else int(total_sampled_steps)
        )
        if requested_target < current_target:
            raise RuntimeError(
                "resume total_sampled_steps cannot be smaller than the existing "
                f"training target {current_target}"
            )
        effective = {
            "seed": original["seed"],
            "num_envs": original["num_envs"],
            "total_sampled_steps": requested_target,
            "device": original["device"] if device is None else str(device),
            "smoke": original["smoke"],
            "legacy_resume": False,
            "original": original,
            "extended_training_target": requested_target > current_target,
        }
    if effective["total_sampled_steps"] <= checkpoint_steps:
        raise RuntimeError(
            f"resume checkpoint already reached {checkpoint_steps} sampled steps; "
            "request a larger --total-sampled-steps target"
        )
    return effective


def checkpoint_sampled_steps(path: Path) -> int | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return int(state["sampled_steps"])
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def reject_stale_resume_checkpoint(run_dir: Path, checkpoint: Path) -> None:
    selected_steps = checkpoint_sampled_steps(checkpoint)
    if selected_steps is None:
        raise RuntimeError(f"resume checkpoint has no readable sampled_steps: {checkpoint}")
    candidates = list(run_dir.glob("checkpoint_*.pt")) + [run_dir / "latest.pt"]
    newer: list[tuple[Path, int]] = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.resolve() == checkpoint.resolve():
            continue
        steps = checkpoint_sampled_steps(candidate)
        if steps is not None and steps > selected_steps:
            newer.append((candidate, steps))
    if newer:
        newest_path, newest_steps = max(newer, key=lambda item: item[1])
        raise RuntimeError(
            "stale resume checkpoint rejected: selected step "
            f"{selected_steps}, but {newest_path.name} is at step {newest_steps}"
        )


def _backup_path(path: Path, timestamp: str) -> Path:
    return path.with_name(f"{path.stem}.pre_resume_{timestamp}{path.suffix}")


def prepare_resume_rollback(
    run_dir: Path, checkpoint: Path, checkpoint_steps: int
) -> dict:
    """Back up and remove durable records newer than the selected checkpoint."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    actions: list[str] = []
    maximum_logged = checkpoint_steps
    for name in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        steps = [int(record["sampled_steps"]) for record in records]
        if steps:
            maximum_logged = max(maximum_logged, max(steps))
        if any(step > checkpoint_steps for step in steps):
            backup = _backup_path(path, timestamp)
            shutil.copy2(path, backup)
            kept = [
                json.dumps(record)
                for record in records
                if int(record["sampled_steps"]) <= checkpoint_steps
            ]
            path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            actions.append(f"truncated {name}; backup={backup.name}")
    evaluation_path = run_dir / "evaluation_history.csv"
    if evaluation_path.exists():
        with evaluation_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "sampled_steps" not in fieldnames:
            raise RuntimeError("evaluation_history.csv missing sampled_steps")
        steps = [int(row["sampled_steps"]) for row in rows]
        if steps:
            maximum_logged = max(maximum_logged, max(steps))
        if any(step > checkpoint_steps for step in steps):
            backup = _backup_path(evaluation_path, timestamp)
            shutil.copy2(evaluation_path, backup)
            with evaluation_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    row for row in rows if int(row["sampled_steps"]) <= checkpoint_steps
                )
            actions.append(
                f"truncated evaluation_history.csv; backup={backup.name}"
            )
    best_path = run_dir / "best_eval.pt"
    if best_path.is_file() and best_path.resolve() != checkpoint.resolve():
        best_steps = checkpoint_sampled_steps(best_path)
        if best_steps is not None and best_steps > checkpoint_steps:
            backup = _backup_path(best_path, timestamp)
            best_path.replace(backup)
            actions.append(f"moved future best_eval.pt; backup={backup.name}")
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        backup = _backup_path(summary_path, timestamp)
        shutil.copy2(summary_path, backup)
        actions.append(f"backed up run_summary.json as {backup.name}")
    return {
        "rollback_performed": bool(actions),
        "rollback_from_max_logged_steps": maximum_logged,
        "rollback_actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-sampled-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--env-config", default="configs/combat_environment.yaml")
    parser.add_argument("--algorithm-config", default="configs/mappo.yaml")
    parser.add_argument("--smoke", action="store_true", default=None)
    args = parser.parse_args()
    env_path = resolved(args.env_config)
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    algorithm_path = resolved(args.algorithm_config)
    algorithm_config = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    configured_seed = int(algorithm_config["training"]["seed"])
    path_seed = configured_seed if args.seed is None else args.seed
    output_dir, resume_checkpoint = resolve_run_paths(
        args.output_dir, args.resume, path_seed
    )
    missing_snapshots: list[tuple[Path, dict]] = []
    resume_state = None
    stored_run_config = None
    if resume_checkpoint is not None:
        reject_stale_resume_checkpoint(output_dir, resume_checkpoint)
        missing_snapshots = validate_resume_config_snapshots(
            output_dir, env_config, algorithm_config
        )
        resume_state = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        validate_checkpoint_for_resume(resume_state, env_config, algorithm_config)
        stored_run_config = load_run_config(output_dir)
    runtime = resolve_runtime_settings(
        algorithm_config,
        seed=args.seed,
        num_envs=args.num_envs,
        total_sampled_steps=args.total_sampled_steps,
        device=args.device,
        smoke=args.smoke,
        run_config=stored_run_config,
        checkpoint_state=resume_state,
    )
    if resume_state is not None and stored_run_config is not None:
        extra = resume_state.get("extra", {})
        expected_checkpoint_protocol = {
            "training_seed": runtime["original"]["seed"],
            "training_num_envs": runtime["original"]["num_envs"],
            "training_smoke": runtime["original"]["smoke"],
            "training_gamma": float(algorithm_config["training"]["gamma"]),
            "environment_config_sha256": config_sha256(env_config),
            "algorithm_config_sha256": config_sha256(algorithm_config),
        }
        for field, expected in expected_checkpoint_protocol.items():
            if field in extra and extra[field] != expected:
                raise RuntimeError(
                    f"resume checkpoint {field} mismatch with original run protocol: "
                    f"expected {expected!r}, got {extra[field]!r}"
                )
    runner = MAPPOTrainingRunner(
        env_config,
        algorithm_config,
        runtime["num_envs"],
        runtime["total_sampled_steps"],
        runtime["device"],
        runtime["seed"],
        output_dir,
        runtime["smoke"],
    )
    if resume_checkpoint is None:
        write_yaml_snapshot(runner.output_dir / "env_config.yaml", env_config)
        write_yaml_snapshot(
            runner.output_dir / "algorithm_config.yaml", algorithm_config
        )
    else:
        for path, config in missing_snapshots:
            write_yaml_snapshot(path, config)
    startup = runner.startup_summary()
    run_config = {
        "device": startup["device"],
        "seed": startup["seed"],
        "num_envs": startup["num_envs_M"],
        "total_sampled_steps": startup["total_sampled_steps"],
        "smoke": runner.smoke,
        "environment_config_path": str(env_path),
        "algorithm_config_path": str(algorithm_path),
        "environment_version": str(env_config["environment_version"]),
        "algorithm": "MAPPO",
        "effective_hidden_dim": startup["effective_hidden_dim"],
        "critic_type":startup["critic_type"],
        "actor_parameter_count":startup["actor_parameter_count"],
        "critic_parameter_count":startup["critic_parameter_count"],
        "total_parameter_count":startup["total_parameter_count"],
        "output_dir": str(runner.output_dir.resolve()),
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "training_gamma": startup["gamma"],
        "environment_config_sha256": config_sha256(env_config),
        "algorithm_config_sha256": config_sha256(algorithm_config),
    }
    if resume_checkpoint is None:
        (runner.output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
    with (runner.output_dir / "train.log").open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout, log_stream)):
            print(runner.start_log_line(), flush=True)
            if resume_checkpoint is not None:
                runner.resume(resume_checkpoint)
                rollback = prepare_resume_rollback(
                    runner.output_dir,
                    resume_checkpoint,
                    int(resume_state.get("sampled_steps", 0)),
                )
                original = runtime["original"] or {
                    "seed": runtime["seed"],
                    "num_envs": runtime["num_envs"],
                    "total_sampled_steps": runtime["total_sampled_steps"],
                }
                resume_record = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "checkpoint": str(resume_checkpoint),
                    "checkpoint_sampled_steps": int(
                        resume_state.get("sampled_steps", 0)
                    ),
                    "original_seed": original["seed"],
                    "effective_seed": startup["seed"],
                    "original_num_envs": original["num_envs"],
                    "effective_num_envs": startup["num_envs_M"],
                    "original_total_sampled_steps": original[
                        "total_sampled_steps"
                    ],
                    "effective_total_sampled_steps": startup[
                        "total_sampled_steps"
                    ],
                    "seed": startup["seed"],
                    "num_envs": startup["num_envs_M"],
                    "total_sampled_steps": startup["total_sampled_steps"],
                    "device": startup["device"],
                    "extended_training_target": runtime[
                        "extended_training_target"
                    ],
                    "environment_config_sha256": config_sha256(env_config),
                    "algorithm_config_sha256": config_sha256(algorithm_config),
                    **rollback,
                }
                with (runner.output_dir / "resume_history.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(resume_record) + "\n")
            summary = runner.run()
            (runner.output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(runner.done_log_line(summary), flush=True)


if __name__ == "__main__":
    main()
