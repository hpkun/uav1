#!/usr/bin/env python3
"""Fail-closed preflight for the matched H3000-vs-H1000 MAPPO branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.checkpoint import validate_checkpoint_for_resume
from algorithm.common.protocol import config_sha256, runtime_source_manifest
from tools.preflight_mappo_critic_baseline_1p5m import build_seeded_trainer


SOURCE_RUN = ROOT / "outputs/mappo_mlp_single_wave_seed5401_1p5m"
SOURCE = SOURCE_RUN / "best_eval.pt"
SOURCE_ENV = SOURCE_RUN / "env_config.yaml"
SOURCE_ALGORITHM = SOURCE_RUN / "algorithm_config.yaml"
SOURCE_RUN_CONFIG = SOURCE_RUN / "run_config.json"
SOURCE_SHA256 = "29f3e12a8774ea2ef2b37d1a2441f857c59ab4416d75dd795ea6e496ced7b6b2"
SOURCE_STEP = 1_105_920
TARGET_STEP = 1_500_000
TRAINING_SEED = 5401
NUM_ENVS = 24
EVALUATION_SEED_BASE = 49_000_000
EVALUATION_EPISODES = 50
EVALUATION_SEED_END = EVALUATION_SEED_BASE + EVALUATION_EPISODES - 1
RUNS = {
    3000: ROOT / "outputs/mappo_mlp_sw5401_best_cont_h3000_1p5m",
    1000: ROOT / "outputs/mappo_mlp_sw5401_best_cont_h1000_1p5m",
}
OUTPUT = ROOT / "outputs/mappo_horizon_continuation_preflight.json"


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected mapping YAML: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def validate_source_sha(actual: str) -> None:
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"source checkpoint SHA mismatch: expected {SOURCE_SHA256}, got {actual}"
        )


def validate_source_state(state: dict, env: dict, algorithm: dict) -> None:
    validate_checkpoint_for_resume(state, env, algorithm)
    extra = state.get("extra", {})
    expected_state = {
        "algorithm": "MAPPO",
        "critic_type": "mlp",
        "sampled_steps": SOURCE_STEP,
    }
    expected_extra = {
        "training_seed": TRAINING_SEED,
        "environment_variant": "persistent_wave_v2",
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
        "training_num_envs": NUM_ENVS,
        "environment_config_sha256": config_sha256(env),
        "algorithm_config_sha256": config_sha256(algorithm),
    }
    bad_state = {
        key: (state.get(key), value)
        for key, value in expected_state.items()
        if state.get(key) != value
    }
    bad_extra = {
        key: (extra.get(key), value)
        for key, value in expected_extra.items()
        if extra.get(key) != value
    }
    if bad_state or bad_extra:
        raise RuntimeError(
            f"source checkpoint identity mismatch: state={bad_state}, extra={bad_extra}"
        )
    if int(state.get("vector_steps", -1)) * NUM_ENVS != SOURCE_STEP:
        raise RuntimeError("source sampled/vector counters are inconsistent")
    source_episode_indices(state)


def source_episode_indices(state: dict) -> np.ndarray:
    previous = np.asarray(state.get("extra", {}).get("episode_indices", []), dtype=np.int64)
    if previous.shape != (NUM_ENVS,):
        raise RuntimeError("source episode_indices must contain exactly 24 entries")
    if np.any(previous < 0):
        raise RuntimeError("source episode_indices contains a negative value")
    return previous


def branch_episode_indices(state: dict) -> np.ndarray:
    return source_episode_indices(state) + 1


def derive_environment(source: dict, horizon: int) -> dict:
    if horizon not in RUNS:
        raise ValueError(f"unsupported horizon: {horizon}")
    value = deepcopy(source)
    value["simulation"]["max_steps"] = int(horizon)
    return value


def derive_algorithm(source: dict) -> dict:
    value = deepcopy(source)
    value["implementation"]["evaluation_seed_base"] = EVALUATION_SEED_BASE
    return value


def changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            paths.extend(changed_paths(a, b, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def validate_environment_pair(source: dict, h3000: dict, h1000: dict) -> None:
    identity = (
        source.get("environment_variant"),
        source.get("persistent_waves", {}).get("total_waves"),
        source.get("simulation", {}).get("max_steps"),
        source.get("scenario", {}).get("team_size"),
    )
    if identity != ("persistent_wave_v2", 1, 3000, 4):
        raise RuntimeError(f"source single-wave environment identity mismatch: {identity}")
    if h3000 != source:
        raise RuntimeError("H3000 branch environment differs from source")
    differences = changed_paths(source, h1000)
    if differences != ["simulation.max_steps"]:
        raise RuntimeError(f"H1000 environment has non-horizon differences: {differences}")
    if h1000["simulation"]["max_steps"] != 1000:
        raise RuntimeError("H1000 max_steps mismatch")


def validate_algorithm_pair(source: dict, branch: dict) -> None:
    differences = changed_paths(source, branch)
    if differences != ["implementation.evaluation_seed_base"]:
        raise RuntimeError(f"branch algorithm config has forbidden differences: {differences}")
    if branch.get("algorithm") != "MAPPO":
        raise RuntimeError("branch algorithm must be MAPPO")
    network = branch.get("network", {})
    if (
        network.get("observation_dim"),
        network.get("action_dim"),
        network.get("num_agents"),
        network.get("critic_type"),
    ) != (52, 3, 4, "mlp"):
        raise RuntimeError("branch MLP network identity mismatch")
    training = branch.get("training", {})
    expected_training = {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_loss_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 0.5,
        "rollout_steps": 256,
        "ppo_epochs": 10,
        "minibatch_size": 512,
        "num_train_envs": NUM_ENVS,
        "total_sampled_steps": TARGET_STEP,
        "evaluation_episodes": EVALUATION_EPISODES,
        "evaluation_interval_sampled_steps": 100_000,
        "device": "cuda",
    }
    bad = {
        key: (training.get(key), value)
        for key, value in expected_training.items()
        if training.get(key) != value
    }
    if bad:
        raise RuntimeError(f"branch training protocol mismatch: {bad}")
    if branch["implementation"].get("evaluation_seed_base") != EVALUATION_SEED_BASE:
        raise RuntimeError("branch evaluation seed base mismatch")


def ensure_output_absent(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing branch output: {path}")


def load_source(device: str = "cpu") -> tuple[dict, dict, dict, dict]:
    required = (SOURCE, SOURCE_ENV, SOURCE_ALGORITHM, SOURCE_RUN_CONFIG)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source artifacts missing: {missing}")
    validate_source_sha(sha256(SOURCE))
    env = load_yaml(SOURCE_ENV)
    algorithm = load_yaml(SOURCE_ALGORITHM)
    run_config = json.loads(SOURCE_RUN_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(run_config, dict):
        raise RuntimeError("source run_config.json is not a mapping")
    if run_config.get("environment_config_sha256") != config_sha256(env):
        raise RuntimeError("source run-local environment SHA mismatch")
    if run_config.get("algorithm_config_sha256") != config_sha256(algorithm):
        raise RuntimeError("source run-local algorithm SHA mismatch")
    if (
        run_config.get("seed"),
        run_config.get("critic_type"),
        run_config.get("environment_variant"),
    ) != (TRAINING_SEED, "mlp", "persistent_wave_v2"):
        raise RuntimeError("source run_config identity mismatch")
    state = torch.load(SOURCE, map_location=device, weights_only=False)
    validate_source_state(state, env, algorithm)
    return env, algorithm, state, run_config


def build_report() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    env, source_algorithm, state, _ = load_source("cuda")
    h3000, h1000 = derive_environment(env, 3000), derive_environment(env, 1000)
    branch_algorithm = derive_algorithm(source_algorithm)
    validate_environment_pair(env, h3000, h1000)
    validate_algorithm_pair(source_algorithm, branch_algorithm)
    existing = [str(path) for path in RUNS.values() if path.exists()]
    if existing:
        raise RuntimeError(f"formal continuation output directories already exist: {existing}")

    control = build_seeded_trainer(branch_algorithm, "cuda", TRAINING_SEED)
    treatment = build_seeded_trainer(branch_algorithm, "cuda", TRAINING_SEED)
    control.load(SOURCE)
    treatment.load(SOURCE)
    parity = {
        "actor": nested_equal(control.actor.state_dict(), treatment.actor.state_dict()),
        "critic": nested_equal(control.critic.state_dict(), treatment.critic.state_dict()),
        "actor_optimizer": nested_equal(
            control.actor_optimizer.state_dict(), treatment.actor_optimizer.state_dict()
        ),
        "critic_optimizer": nested_equal(
            control.critic_optimizer.state_dict(), treatment.critic_optimizer.state_dict()
        ),
        "counters": all(
            getattr(control, name) == getattr(treatment, name)
            for name in (
                "sampled_steps",
                "vector_steps",
                "ppo_update_count",
                "actor_update_count",
                "critic_update_count",
            )
        ),
    }
    if not all(parity.values()):
        raise RuntimeError(f"source branch parity failed: {parity}")
    if control.sampled_steps != SOURCE_STEP:
        raise RuntimeError("loaded source counter mismatch")
    previous = source_episode_indices(state)
    next_indices = branch_episode_indices(state)
    if not np.array_equal(next_indices, previous + 1):
        raise RuntimeError("branch reset episode-index rule failed")

    manifest = runtime_source_manifest(ROOT)
    return {
        "status": "READY_FOR_MAPPO_HORIZON_CONTINUATION",
        "cuda": torch.cuda.get_device_name(0),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": sha256(SOURCE),
        "source_sampled_steps": SOURCE_STEP,
        "target_sampled_steps": TARGET_STEP,
        "continuation_sampled_steps": TARGET_STEP - SOURCE_STEP,
        "source_identity": {
            "algorithm": "MAPPO",
            "critic_type": "mlp",
            "training_seed": TRAINING_SEED,
            "observation_dim": 52,
            "action_dim": 3,
            "num_agents": 4,
            "environment_variant": "persistent_wave_v2",
            "total_waves": 1,
            "max_steps": 3000,
        },
        "source_run_local": {
            "environment_config_file_sha256": sha256(SOURCE_ENV),
            "algorithm_config_file_sha256": sha256(SOURCE_ALGORITHM),
            "environment_config_sha256": config_sha256(env),
            "algorithm_config_sha256": config_sha256(source_algorithm),
        },
        "branch_config": {
            "algorithm_changed_paths": changed_paths(source_algorithm, branch_algorithm),
            "h3000_environment_changed_paths": changed_paths(env, h3000),
            "h1000_environment_changed_paths": changed_paths(env, h1000),
            "h3000_environment_config_sha256": config_sha256(h3000),
            "h1000_environment_config_sha256": config_sha256(h1000),
            "branch_algorithm_config_sha256": config_sha256(branch_algorithm),
        },
        "pre_intervention_parity": parity,
        "runtime_training_seed": TRAINING_SEED,
        "source_yaml_seed_preserved": source_algorithm["training"]["seed"],
        "reset_semantics": "episode_indices = source episode_indices + 1, then fresh reset",
        "not_bitwise_continuation_of_original_run": True,
        "episode_indices_restored_and_incremented": True,
        "evaluation_seed_range": [EVALUATION_SEED_BASE, EVALUATION_SEED_END],
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "formal_outputs_absent": "2/2",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build_report()
    if not args.print_only:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(report["status"])


if __name__ == "__main__":
    main()
