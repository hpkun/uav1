#!/usr/bin/env python3
"""Train one fixed-source MAPPO single-wave horizon continuation branch."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.train_mappo import TeeOutput, write_yaml_snapshot
from tools.preflight_mappo_horizon_continuation import (
    EVALUATION_EPISODES,
    EVALUATION_SEED_BASE,
    EVALUATION_SEED_END,
    NUM_ENVS,
    RUNS,
    SOURCE,
    SOURCE_SHA256,
    SOURCE_STEP,
    TARGET_STEP,
    TRAINING_SEED,
    branch_episode_indices,
    derive_algorithm,
    derive_environment,
    ensure_output_absent,
    load_source,
    sha256,
    validate_algorithm_pair,
    validate_environment_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(3000, 1000), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    expected_output = RUNS[args.horizon].resolve()
    output = (ROOT / args.output_dir).resolve()
    if output != expected_output:
        raise RuntimeError(
            f"formal output-dir is fixed for H{args.horizon}: expected {expected_output}, got {output}"
        )
    ensure_output_absent(output)

    source_env, source_algorithm, state, _ = load_source("cpu")
    env = derive_environment(source_env, args.horizon)
    algorithm = derive_algorithm(source_algorithm)
    validate_environment_pair(
        source_env,
        derive_environment(source_env, 3000),
        derive_environment(source_env, 1000),
    )
    validate_algorithm_pair(source_algorithm, algorithm)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source checkpoint changed after protocol validation")

    output.mkdir(parents=True)
    write_yaml_snapshot(output / "env_config.yaml", env)
    write_yaml_snapshot(output / "algorithm_config.yaml", algorithm)
    manifest = runtime_source_manifest(ROOT)

    runner = MAPPOTrainingRunner(
        env,
        algorithm,
        NUM_ENVS,
        TARGET_STEP,
        "cuda",
        TRAINING_SEED,
        output,
        False,
    )
    extra = runner.trainer.load(SOURCE)
    if runner.trainer.sampled_steps != SOURCE_STEP:
        raise RuntimeError("source sampled_steps changed during load")
    if runner.trainer.vector_steps * NUM_ENVS != SOURCE_STEP:
        raise RuntimeError("source vector/sample counter mismatch during load")
    if extra != state.get("extra", {}):
        raise RuntimeError("source checkpoint extra metadata changed during load")
    actor_lrs = {float(group["lr"]) for group in runner.trainer.actor_optimizer.param_groups}
    critic_lrs = {float(group["lr"]) for group in runner.trainer.critic_optimizer.param_groups}
    if actor_lrs != {float(algorithm["training"]["actor_learning_rate"])}:
        raise RuntimeError(f"restored Actor optimizer LR mismatch: {actor_lrs}")
    if critic_lrs != {float(algorithm["training"]["critic_learning_rate"])}:
        raise RuntimeError(f"restored Critic optimizer LR mismatch: {critic_lrs}")

    reset_indices = branch_episode_indices(state)
    runner.vector.episode_indices = reset_indices.copy()
    runner.observations = runner.vector.reset()
    runner.alive_masks = runner.vector.current_alive_masks.copy()
    if not np.array_equal(runner.vector.episode_indices, reset_indices):
        raise RuntimeError("branch reset mutated episode_indices")
    runner.evaluation_history = []
    runner.best_evaluation = None
    runner.next_console_log = (
        SOURCE_STEP // runner.console_interval + 1
    ) * runner.console_interval
    runner.next_evaluation = (
        SOURCE_STEP // runner.evaluation_interval + 1
    ) * runner.evaluation_interval
    runner.next_checkpoint = (
        SOURCE_STEP // runner.checkpoint_interval + 1
    ) * runner.checkpoint_interval

    branch = {
        "experiment": "matched_h3000_vs_h1000_mappo_continuation",
        "parent_checkpoint": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "parent_checkpoint_sha256": sha256(SOURCE),
        "source_sampled_steps": SOURCE_STEP,
        "target_sampled_steps": TARGET_STEP,
        "continuation_sampled_steps": TARGET_STEP - SOURCE_STEP,
        "training_seed": TRAINING_SEED,
        "training_horizon": args.horizon,
        "source_horizon": 3000,
        "total_waves": 1,
        "reset_semantics": "episode_indices = source episode_indices + 1, then fresh reset",
        "not_bitwise_continuation_of_original_run": True,
        "restored_states": {
            "actor": True,
            "critic": True,
            "actor_optimizer": True,
            "critic_optimizer": True,
            "sampled_steps": True,
            "vector_steps": True,
            "ppo_update_count": True,
            "actor_update_count": True,
            "critic_update_count": True,
            "episode_indices_source": True,
            "live_environments": False,
            "global_rng_state": False,
        },
        "source_episode_indices": state["extra"]["episode_indices"],
        "branch_reset_episode_indices": reset_indices.tolist(),
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "evaluation_seed_end": EVALUATION_SEED_END,
        "evaluation_episodes": EVALUATION_EPISODES,
        "source_environment_config_sha256": config_sha256(source_env),
        "branch_environment_config_sha256": config_sha256(env),
        "source_algorithm_config_sha256": config_sha256(source_algorithm),
        "branch_algorithm_config_sha256": config_sha256(algorithm),
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
    }
    (output / "branch_from.json").write_text(
        json.dumps(branch, indent=2), encoding="utf-8"
    )

    startup = runner.startup_summary()
    run_config = {
        "device": "cuda",
        "seed": TRAINING_SEED,
        "num_envs": NUM_ENVS,
        "total_sampled_steps": TARGET_STEP,
        "source_sampled_steps": SOURCE_STEP,
        "smoke": False,
        "environment_config_path": str((output / "env_config.yaml").resolve()),
        "algorithm_config_path": str((output / "algorithm_config.yaml").resolve()),
        "environment_variant": "persistent_wave_v2",
        "environment_version": str(env["environment_version"]),
        "algorithm": "MAPPO",
        "critic_type": "mlp",
        "training_horizon": args.horizon,
        "total_waves": 1,
        "actor_learning_rate": float(algorithm["training"]["actor_learning_rate"]),
        "critic_learning_rate": float(algorithm["training"]["critic_learning_rate"]),
        "actor_parameter_count": startup["actor_parameter_count"],
        "critic_parameter_count": startup["critic_parameter_count"],
        "total_parameter_count": startup["total_parameter_count"],
        "output_dir": str(output),
        "resume_checkpoint": str(SOURCE),
        "parent_checkpoint_sha256": sha256(SOURCE),
        "environment_config_sha256": config_sha256(env),
        "algorithm_config_sha256": config_sha256(algorithm),
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "runtime_source_manifest_file_count": manifest["runtime_source_manifest_file_count"],
        "runtime_source_manifest_files": manifest["runtime_source_manifest_files"],
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "evaluation_seed_end": EVALUATION_SEED_END,
        "evaluation_episodes": EVALUATION_EPISODES,
        "not_bitwise_continuation_of_original_run": True,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    with (output / "train.log").open("a", encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout, log_stream)):
            print(runner.start_log_line(), flush=True)
            print(
                f"[BRANCH] source_steps={SOURCE_STEP} | target_steps={TARGET_STEP} "
                f"| horizon={args.horizon} | reset=episode_indices_plus_one "
                "| bitwise_original_continuation=false",
                flush=True,
            )
            summary = runner.run()
            if runner.trainer.sampled_steps != TARGET_STEP:
                raise RuntimeError(
                    f"continuation did not stop exactly at {TARGET_STEP}: "
                    f"{runner.trainer.sampled_steps}"
                )
            runner.save_checkpoint(output / "final.pt")
            summary.update(
                {
                    "source_sampled_steps": SOURCE_STEP,
                    "target_sampled_steps": TARGET_STEP,
                    "training_horizon": args.horizon,
                    "parent_checkpoint_sha256": sha256(SOURCE),
                    "evaluation_seed_base": EVALUATION_SEED_BASE,
                    "evaluation_seed_end": EVALUATION_SEED_END,
                    "not_bitwise_continuation_of_original_run": True,
                }
            )
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(runner.done_log_line(summary), flush=True)


if __name__ == "__main__":
    main()
