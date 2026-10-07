"""Verify that training environments are persistent independent processes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from algorithm.common.vector_env import ParallelVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--env-config", default="configs/combat_environment.yaml")
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        raise ValueError("num-envs and steps must be positive")

    config_path = Path(args.env_config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actions = np.zeros((args.num_envs, 4, 3), dtype=np.float32)
    with ParallelVectorEnv(args.num_envs, config, base_seed=91_000_000) as vector:
        vector.reset()
        started = time.perf_counter()
        for _ in range(args.steps):
            vector.step_batch(actions, auto_reset=False)
        elapsed = time.perf_counter() - started
        print(f"parent_pid={os.getpid()}")
        print(f"backend={vector.backend}")
        print(f"workers={vector.num_workers}")
        print("worker_pids=" + ",".join(map(str, vector.worker_pids)))
        print("worker_environment_classes=" + ",".join(
            vector.worker_environment_classes
        ))
        print("worker_environment_variants=" + ",".join(
            vector.worker_environment_variants
        ))
        print(f"unique_worker_pids={len(set(vector.worker_pids))}")
        print(f"batch_steps={args.steps}")
        print(f"environment_transitions={args.steps * args.num_envs}")
        print(f"elapsed_seconds={elapsed:.6f}")
        print(
            "environment_transitions_per_second="
            f"{args.steps * args.num_envs / elapsed:.3f}"
        )


if __name__ == "__main__":
    main()
