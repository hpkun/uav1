"""Tiny CUDA-only end-to-end FireReady-Plain shape and checkpoint smoke."""
from __future__ import annotations

from copy import deepcopy
import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/fire_ready_plain_cuda_smoke")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for FireReady-Plain real smoke")
    output = ROOT / args.output_dir
    if output.exists():
        raise RuntimeError(f"refuse overwrite smoke output: {output}")
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_fire_ready_environment.yaml").read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs/dev_fire_ready_plain_3m.yaml")
    config = deepcopy(config)
    config["training"]["evaluation_interval_sampled_steps"] = 16
    config["implementation"]["checkpoint_interval_sampled_steps"] = 16
    config["runtime_logging"]["console_interval_sampled_steps"] = 16
    config["implementation"]["evaluation_seed_base"] = 88_000_100
    runner = ModularMAPPOTrainingRunner(env, config, num_envs=4, total_sampled_steps=32,
                                        device="cuda", seed=88_000_001,
                                        output_dir=output, smoke=True)
    result = runner.run()
    checkpoint = output / "final.pt"
    if not checkpoint.is_file():
        raise RuntimeError("smoke did not save final.pt")
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if int(state["extra"]["observation_dim"]) != 53:
        raise RuntimeError("checkpoint did not preserve 53D protocol")
    reloaded = ModularMAPPOTrainingRunner(env, config, num_envs=4, total_sampled_steps=32,
                                          device="cuda", seed=88_000_001,
                                          output_dir=output / "roundtrip", smoke=True,
                                          resume_mode=True)
    try:
        reloaded.trainer.load(checkpoint, strict_protocol=True, restore_rng=False)
        if reloaded.trainer.sampled_steps != 32:
            raise RuntimeError("checkpoint roundtrip sampled_steps mismatch")
    finally:
        reloaded.vector.close()
    if not result.get("latest_evaluation"):
        raise RuntimeError("smoke did not execute evaluation")
    print("FIRE_READY_PLAIN_REAL_SMOKE_PASS")


if __name__ == "__main__":
    main()
