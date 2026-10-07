#!/usr/bin/env python3
"""Strict preflight for the corrected 10-epoch Actor-only clip screen."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.train_modular_mappo import load_config
from algorithm.common.protocol import runtime_source_manifest
from algorithm.modular_mappo.protocol import (
    validate_actor_grad_clip_branch,
    validate_actor_grad_clip_config_pair,
)

SEEDS = (5301, 5302, 5303)
SOURCE = lambda seed: ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
OUTPUTS = lambda seed: (
    ROOT / f"outputs/dev_actor_clip05_fixed10_seed{seed}_300k",
    ROOT / f"outputs/dev_actor_clip10_fixed10_seed{seed}_300k",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_epoch_early_return() -> None:
    path = ROOT / "algorithm/modular_mappo/trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_update_flat")
    epoch_loops = [node for node in function.body if isinstance(node, ast.For)]
    if len(epoch_loops) != 1:
        raise RuntimeError("_update_flat must contain one top-level PPO epoch loop")
    if any(isinstance(node, ast.Return) for node in ast.walk(epoch_loops[0])):
        raise RuntimeError("_update_flat still returns from inside its PPO epoch loop")
    if not any(isinstance(node, ast.Return) for node in function.body):
        raise RuntimeError("_update_flat has no top-level return")


def main() -> None:
    assert_no_epoch_early_return()
    env = load_config(ROOT / "configs/persistent_wave_v2_environment.yaml")
    control = load_config(ROOT / "configs/dev_actor_grad_clip_05_control_300k.yaml")
    treatment = load_config(ROOT / "configs/dev_actor_grad_clip_10_300k.yaml")
    validate_actor_grad_clip_config_pair(control, treatment)
    expected = {
        "total_sampled_steps": 1_805_280, "num_train_envs": 24, "rollout_steps": 256,
        "gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
        "entropy_coefficient": .01, "value_loss_coefficient": .5,
        "max_grad_norm": .5, "ppo_epochs": 10, "minibatch_size": 512,
    }
    for name, config, limit in (("control", control, .5), ("treatment", treatment, 1.0)):
        enabled = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        if enabled != ["actor_gradient_clipping", "actor_lr_decay"]:
            raise RuntimeError(f"{name} enabled modules mismatch: {enabled}")
        for key, value in expected.items():
            if config["training"].get(key) != value:
                raise RuntimeError(f"{name} training.{key} mismatch")
        module = config["modules"]["actor_gradient_clipping"]
        if float(module["actor_max_grad_norm"]) != limit or float(module["critic_max_grad_norm"]) != .5:
            raise RuntimeError(f"{name} clipping mismatch")
    if env["environment_variant"] != "persistent_wave_v2" or env["persistent_waves"]["total_waves"] != 3:
        raise RuntimeError("environment mismatch")
    sources = []
    for seed in SEEDS:
        source = SOURCE(seed)
        state = torch.load(source, map_location="cpu", weights_only=False)
        rng = state.get("rng_state", {})
        if int(state["sampled_steps"]) != 1_505_280 or int(state["extra"]["training_seed"]) != seed:
            raise RuntimeError(f"source identity mismatch: {seed}")
        if state["enabled_modules"] != ["actor_lr_decay"]:
            raise RuntimeError(f"source modules mismatch: {seed}")
        if not state["actor_optimizer"]["state"] or not state["critic_optimizer"]["state"]:
            raise RuntimeError(f"source optimizer incomplete: {seed}")
        required_rng = ("python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all", "trainer_permutation_rng_state")
        if any(key not in rng for key in required_rng):
            raise RuntimeError(f"source RNG incomplete: {seed}")
        if float(state["actor_optimizer"]["param_groups"][0]["lr"]) != 1e-4:
            raise RuntimeError(f"source actor LR mismatch: {seed}")
        runtime = {"training_seed": seed, "training_num_envs": 24, "training_smoke": False}
        validate_actor_grad_clip_branch(state, env, control, runtime)
        validate_actor_grad_clip_branch(state, env, treatment, runtime)
        sources.append({"seed": seed, "sha256": sha256(source), "status": "PASS"})
    existing = [str(path) for seed in SEEDS for path in OUTPUTS(seed) if path.exists()]
    if existing:
        raise RuntimeError(f"fixed10 formal output directories already exist: {existing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    full_updates = 48
    full_minibatches = math.ceil(6144 / 512)
    partial_minibatches = math.ceil(5088 / 512)
    optimizer_steps = 10 * (full_updates * full_minibatches + partial_minibatches)
    if optimizer_steps != 5860:
        raise RuntimeError("300k optimizer-step arithmetic mismatch")
    report = {
        "status": "READY_FOR_ACTOR_GRAD_CLIP_FIXED10_300K_SCREEN",
        "cuda": torch.cuda.get_device_name(0), "flat_epoch_control_flow": "PASS",
        "configured_epochs": 10, "expected_ppo_updates": 49,
        "expected_actor_optimizer_steps": optimizer_steps,
        "expected_critic_optimizer_steps": optimizer_steps,
        "config_pair_single_variable": "PASS", "sources": sources,
        "runtime_source_manifest_sha256": runtime_source_manifest(ROOT)["runtime_source_manifest_sha256"],
        "runtime_trainer_sha256": sha256(ROOT / "algorithm/modular_mappo/trainer.py"),
        "new_formal_outputs_absent": "6/6", "old_outputs_allowed": True,
        "uses_45m": False,
    }
    print(json.dumps(report, indent=2))
    print(report["status"])


if __name__ == "__main__":
    main()
