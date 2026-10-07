"""CUDA/static preflight for the matched 18-branch PWTR 300k ablation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import runtime_source_manifest
from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.train_modular_mappo import load_config

SEEDS = (5301, 5302, 5303)
SOURCE = 1_505_280
TARGET = 1_805_280
BRANCHES = ("plain", "stratified", "current_extra", "uniform_recent", "priority_recent", "full")
ENV_PATH = ROOT / "configs/persistent_wave_v2_environment.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for PWTR preflight")
    environment = yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))
    identity = (
        environment["environment_variant"], environment["persistent_waves"]["total_waves"],
        environment["simulation"]["max_steps"], environment["scenario"]["team_size"],
    )
    if identity != ("persistent_wave_v2", 3, 3000, 4):
        raise RuntimeError(f"PWTR environment mismatch: {identity}")
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in BRANCHES}
    expected_training = {
        "gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
        "entropy_coefficient": .01, "value_loss_coefficient": .5,
        "max_grad_norm": .5, "rollout_steps": 256, "ppo_epochs": 10,
        "minibatch_size": 512, "num_train_envs": 24,
        "evaluation_episodes": 50, "evaluation_interval_sampled_steps": 100000,
    }
    expected_modules = {
        "plain": ["actor_lr_decay"],
        **{name: ["actor_lr_decay", "persistent_wave_trajectory_replay"] for name in BRANCHES if name != "plain"},
    }
    for name, config in configs.items():
        if any(config["training"][key] != value for key, value in expected_training.items()):
            raise RuntimeError(f"{name}: training protocol mismatch")
        if (config["network"]["observation_dim"], config["network"]["action_dim"], config["network"]["num_agents"]) != (52, 3, 4):
            raise RuntimeError(f"{name}: network protocol mismatch")
        if int(config["implementation"]["evaluation_seed_base"]) != 44_000_000:
            raise RuntimeError(f"{name}: evaluation seed mismatch")
        branch = config["development_branch"]
        if (config["training"]["total_sampled_steps"], branch["source_sampled_steps"],
                branch["additional_sampled_steps"], branch["target_sampled_steps"]) != (TARGET, SOURCE, 300_000, TARGET):
            raise RuntimeError(f"{name}: branch budget mismatch")
        enabled = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        if enabled != expected_modules[name]:
            raise RuntimeError(f"{name}: enabled modules mismatch: {enabled}")

    source_hashes = {}
    validations = {}
    for seed in SEEDS:
        checkpoint = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_{SOURCE}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extra = state["extra"]
        if int(state["sampled_steps"]) != SOURCE or int(extra["training_seed"]) != seed:
            raise RuntimeError(f"seed{seed}: source identity mismatch")
        if state.get("enabled_modules") != ["actor_lr_decay"]:
            raise RuntimeError(f"seed{seed}: source is not Plain actor_lr_decay")
        if float(state["actor_optimizer"]["param_groups"][0]["lr"]) != 1e-4:
            raise RuntimeError(f"seed{seed}: source actor LR is not 1e-4")
        source_hashes[str(seed)] = {"absolute_path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}
        validations[str(seed)] = {
            name: validate_pwtr_branch(state, environment, config, {
                "training_seed": seed, "training_num_envs": 24, "training_smoke": False,
            })
            for name, config in configs.items()
        }

    registry = json.loads((ROOT / "experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    future = registry["evaluation_ranges"]["45000000..45000199"]
    if future.get("executed") is not False or future.get("status") != "CURRENT_FUTURE_FINAL_BLOCK":
        raise RuntimeError("45M final range is not untouched")
    outputs = [ROOT / f"outputs/dev_pwtr_{branch}_seed{seed}_300k" for seed in SEEDS for branch in BRANCHES]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(f"formal PWTR outputs already exist: {existing}")
    manifest = runtime_source_manifest(ROOT)
    result = {
        "status": "READY_FOR_PWTR_300K_ABLATION",
        "cuda": torch.cuda.get_device_name(0),
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "runtime_source_manifest_file_count": manifest["runtime_source_manifest_file_count"],
        "source_checkpoint_hashes": source_hashes,
        "validations": validations,
        "formal_output_count": 18,
        "formal_outputs_absent": True,
        "evaluation_seed_range": [44_000_000, 44_000_049],
        "reserved_45m_executed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
