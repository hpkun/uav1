#!/usr/bin/env python3
"""Fail-closed preflight for the matched Fixed10 Control versus RV V1 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.modular_mappo.protocol import validate_rv_branch, validate_rv_config_pair
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.train_modular_mappo import load_config

SEEDS = (5301, 5302, 5303)
SOURCE_STEP = 1_505_280
TARGET = 1_805_280
DEFAULT_OUTPUT = ROOT / "outputs/rv_fixed10_300k_preflight.json"
ENVIRONMENT = ROOT / "configs/persistent_wave_v2_environment.yaml"
CONTROL = ROOT / "configs/dev_rv_fixed10_control_300k.yaml"
TREATMENT = ROOT / "configs/dev_rv_mappo_v1_300k.yaml"


def source_path(seed: int) -> Path:
    return ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"


def formal_outputs(seed: int) -> tuple[Path, Path]:
    return (
        ROOT / f"outputs/dev_rv_control_seed{seed}_300k",
        ROOT / f"outputs/dev_rv_v1_seed{seed}_300k",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evaluation_protocol(control: dict, treatment: dict) -> dict:
    expected_validation = {
        "seed_start": 44_000_000,
        "seed_end": 44_000_049,
        "episodes": 50,
        "deterministic": True,
        "common_scenarios": True,
        "is_holdout": False,
    }
    expected_reserved = {
        "seed_start": 45_000_000,
        "seed_end": 45_000_199,
        "executed": False,
    }
    snapshots = []
    for name, config in (("Control", control), ("RV", treatment)):
        implementation = config.get("implementation", {})
        training = config.get("training", {})
        protocol = config.get("development_protocol", {})
        validation = protocol.get("validation")
        reserved = protocol.get("reserved_future_final_test")
        if implementation.get("evaluation_seed_base") != 44_000_000:
            raise RuntimeError(f"{name} evaluation_seed_base mismatch")
        if training.get("evaluation_episodes") != 50:
            raise RuntimeError(f"{name} evaluation_episodes mismatch")
        if training.get("evaluation_interval_sampled_steps") != 100_000:
            raise RuntimeError(f"{name} evaluation interval mismatch")
        if validation != expected_validation:
            raise RuntimeError(f"{name} development validation protocol mismatch: {validation}")
        if reserved != expected_reserved:
            raise RuntimeError(f"{name} reserved 45M protocol mismatch: {reserved}")
        snapshots.append({
            "evaluation_seed_base": implementation["evaluation_seed_base"],
            "evaluation_episodes": training["evaluation_episodes"],
            "evaluation_interval_sampled_steps": training["evaluation_interval_sampled_steps"],
            "validation": validation,
            "reserved_future_final_test": reserved,
        })
    if snapshots[0] != snapshots[1]:
        raise RuntimeError("Control and RV evaluation protocols differ")
    return snapshots[0]


def validate_plain_feedforward_architecture(architecture: dict) -> None:
    expected = {
        "actor_class": "ModularMAPPOActor",
        "critic_class": "ModularCentralizedCritic",
        "actor_input_dim": 52,
        "critic_input_dim": 52,
        "actor_context_dim": 0,
        "critic_context_dim": 0,
        "actor_gru_hidden_dim": 0,
        "critic_gru_hidden_dim": 0,
        "entity_attention_enabled": False,
        "entity_attention_mode": "disabled",
        "base_actor_frozen": False,
        "entity_mean_residual_enabled": False,
        "dual_bound_enabled": False,
        "log_std_source": "actor_head",
    }
    mismatches = {key: (architecture.get(key), value) for key, value in expected.items() if architecture.get(key) != value}
    if mismatches:
        raise RuntimeError(f"source network architecture mismatch: {mismatches}")


def build_report() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    environment = load_config(ENVIRONMENT)
    control = load_config(CONTROL)
    treatment = load_config(TREATMENT)
    validate_rv_config_pair(control, treatment)
    evaluation_protocol = validate_evaluation_protocol(control, treatment)

    expected_training = {
        "total_sampled_steps": TARGET, "num_train_envs": 24, "rollout_steps": 256,
        "gamma": 0.999, "gae_lambda": 0.95, "clip_ratio": 0.2,
        "entropy_coefficient": 0.01, "value_loss_coefficient": 0.5,
        "max_grad_norm": 0.5, "ppo_epochs": 10, "minibatch_size": 512,
    }
    for name, config, enabled in (("Control", control, False), ("RV", treatment, True)):
        mismatches = {key: (config["training"].get(key), value) for key, value in expected_training.items() if config["training"].get(key) != value}
        if mismatches:
            raise RuntimeError(f"{name} training protocol mismatch: {mismatches}")
        modules = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        wanted = ["actor_gradient_clipping", "actor_lr_decay"] + (["reference_variance"] if enabled else [])
        if modules != wanted:
            raise RuntimeError(f"{name} enabled modules mismatch: {modules}")
        rv = config["modules"]["reference_variance"]
        expected_rv = {"enabled": enabled, "mode": "frozen_source_state_dependent_variance",
                       "source_sampled_steps": 1_505_280, "freeze_current_log_std_head": True,
                       "behavior_uses_reference_mean": False}
        if rv != expected_rv:
            raise RuntimeError(f"{name} RV config mismatch")
        clip = config["modules"]["actor_gradient_clipping"]
        expected_clip = {"enabled": True, "mode": "actor_only_fixed_norm", "actor_max_grad_norm": 0.5, "critic_max_grad_norm_unchanged": True, "critic_max_grad_norm": 0.5}
        if clip != expected_clip:
            raise RuntimeError(f"{name} actor clip mismatch")

    env_identity = (environment.get("environment_variant"), environment["persistent_waves"]["total_waves"], environment["simulation"]["max_steps"], environment["scenario"]["team_size"])
    if env_identity != ("persistent_wave_v2", 3, 3000, 4):
        raise RuntimeError(f"environment mismatch: {env_identity}")
    if (control["network"]["observation_dim"], control["network"]["action_dim"], control["network"]["num_agents"]) != (52, 3, 4):
        raise RuntimeError("network dimensions mismatch")

    sources = []
    for seed in SEEDS:
        path = source_path(seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cuda", weights_only=False)
        extra = state.get("extra", {})
        rng = state.get("rng_state", {})
        if state.get("algorithm") != "modular_mappo":
            raise RuntimeError(f"seed{seed} source algorithm mismatch")
        if state.get("modular_mappo_impl_version") != MODULAR_MAPPO_IMPL_VERSION:
            raise RuntimeError(f"seed{seed} modular implementation version mismatch")
        if state.get("baseline_mappo_impl_version") != MAPPO_IMPL_VERSION:
            raise RuntimeError(f"seed{seed} baseline implementation version mismatch")
        if int(state.get("sampled_steps", -1)) != SOURCE_STEP or int(extra.get("training_seed", -1)) != seed:
            raise RuntimeError(f"seed{seed} source identity mismatch")
        if state.get("enabled_modules") != ["actor_lr_decay"]:
            raise RuntimeError(f"seed{seed} source modules mismatch")
        validate_plain_feedforward_architecture(extra.get("network_architecture", {}))
        if not state.get("actor_optimizer", {}).get("state") or not state.get("critic_optimizer", {}).get("state"):
            raise RuntimeError(f"seed{seed} optimizer state incomplete")
        required_rng = ("python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all", "trainer_permutation_rng_state")
        if any(key not in rng for key in required_rng):
            raise RuntimeError(f"seed{seed} RNG state incomplete")
        if float(state["actor_optimizer"]["param_groups"][0]["lr"]) != 1e-4:
            raise RuntimeError(f"seed{seed} source actor LR mismatch")
        runtime = {"training_seed": seed, "training_num_envs": 24, "training_smoke": False}
        validate_rv_branch(state, environment, control, runtime)
        validate_rv_branch(state, environment, treatment, runtime)
        sources.append({
            "training_seed": seed,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(path),
            "sampled_steps": int(state["sampled_steps"]),
            "status": "PASS",
        })

    existing = [str(path.relative_to(ROOT)).replace("\\", "/") for seed in SEEDS for path in formal_outputs(seed) if path.exists()]
    if existing:
        raise RuntimeError(f"formal output directories already exist: {existing}")
    manifest = runtime_source_manifest(ROOT)
    return {
        "status": "READY_FOR_RV_FIXED10_300K_SCREEN",
        "cuda": torch.cuda.get_device_name(0),
        "environment_config_sha256": config_sha256(environment),
        "control_algorithm_config_sha256": config_sha256(control),
        "treatment_algorithm_config_sha256": config_sha256(treatment),
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "runtime_source_manifest_files": manifest["runtime_source_manifest_files"],
        "sources": sources,
        "config_pair_only_three_registered_differences": "PASS",
        "evaluation_protocol": "PASS",
        "development_eval_seed_range": [44_000_000, 44_000_049],
        "development_eval_episodes": 50,
        "development_eval_deterministic": True,
        "evaluation_interval_sampled_steps": evaluation_protocol["evaluation_interval_sampled_steps"],
        "reserved_45m_untouched": "PASS",
        "expected_ppo_updates": 49,
        "expected_actor_optimizer_steps": 5860,
        "expected_critic_optimizer_steps": 5860,
        "formal_outputs_absent": "6/6",
        "uses_45m": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    if not args.print_only:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if output.exists():
            raise FileExistsError(f"refusing to overwrite preflight report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(output)
    print(json.dumps(report, indent=2))
    print(report["status"])


if __name__ == "__main__":
    main()

