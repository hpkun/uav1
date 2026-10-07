"""Static fail-closed preflight for HTA-MAPPO V2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modules import HTA_MAPPO_VERSION, HTA_WORKER_CONSOLIDATION_VERSION


def identical(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def optimizer_parameter_ids(optimizer):
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for formal HTA V2 preflight")

    v1_config = load_config("configs/dev_hta_mappo_v1_3m.yaml")
    v2_config = load_config("configs/dev_hta_mappo_v2_3m.yaml")
    environment = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "experiments/hta_mappo_v2_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "experiments/current_seed_provenance.json").read_text(encoding="utf-8"))

    plain_path = ROOT / "outputs/diag_mappo_learnability/l3_seed5301/checkpoint_3000000.pt"
    if not plain_path.exists():
        raise FileNotFoundError("matched Plain exact-3M checkpoint is unavailable")
    plain = torch.load(plain_path, map_location="cuda", weights_only=False)
    embedded_environment = plain.get("extra", {}).get("environment_config")
    if not isinstance(embedded_environment, dict):
        raise RuntimeError("matched Plain checkpoint lacks embedded environment")

    v1_config["training"]["seed"] = 5301
    v2_config["training"]["seed"] = 5301
    v1 = build_modular_mappo_trainer(v1_config, "cuda", 256, 3_000_000)
    v1_cpu_rng = torch.get_rng_state().clone()
    v1_cuda_rng = torch.cuda.get_rng_state_all()
    v2 = build_modular_mappo_trainer(v2_config, "cuda", 256, 3_000_000)
    v2_cpu_rng = torch.get_rng_state().clone()
    v2_cuda_rng = torch.cuda.get_rng_state_all()
    architecture = checkpoint_architecture(v2)

    enabled = {
        key for key, value in v2_config["modules"].items()
        if isinstance(value, dict) and value.get("enabled", False)
    }
    training = v2_config["training"]
    hta = v2_config["modules"]["hierarchical_temporal_abstraction"]
    decay = v2_config["modules"]["actor_lr_decay"]
    consolidation = v2_config["modules"]["hta_worker_consolidation"]
    targets = (0, 600_000, 900_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000)
    expected_worker = (3e-4, 3e-4, 1e-4, 1e-4, 6.25e-5, 2.5e-5, 2.5e-5)
    expected_manager = (3e-4, 3e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    actual_manager = [v2.actor_lr_decay.learning_rate(step, 3e-4) for step in targets]
    actual_worker = [
        baseline * v2.hta_worker_consolidation.multiplier(step)
        for step, baseline in zip(targets, actual_manager)
    ]

    checks = {
        "environment_hash_identical": config_sha256(environment) == config_sha256(embedded_environment),
        "persistent_wave_v2": environment.get("environment_variant") == "persistent_wave_v2",
        "three_waves": environment["persistent_waves"]["total_waves"] == 3,
        "max_steps_3000": environment["simulation"]["max_steps"] == 3000,
        "observation_52": v2_config["network"]["observation_dim"] == 52,
        "action_3": v2_config["network"]["action_dim"] == 3,
        "num_envs_24": training["num_train_envs"] == 24,
        "rollout_256": training["rollout_steps"] == 256,
        "budget_3m": training["total_sampled_steps"] == 3_000_000,
        "hta_core_version_1": HTA_MAPPO_VERSION == 1 and architecture["hta_version"] == 1,
        "k16": hta["decision_interval_steps"] == 16,
        "four_options": hta["num_options"] == 4,
        "v1_core_config_identical": v1_config["modules"]["hierarchical_temporal_abstraction"] == hta,
        "enabled_modules_exact": enabled == {"actor_lr_decay", "hierarchical_temporal_abstraction", "hta_worker_consolidation"},
        "actor_decay_exact": decay == {"enabled": True, "schedule": "delayed_linear", "start_step": 600_000, "end_step": 900_000, "start_lr": 3e-4, "end_lr": 1e-4},
        "consolidation_exact": consolidation == {"enabled": True, "schedule": "progressive_linear", "start_step": 1_500_000, "end_step": 2_500_000, "final_lr_multiplier": 0.25},
        "consolidation_version_1": HTA_WORKER_CONSOLIDATION_VERSION == 1,
        "worker_lr_math_exact": actual_worker == list(expected_worker),
        "manager_lr_math_exact": actual_manager == list(expected_manager),
        "critic_lr_constant": training["critic_learning_rate"] == 3e-4 and all(group["lr"] == 3e-4 for group in v2.critic_optimizer.param_groups) and all(group["lr"] == 3e-4 for group in v2.manager_critic_optimizer.param_groups),
        "worker_actor_initial_exact": identical(v1.actor.state_dict(), v2.actor.state_dict()),
        "tactical_critic_initial_exact": identical(v1.critic.state_dict(), v2.critic.state_dict()),
        "manager_actor_initial_exact": identical(v1.manager_actor.state_dict(), v2.manager_actor.state_dict()),
        "manager_critic_initial_exact": identical(v1.manager_critic.state_dict(), v2.manager_critic.state_dict()),
        "global_cpu_rng_identical": torch.equal(v1_cpu_rng, v2_cpu_rng),
        "global_cuda_rng_identical": len(v1_cuda_rng) == len(v2_cuda_rng) and all(torch.equal(a, b) for a, b in zip(v1_cuda_rng, v2_cuda_rng)),
        "no_new_optimizer": optimizer_parameter_ids(v2.actor_optimizer) == {id(parameter) for parameter in v2.actor.trainable_policy_parameters()},
        "architecture_metadata": architecture.get("hta_worker_consolidation_enabled") is True and architecture.get("worker_final_lr_multiplier") == 0.25,
        "future_45m_untouched": manifest["future_final"] == {"seed_start": 45_000_000, "seed_end": 45_000_199, "executed": False} and registry["evaluation_ranges"]["45000000..45000199"].get("executed") is False and registry["evaluation_ranges"]["45000000..45000199"].get("status") == "CURRENT_FUTURE_FINAL_BLOCK",
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError("HTA V2 static preflight failed: " + ", ".join(failed))
    print("HTA_V2_STATIC_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
