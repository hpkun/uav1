"""CUDA construction and matched-branch parity preflight for SWGP-MAPPO V1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture, validate_swgp_branch
from algorithm.train_modular_mappo import load_config
from tools.audit_wave_gradient_conflict import state_sha256

SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"


def run_preflight(device: str = "cuda") -> dict:
    if device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory for SWGP preflight")
    env = load_config("configs/persistent_wave_v2_environment.yaml")
    plain = load_config("configs/diag_mappo_learnability_common_3m.yaml")
    control_cfg = load_config("configs/dev_swgp_branch_control_3m.yaml")
    swgp_cfg = load_config("configs/dev_swgp_branch_3m.yaml")
    state = torch.load(SOURCE, map_location=device, weights_only=False); extra = state.get("extra", {})
    expected_runtime = {"training_seed": 5302, "training_num_envs": 24, "training_smoke": False}
    control_validation = validate_swgp_branch(state, env, control_cfg, expected_runtime)
    swgp_validation = validate_swgp_branch(state, env, swgp_cfg, expected_runtime)
    control = build_modular_mappo_trainer(control_cfg, device, 256, 3_000_000)
    swgp = build_modular_mappo_trainer(swgp_cfg, device, 256, 3_000_000)
    control.load(SOURCE, strict_protocol=False, restore_rng=True)
    control_rng = control.rng.bit_generator.state == state["rng_state"]["trainer_permutation_rng_state"]
    swgp.load(SOURCE, strict_protocol=False, restore_rng=True)
    swgp_rng = swgp.rng.bit_generator.state == state["rng_state"]["trainer_permutation_rng_state"]
    source_actor = state_sha256(state["actor"]); source_critic = state_sha256(state["critic"])
    source_actor_opt = state_sha256(state["actor_optimizer"]); source_critic_opt = state_sha256(state["critic_optimizer"])
    hyper = swgp_cfg["training"]
    enabled_control = sorted(k for k, v in control_cfg["modules"].items() if isinstance(v, dict) and v.get("enabled", False))
    enabled_swgp = sorted(k for k, v in swgp_cfg["modules"].items() if isinstance(v, dict) and v.get("enabled", False))
    plain_arch = checkpoint_architecture(control); swgp_arch = checkpoint_architecture(swgp)
    source_env = extra.get("environment_config", {})
    checks = {
        "source_exact_step": int(state.get("sampled_steps", -1)) == 1_505_280,
        "source_seed5302": int(extra.get("training_seed", -1)) == 5302,
        "environment_exact": env.get("environment_variant") == "persistent_wave_v2" and env["persistent_waves"]["total_waves"] == 3 and env["simulation"]["max_steps"] == 3000,
        "environment_hash_unchanged": config_sha256(env) == config_sha256(source_env),
        "reward_unchanged": config_sha256(env["reward"]) == config_sha256(source_env["reward"]),
        "observation_action_agents": swgp_cfg["network"]["observation_dim"] == 52 and swgp_cfg["network"]["action_dim"] == 3 and swgp_cfg["network"]["num_agents"] == 4,
        "hyperparameters_matched": (hyper["gamma"], hyper["gae_lambda"], hyper["clip_ratio"], hyper["entropy_coefficient"], hyper["value_loss_coefficient"], hyper["max_grad_norm"], hyper["rollout_steps"], hyper["ppo_epochs"], hyper["minibatch_size"], hyper["num_train_envs"], hyper["total_sampled_steps"], hyper["critic_learning_rate"]) == (.999, .95, .2, .01, .5, .5, 256, 10, 512, 24, 3_000_000, 3e-4),
        "control_modules_exact": enabled_control == ["actor_lr_decay"],
        "swgp_modules_exact": enabled_swgp == ["actor_lr_decay", "sequential_wave_gradient_projection"],
        "topology_identical": plain_arch == swgp_arch,
        "parameter_counts_identical": sum(p.numel() for p in control.actor.parameters()) == sum(p.numel() for p in swgp.actor.parameters()) and sum(p.numel() for p in control.critic.parameters()) == sum(p.numel() for p in swgp.critic.parameters()),
        "control_actor_exact_source": state_sha256(control.actor.state_dict()) == source_actor,
        "swgp_actor_exact_source": state_sha256(swgp.actor.state_dict()) == source_actor,
        "control_critic_exact_source": state_sha256(control.critic.state_dict()) == source_critic,
        "swgp_critic_exact_source": state_sha256(swgp.critic.state_dict()) == source_critic,
        "control_actor_optimizer_exact": state_sha256(control.actor_optimizer.state_dict()) == source_actor_opt,
        "swgp_actor_optimizer_exact": state_sha256(swgp.actor_optimizer.state_dict()) == source_actor_opt,
        "control_critic_optimizer_exact": state_sha256(control.critic_optimizer.state_dict()) == source_critic_opt,
        "swgp_critic_optimizer_exact": state_sha256(swgp.critic_optimizer.state_dict()) == source_critic_opt,
        "steps_and_update_counts_exact": all(getattr(trainer, name) == int(state[key]) for trainer in (control, swgp) for name, key in (("sampled_steps", "sampled_steps"), ("actor_update_count", "actor_updates"), ("critic_update_count", "critic_updates"), ("ppo_update_count", "ppo_updates"))),
        "source_rng_restored": control_rng and swgp_rng and control.rng_restore_metadata["rng_state_restored"] and swgp.rng_restore_metadata["rng_state_restored"],
        "terminal_actor_lr_preserved": control.actor_optimizer.param_groups[0]["lr"] == 1e-4 and swgp.actor_optimizer.param_groups[0]["lr"] == 1e-4,
        "swgp_counters_zero": all(value == 0 for key, value in swgp.sequential_wave_gradient_projection.state_dict().items() if key.endswith("count") or key.endswith("minibatches")),
        "matched_branch_validation": control_validation["matched_causal_continuation"] and swgp_validation["matched_causal_continuation"],
        "44m_reserved_development": swgp_cfg["implementation"]["evaluation_seed_base"] == 44_000_000,
        "45m_unused": swgp_cfg["development_protocol"]["reserved_future_final_test"]["executed"] is False,
    }
    passed = all(checks.values())
    report = {"status": "SWGP_BRANCH_PREFLIGHT_PASS" if passed else "SWGP_BRANCH_PREFLIGHT_FAIL",
              "source": str(SOURCE.relative_to(ROOT)), "checks": checks,
              "branch_semantics": "matched causal continuation; not a bitwise continuation of the historical Plain run because environment episode state is not checkpointed",
              "control_validation": control_validation, "swgp_validation": swgp_validation,
              "future_final_45m_used": False}
    print(json.dumps(report, indent=2)); return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda", choices=("cuda",)); args = parser.parse_args()
    raise SystemExit(0 if run_preflight(args.device)["status"].endswith("PASS") else 2)


if __name__ == "__main__": main()
