"""Read-only protocol preflight for Wave-Balanced Curriculum MAPPO."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.train_modular_mappo import load_config
from env.config import load_config as load_env_config

PLAIN = ROOT / "configs/diag_mappo_learnability_common_3m.yaml"
CONFIGS = {
    "plain": PLAIN,
    "wb_only": ROOT / "configs/dev_wave_balance_3m.yaml",
    "wec_only": ROOT / "configs/dev_wave_entry_curriculum_3m.yaml",
    "proposed": ROOT / "configs/dev_wave_balanced_curriculum_3m.yaml",
}
ENV = ROOT / "configs/persistent_wave_v2_environment.yaml"
RESEARCH_MODULES = {
    "wave_context", "recurrent_memory", "popart", "multi_wave_reward",
    "wave_survival_pbrs", "warm_start", "curriculum", "policy_anchor",
    "entity_attention", "advantage_priority", "ppo_stabilization",
    "critic_mission_context", "mission_film", "actor_kl_guard",
    "inter_wave_credit", "counterfactual_inter_wave_credit",
    "boundary_redistributed_segment_credit", "hierarchical_temporal_abstraction",
    "hta_worker_consolidation",
}


def enabled(config: dict) -> set[str]:
    return {key for key, value in config["modules"].items() if bool(value.get("enabled", False))}


def normalized(config: dict, removable: set[str]) -> dict:
    result = deepcopy(config)
    result.pop("development_method", None)
    for key in removable:
        result.get("modules", {}).pop(key, None)
    return result


def validate() -> dict:
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    expected = {
        "plain": {"actor_lr_decay"},
        "wb_only": {"actor_lr_decay", "wave_balancing"},
        "wec_only": {"actor_lr_decay", "wave_entry_curriculum"},
        "proposed": {"actor_lr_decay", "wave_balancing", "wave_entry_curriculum"},
    }
    for name, config in configs.items():
        if enabled(config) != expected[name]:
            raise RuntimeError(f"{name} enabled modules mismatch: {sorted(enabled(config))}")
        if enabled(config) & RESEARCH_MODULES:
            raise RuntimeError(f"{name} enables a forbidden research module")
    plain = configs["plain"]
    if normalized(configs["wb_only"], {"wave_balancing"}) != normalized(plain, {"wave_balancing"}):
        raise RuntimeError("WB-only differs from Plain outside wave_balancing")
    if normalized(configs["wec_only"], {"wave_entry_curriculum"}) != normalized(plain, {"wave_entry_curriculum"}):
        raise RuntimeError("WEC-only differs from Plain outside wave_entry_curriculum")
    if normalized(configs["proposed"], {"wave_balancing", "wave_entry_curriculum"}) != normalized(plain, {"wave_balancing", "wave_entry_curriculum"}):
        raise RuntimeError("Proposed differs from Plain outside WB/WEC")
    env = load_env_config(ENV)
    registry = json.loads((ROOT / "experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    future = registry["evaluation_ranges"]["45000000..45000199"]
    training = plain["training"]; protocol = plain["development_protocol"]
    checks = {
        "observation_dim": plain["network"]["observation_dim"] == 52,
        "action_dim": plain["network"]["action_dim"] == 3,
        "num_agents": plain["network"]["num_agents"] == 4,
        "gamma": training["gamma"] == 0.999,
        "three_waves": env["persistent_waves"]["total_waves"] == 3,
        "max_steps": env["simulation"]["max_steps"] == 3000,
        "development_44m": protocol["validation"]["seed_start"] == 44_000_000 and protocol["validation"]["seed_end"] == 44_000_049,
        "future_45m_untouched": (
            protocol["reserved_future_final_test"] == {"seed_start": 45_000_000, "seed_end": 45_000_199, "executed": False}
            and future.get("status") == "CURRENT_FUTURE_FINAL_BLOCK"
            and future.get("executed") is False
            and all(int(value) == 0 for key, value in future.get("freshness_evidence", {}).items() if key.endswith("_hits"))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen protocol mismatch: {checks}")
    return {"status": "READY_FOR_WAVE_BALANCED_CURRICULUM_DEVELOPMENT",
            "configs": {name: {"path": str(CONFIGS[name]), "enabled_modules": sorted(enabled(config)),
                               "sha256": config_sha256(config)} for name, config in configs.items()},
            "environment": {"path": str(ENV), "sha256": config_sha256(env),
                "reward_sha256": config_sha256(env["reward"]), "blue_sha256": config_sha256(env["blue_policy"]),
                "weapon_sha256": config_sha256(env["weapon"]), "spawn_sha256": config_sha256(env["persistent_waves"]),
                "arena_sha256": config_sha256(env["arena"])}, "checks": checks}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
