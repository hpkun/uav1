"""Strict MADSAC checkpoint and experiment identity validation."""
from __future__ import annotations

from algorithm.common.protocol import config_sha256
from env.observation import observation_dim_from_config
from .trainer import MADSAC_IMPL_VERSION


def validate_madsac_config(env_config: dict, algorithm_config: dict) -> None:
    n, t, i = algorithm_config["network"], algorithm_config["training"], algorithm_config["implementation"]
    if algorithm_config.get("algorithm") != "madsac":
        raise RuntimeError("algorithm config is not madsac")
    if env_config.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError("MADSAC formal environment must be persistent_wave_v2")
    if int(env_config["persistent_waves"]["total_waves"]) != 3 or int(env_config["simulation"]["max_steps"]) != 3000:
        raise RuntimeError("MADSAC formal task must be three-wave max_steps=3000")
    expected = (observation_dim_from_config(env_config.get("observation", {})), 3, 4)
    actual = (int(n["observation_dim"]), int(n["action_dim"]), int(n["num_agents"]))
    if actual != expected or actual != (52, 3, 4):
        raise RuntimeError(f"MADSAC dimension mismatch: {actual} vs {expected}")
    if int(t["evaluation_episodes"]) != 50 or int(i["evaluation_seed_base"]) != 44_000_000:
        raise RuntimeError("MADSAC development evaluation must be fixed 50-episode 44M")
    if int(i["evaluation_policy_seed"]) != 770001 or i["evaluation_mode"] != "stochastic":
        raise RuntimeError("MADSAC primary evaluation protocol mismatch")
    if any(45_000_000 <= seed <= 45_000_199 for seed in range(int(i["evaluation_seed_base"]), int(i["evaluation_seed_base"]) + int(t["evaluation_episodes"]))):
        raise RuntimeError("45M future-final range is forbidden")


def validate_madsac_checkpoint(
    state: dict,
    env_config: dict,
    algorithm_config: dict,
    *,
    expected_training_seed: int | None = None,
) -> None:
    validate_madsac_config(env_config, algorithm_config)
    if state.get("algorithm") != "madsac" or state.get("implementation_version") != MADSAC_IMPL_VERSION:
        raise RuntimeError("MADSAC checkpoint identity/version mismatch")
    extra = state.get("extra", {})
    checks = {
        "training_seed": int(extra.get("training_seed", -1)),
        "environment_config_sha256": extra.get("environment_config_sha256"),
        "algorithm_config_sha256": extra.get("algorithm_config_sha256"),
        "observation_dim": int(extra.get("observation_dim", -1)),
        "action_dim": int(extra.get("action_dim", -1)),
        "num_agents": int(extra.get("num_agents", -1)),
    }
    expected = {
        "training_seed": int(
            algorithm_config["training"]["seed"]
            if expected_training_seed is None
            else expected_training_seed
        ),
        "environment_config_sha256": config_sha256(env_config),
        "algorithm_config_sha256": config_sha256(algorithm_config),
        "observation_dim": 52, "action_dim": 3, "num_agents": 4,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(f"MADSAC checkpoint {key} mismatch")
    if state.get("formal_exact_resume_supported") is not False or state.get("replay_buffer_included") is not False:
        raise RuntimeError("MADSAC checkpoint resume/replay provenance mismatch")


__all__ = ["validate_madsac_checkpoint", "validate_madsac_config"]
