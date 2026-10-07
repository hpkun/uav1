"""Explicit resume and evaluation compatibility checks for MAPPO checkpoints."""
from __future__ import annotations

from typing import Any

from env.config import ENVIRONMENT_VERSION, validate_config
from env.combat_env import MultiUAVCombatEnv
from algorithm.common.protocol import config_sha256


def _checkpoint_extra(state: dict[str, Any]) -> dict[str, Any]:
    extra = state.get("extra", {})
    return extra if isinstance(extra, dict) else {}




def _configured_dimensions(
    algorithm_config: dict[str, Any],
) -> tuple[int, int, int]:
    network = algorithm_config["network"]
    return (
        int(network["observation_dim"]),
        int(network["action_dim"]),
        int(network["num_agents"]),
    )


def _inferred_checkpoint_dimensions(
    state: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    extra = _checkpoint_extra(state)
    observation_dim = extra.get("observation_dim")
    action_dim = extra.get("action_dim")
    num_agents = extra.get("num_agents")
    actor = state.get("actor", {})
    if isinstance(actor, dict):
        first_weight = actor.get("backbone.0.weight")
        mean_weight = actor.get("mean.weight")
        if observation_dim is None and hasattr(first_weight, "shape"):
            observation_dim = int(first_weight.shape[1])
        if action_dim is None and hasattr(mean_weight, "shape"):
            action_dim = int(mean_weight.shape[0])
    return (
        None if observation_dim is None else int(observation_dim),
        None if action_dim is None else int(action_dim),
        None if num_agents is None else int(num_agents),
    )


def _validate_common_checkpoint_contract(
    state: dict[str, Any],
    env_config: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> None:
    validate_config(env_config)
    from algorithm.mappo.trainer import MAPPO_IMPL_VERSION

    if state.get("algorithm") != "MAPPO":
        raise RuntimeError("checkpoint is not a MAPPO checkpoint")
    extra = _checkpoint_extra(state)
    expected_version = str(env_config.get("environment_version", ENVIRONMENT_VERSION))
    if expected_version != ENVIRONMENT_VERSION:
        raise RuntimeError("unsupported combat environment_version")
    checkpoint_version = extra.get("environment_version")
    if checkpoint_version != expected_version:
        raise RuntimeError(
            "checkpoint environment_version mismatch: expected "
            f"{expected_version!r}, got {checkpoint_version!r}; environment "
            "semantics are incompatible"
        )
    implementation_version = state.get("mappo_impl_version")
    if implementation_version != MAPPO_IMPL_VERSION:
        raise RuntimeError(
            "checkpoint MAPPO implementation mismatch: expected "
            f"{MAPPO_IMPL_VERSION}, got {implementation_version!r}"
        )
    configured_critic_type=str(algorithm_config.get("network",{}).get("critic_type","attention"))
    checkpoint_critic_type=str(state.get("critic_type",extra.get("critic_type","attention")))
    if checkpoint_critic_type!=configured_critic_type:
        raise RuntimeError(f"checkpoint critic_type mismatch: expected {configured_critic_type!r}, got {checkpoint_critic_type!r}")
    configured = _configured_dimensions(algorithm_config)
    environment = (
        MultiUAVCombatEnv.observation_dim,
        MultiUAVCombatEnv.action_dim,
        int(env_config.get("scenario", {}).get("team_size", MultiUAVCombatEnv.team_size)),
    )
    if configured != environment:
        raise RuntimeError(
            "algorithm/environment dimensions mismatch: configured "
            f"obs/action/agents={configured}, environment={environment}"
        )
    checkpoint = _inferred_checkpoint_dimensions(state)
    labels = ("observation_dim", "action_dim", "num_agents")
    for label, checkpoint_value, expected_value in zip(labels, checkpoint, configured):
        if checkpoint_value is not None and checkpoint_value != expected_value:
            raise RuntimeError(
                f"checkpoint {label} mismatch: expected {expected_value}, "
                f"got {checkpoint_value}"
            )


def validate_checkpoint_environment(
    state: dict[str, Any], env_config: dict[str, Any]
) -> None:
    validate_config(env_config)
    extra = _checkpoint_extra(state)
    version = extra.get("environment_version")
    if version != ENVIRONMENT_VERSION:
        raise RuntimeError(
            "checkpoint environment_version mismatch: expected "
            f"{ENVIRONMENT_VERSION}, got {version!r}; environment semantics "
            "are incompatible"
        )


def validate_checkpoint_for_resume(
    state: dict[str, Any],
    env_config: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> None:
    """Validate a checkpoint for strict continuation of the original run."""
    _validate_common_checkpoint_contract(state, env_config, algorithm_config)
    extra = _checkpoint_extra(state)
    for label, expected in (
        ("environment_config_sha256", config_sha256(env_config)),
        ("algorithm_config_sha256", config_sha256(algorithm_config)),
    ):
        recorded = extra.get(label)
        if recorded is not None and recorded != expected:
            raise RuntimeError(
                f"checkpoint {label} mismatch: expected {expected!r}, "
                f"got {recorded!r}"
            )


def validate_checkpoint_for_evaluation(
    state: dict[str, Any], env_config: dict[str, Any], algorithm_config: dict[str, Any],
) -> None:
    """Validate the combat and network contract for evaluation."""
    _validate_common_checkpoint_contract(state, env_config, algorithm_config)
    recorded = _checkpoint_extra(state).get("environment_config_sha256")
    if recorded is not None and recorded != config_sha256(env_config):
        raise RuntimeError("checkpoint environment_config_sha256 mismatch")



def evaluation_selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    """Rank checkpoints by Red win rate, return, then fewer Red losses."""
    return (float(record["win_rate"]), float(record["average_return"]),
            -float(record["average_red_loss"]))



__all__ = [
    "evaluation_selection_key",
    "validate_checkpoint_environment",
    "validate_checkpoint_for_evaluation",
    "validate_checkpoint_for_resume",
]
