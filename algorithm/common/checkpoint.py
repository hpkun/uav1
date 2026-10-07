"""Explicit resume and evaluation compatibility checks for MAPPO checkpoints."""
from __future__ import annotations

from typing import Any

from env.config import ENVIRONMENT_VERSION
from env.combat_env import MultiUAVCombatEnv
from env.observation import observation_dim_from_config
from algorithm.common.protocol import config_sha256


def _checkpoint_extra(state: dict[str, Any]) -> dict[str, Any]:
    extra = state.get("extra", {})
    return extra if isinstance(extra, dict) else {}


def _environment_variant(config: dict[str, Any]) -> str:
    return str(config.get("environment_variant", "direct_v2_3"))


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
    from algorithm.mappo.trainer import MAPPO_IMPL_VERSION

    if state.get("algorithm") != "MAPPO":
        raise RuntimeError("checkpoint is not a MAPPO checkpoint")
    extra = _checkpoint_extra(state)
    expected_version = str(env_config.get("environment_version", ENVIRONMENT_VERSION))
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
        observation_dim_from_config(env_config.get("observation", {})),
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
    extra = _checkpoint_extra(state)
    version = extra.get("environment_version")
    if version != ENVIRONMENT_VERSION:
        raise RuntimeError(
            "checkpoint environment_version mismatch: expected "
            f"{ENVIRONMENT_VERSION}, got {version!r}; environment semantics "
            "are incompatible"
        )
    expected_variant = _environment_variant(env_config)
    checkpoint_variant = str(extra.get("environment_variant", "direct_v2_3"))
    if checkpoint_variant != expected_variant:
        raise RuntimeError(
            "checkpoint environment_variant mismatch: expected "
            f"{expected_variant!r}, got {checkpoint_variant!r}"
        )


def validate_checkpoint_for_resume(
    state: dict[str, Any],
    env_config: dict[str, Any],
    algorithm_config: dict[str, Any],
) -> None:
    """Validate a checkpoint for strict continuation of the original run."""
    _validate_common_checkpoint_contract(state, env_config, algorithm_config)
    expected_variant = _environment_variant(env_config)
    checkpoint_variant = str(
        _checkpoint_extra(state).get("environment_variant", "direct_v2_3")
    )
    if checkpoint_variant != expected_variant:
        raise RuntimeError(
            "checkpoint environment_variant mismatch: expected "
            f"{expected_variant!r}, got {checkpoint_variant!r}"
        )
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
    state: dict[str, Any],
    env_config: dict[str, Any],
    algorithm_config: dict[str, Any],
    allow_cross_variant: bool = False,
) -> None:
    """Validate strict evaluation, optionally allowing only variant transfer."""
    _validate_common_checkpoint_contract(state, env_config, algorithm_config)
    target_variant = _environment_variant(env_config)
    checkpoint_variant = str(
        _checkpoint_extra(state).get("environment_variant", "direct_v2_3")
    )
    if checkpoint_variant != target_variant and not allow_cross_variant:
        raise RuntimeError(
            "checkpoint environment_variant mismatch: expected "
            f"{target_variant!r}, got {checkpoint_variant!r}; pass explicit "
            "allow_cross_variant=True only for policy-transfer evaluation"
        )


def evaluation_selection_key(
    record: dict[str, Any], environment_variant: str
) -> tuple[float, ...]:
    """Return the variant-specific lexicographic best-checkpoint key."""
    if environment_variant in {"persistent_wave_v1", "persistent_wave_v2"}:
        waves_cleared = record.get(
            "average_waves_cleared", record.get("mean_waves_cleared", 0.0)
        )
        final_clear = record.get(
            "clear_wave_3_probability", record.get("win_rate", 0.0)
        )
        return (
            float(final_clear),
            float(waves_cleared),
            float(record["average_return"]),
            -float(record["average_red_loss"]),
        )
    return (
        float(record["win_rate"]),
        float(record["average_return"]),
        -float(record["average_red_loss"]),
    )


__all__ = [
    "evaluation_selection_key",
    "validate_checkpoint_environment",
    "validate_checkpoint_for_evaluation",
    "validate_checkpoint_for_resume",
]
