"""Reusable, metadata-complete evaluation of MAPPO checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import warnings

import torch

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.evaluator import evaluate
from algorithm.common.protocol import config_sha256
from .factory import build_mappo_trainer


def evaluate_mappo_checkpoint(
    checkpoint_path: str | Path,
    algorithm_config: dict[str, Any],
    environment_config: dict[str, Any],
    device: str,
    evaluation_seeds: Iterable[int],
    allow_cross_variant: bool = False,
) -> dict[str, Any]:
    """Load one checkpoint and evaluate it under an explicit target protocol."""
    checkpoint = Path(checkpoint_path)
    seeds = [int(seed) for seed in evaluation_seeds]
    if not seeds:
        raise ValueError("evaluation_seeds must not be empty")
    if any(right != left + 1 for left, right in zip(seeds, seeds[1:])):
        raise ValueError(
            "evaluation_seeds must be strictly increasing and contiguous by 1"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_for_evaluation(
        state,
        environment_config,
        algorithm_config,
        allow_cross_variant=allow_cross_variant,
    )
    extra = state.get("extra", {})
    required_protocol_fields = (
        "environment_version",
        "environment_variant",
        "observation_dim",
        "action_dim",
        "num_agents",
        "training_seed",
        "training_gamma",
        "training_num_envs",
        "training_total_sampled_steps",
        "training_smoke",
        "effective_hidden_dim",
        "environment_config_sha256",
        "algorithm_config_sha256",
    )
    protocol_complete = isinstance(extra, dict) and all(
        field in extra for field in required_protocol_fields
    )
    provided_algorithm_hash = config_sha256(algorithm_config)
    if protocol_complete:
        if extra["algorithm_config_sha256"] != provided_algorithm_hash:
            raise RuntimeError(
                "provided algorithm config fingerprint does not match checkpoint "
                f"algorithm_config_sha256: expected "
                f"{extra['algorithm_config_sha256']!r}, got "
                f"{provided_algorithm_hash!r}"
            )
    else:
        warnings.warn(
            "legacy/incomplete checkpoint protocol metadata; evaluation is "
            "diagnostic only and cannot be formally aggregated",
            RuntimeWarning,
        )
    effective_hidden_dim = extra.get("effective_hidden_dim")
    if effective_hidden_dim is None:
        first_weight = state.get("actor", {}).get("backbone.0.weight")
        if hasattr(first_weight, "shape"):
            effective_hidden_dim = int(first_weight.shape[0])
    configured_hidden_dim = int(
        algorithm_config["network"]["actor_hidden_layers"][0]
    )
    if (
        effective_hidden_dim is not None
        and int(effective_hidden_dim) != configured_hidden_dim
    ):
        trainer = build_mappo_trainer(
            algorithm_config, device, hidden_dim=int(effective_hidden_dim)
        )
    else:
        trainer = build_mappo_trainer(algorithm_config, device)
    trainer.load(checkpoint)
    result: dict[str, Any] = dict(
        evaluate(trainer, environment_config, seeds)
    )
    checkpoint_variant = str(extra.get("environment_variant", "direct_v2_3"))
    evaluation_variant = str(
        environment_config.get("environment_variant", "direct_v2_3")
    )
    network = algorithm_config["network"]
    result.update({
        "algorithm": "MAPPO",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sampled_steps": int(state.get("sampled_steps", 0)),
        "checkpoint_training_seed": extra.get("training_seed"),
        "checkpoint_training_gamma": extra.get("training_gamma"),
        "checkpoint_training_num_envs": extra.get("training_num_envs"),
        "checkpoint_training_total_sampled_steps": extra.get(
            "training_total_sampled_steps"
        ),
        "checkpoint_training_smoke": extra.get("training_smoke"),
        "checkpoint_effective_hidden_dim": effective_hidden_dim,
        "checkpoint_environment_config_sha256": extra.get(
            "environment_config_sha256"
        ),
        "checkpoint_algorithm_config_sha256": extra.get(
            "algorithm_config_sha256"
        ),
        "evaluation_environment_config_sha256": config_sha256(
            environment_config
        ),
        "provided_algorithm_config_sha256": provided_algorithm_hash,
        "protocol_complete": protocol_complete,
        "checkpoint_environment_version": extra.get("environment_version"),
        "checkpoint_environment_variant": checkpoint_variant,
        "evaluation_environment_version": str(
            environment_config["environment_version"]
        ),
        "evaluation_environment_variant": evaluation_variant,
        "cross_variant_evaluation": checkpoint_variant != evaluation_variant,
        "mappo_impl_version": state.get("mappo_impl_version"),
        "holdout_seed_base": seeds[0],
        "holdout_seed_end": seeds[-1],
        "evaluation_episodes": len(seeds),
        "device": str(device),
        "observation_dim": int(network["observation_dim"]),
        "action_dim": int(network["action_dim"]),
        "num_agents": int(network["num_agents"]),
    })
    return result


__all__ = ["evaluate_mappo_checkpoint"]
