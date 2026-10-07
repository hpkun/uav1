"""Construct MAPPO trainers from the existing YAML configuration schema."""
from __future__ import annotations

from typing import Any

from .trainer import MAPPOTrainer


def build_mappo_trainer(
    config: dict[str, Any], device: str, hidden_dim: int | None = None
) -> MAPPOTrainer:
    """Build a formal MAPPO trainer without changing configured parameters."""
    network = config["network"]
    training = config["training"]
    implementation = config["implementation"]
    common = {
        "observation_dim": int(network["observation_dim"]),
        "action_dim": int(network["action_dim"]),
        "num_agents": int(network["num_agents"]),
        "hidden_dim": int(
            network["actor_hidden_layers"][0] if hidden_dim is None else hidden_dim
        ),
        "attention_heads": int(network["attention_heads"]),
        "critic_type": str(network.get("critic_type", "attention")),
        "device": device,
        "actor_activation": implementation["actor_activation"],
        "critic_activation": implementation["critic_activation"],
        "log_std_min": float(implementation["log_std_min"]),
        "log_std_max": float(implementation["log_std_max"]),
    }
    return MAPPOTrainer(
        **common,
        actor_learning_rate=float(training["actor_learning_rate"]),
        critic_learning_rate=float(training["critic_learning_rate"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_ratio=float(training["clip_ratio"]),
        value_loss_coefficient=float(training["value_loss_coefficient"]),
        entropy_coefficient=float(training["entropy_coefficient"]),
        max_grad_norm=float(training["max_grad_norm"]),
        ppo_epochs=int(training["ppo_epochs"]),
        minibatch_size=int(training["minibatch_size"]),
        normalize_advantages=bool(implementation["normalize_advantages"]),
        clip_value_loss=bool(implementation["clip_value_loss"]),
    )


__all__ = ["build_mappo_trainer"]
