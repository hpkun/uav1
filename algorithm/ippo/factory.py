"""Construct IPPO trainers from the existing YAML configuration schema."""
from __future__ import annotations

from typing import Any

from .trainer import IPPOTrainer
from algorithm.common.critic_protocol import layer_width
from algorithm.common.protocol import config_sha256


def build_ippo_trainer(
    config: dict[str, Any], device: str, hidden_dim: int | None = None,
    critic_hidden_dim: int | None = None, seed: int | None = None,
) -> IPPOTrainer:
    """Build a formal IPPO trainer without changing configured parameters."""
    from .protocol import validate_algorithm_config
    validate_algorithm_config(config)
    network = config["network"]
    training = config["training"]
    implementation = config["implementation"]
    actor_width = layer_width(network, "actor_hidden_layers")
    critic_width = layer_width(network, "critic_hidden_layers")
    if hidden_dim not in (None, actor_width) or critic_hidden_dim not in (None, critic_width):
        raise ValueError('IPPO cannot silently override configured network widths')
    common = {
        "observation_dim": int(network["observation_dim"]),
        "action_dim": int(network["action_dim"]),
        "num_agents": int(network["num_agents"]),
        "hidden_dim": int(
            actor_width if hidden_dim is None else hidden_dim
        ),
        "critic_hidden_dim": critic_width if critic_hidden_dim is None else int(critic_hidden_dim),
        "attention_heads": 1,
        "critic_type": str(network.get("critic_type", "attention")),
        "device": device,
        "actor_activation": implementation["actor_activation"],
        "critic_activation": implementation["critic_activation"],
        "log_std_min": float(implementation["log_std_min"]),
        "log_std_max": float(implementation["log_std_max"]),
        "policy_std_mode": implementation.get("policy_std_mode", "state_dependent"),
        "log_std_init": float(implementation.get("log_std_init", -.5)),
        "mean_head_init_gain": float(implementation.get("mean_head_init_gain", .01)),
        "target_kl": training.get("target_kl"),
    }
    trainer = IPPOTrainer(
        **common,
        seed=int(training["seed"] if seed is None else seed),
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
    trainer.algorithm_config = config
    trainer.algorithm_config_sha256 = config_sha256(config)
    return trainer


__all__ = ["build_ippo_trainer"]
