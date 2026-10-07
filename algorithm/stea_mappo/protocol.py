"""Strict independent checkpoint identity, architecture and config provenance."""
from algorithm.common.protocol import config_sha256
from env.config import ENVIRONMENT_VERSION
from .trainer import STEA_MAPPO_IMPL_VERSION
from .factory import validate_config


def require_cuda(device):
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for STEA-MAPPO training and checkpoint evaluation")


def architecture_from_config(config, smoke=False):
    n = config["network"]
    return {**{key:n[key] for key in ("actor_type","entity_dim","entity_attention_heads",
        "spatial_hidden_dim","gru_hidden_dim","gru_layers","recurrent_sequence_length","critic_type")},
        "critic_hidden_dim":64 if smoke else int(n["critic_hidden_layers"][0]),
        "critic_attention_heads":int(n["attention_heads"])}


def validate_checkpoint(state, env_config, algorithm_config):
    validate_config(algorithm_config)
    if state.get("algorithm") != "STEA-MAPPO":
        raise RuntimeError("checkpoint is not a STEA-MAPPO checkpoint")
    if state.get("stea_mappo_impl_version") != STEA_MAPPO_IMPL_VERSION:
        raise RuntimeError("STEA-MAPPO implementation version mismatch")
    extra = state.get("extra",{})
    required = ("environment_version","observation_dim","action_dim","num_agents","training_seed",
        "training_gamma","training_num_envs","training_total_sampled_steps","training_smoke",
        "effective_hidden_dim","environment_config_sha256","algorithm_config_sha256","network_architecture",
        "stea_mappo_impl_version","actor_parameter_count","critic_parameter_count","total_parameter_count")
    if any(key not in extra for key in required):
        raise RuntimeError("incomplete STEA-MAPPO checkpoint protocol metadata")
    expected = {"environment_version":ENVIRONMENT_VERSION,"observation_dim":52,"action_dim":3,"num_agents":4,
        "training_gamma":float(algorithm_config["training"]["gamma"]),
        "environment_config_sha256":config_sha256(env_config),
        "algorithm_config_sha256":config_sha256(algorithm_config),
        "stea_mappo_impl_version":STEA_MAPPO_IMPL_VERSION}
    for key,value in expected.items():
        if extra[key] != value:
            raise RuntimeError(f"STEA-MAPPO checkpoint {key} mismatch")
    architecture = architecture_from_config(algorithm_config,bool(extra["training_smoke"]))
    if state.get("network_architecture") != architecture or extra["network_architecture"] != architecture:
        raise RuntimeError("STEA-MAPPO network_architecture mismatch")
    if state.get("critic_type") != "attention" or extra["effective_hidden_dim"] != architecture["critic_hidden_dim"]:
        raise RuntimeError("STEA-MAPPO critic architecture mismatch")
    actor_count = sum(value.numel() for value in state["actor"].values())
    critic_count = sum(value.numel() for value in state["critic"].values())
    if (extra["actor_parameter_count"],extra["critic_parameter_count"],extra["total_parameter_count"]) != (actor_count,critic_count,actor_count+critic_count):
        raise RuntimeError("STEA-MAPPO checkpoint parameter count mismatch")
    return extra
