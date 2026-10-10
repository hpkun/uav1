"""Strict independent checkpoint identity, architecture and config provenance."""
from algorithm.common.protocol import config_sha256
from env.config import environment_dimensions
from .trainer import RMAPPO_IMPL_VERSION
from .factory import validate_config
from algorithm.common.policy_protocol import validate_policy_protocol, actor_architecture_protocol


def require_cuda(device):
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for RMAPPO training and checkpoint evaluation")


def architecture_from_config(config,smoke=False):
    n=config["network"]
    return {**{key:n[key] for key in ('actor_type', 'flat_encoder_dim', 'gru_hidden_dim', 'gru_layers', 'recurrent_sequence_length', 'critic_type')},
        "critic_hidden_dim":int(n["critic_hidden_layers"][0]),**actor_architecture_protocol(config)}


def validate_checkpoint(state, env_config, algorithm_config):
    validate_config(algorithm_config)
    if str(env_config["environment_version"]) not in {"2.4","2.5","2.6","2.7","2.8","2.9","3.0","3.1","3.2","3.3","3.4"}:
        raise RuntimeError("formal control requires environment_version 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3 or 3.4")
    dimensions = environment_dimensions(env_config)
    network = algorithm_config["network"]
    if dimensions != tuple(network[key] for key in ("observation_dim","action_dim","num_agents")):
        raise RuntimeError("RMAPPO network/environment dimensions mismatch")
    if state.get("algorithm") != "RMAPPO":
        raise RuntimeError("checkpoint is not a RMAPPO checkpoint")
    if state.get("rmappo_impl_version") != RMAPPO_IMPL_VERSION or state.get("implementation_version") != RMAPPO_IMPL_VERSION:
        raise RuntimeError("RMAPPO implementation version mismatch")
    extra = state.get("extra",{})
    required = ("environment_version","observation_dim","action_dim","num_agents","training_seed",
        "training_gamma","training_num_envs","training_total_sampled_steps","training_smoke",
        "effective_hidden_dim","environment_config_sha256","algorithm_config_sha256","network_architecture",
        "rmappo_impl_version","actor_parameter_count","critic_parameter_count","total_parameter_count")
    if any(key not in extra for key in required):
        raise RuntimeError("incomplete RMAPPO checkpoint protocol metadata")
    if extra["environment_version"] != env_config["environment_version"]:
        raise RuntimeError("RMAPPO checkpoint environment_version mismatch")
    if state.get("critic_type") != network["critic_type"]:
        raise RuntimeError("RMAPPO checkpoint critic_type mismatch")
    if "critic_type" in extra and extra["critic_type"] != network["critic_type"]:
        raise RuntimeError("RMAPPO checkpoint extra critic_type mismatch")
    validate_policy_protocol(state, algorithm_config)
    expected = {"environment_version":env_config["environment_version"],"observation_dim":dimensions[0],"action_dim":dimensions[1],"num_agents":dimensions[2],
        "training_gamma":float(algorithm_config["training"]["gamma"]),
        "environment_config_sha256":config_sha256(env_config),
        "algorithm_config_sha256":config_sha256(algorithm_config),
        "rmappo_impl_version":RMAPPO_IMPL_VERSION}
    for key,value in expected.items():
        if extra[key] != value:
            raise RuntimeError(f"RMAPPO checkpoint {key} mismatch")
    architecture = architecture_from_config(algorithm_config,bool(extra["training_smoke"]))
    if state.get("network_architecture") != architecture or extra["network_architecture"] != architecture:
        raise RuntimeError("RMAPPO network_architecture mismatch")
    if state.get("critic_type") != architecture["critic_type"] or extra["effective_hidden_dim"] != architecture["critic_hidden_dim"]:
        raise RuntimeError("RMAPPO critic architecture mismatch")
    actor_count = sum(value.numel() for value in state["actor"].values())
    critic_count = sum(value.numel() for value in state["critic"].values())
    if (extra["actor_parameter_count"],extra["critic_parameter_count"],extra["total_parameter_count"]) != (actor_count,critic_count,actor_count+critic_count):
        raise RuntimeError("RMAPPO checkpoint parameter count mismatch")
    return extra
