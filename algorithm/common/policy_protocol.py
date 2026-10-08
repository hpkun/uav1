"""Gaussian policy metadata shared by MAPPO and STEA without legacy migration."""
from algorithm.common.protocol import config_sha256

FIELDS = ("policy_std_mode", "log_std_init", "mean_head_init_gain", "target_kl")


def configured_policy_protocol(config):
    implementation = config["implementation"]
    return dict(policy_std_mode=implementation.get("policy_std_mode", "state_dependent"),
                log_std_init=float(implementation.get("log_std_init", -.5)),
                mean_head_init_gain=float(implementation.get("mean_head_init_gain", .01)),
                target_kl=config["training"].get("target_kl"))


def validate_policy_protocol(state, config):
    expected = configured_policy_protocol(config)
    actor = state.get("actor", {})
    actual_mode = "state_independent" if "log_std_parameter" in actor else "state_dependent"
    if actual_mode != expected["policy_std_mode"]:
        raise RuntimeError("checkpoint policy_std_mode mismatch")
    extra = state.get("extra", {})
    recorded = [name in extra for name in FIELDS]
    explicit = any(name in config["implementation"] for name in FIELDS[:-1]) or "target_kl" in config["training"]
    if any(recorded) or explicit:
        if not all(recorded):
            raise RuntimeError("incomplete checkpoint Gaussian/PPO protocol metadata")
        for name in FIELDS:
            if extra[name] != expected[name]:
                raise RuntimeError(f"checkpoint {name} mismatch")
    elif actual_mode != "state_dependent":
        raise RuntimeError("state-independent checkpoint requires explicit protocol metadata")
    recorded_hash = extra.get("algorithm_config_sha256")
    if recorded_hash is not None and recorded_hash != config_sha256(config):
        raise RuntimeError("checkpoint algorithm_config_sha256 mismatch")


def actor_architecture_protocol(config):
    protocol = configured_policy_protocol(config)
    return {key: protocol[key] for key in FIELDS[:-1]} if protocol["policy_std_mode"] == "state_independent" else {}


def validate_trainer_policy_protocol(state, expected):
    mode = "state_independent" if "log_std_parameter" in state.get("actor", {}) else "state_dependent"
    if mode != expected["policy_std_mode"]:
        raise RuntimeError("checkpoint policy_std_mode mismatch")
    extra = state.get("extra", {})
    if any(name in extra for name in FIELDS):
        for name in FIELDS:
            if name not in extra or extra[name] != expected[name]:
                raise RuntimeError(f"checkpoint {name} mismatch")
    elif mode != "state_dependent" or expected["target_kl"] is not None:
        raise RuntimeError("checkpoint missing Gaussian/PPO protocol metadata")
