"""Independent actor/critic widths and legacy checkpoint width inference."""


def layer_width(network, key):
    layers = network[key]
    if (len(layers) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in layers)
            or layers[0] != layers[1]):
        raise ValueError(f"{key} must contain two equal positive integer widths")
    return layers[0]


def checkpoint_widths(state):
    """Infer actual tensor widths, including legacy reduced-width smoke runs."""
    actor = int(state["actor"]["backbone.0.weight"].shape[0])
    critic_state = state["critic"]
    key = "embedding.0.weight" if "embedding.0.weight" in critic_state else "value_network.0.weight"
    return actor, int(critic_state[key].shape[0])


def validate_mappo_architecture(state, config):
    """Old metadata remains readable; new metadata must match tensors/config."""
    extra = state.get("extra", {})
    architecture = extra.get("network_architecture", {})
    if not any(key in architecture or key in extra for key in ("actor_hidden_dim", "critic_hidden_dim")):
        return
    actor, critic = checkpoint_widths(state)
    n = config["network"]
    expected = {
        "actor_hidden_dim": 64 if extra.get("training_smoke") else layer_width(n, "actor_hidden_layers"),
        "critic_hidden_dim": 64 if extra.get("training_smoke") else layer_width(n, "critic_hidden_layers"),
        "critic_type": n.get("critic_type", "attention"),
    }
    if expected["critic_type"] == "attention":
        expected["attention_heads"] = int(n["attention_heads"])
    if architecture != expected or (actor, critic) != (expected["actor_hidden_dim"], expected["critic_hidden_dim"]):
        raise RuntimeError("checkpoint network_architecture mismatch")
    for key, count in (("actor_parameter_count", sum(v.numel() for v in state["actor"].values())),
                       ("critic_parameter_count", sum(v.numel() for v in state["critic"].values()))):
        if extra.get(key) != count:
            raise RuntimeError(f"checkpoint {key} mismatch")
    for key in ("actor_hidden_dim", "critic_hidden_dim", "critic_type"):
        if extra.get(key) != expected[key]:
            raise RuntimeError(f"checkpoint {key} mismatch")
    if "total_parameter_count" in extra and extra["total_parameter_count"] != extra["actor_parameter_count"] + extra["critic_parameter_count"]:
        raise RuntimeError("checkpoint total_parameter_count mismatch")
