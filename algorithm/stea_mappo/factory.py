"""Validated STEA-MAPPO construction; baseline factories remain unchanged."""
from .trainer import STEAMAPPOTrainer
from algorithm.common.critic_protocol import layer_width


def validate_config(config):
    if config.get("algorithm") != "STEA-MAPPO":
        raise ValueError("algorithm must be STEA-MAPPO")
    n,t,i = (config[key] for key in ("network","training","implementation"))
    if n["num_agents"] not in (4,5,8) or n["observation_dim"] != 13*n["num_agents"]+n.get("self_feature_dim",7)-7 or n["action_dim"] != 3:
        raise ValueError("STEA-MAPPO requires dimensions 52/3/4, 65/3/5 or 104/3/8")
    if n.get("self_feature_dim",7) not in (7,8) or (n.get("self_feature_dim",7)==8 and n["num_agents"]!=5):
        raise ValueError("STEA self feature layout mismatch")
    if n["actor_type"] != "stea" or n["critic_type"] not in {"attention", "mlp"}:
        raise ValueError("STEA-MAPPO requires actor_type=stea and critic_type=attention or mlp")
    if i["actor_activation"] != "relu" or i["critic_activation"] != "relu":
        raise ValueError("STEA-MAPPO requires baseline ReLU activations")
    length = int(n["recurrent_sequence_length"])
    if length <= 0 or int(t["minibatch_size"]) <= 0 or int(t["minibatch_size"]) % length:
        raise ValueError("minibatch_size must be positive and divisible by recurrent_sequence_length")
    if int(t["rollout_steps"]) <= 0 or int(t["rollout_steps"]) % length:
        raise ValueError("rollout_steps must be positive and divisible by recurrent_sequence_length")
    if int(n["gru_layers"]) != 1:
        raise ValueError("only one unidirectional GRU layer is supported")
    width = layer_width(n, "critic_hidden_layers")
    attention_dimensions = [(n["entity_dim"], n["entity_attention_heads"])]
    if n["critic_type"] == "attention":
        attention_dimensions.append((width, n["attention_heads"]))
    for dim,heads in attention_dimensions:
        if int(dim) <= 0 or int(heads) <= 0 or int(dim)%int(heads):
            raise ValueError("attention dimensions must be positive and divisible by heads")
    if min(int(n["spatial_hidden_dim"]),int(n["gru_hidden_dim"]),int(t["ppo_epochs"])) <= 0:
        raise ValueError("network dimensions and ppo_epochs must be positive")
    if t["optimizer"] != "Adam":
        raise ValueError("STEA-MAPPO uses the baseline Adam optimizer")


def build_stea_mappo_trainer(config, device, *, seed=None, smoke=False):
    validate_config(config)
    n,t,i = (config[key] for key in ("network","training","implementation"))
    kwargs = {key:n[key] for key in ("entity_dim","entity_attention_heads","spatial_hidden_dim",
        "gru_hidden_dim","gru_layers","recurrent_sequence_length","observation_dim","action_dim","num_agents","attention_heads","critic_type")}
    kwargs.update({key:t[key] for key in ("actor_learning_rate","critic_learning_rate","gamma","gae_lambda",
        "clip_ratio","value_loss_coefficient","entropy_coefficient","max_grad_norm","ppo_epochs","minibatch_size")})
    kwargs.update({key:i[key] for key in ("actor_activation","critic_activation","log_std_min","log_std_max",
        "normalize_advantages","clip_value_loss")})
    kwargs.update(policy_std_mode=i.get("policy_std_mode", "state_dependent"),
        log_std_init=float(i.get("log_std_init", -.5)),
        mean_head_init_gain=float(i.get("mean_head_init_gain", .01)), target_kl=t.get("target_kl"))
    kwargs["self_feature_dim"]=n.get("self_feature_dim",7)
    kwargs.update(hidden_dim=64 if smoke else int(n["critic_hidden_layers"][0]),device=device,
                  critic_hidden_dim=64 if smoke else int(n["critic_hidden_layers"][0]),
                  seed=int(t["seed"] if seed is None else seed))
    if smoke:
        kwargs.update(ppo_epochs=2,minibatch_size=2*int(n["recurrent_sequence_length"]))
    return STEAMAPPOTrainer(**kwargs)
