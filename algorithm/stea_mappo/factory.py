"""Validated STEA-MAPPO construction; baseline factories remain unchanged."""
from .trainer import STEAMAPPOTrainer


def validate_config(config):
    if config.get("algorithm") != "STEA-MAPPO":
        raise ValueError("algorithm must be STEA-MAPPO")
    n,t,i = (config[key] for key in ("network","training","implementation"))
    if n["num_agents"] not in (4,8) or n["observation_dim"] != 13*n["num_agents"] or n["action_dim"] != 3:
        raise ValueError("STEA-MAPPO requires dimensions 52/3/4 or 104/3/8")
    if n["actor_type"] != "stea" or n["critic_type"] != "attention":
        raise ValueError("STEA-MAPPO requires actor_type=stea and critic_type=attention")
    if i["actor_activation"] != "relu" or i["critic_activation"] != "relu":
        raise ValueError("STEA-MAPPO requires baseline ReLU activations")
    length = int(n["recurrent_sequence_length"])
    if length <= 0 or int(t["minibatch_size"]) <= 0 or int(t["minibatch_size"]) % length:
        raise ValueError("minibatch_size must be positive and divisible by recurrent_sequence_length")
    if int(t["rollout_steps"]) <= 0 or int(t["rollout_steps"]) % length:
        raise ValueError("rollout_steps must be positive and divisible by recurrent_sequence_length")
    if int(n["gru_layers"]) != 1:
        raise ValueError("only one unidirectional GRU layer is supported")
    hidden = n["critic_hidden_layers"]
    if len(hidden) != 2 or hidden[0] != hidden[1]:
        raise ValueError("critic_hidden_layers must match the baseline two equal widths")
    for dim,heads in ((n["entity_dim"],n["entity_attention_heads"]),(hidden[0],n["attention_heads"])):
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
    kwargs.update(hidden_dim=64 if smoke else int(n["critic_hidden_layers"][0]),device=device,
                  seed=int(t["seed"] if seed is None else seed))
    if smoke:
        kwargs.update(ppo_epochs=2,minibatch_size=2*int(n["recurrent_sequence_length"]))
    return STEAMAPPOTrainer(**kwargs)
