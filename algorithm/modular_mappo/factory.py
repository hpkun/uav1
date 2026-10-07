"""Configuration-driven modular trainer construction."""
from .trainer import ModularMAPPOTrainer

def build_modular_mappo_trainer(config,device=None,hidden_dim=None,total_sampled_steps=None):
 n,t,i=config["network"],config["training"],config["implementation"]
 from .protocol import validate_marc_gru_screen_config,validate_marc_state_memory_config,validate_marc_factorial_config,validate_jiao2025_3m_config,JIAO_3M_METHODS
 validate_jiao2025_3m_config(config)
 if config.get("development_method") in JIAO_3M_METHODS and hidden_dim is not None and int(hidden_dim)!=256:
  raise ValueError("Jiao 3M cannot override the fixed 256-wide architecture")
 validate_marc_factorial_config(config)
 validate_marc_gru_screen_config(config)
 validate_marc_state_memory_config(config)
 if config.get("development_method") in {"marc_core_factorial_v1","marc_credit_balance_ablation","marc_mappo_wsgru_v1","marc_mappo_state_memory_v1"} and hidden_dim is not None and int(hidden_dim)!=256:
  raise ValueError("MARC GRU matched screen cannot override the 256-wide baseline")
 return ModularMAPPOTrainer(observation_dim=int(n["observation_dim"]),action_dim=int(n["action_dim"]),num_agents=int(n["num_agents"]),hidden_dim=int(hidden_dim or n["actor_hidden_layers"][0]),attention_heads=int(n["attention_heads"]),actor_learning_rate=float(t["actor_learning_rate"]),critic_learning_rate=float(t["critic_learning_rate"]),gamma=float(t["gamma"]),gae_lambda=float(t["gae_lambda"]),clip_ratio=float(t["clip_ratio"]),value_loss_coefficient=float(t["value_loss_coefficient"]),entropy_coefficient=float(t["entropy_coefficient"]),max_grad_norm=float(t["max_grad_norm"]),ppo_epochs=int(t["ppo_epochs"]),minibatch_size=int(t["minibatch_size"]),normalize_advantages=bool(i["normalize_advantages"]),clip_value_loss=bool(i["clip_value_loss"]),device=str(device or t["device"]),seed=int(t["seed"]),actor_activation=i["actor_activation"],critic_activation=i["critic_activation"],log_std_min=float(i["log_std_min"]),log_std_max=float(i["log_std_max"]),modules_config=config.get("modules",{}),total_sampled_steps=int(total_sampled_steps or t["total_sampled_steps"]),development_method=config.get("development_method"))

__all__=["build_modular_mappo_trainer"]
