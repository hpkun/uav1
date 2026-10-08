"""Ordinary transition-shuffled MAPPO PPO, with a stateless entity actor."""
from pathlib import Path
import torch
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.common.policy_protocol import validate_trainer_policy_protocol
from .networks import EntityAttentionActor, spatial_diagnostics

EAMAPPO_IMPL_VERSION = 1


class EAMAPPOTrainer(MAPPOTrainer):
    def __init__(self, *, entity_dim=64, entity_attention_heads=2, spatial_hidden_dim=128, **kwargs):
        if kwargs.get("critic_type") != "mlp":
            raise ValueError("EA-MAPPO requires CentralizedMLPCritic")
        super().__init__(**kwargs)
        self.actor = EntityAttentionActor(entity_dim,entity_attention_heads,spatial_hidden_dim,
            self.action_dim,kwargs.get("log_std_min",-5.),kwargs.get("log_std_max",.5),
            kwargs.get("policy_std_mode","state_independent"),kwargs.get("log_std_init",-.5),
            kwargs.get("mean_head_init_gain",.01)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=kwargs.get("actor_learning_rate",3e-4))
        self.actor_hidden_dim = 128
        self.network_architecture = dict(actor_type="entity_attention",entity_dim=64,
            entity_attention_heads=2,spatial_hidden_dim=128,critic_type="mlp",
            critic_hidden_dim=self.critic_hidden_dim,
            **{k:v for k,v in self.policy_protocol().items() if k != "target_kl"})

    def update(self, rollout):
        metrics = super().update(rollout)
        # Pure no-grad diagnostics after PPO: no sampling, state, or optimizer changes.
        with torch.no_grad():
            obs = torch.as_tensor(rollout.observations,dtype=torch.float32,device=self.device)
            alive = torch.as_tensor(rollout.alive_masks,dtype=torch.float32,device=self.device)
            _,attention = self.actor.spatial(obs,alive)
            metrics.update({k:float(v) for k,v in spatial_diagnostics(attention,alive).items()})
        return metrics

    def checkpoint_state(self, extra=None):
        state = super().checkpoint_state(extra)
        state.pop("mappo_impl_version")
        state.update(algorithm="EA-MAPPO", ea_mappo_impl_version=EAMAPPO_IMPL_VERSION,
                     implementation_version=EAMAPPO_IMPL_VERSION, network_architecture=dict(self.network_architecture))
        return state

    def load(self, path: str | Path):
        state = torch.load(path,map_location=self.device,weights_only=False)
        if state.get("algorithm") != "EA-MAPPO":
            raise RuntimeError("checkpoint is not a EAMAPPO checkpoint")
        if state.get("ea_mappo_impl_version") != EAMAPPO_IMPL_VERSION or state.get("implementation_version") != EAMAPPO_IMPL_VERSION:
            raise RuntimeError("EAMAPPO implementation version mismatch")
        validate_trainer_policy_protocol(state, self.policy_protocol())
        if state.get("network_architecture") != self.network_architecture:
            raise RuntimeError("EAMAPPO network_architecture mismatch")
        if state.get("critic_type") != self.critic_type:
            raise RuntimeError("EAMAPPO critic_type mismatch")
        extra = state.get("extra",{})
        for key in ("observation_dim","action_dim","num_agents"):
            if key in extra and extra[key] != getattr(self,key):
                raise RuntimeError(f"EAMAPPO checkpoint {key} mismatch")
        if "network_architecture" in extra and extra["network_architecture"] != self.network_architecture:
            raise RuntimeError("EAMAPPO extra network_architecture mismatch")
        if "ea_mappo_impl_version" in extra and extra["ea_mappo_impl_version"] != EAMAPPO_IMPL_VERSION:
            raise RuntimeError("EAMAPPO extra implementation version mismatch")
        self.actor.load_state_dict(state["actor"],strict=True)
        self.critic.load_state_dict(state["critic"],strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for name,key in (("ppo_update_count","ppo_updates"),("actor_update_count","actor_updates"),
                         ("critic_update_count","critic_updates"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):
            setattr(self,name,int(state[key]))
        return dict(state.get("extra",{}))
