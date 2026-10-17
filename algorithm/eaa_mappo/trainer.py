"""Inherit MAPPO optimization unchanged; add representation and pure diagnostics."""
from copy import deepcopy
import numpy as np
import torch
from algorithm.mappo.trainer import MAPPOTrainer
from .networks import EntityAttentionMAPPOActor
from .protocol import contract, validate_checkpoint, require_cuda, checkpoint_family


class EAAMAPPOTrainer(MAPPOTrainer):
    def __init__(self, env_config, algorithm_config, **kwargs):
        super().__init__(**kwargs)
        self.env_config, self.algorithm_config = deepcopy(env_config), deepcopy(algorithm_config)
        self.variant = algorithm_config['algorithm']
        self.actor_type = algorithm_config['network']['actor_type']
        if self.actor_type == 'entity_attention':
            i = algorithm_config['implementation']
            self.actor = EntityAttentionMAPPOActor(i['log_std_min'], i['log_std_max'],
                i['log_std_init'], i['mean_head_init_gain']).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=kwargs['actor_learning_rate'])

    def parameter_counts(self):
        a = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        c = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        return dict(actor=a, critic=c, total=a+c)

    @torch.no_grad()
    def attention_diagnostics(self, observations, alive):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device).reshape(-1,4,45)
        mask = torch.as_tensor(alive, dtype=torch.float32, device=self.device).reshape(-1,4) > .5
        result = {}
        if self.actor_type == 'entity_attention' and mask.any():
            _, weights = self.actor.distribution(obs[mask], return_attention=True)
            masses = dict(self=weights[...,0], ally=weights[...,1:4].sum(-1), enemy=weights[...,4:8].sum(-1))
            for name, values in masses.items():
                result[f'actor_attention_{name}_mass'] = float(values.mean())
                for h, value in enumerate(values.mean(0)):
                    result[f'actor_attention_head_{h}_{name}_mass'] = float(value)
            ent = -(weights*weights.clamp_min(1e-12).log()).sum(-1)
            result['actor_attention_entropy'] = float(ent.mean())
            for h, value in enumerate(ent.mean(0)):result[f'actor_attention_head_{h}_entropy'] = float(value)
        if self.critic_type == 'attention':
            _, weights = self.critic(obs, mask.float(), return_attention=True)
            valid_queries = mask & (mask.sum(-1,keepdim=True)>1)
            # A sole survivor has zero context/zero weights, not a normalized attention row.
            live = weights.transpose(1,2)[valid_queries]
            result['critic_attention_entropy'] = float(-(live*live.clamp_min(1e-12).log()).sum(-1).mean()) if live.numel() else 0.
            result['critic_attention_max_weight'] = float(live.max(-1).values.mean()) if live.numel() else 0.
            result['critic_attention_valid_queries'] = float(valid_queries.sum())
        return result

    def update(self, rollout):
        metrics = super().update(rollout)
        metrics.update(self.attention_diagnostics(rollout.observations, rollout.alive_masks))
        if not np.isfinite(list(metrics.values())).all():raise FloatingPointError('non-finite attention diagnostics')
        return metrics

    def checkpoint_state(self, extra=None):
        state = super().checkpoint_state(extra)
        state.pop('mappo_impl_version')
        state['extra'].pop('mappo_impl_version',None)
        identity = contract(self.env_config, self.algorithm_config)
        state.update(algorithm=self.variant, checkpoint_family=checkpoint_family(self.env_config),
                     parameter_counts=self.parameter_counts(), **identity)
        state['extra'].update(identity)
        return state

    def load(self, path, **kwargs):
        require_cuda(self.device)
        state = torch.load(path, map_location=self.device, weights_only=False)
        extra = validate_checkpoint(state, self.env_config, self.algorithm_config)
        for name in ('actor','critic'):getattr(self,name).load_state_dict(state[name],strict=True)
        for name in ('actor_optimizer','critic_optimizer'):getattr(self,name).load_state_dict(state[name])
        for attr, key in [('ppo_update_count','ppo_updates'),('actor_update_count','actor_updates'),
                          ('critic_update_count','critic_updates'),('sampled_steps','sampled_steps'),('vector_steps','vector_steps')]:
            setattr(self,attr,int(state[key]))
        return extra
