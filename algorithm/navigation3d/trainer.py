"""Strict navigation identities; inherited optimization equations stay unchanged."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from algorithm.mappo.trainer import MAPPOTrainer, MAPPO_IMPL_VERSION
from algorithm.ippo.trainer import IPPOTrainer, IPPO_IMPL_VERSION
from algorithm.common.protocol import config_sha256
from algorithm.common.policy_protocol import validate_trainer_policy_protocol
from env.navigation3d.config import validate_config, observation_dim, IDENTITY, OBSERVATION_SCHEMA


def require_cuda(device='cuda'):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for navigation training/evaluation/checkpoint audits')


def validate_algorithm(config, env_config):
    validate_config(env_config)
    n, t, i = (config[k] for k in ('network', 'training', 'implementation'))
    if config['algorithm'] not in ('MAPPO', 'ippo'):
        raise ValueError('navigation adapter supports MAPPO or ippo')
    expected_identity = dict(environment_id=IDENTITY, observation_schema=OBSERVATION_SCHEMA, num_agents=env_config['num_agents'])
    if config.get('environment') != expected_identity:
        raise ValueError('navigation algorithm/environment identity mismatch')
    if (n['observation_dim'], n['action_dim'], n['num_agents']) != (observation_dim(env_config), 3, env_config['num_agents']):
        raise ValueError('navigation network dimension mismatch')
    if n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]:
        raise ValueError('navigation smoke retains formal [256,256] networks')
    if n['critic_type'] != ('local' if config['algorithm'] == 'ippo' else 'mlp'):
        raise ValueError('navigation critic scope mismatch')
    if t['gamma'] != env_config['reward']['gamma']:
        raise ValueError('navigation potential gamma must equal PPO gamma')
    if t['optimizer'] != 'Adam' or t['device'] != 'cuda':
        raise ValueError('navigation requires configured Adam/CUDA')
    for key in ('rollout_steps', 'ppo_epochs', 'minibatch_size', 'num_train_envs'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f'invalid navigation {key}')
    for key in ('actor_learning_rate', 'critic_learning_rate', 'max_grad_norm', 'clip_ratio'):
        if not np.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'invalid navigation {key}')
    for key in ('gamma', 'gae_lambda'):
        if not np.isfinite(t[key]) or not 0 <= t[key] <= 1:
            raise ValueError(f'invalid navigation {key}')
    for key in ('entropy_coefficient', 'value_loss_coefficient'):
        if not np.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f'invalid navigation {key}')
    if t['target_kl'] is not None and (not np.isfinite(t['target_kl']) or t['target_kl'] <= 0):
        raise ValueError('invalid navigation target_kl')
    if i['actor_activation'] != 'relu' or i['critic_activation'] != 'relu' or i['policy_std_mode'] != 'state_independent':
        raise ValueError('navigation requires original ReLU/state-independent Gaussian')
    if not all(np.isfinite(i[k]) for k in ('log_std_min', 'log_std_init', 'log_std_max', 'mean_head_init_gain')) or not i['log_std_min'] <= i['log_std_init'] <= i['log_std_max'] or i['mean_head_init_gain'] <= 0:
        raise ValueError('invalid navigation Gaussian')
    if any(type(i[k]) is not bool for k in ('normalize_advantages', 'clip_value_loss')):
        raise ValueError('navigation advantage/value flags must be boolean')
    if config['algorithm'] == 'ippo' and (n.get('parameter_sharing') is not True or n.get('critic_scope') != 'local_observation'):
        raise ValueError('navigation IPPO requires shared local actor/critic')


class NavigationLocalCritic(nn.Module):
    """Same two-layer local value architecture, with navigation input size."""
    def __init__(self, features, hidden):
        super().__init__()
        self.features = features
        self.value_network = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, observations, alive_mask=None):
        if observations.ndim < 2 or observations.shape[-1] != self.features:
            raise ValueError('navigation local critic observation shape mismatch')
        value = self.value_network(observations).squeeze(-1)
        if alive_mask is not None:
            if value.shape != alive_mask.shape:
                raise ValueError('navigation local critic mask mismatch')
            value = value * alive_mask
        return value


class NavigationTrainer(MAPPOTrainer):
    # Reuse the actual PS-IPPO optimizer, including telemetry and KL early stop.
    _ppo_update = IPPOTrainer._ppo_update

    def __init__(self, env_config, algorithm_config, device='cuda', seed=0):
        require_cuda(device)
        validate_algorithm(algorithm_config, env_config)
        self.env_config, self.algorithm_config = copy.deepcopy(env_config), copy.deepcopy(algorithm_config)
        self.kind = algorithm_config['algorithm']
        n, t, i = (algorithm_config[k] for k in ('network', 'training', 'implementation'))
        # Base construction uses the existing generic MAPPO factory contract.
        # IPPO then installs its own local critic; no centralized value is used.
        super().__init__(observation_dim=n['observation_dim'], action_dim=n['action_dim'],
            num_agents=n['num_agents'], hidden_dim=n['actor_hidden_layers'][0],
            critic_hidden_dim=n['critic_hidden_layers'][0], attention_heads=n.get('attention_heads', 2),
            critic_type='mlp', device=device, seed=seed,
            **{k: t[k] for k in ('actor_learning_rate', 'critic_learning_rate', 'gamma', 'gae_lambda',
                'clip_ratio', 'value_loss_coefficient', 'entropy_coefficient', 'max_grad_norm', 'ppo_epochs', 'minibatch_size', 'target_kl')},
            **{k: i[k] for k in ('actor_activation', 'critic_activation', 'log_std_min', 'log_std_max',
                'policy_std_mode', 'log_std_init', 'mean_head_init_gain', 'normalize_advantages', 'clip_value_loss')})
        if self.kind == 'ippo':
            self.critic_type = 'local'
            self.critic = NavigationLocalCritic(self.observation_dim, self.critic_hidden_dim).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=t['critic_learning_rate'])
        self.algorithm_config_sha256 = config_sha256(algorithm_config)
        self.training_seed = int(seed)

    def update(self, rollout):
        if self.kind == 'ippo' or not (rollout.alive_masks > .5).any():
            return IPPOTrainer.update(self, rollout)
        return MAPPOTrainer.update(self, rollout)

    def checkpoint_state(self, extra=None):
        state = MAPPOTrainer.checkpoint_state(self)
        state.update(algorithm='navigation3d_'+self.kind.lower(), checkpoint_family=IDENTITY,
            navigation_adapter_version=1, optimizer_source_version=IPPO_IMPL_VERSION if self.kind == 'ippo' else MAPPO_IMPL_VERSION)
        state['extra'].update(extra or {})
        # Required identity fields cannot be overridden by caller metadata.
        state['extra'].update(environment_id=IDENTITY, observation_schema=OBSERVATION_SCHEMA,
            environment_config_sha256=config_sha256(self.env_config),
            algorithm_config_sha256=config_sha256(self.algorithm_config),
            environment_config=copy.deepcopy(self.env_config), algorithm_config=copy.deepcopy(self.algorithm_config),
            network_architecture=copy.deepcopy(self.algorithm_config['network']),
            training_seed=self.training_seed, resume_capability='evaluation_reload_only',
            actor_parameter_count=sum(p.numel() for p in self.actor.parameters()),
            critic_parameter_count=sum(p.numel() for p in self.critic.parameters()))
        return state

    def load(self, path):
        require_cuda(self.device)
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        expected = self.checkpoint_state()
        for key in ('algorithm', 'checkpoint_family', 'navigation_adapter_version', 'optimizer_source_version', 'mappo_impl_version', 'critic_type'):
            if state.get(key) != expected[key]:
                raise RuntimeError(f'navigation checkpoint {key} mismatch')
        for key in ('environment_id', 'observation_schema', 'environment_config_sha256',
            'algorithm_config_sha256', 'environment_config', 'algorithm_config', 'network_architecture',
            'actor_parameter_count', 'critic_parameter_count', 'training_gamma', 'training_gae_lambda'):
            if state.get('extra', {}).get(key) != expected['extra'][key]:
                raise RuntimeError(f'navigation checkpoint {key} mismatch')
        validate_trainer_policy_protocol(state, self.policy_protocol())
        # Validate all tensors before mutating a live model.
        for component in ('actor', 'critic'):
            current = getattr(self, component).state_dict()
            saved = state.get(component, {})
            if saved.keys() != current.keys() or any(saved[k].shape != current[k].shape or not torch.isfinite(saved[k]).all() for k in current):
                raise RuntimeError(f'navigation checkpoint {component} tensors mismatch')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.training_seed = state['extra']['training_seed']
        for attr, key in (('ppo_update_count', 'ppo_updates'), ('actor_update_count', 'actor_updates'),
            ('critic_update_count', 'critic_updates'), ('sampled_steps', 'sampled_steps'), ('vector_steps', 'vector_steps')):
            setattr(self, attr, int(state[key]))
        return copy.deepcopy(state['extra'])
