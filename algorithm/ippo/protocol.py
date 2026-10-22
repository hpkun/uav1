"""Fail-closed PS-IPPO v3.41 identity and artifact contract."""
import torch
import math
from env.config import validate_config, environment_dimensions
from algorithm.common.protocol import config_sha256
from algorithm.common.policy_protocol import validate_policy_protocol
from algorithm.common.critic_protocol import layer_width

def require_cuda(device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for IPPO training/evaluation/audits')

def validate_algorithm_config(config, env_config=None):
    n, t, i = (config[key] for key in ('network','training','implementation'))
    if config.get('algorithm') != 'ippo':
        raise ValueError('algorithm must be ippo')
    version = config.get('environment',{}).get('environment_version')
    if version not in ('3.41','3.42','3.43','3.44','3.46') or config['environment'] != dict(environment_version=version,team_size=4):
        raise ValueError('IPPO requires a supported 4v4 identity')
    obs_dim = 49 if version in ('3.43','3.44','3.46') else 45
    if [n[k] for k in ('observation_dim','action_dim','num_agents')] != [obs_dim,3,4]:
        raise ValueError(f'IPPO requires {obs_dim}/3/4 dimensions')
    if n.get('parameter_sharing') is not True or n.get('critic_scope') != 'local_observation' or n.get('critic_type') != 'local':
        raise ValueError('IPPO requires parameter sharing and local_observation critic')
    for key in ('actor_hidden_layers','critic_hidden_layers'):
        if layer_width(n,key) != 256:
            raise ValueError('formal IPPO requires [256,256] networks')
    if t['optimizer'] != 'Adam' or i['actor_activation'] != 'relu' or i['critic_activation'] != 'relu':
        raise ValueError('IPPO requires Adam/ReLU')
    if i['policy_std_mode'] != 'state_independent':
        raise ValueError('IPPO requires state_independent Gaussian')
    for key in ('rollout_steps','ppo_epochs','minibatch_size','num_train_envs','total_sampled_steps',
                'evaluation_episodes','evaluation_interval_sampled_steps'):
        if not isinstance(t[key],int) or isinstance(t[key],bool) or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('actor_learning_rate','critic_learning_rate','clip_ratio','max_grad_norm'):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('gamma','gae_lambda'):
        if not math.isfinite(t[key]) or not 0 <= t[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    for key in ('value_loss_coefficient','entropy_coefficient'):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if t['target_kl'] is not None and (not math.isfinite(t['target_kl']) or t['target_kl'] <= 0):
        raise ValueError('target_kl must be positive or null')
    if not i['log_std_min'] <= i['log_std_init'] <= i['log_std_max'] or not all(
        math.isfinite(i[k]) for k in ('log_std_min','log_std_init','log_std_max','mean_head_init_gain')) or i['mean_head_init_gain'] <= 0:
        raise ValueError('invalid Gaussian configuration')
    for key in ('normalize_advantages','clip_value_loss'):
        if type(i[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    base = i['evaluation_seed_base']
    if not 10_000_000 <= base < base+t['evaluation_episodes'] <= 20_000_000:
        raise ValueError('development evaluation seeds must be below reserved holdout range')
    if env_config is not None:
        validate_config(env_config)
        if str(env_config['environment_version']) != version or environment_dimensions(env_config) != (obs_dim,3,4):
            raise ValueError('IPPO algorithm/environment identity mismatch')
        from env.v343_config import validate_shaping_gamma
        validate_shaping_gamma(env_config,config)

def validate_checkpoint(state, env_config, config):
    from .trainer import IPPO_IMPL_VERSION
    validate_algorithm_config(config, env_config)
    if (state.get('algorithm') != 'ippo' or state.get('ippo_impl_version') != IPPO_IMPL_VERSION
        or state.get('parameter_sharing') is not True or state.get('critic_scope') != 'local_observation'
        or state.get('critic_type') != 'local'):
        raise RuntimeError('checkpoint is not compatible PS-IPPO')
    e = state.get('extra',{})
    for key, expected in [('environment_config_sha256', config_sha256(env_config)),
                          ('algorithm_config_sha256', config_sha256(config)),
                          ('environment_version',env_config['environment_version']), ('parameter_sharing',True),
                          ('critic_scope','local_observation'), ('bit_exact_training_resume',False)]:
        if e.get(key) != expected:
            raise RuntimeError(f'checkpoint {key} mismatch/missing')
    validate_policy_protocol(state, config)
    obs_dim = config['network']['observation_dim']
    architecture = dict(observation_dim=obs_dim, action_dim=3, num_agents=4,
        actor_hidden_layers=config['network']['actor_hidden_layers'],
        critic_hidden_layers=config['network']['critic_hidden_layers'],
        actor_activation='relu', critic_activation='relu', critic_type='local')
    if e.get('network_architecture') != architecture:
        raise RuntimeError('checkpoint network_architecture mismatch')
    for key, expected in [('actor_parameter_count',78342+256*(obs_dim-45)),('critic_parameter_count',77825+256*(obs_dim-45))]:
        component = key.split('_')[0]
        if e.get(key) != expected or sum(v.numel() for v in state[component].values()) != expected:
            raise RuntimeError(f'checkpoint {key} mismatch')
    if state['critic']['value_network.0.weight'].shape != (256,obs_dim):
        raise RuntimeError('checkpoint local critic observation dimension mismatch')
    required = ('training_seed','training_num_envs','training_total_sampled_steps','training_smoke',
                'best_evaluation','evaluation_history','ppo_protocol','algorithm_config', 'evaluation_seed_base')
    if any(key not in e for key in required):
        raise RuntimeError('checkpoint incomplete training/evaluation provenance')
    if e['algorithm_config'] != config:
        raise RuntimeError('checkpoint algorithm snapshot mismatch')
    t, i = config['training'], config['implementation']
    protocol_keys = ('gamma','gae_lambda','clip_ratio','value_loss_coefficient','entropy_coefficient',
        'max_grad_norm','ppo_epochs','minibatch_size','normalize_advantages','clip_value_loss',
        'log_std_min','log_std_max')
    expected_protocol = {key:(t[key] if key in t else i[key]) for key in protocol_keys}
    if e['ppo_protocol'] != expected_protocol:
        raise RuntimeError('checkpoint PPO/Gaussian protocol mismatch')
    for key in ('actor','critic','actor_optimizer','critic_optimizer','ppo_updates','actor_updates',
                'critic_updates','sampled_steps','vector_steps'):
        if key not in state:
            raise RuntimeError(f'checkpoint missing {key}')
