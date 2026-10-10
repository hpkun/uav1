"""Strict independent identity, dimensions, architecture and config fingerprints."""
from algorithm.common.protocol import config_sha256
from env.config import validate_config as validate_environment, environment_dimensions
from .trainer import MADDPG_IMPL_VERSION


def require_cuda(device):
    import torch
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for MADDPG training and checkpoint/evaluation audits')


def validate_config(env, config):
    validate_environment(env)
    if config.get('algorithm') != 'maddpg':
        raise RuntimeError('algorithm config is not maddpg')
    n, t, i = (config[key] for key in ('network', 'training', 'implementation'))
    dimensions = tuple(n[key] for key in ('observation_dim', 'action_dim', 'num_agents'))
    if dimensions != environment_dimensions(env) or (str(env['environment_version']), dimensions) not in (
            ('2.5', (65, 3, 5)), ('2.4', (104, 3, 8)), ('3.3', (66, 3, 5)), ('3.4', (66, 3, 5)), ('3.5', (66, 3, 5)), ('3.6', (66, 3, 5)), ('3.7', (66, 3, 5))):
        raise RuntimeError('MADDPG requires v2.5 65/3/5, v2.4 104/3/8 or v3.3/v3.4/v3.5/v3.6/v3.7 66/3/5')
    if n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]:
        raise RuntimeError('MADDPG requires independent 256/256 MLPs')
    forbidden = {'attention_heads', 'actor_type', 'gru_hidden_dim', 'policy_delay', 'alpha',
                 'target_policy_noise', 'log_std_min', 'policy_std_mode', 'entropy_coefficient'}
    if any(forbidden.intersection(section) for section in (n, t, i)):
        raise RuntimeError('unsupported mechanism in classic MADDPG config')
    if t['optimizer'] != 'Adam' or i['noise_type'] != 'ou' or i['evaluation_mode'] != 'deterministic':
        raise RuntimeError('MADDPG requires Adam, OU exploration and deterministic evaluation')
    for key in ('replay_capacity', 'minibatch_size', 'learning_starts', 'gradient_steps_per_vector_step',
                'num_train_envs', 'total_sampled_steps', 'evaluation_episodes', 'evaluation_interval_sampled_steps'):
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if t['minibatch_size'] > t['replay_capacity']:
        raise ValueError('minibatch exceeds replay capacity')
    if min(t['actor_learning_rate'], t['critic_learning_rate']) <= 0 or not 0 <= t['gamma'] <= 1 or not 0 < t['tau'] <= 1:
        raise ValueError('invalid MADDPG optimization hyperparameters')
    if min(i['ou_theta'], i['ou_sigma'], i['exploration_scale_start'], i['exploration_scale_end']) < 0:
        raise ValueError('invalid exploration parameters')
    if min(i['exploration_decay_sampled_steps'], i['checkpoint_interval_sampled_steps']) <= 0:
        raise ValueError('invalid schedule interval')
    if not 10_000_000 <= i['evaluation_seed_base'] < 70_000_000:
        raise ValueError('evaluation must use the development seed block')


def validate_checkpoint(state, env, config):
    if state.get('algorithm') != 'maddpg' or state.get('implementation_version') != MADDPG_IMPL_VERSION:
        raise RuntimeError('MADDPG checkpoint identity/implementation version mismatch')
    validate_config(env, config)
    extra = state.get('extra', {})
    n, t = config['network'], config['training']
    expected = {'environment_version': env['environment_version'],
        **{key: n[key] for key in ('observation_dim', 'action_dim', 'num_agents')},
        'environment_config_sha256': config_sha256(env), 'algorithm_config_sha256': config_sha256(config),
        'gamma': t['gamma'], 'tau': t['tau']}
    for key, value in expected.items():
        if extra.get(key) != value:
            raise RuntimeError(f'MADDPG checkpoint {key} mismatch')
    required = ('training_seed', 'training_num_envs', 'training_total_sampled_steps', 'training_smoke',
                'network_architecture', 'parameter_counts', 'ou_noise_state')
    if any(key not in extra for key in required):
        raise RuntimeError('incomplete MADDPG checkpoint protocol metadata')
    if (state.get('training_seed'), state.get('gamma'), state.get('tau')) != (extra['training_seed'], t['gamma'], t['tau']):
        raise RuntimeError('MADDPG top-level training protocol mismatch')
    architecture = {'observation_dim': n['observation_dim'], 'action_dim': 3, 'num_agents': n['num_agents'],
        'actor_layers': [n['observation_dim'], 256, 256, 3],
        'critic_layers': [n['num_agents']*(n['observation_dim']+3), 256, 256, 1],
        'actor_activation': 'relu', 'actor_output': 'tanh', 'critic_activation': 'relu',
        'independent_actors': n['num_agents'], 'independent_critics': n['num_agents']}
    if state.get('network_architecture') != architecture or extra['network_architecture'] != architecture:
        raise RuntimeError('MADDPG checkpoint network architecture mismatch')
    for key in ('actors', 'target_actors', 'critics', 'target_critics', 'actor_optimizers', 'critic_optimizers'):
        if not isinstance(state.get(key), list) or len(state[key]) != n['num_agents']:
            raise RuntimeError(f'MADDPG checkpoint {key} must have N independent entries')
    a, q = (sum(v.numel() for v in state[key][0].values()) for key in ('actors', 'critics'))
    counts = {'actor_per_agent': a, 'actors_trainable': a*n['num_agents'], 'critic_per_agent': q,
        'critics_trainable': q*n['num_agents'], 'trainable_total': (a+q)*n['num_agents'],
        'target_total_nontrainable': (a+q)*n['num_agents']}
    if extra['parameter_counts'] != counts:
        raise RuntimeError('MADDPG checkpoint parameter count mismatch')
    if state.get('formal_exact_resume_supported') is not False or state.get('replay_buffer_included') is not False:
        raise RuntimeError('MADDPG replay/resume provenance mismatch')
    if state.get('actor_update_count') != state.get('critic_update_count'):
        raise RuntimeError('MADDPG updates must not use policy delay')
    if state.get('sampled_steps') != state.get('vector_steps', -1)*extra['training_num_envs']:
        raise RuntimeError('MADDPG sampled-step unit mismatch')
    return extra
