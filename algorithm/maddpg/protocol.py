"""Strict independent identity, dimensions, architecture and config fingerprints."""
from algorithm.common.protocol import config_sha256
from env.config import validate_config as validate_environment, environment_dimensions
import math

MADDPG_IMPL_VERSION = 3
SAMPLED_STEPS_UNIT = 'environment_transitions_not_agent_transitions'
SUPPORTED_DIMENSIONS = {'2.5': (65, 3, 5), '2.4': (104, 3, 8),
    **{f'3.{v}': (66, 3, 5) for v in range(3, 12)},
    **{f'3.{v}': (56, 3, 5) for v in range(12, 19)}, '3.38': (45, 3, 4)}


def validate_dimensions(observation_dim, action_dim, num_agents):
    dimensions = (observation_dim, action_dim, num_agents)
    if any(type(v) is not int for v in dimensions) or dimensions not in SUPPORTED_DIMENSIONS.values():
        raise ValueError('MADDPG dimensions are not registered in protocol.SUPPORTED_DIMENSIONS')


def schedule_contract(config):
    t = config['training']
    if 'gradient_steps_per_vector_step' in t or 'sampled_steps_per_update' not in t:
        raise RuntimeError('MADDPG-v3 requires explicit sampled_steps_per_update; legacy vector-step schedule is not reinterpreted')
    return {'sampled_steps_unit': SAMPLED_STEPS_UNIT,
        **{k: t[k] for k in ('sampled_steps_per_update', 'learning_starts', 'replay_capacity', 'minibatch_size')},
        'gradient_updates_per_event':t['gradient_updates_per_event'], 'gradient_clip_norm':t['gradient_clip_norm'],
        'gradient_clipping_scope':'per_agent_global_l2',
        'nominal_utd':t['gradient_updates_per_event']/t['sampled_steps_per_update'],
        'exploration_type': 'OU', 'evaluation_mode': 'deterministic',
        'formal_exact_resume_supported': False, 'replay_buffer_in_checkpoint': False}


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
    if dimensions != environment_dimensions(env) or SUPPORTED_DIMENSIONS.get(str(env['environment_version'])) != dimensions:
        raise RuntimeError('MADDPG requires a registered environment version/dimensions pair; v3.38 requires 45/3/4')
    identity = config.get('environment')
    expected_identity = {'environment_version': env['environment_version'], 'team_size': dimensions[2]}
    if (identity is not None and identity != expected_identity) or (env['environment_version'] == '3.38' and identity is None):
        raise RuntimeError('MADDPG algorithm/environment identity mismatch (environment_version/team_size)')
    if n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]:
        raise RuntimeError('MADDPG requires independent 256/256 MLPs')
    forbidden = {'attention_heads', 'actor_type', 'gru_hidden_dim', 'policy_delay', 'alpha',
                 'target_policy_noise', 'log_std_min', 'log_std_max', 'log_std', 'policy_std_mode', 'entropy_coefficient',
                 'shared_actor', 'shared_critic', 'attention', 'twin_critic', 'maximum_entropy', 'target_policy_smoothing', 'transformer'}
    def forbidden_key(section):
        return any(key in forbidden or any(token in key.lower() for token in ('attention', 'log_std', 'transformer', 'gru'))
                   or isinstance(value, dict) and forbidden_key(value) for key, value in section.items())
    if any(forbidden_key(section) for section in (n, t, i)):
        raise RuntimeError('unsupported mechanism in classic MADDPG config')
    if t['optimizer'] != 'Adam' or i['noise_type'] != 'ou' or i['evaluation_mode'] != 'deterministic':
        raise RuntimeError('MADDPG requires Adam, OU exploration and deterministic evaluation')
    schedule_key = 'sampled_steps_per_update' if 'sampled_steps_per_update' in t else 'gradient_steps_per_vector_step'
    if env['environment_version'] == '3.38' or schedule_key == 'sampled_steps_per_update':
        schedule_contract(config)
    for key in ('replay_capacity', 'minibatch_size', 'learning_starts', schedule_key,
                'num_train_envs', 'total_sampled_steps', 'evaluation_episodes', 'evaluation_interval_sampled_steps'):
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if schedule_key == 'sampled_steps_per_update':
        if type(t['gradient_updates_per_event']) is not int or t['gradient_updates_per_event'] <= 0:
            raise ValueError('gradient_updates_per_event must be a positive integer')
        if not math.isfinite(t['gradient_clip_norm']) or t['gradient_clip_norm'] <= 0:
            raise ValueError('gradient_clip_norm must be finite and positive')
        if type(t.get('diagnostics_interval_updates',100)) is not int or t.get('diagnostics_interval_updates',100) <= 0:
            raise ValueError('diagnostics_interval_updates must be a positive integer')
    if t['minibatch_size'] > t['replay_capacity']:
        raise ValueError('minibatch exceeds replay capacity')
    if schedule_key == 'sampled_steps_per_update' and t['learning_starts'] < t['minibatch_size']:
        raise ValueError('learning_starts must cover minibatch_size so eligible update credit is unambiguous')
    numeric = [t[k] for k in ('actor_learning_rate', 'critic_learning_rate', 'gamma', 'tau')]
    numeric += [i[k] for k in ('ou_theta', 'ou_sigma', 'ou_mu', 'exploration_scale_start', 'exploration_scale_end')]
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError('non-finite MADDPG hyperparameter')
    if min(t['actor_learning_rate'], t['critic_learning_rate']) <= 0 or not 0 <= t['gamma'] <= 1 or not 0 < t['tau'] <= 1:
        raise ValueError('invalid MADDPG optimization hyperparameters')
    if min(i['ou_theta'], i['ou_sigma'], i['exploration_scale_start'], i['exploration_scale_end']) < 0:
        raise ValueError('invalid exploration parameters')
    if min(i['exploration_decay_sampled_steps'], i['checkpoint_interval_sampled_steps']) <= 0:
        raise ValueError('invalid schedule interval')
    if not 10_000_000 <= i['evaluation_seed_base'] < 70_000_000:
        raise ValueError('evaluation must use the development seed block')
    for key in ('console_interval_sampled_steps', 'recent_episode_window'):
        if type(config['runtime_logging'][key]) is not int or config['runtime_logging'][key] <= 0:
            raise ValueError(f'{key} must be a positive integer')


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
    contract = schedule_contract(config)
    for key, value in contract.items():
        if extra.get(key) != value or state.get(key) != value:
            raise RuntimeError(f'MADDPG checkpoint schedule metadata {key} mismatch')
    remainder = extra.get('transitions_since_update')
    if type(remainder) is not int or not 0 <= remainder < contract['sampled_steps_per_update'] or state.get('transitions_since_update') != remainder:
        raise RuntimeError('MADDPG checkpoint transitions_since_update mismatch')
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
    events, credit = divmod(max(0, state['sampled_steps']-t['learning_starts']), t['sampled_steps_per_update'])
    if state['actor_update_count'] != events*t['gradient_updates_per_event'] or remainder != credit or state.get('update_event_count') != events or extra.get('update_event_count') != events:
        raise RuntimeError('MADDPG checkpoint update schedule counts/remainder mismatch')
    expected_utd = {name:state[name+'_update_count']/max(1,state['sampled_steps']) for name in ('actor','critic')}
    if state.get('effective_utd') != expected_utd:
        raise RuntimeError('MADDPG checkpoint effective UTD mismatch')
    for name,width in [('actors',counts['actor_per_agent']),('critics',counts['critic_per_agent'])]:
        values=state.get('initial_parameter_references',{}).get(name,[])
        if len(values)!=n['num_agents'] or any(v.numel()!=width or v.requires_grad for v in values):
            raise RuntimeError('MADDPG initial parameter reference mismatch')
    for key in ('actor_updates_per_agent', 'critic_updates_per_agent'):
        values = state.get(key)
        if not isinstance(values, list) or len(values) != n['num_agents'] or any(type(v) is not int or not 0 <= v <= events*t['gradient_updates_per_event'] for v in values):
            raise RuntimeError(f'MADDPG checkpoint {key} mismatch')
    return extra
