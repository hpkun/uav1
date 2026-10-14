"""Validate combat dimensions and MADSAC checkpoint identity."""
from algorithm.common.protocol import config_sha256
from env.config import ENVIRONMENT_VERSION, validate_config, environment_dimensions
from .trainer import MADSAC_IMPL_VERSION
from .critic_contract import CRITIC_ARCHITECTURE, validate_critic_shapes


def validate_madsac_config(env_config: dict, algorithm_config: dict) -> None:
    validate_config(env_config)
    if algorithm_config.get('algorithm') != 'madsac':
        raise RuntimeError('algorithm config is not madsac')
    if str(env_config.get('environment_version')) not in (ENVIRONMENT_VERSION,'3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15','3.16','3.17','3.18','3.26','3.27','3.28','3.29','3.30'):
        raise RuntimeError('MADSAC environment_version mismatch: unsupported protocol; MADSAC supports only v2.3 4v4/52D or v3.3/v3.4/v3.5/v3.6/v3.7/v3.8/v3.9/v3.10/v3.11 5v5/66D or v3.12-v3.18 and v3.26-v3.29 5v5/56D or v3.30 4v4/45D')
    n, t, i = algorithm_config['network'], algorithm_config['training'], algorithm_config['implementation']
    if env_config['environment_version'] in ('3.26', '3.27', '3.28','3.29','3.30'):
        if algorithm_config.get('environment') != {'environment_version': env_config['environment_version'], 'team_size': environment_dimensions(env_config)[2]}:
            raise RuntimeError(f"MADSAC environment identity must be {env_config['environment_version']}/{environment_dimensions(env_config)[2]}v{environment_dimensions(env_config)[2]}")
        dimensions = tuple(n[key] for key in ('observation_dim', 'action_dim', 'num_agents'))
        if dimensions != environment_dimensions(env_config) or any(type(value) is not int for value in dimensions):
            raise RuntimeError(f'MADSAC requires exact dimensions {environment_dimensions(env_config)}')
        if (n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]
                or n['attention_heads'] != 2):
            raise RuntimeError('MADSAC v3.26-v3.30 requires 256/256 shared actor and two-head attention critics')
    actual = tuple(int(n[key]) for key in ('observation_dim', 'action_dim', 'num_agents'))
    expected=environment_dimensions(env_config)
    if actual != expected:
        raise RuntimeError(f'MADSAC dimension mismatch: {actual} vs {expected}')
    if int(t['evaluation_episodes']) <= 0:
        raise RuntimeError('evaluation_episodes must be positive')
    if i['evaluation_mode'] not in ('stochastic', 'deterministic'):
        raise RuntimeError('MADSAC evaluation mode must be stochastic or deterministic')
    for key in ('num_train_envs', 'total_sampled_steps', 'replay_capacity', 'minibatch_size',
                'gradient_steps_per_vector_step', 'policy_delay', 'evaluation_interval_sampled_steps'):
        if int(t[key]) <= 0:
            raise RuntimeError(f'{key} must be positive')


def validate_madsac_checkpoint(state: dict, env_config: dict, algorithm_config: dict,
                               *, expected_training_seed: int | None = None) -> None:
    validate_madsac_config(env_config, algorithm_config)
    if state.get('algorithm') != 'madsac' or state.get('implementation_version') != MADSAC_IMPL_VERSION:
        raise RuntimeError('MADSAC checkpoint identity/version mismatch')
    extra = state.get('extra', {})
    expected = {
        'environment_version': env_config['environment_version'],
        'environment_config_sha256': config_sha256(env_config),
        'algorithm_config_sha256': config_sha256(algorithm_config),
        'training_seed': int(algorithm_config['training']['seed'] if expected_training_seed is None else expected_training_seed),
        **dict(zip(('observation_dim','action_dim','num_agents'),environment_dimensions(env_config))),
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            raise RuntimeError(f'MADSAC checkpoint {key} mismatch')
    architecture = extra.get('network_architecture', {})
    for key, value in CRITIC_ARCHITECTURE.items():
        if architecture.get(key) != value:
            raise RuntimeError(f'MADSAC checkpoint architecture {key} mismatch')
    n = algorithm_config['network']
    if env_config['environment_version'] in ('3.26', '3.27', '3.28','3.29','3.30'):
        for key, value in {'shared_actor': True, 'twin_independent_critics': True,
                           'target_actor': True, 'target_critics': 2,
                           **dict(zip(('observation_dim','action_dim','num_agents'),environment_dimensions(env_config)))}.items():
            if architecture.get(key) != value:
                raise RuntimeError(f'MADSAC checkpoint architecture {key} mismatch')
        if extra.get('formal_exact_resume_supported') is not False or extra.get('replay_buffer_in_checkpoint') is not False:
            raise RuntimeError('MADSAC checkpoint extra resume/replay provenance mismatch')
    if architecture.get('attention_heads') != n['attention_heads']:
        raise RuntimeError('MADSAC checkpoint attention_heads mismatch')
    expected_hidden = 32 if extra.get('training_smoke') else int(n['actor_hidden_layers'][0])
    if architecture.get('hidden_dim') != expected_hidden:
        raise RuntimeError('MADSAC checkpoint hidden_dim mismatch')
    validate_critic_shapes(state, expected_hidden)
    actor_weight = state.get('actor', {}).get('backbone.0.weight')
    if actor_weight is not None and tuple(actor_weight.shape) != (expected_hidden, expected['observation_dim']):
        raise RuntimeError('MADSAC checkpoint actor architecture mismatch')
    if state.get('formal_exact_resume_supported') is not False or state.get('replay_buffer_included') is not False:
        raise RuntimeError('MADSAC checkpoint resume/replay provenance mismatch')


__all__ = ['validate_madsac_checkpoint', 'validate_madsac_config']
