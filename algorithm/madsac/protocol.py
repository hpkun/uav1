"""Validate combat dimensions and MADSAC checkpoint identity."""
from algorithm.common.protocol import config_sha256
from env.config import ENVIRONMENT_VERSION, validate_config
from .trainer import MADSAC_IMPL_VERSION


def validate_madsac_config(env_config: dict, algorithm_config: dict) -> None:
    validate_config(env_config)
    if algorithm_config.get('algorithm') != 'madsac':
        raise RuntimeError('algorithm config is not madsac')
    if str(env_config.get('environment_version')) != ENVIRONMENT_VERSION:
        raise RuntimeError('MADSAC environment_version mismatch')
    n, t, i = algorithm_config['network'], algorithm_config['training'], algorithm_config['implementation']
    actual = tuple(int(n[key]) for key in ('observation_dim', 'action_dim', 'num_agents'))
    if actual != (52, 3, 4) or int(env_config['scenario']['team_size']) != 4:
        raise RuntimeError(f'MADSAC dimension mismatch: {actual} vs (52, 3, 4)')
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
        'environment_version': ENVIRONMENT_VERSION,
        'environment_config_sha256': config_sha256(env_config),
        'algorithm_config_sha256': config_sha256(algorithm_config),
        'training_seed': int(algorithm_config['training']['seed'] if expected_training_seed is None else expected_training_seed),
        'observation_dim': 52, 'action_dim': 3, 'num_agents': 4,
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            raise RuntimeError(f'MADSAC checkpoint {key} mismatch')
    architecture = extra.get('network_architecture', {})
    n = algorithm_config['network']
    if architecture.get('attention_heads') != n['attention_heads']:
        raise RuntimeError('MADSAC checkpoint attention_heads mismatch')
    expected_hidden = 32 if extra.get('training_smoke') else int(n['actor_hidden_layers'][0])
    if architecture.get('hidden_dim') != expected_hidden:
        raise RuntimeError('MADSAC checkpoint hidden_dim mismatch')
    actor_weight = state.get('actor', {}).get('backbone.0.weight')
    if actor_weight is not None and tuple(actor_weight.shape) != (expected_hidden, 52):
        raise RuntimeError('MADSAC checkpoint actor architecture mismatch')
    if state.get('formal_exact_resume_supported') is not False or state.get('replay_buffer_included') is not False:
        raise RuntimeError('MADSAC checkpoint resume/replay provenance mismatch')


__all__ = ['validate_madsac_checkpoint', 'validate_madsac_config']
