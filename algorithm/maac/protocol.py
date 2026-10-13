"""Strict v3.29 identity; never accepts MAPPO/MADSAC or alternate MAAC variants."""
from env.config import validate_config, environment_dimensions
from algorithm.common.protocol import config_sha256

IMPLEMENTATION_VERSION = 1
VARIANT = 'self_inclusive_context_only_on_policy_deterministic_v1'


def validate_config_pair(env_config, config):
    validate_config(env_config)
    if env_config.get('environment_version') != '3.29' or environment_dimensions(env_config) != (56,3,5):
        raise RuntimeError('MAAC requires exact environment_version 3.29 and 56/3/5')
    if config.get('algorithm') != 'MAAC' or config.get('environment') != {'environment_version':'3.29','team_size':5}:
        raise RuntimeError('MAAC algorithm/environment identity mismatch')
    n, t = config['network'], config['training']
    if tuple(n[k] for k in ('observation_dim','action_dim','num_agents')) != (56,3,5):
        raise RuntimeError('MAAC exact dimensions must be 56/3/5')
    if any(type(n[k]) is not int for k in ('observation_dim','action_dim','num_agents','hidden_dim','attention_heads')):
        raise RuntimeError('MAAC dimensions must be integers')
    if n['hidden_dim'] != 256 or n['attention_heads'] != 2:
        raise RuntimeError('MAAC formal network must be hidden_dim=256 and heads=2')
    if config.get('implementation', {}).get('variant') != VARIANT:
        raise RuntimeError('MAAC variant identity mismatch')
    for key in ('rollout_length','num_train_envs','total_sampled_steps','minibatch_size',
                'evaluation_episodes','evaluation_interval_sampled_steps','checkpoint_interval_sampled_steps',
                'console_interval_sampled_steps','recent_episode_window'):
        if type(t[key]) is not int or t[key] <= 0:
            raise RuntimeError(f'MAAC {key} must be a positive integer')
    if not 0 <= t['gamma'] <= 1 or min(t['lr_actor'],t['lr_critic']) <= 0:
        raise RuntimeError('invalid MAAC gamma/learning rates')


def identity(env_config, config, smoke=False):
    n=config['network']
    return dict(environment_version=env_config['environment_version'],
        environment_config_sha256=config_sha256(env_config), algorithm_config_sha256=config_sha256(config),
        observation_dim=56, action_dim=3, num_agents=5, variant=VARIANT, training_smoke=bool(smoke),
        network_architecture=dict(hidden_dim=32 if smoke else n['hidden_dim'], attention_heads=n['attention_heads'],
            shared_actor=True, single_critic=True, self_attention=True, context_only=True, value_activation='leaky_relu',
            target_actor=False, target_critic=False, replay_buffer=False, entropy_regularization=False),
        formal_exact_resume_supported=False)


def validate_checkpoint(state, env_config, config):
    validate_config_pair(env_config, config)
    if state.get('algorithm') != 'MAAC' or state.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise RuntimeError('MAAC checkpoint algorithm/implementation identity mismatch')
    extra=state.get('extra',{})
    if type(extra.get('training_smoke')) is not bool:
        raise RuntimeError('MAAC checkpoint training_smoke missing/invalid')
    for key, value in identity(env_config, config, extra['training_smoke']).items():
        if extra.get(key) != value:
            raise RuntimeError(f'MAAC checkpoint {key} mismatch')
