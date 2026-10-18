"""Explicit fixed-4v4 identities and exact network/PPO/config fingerprints."""
from copy import deepcopy
from pathlib import Path
import yaml
import torch
from env.config import validate_config as validate_environment
from algorithm.common.protocol import config_sha256
from algorithm.common.policy_protocol import validate_trainer_policy_protocol, configured_policy_protocol
from .networks import OBSERVATION_LAYOUT

IMPLEMENTATION_VERSION = 1
VARIANTS = {'MAPPO-CA': ('mlp', 'attention'), 'EA-MAPPO': ('entity_attention', 'mlp'),
            'EAA-MAPPO': ('entity_attention', 'attention')}
ROOT = Path(__file__).resolve().parents[2]


def require_cuda(device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for EAA family training and checkpoint/evaluation audits')


def validate_config(env, config):
    validate_environment(env)
    version = str(env['environment_version'])
    if version not in ('3.38', '3.39', '3.41'):
        raise ValueError('attention v1 requires v3.38, v3.39 or v3.41 / team_size=4')
    baseline = yaml.safe_load((ROOT/f'configs/mappo_4v4_v{version.replace(".", "")}.yaml').read_text(encoding='utf-8'))
    if config.get('environment') != baseline['environment']:
        raise ValueError('attention v1 requires matching environment identity / team_size=4')
    variant = config.get('algorithm')
    if variant not in VARIANTS:
        raise ValueError('unknown EAA family variant')
    expected = deepcopy(baseline['network'])
    actor_type, critic_type = VARIANTS[variant]
    expected.update(actor_type=actor_type, critic_type=critic_type)
    if actor_type == 'entity_attention':
        expected.update(entity_embedding_dim=64, entity_attention_heads=4, entity_attention_head_dim=16)
    if config['network'] != expected:
        raise ValueError('variant network mismatch: fixed 45/3/4, hidden256, entity64/4heads, critic2heads')
    # Pin the entire existing formal numerical/implementation protocol, not a hand-picked subset.
    for section in ('training', 'implementation', 'runtime_logging'):
        if config[section] != baseline[section]:
            raise ValueError(f'{section} differs from frozen MAPPO baseline')


def architecture(config):
    n = config['network']
    return dict(observation_layout=dict(OBSERVATION_LAYOUT), observation_dim=45, action_dim=3, num_agents=4,
        actor_type=n['actor_type'], critic_type=n['critic_type'], actor_hidden_layers=[256,256],
        critic_hidden_layers=[256,256], critic_attention_heads=2 if n['critic_type']=='attention' else None,
        entity_embedding_dim=64 if n['actor_type']=='entity_attention' else None,
        entity_attention_heads=4 if n['actor_type']=='entity_attention' else None,
        entity_attention_head_dim=16 if n['actor_type']=='entity_attention' else None,
        slot_positional_embedding=False, recurrent=False, biased_random_mask=False)


def contract(env, config):
    return dict(algorithm_variant=config['algorithm'], implementation_version=IMPLEMENTATION_VERSION,
        environment_version=env['environment_version'], environment_config_sha256=config_sha256(env),
        algorithm_config_sha256=config_sha256(config), network_architecture=architecture(config),
        ppo_hyperparameters=deepcopy(config['training']), policy_protocol=configured_policy_protocol(config))


def validate_checkpoint(state, env, config):
    validate_config(env, config)
    expected = contract(env, config)
    if state.get('algorithm') != config['algorithm'] or state.get('checkpoint_family') != checkpoint_family(env):
        raise RuntimeError('checkpoint variant/family mismatch; baseline and legacy EA are not interchangeable')
    for key, value in expected.items():
        if state.get(key) != value or state.get('extra', {}).get(key) != value:
            raise RuntimeError(f'EAA checkpoint {key} mismatch')
    validate_trainer_policy_protocol(state, configured_policy_protocol(config))
    actual = {name:sum(v.numel() for v in state[name].values()) for name in ('actor','critic')}
    if state.get('parameter_counts') != dict(actor=actual['actor'], critic=actual['critic'], total=sum(actual.values())):
        raise RuntimeError('checkpoint parameter counts mismatch')
    return state['extra']


def checkpoint_family(env):
    return {'3.38': 'eaa_mappo_v338', '3.39': 'eaa_mappo_v339', '3.41': 'eaa_mappo_v341'}[env['environment_version']]
