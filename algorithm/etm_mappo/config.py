"""Explicit standalone navigation experiment contract."""
import copy
import hashlib
import json
import math
import torch

IMPLEMENTATION_VERSION = 1
ENVIRONMENT_VERSION = 'cooperative_navigation_v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require_cuda(device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for ETM-MAPPO training/evaluation/audits')


def validate_config(config):
    if config.get('algorithm') != 'ETM-MAPPO':
        raise ValueError('algorithm must be ETM-MAPPO')
    n,t,e = (config[key] for key in ('network','training','environment'))
    if e.get('version') != ENVIRONMENT_VERSION:
        raise ValueError('only the standalone cooperative navigation environment is supported')
    if e.get('observation_schema') != 'local_task_state_v1':
        raise ValueError('navigation requires the explicit local-task observation schema')
    if n['variant'] not in ('B0','B1','B2','B3'):
        raise ValueError('variant must be B0, B1, B2 or B3')
    for key in ('self_dim','peer_dim','obstacle_dim','action_dim','memory_dim','fusion_dim','attention_heads','critic_dim'):
        if type(n[key]) is not int or n[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if n['memory_dim'] % n['attention_heads'] or n['critic_dim'] % n['attention_heads']:
        raise ValueError('attention dimensions must be divisible by heads')
    if not n['log_std_min'] <= n['log_std_init'] <= n['log_std_max'] or not all(
        math.isfinite(n[key]) for key in ('log_std_min','log_std_init','log_std_max')):
        raise ValueError('invalid Gaussian std configuration')
    for key in ('num_envs','total_sampled_steps','rollout_steps','sequence_length','ppo_epochs','minibatch_joint_transitions',
                'evaluation_episodes','evaluation_interval','checkpoint_interval','torch_threads'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if t['minibatch_joint_transitions'] % t['sequence_length']:
        raise ValueError('minibatch joint transitions must be divisible by sequence length')
    if t['total_sampled_steps'] % t['num_envs']:
        raise ValueError('sample budget must be divisible by num_envs')
    for key in ('gamma','gae_lambda'):
        if not math.isfinite(t[key]) or not 0 <= t[key] <= 1:
            raise ValueError(f'invalid {key}')
    for key in ('actor_learning_rate','critic_learning_rate','clip_ratio','max_grad_norm'):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('entropy_coefficient','value_loss_coefficient'):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f'invalid {key}')
    if t['target_kl'] is not None and (not math.isfinite(t['target_kl']) or t['target_kl'] <= 0):
        raise ValueError('target_kl must be positive or null')
    if type(t['memory_reset_each_step']) is not bool or type(t['normalize_advantages']) is not bool or type(t['clip_value_loss']) is not bool:
        raise ValueError('PPO feature switches must be booleans')
    if type(t['seed']) is not int or t['seed'] < 0 or t['seed']+t['total_sampled_steps']+t['num_envs'] >= t['evaluation_seed_base']:
        raise ValueError('training and evaluation seed ranges overlap or are invalid')
    if type(e['num_agents']) is not int or e['num_agents'] < 1 or type(e['num_obstacles']) is not int or e['num_obstacles'] < 0:
        raise ValueError('invalid entity counts')
    if e['scene'] not in ('simple','random'):
        raise ValueError('scene must be simple or random')
    for key in ('dt','max_speed','max_acceleration','visibility_radius','robot_radius','obstacle_radius','goal_tolerance'):
        if not math.isfinite(e[key]) or e[key] <= 0:
            raise ValueError(f'invalid environment {key}')
    if type(e['max_steps']) is not int or e['max_steps'] <= 0 or type(e['shuffle_slots']) is not bool:
        raise ValueError('invalid environment horizon/slot shuffle')
    if set(e['reward']) != {'progress','formation','collision','smoothness'} or any(not math.isfinite(v) or v < 0 for v in e['reward'].values()):
        raise ValueError('navigation reward has only four finite nonnegative weights')
    return config


def runtime_config(config, **overrides):
    result = copy.deepcopy(config)
    for key,value in overrides.items():
        if value is not None:
            if key == 'variant':
                result['network'][key] = value
            else:
                result['training'][key] = value
    return validate_config(result)

