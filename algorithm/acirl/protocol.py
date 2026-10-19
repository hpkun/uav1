"""Independent ACIRL identity, ablation, schema and checkpoint contracts."""
import math
import torch
from algorithm.common.protocol import config_sha256
from env.config import validate_config as validate_environment,environment_dimensions

ACIRL_IMPL_VERSION=1
MODES=('full','no_intent','fixed_interval','no_gate_q','no_teammate_inference','no_temporal_critic','no_switching_cost')


def require_cuda(device):
    if torch.device(device).type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for ACIRL training, evaluation and checkpoint audits')


def architecture(config):
    return dict(network=config['network'],intention=config['intention'],ablation=config['ablation'],
        entity_layout='self_then_sorted_other_allies_then_fixed_enemies',
        actor_information='own_observation_and_own_history_and_inferred_distributions_only',
        macro_cut='bootstrap_within_behavior_rollout_no_future_policy_rewards',
        critic_state='pre_gate_observation_alive_post_observation_GRU_previous_intent_age',
        timeout_semantics='terminated_or_truncated_stops_bootstrap',implementation_version=ACIRL_IMPL_VERSION)


def validate_config(env,config):
    validate_environment(env)
    if config.get('algorithm')!='acirl':raise RuntimeError('algorithm config is not acirl')
    n,t,p=config['network'],config['training'],config['intention']
    for section in (n,t,p):
        if any(isinstance(value,(int,float)) and not isinstance(value,bool) and not math.isfinite(value) for value in section.values()):
            raise ValueError('ACIRL numerical configuration values must be finite')
    dims=tuple(n[k] for k in ('observation_dim','action_dim','num_agents'))
    # Register compatible semantics explicitly; new schemas require a new adapter.
    if str(env['environment_version']) not in ('3.39','3.41') or dims!=environment_dimensions(env):
        raise ValueError('ACIRL requires registered 7/6/5 entity semantics and matching environment dimensions')
    if config['environment']!={'environment_version':env['environment_version'],'team_size':dims[2]}:
        raise RuntimeError('ACIRL environment identity mismatch')
    if tuple(n[k] for k in ('self_features','ally_features','enemy_features'))!=(7,6,5):raise ValueError('incorrect local entity semantics')
    if dims[0]!=n['self_features']+(dims[2]-1)*n['ally_features']+dims[2]*n['enemy_features'] or dims[1]!=3:
        raise ValueError('observation/action layout mismatch')
    mode=config['ablation']['mode']
    if mode not in MODES:raise ValueError('unknown ACIRL ablation mode')
    for k in ('observation_dim','action_dim','num_agents','entity_embedding','attention_heads','gru_hidden','intentions','intention_embedding'):
        if type(n[k]) is not int or n[k]<=0:raise ValueError(f'network.{k} must be a positive integer')
    if n['num_agents']<2 or n['entity_embedding']%n['attention_heads']:raise ValueError('invalid attention dimensions')
    for k in ('actor_hidden','critic_hidden'):
        if not n[k] or any(type(v) is not int or v<=0 for v in n[k]):raise ValueError('invalid MLP widths')
    if not n['log_std_min']<=n['log_std_init']<=n['log_std_max']:raise ValueError('invalid log standard deviation bounds')
    if not 0<n['mean_head_gain']<=1:raise ValueError('invalid mean head initialization gain')
    for k in ('min_duration','max_duration','fixed_interval'):
        if type(p[k]) is not int or p[k]<=0:raise ValueError('duration must be a positive integer')
    if not p['min_duration']<=p['fixed_interval']<=p['max_duration']:raise ValueError('invalid duration interval')
    if n['max_duration_scale']!=p['max_duration']:raise ValueError('duration normalization mismatch')
    for k in ('num_train_envs','total_sampled_steps','rollout_steps','sequence_length','minibatch_size','critic_minibatch_size','ppo_epochs',
              'evaluation_episodes','evaluation_interval','checkpoint_interval'):
        if type(t[k]) is not int or t[k]<=0:raise ValueError(f'training.{k} must be positive integer')
    sequence_agents=t['sequence_length']*n['num_agents']
    if t['minibatch_size']<sequence_agents or t['minibatch_size']%sequence_agents:
        raise ValueError('minibatch must contain whole joint recurrent sequences, counted as agent-time samples')
    if t['total_sampled_steps']%t['num_train_envs']:raise ValueError('budget must be divisible by environment count')
    for k in ('gamma','gae_lambda','lambda_high'):
        if not 0<=t[k]<=1:raise ValueError(f'invalid {k}')
    for k in ('actor_learning_rate','critic_learning_rate','max_grad_norm','target_kl','clip_ratio'):
        if not math.isfinite(t[k]) or t[k]<=0:raise ValueError(f'invalid {k}')
    if t['switching_cost']<0 or (mode=='no_switching_cost' and t['switching_cost']!=0):raise ValueError('invalid switching cost')
    if t['reward_reduction'] not in ('mean','sum') or t['evaluation_mode'] not in ('deterministic','stochastic'):raise ValueError('invalid reward/evaluation protocol')
    if type(t['seed']) is not int or t['seed']<0 or t['seed']+t['total_sampled_steps']+t['num_train_envs']>=t['evaluation_seed_base']:
        raise ValueError('invalid or overlapping training/evaluation seeds')
    if not 10_000_000<=t['evaluation_seed_base']<70_000_000:raise ValueError('formal selection uses development seeds')
    for section in ('loss_weights','entropy'):
        if any(not math.isfinite(v) or v<0 for v in config[section].values()):raise ValueError(f'invalid {section}')
    required=['action','team_value']
    if mode!='no_intent':required.append('intention')
    if mode not in ('no_intent','fixed_interval'):required.append('gate')
    if mode not in ('no_intent','fixed_interval','no_gate_q'):required.append('gate_q')
    if mode not in ('no_intent','no_teammate_inference'):required.append('inference')
    if any(config['loss_weights'][k]<=0 for k in required):raise ValueError('active mechanisms require positive loss weights')


def validate_checkpoint(state,env,config):
    if state.get('algorithm')!='acirl' or state.get('implementation_version')!=ACIRL_IMPL_VERSION:
        raise RuntimeError('ACIRL checkpoint identity/implementation mismatch')
    validate_config(env,config)
    expected={'environment_config_sha256':config_sha256(env),'algorithm_config_sha256':config_sha256(config),
              'network_architecture':architecture(config),'ablation_mode':config['ablation']['mode']}
    for key,value in expected.items():
        if state.get(key)!=value:raise RuntimeError(f'ACIRL checkpoint {key} mismatch')
    for key in ('environment_config_sha256','algorithm_config_sha256'):
        if state.get('extra',{}).get(key)!=expected[key]:raise RuntimeError(f'ACIRL common checkpoint {key} mismatch')
    if config_sha256(state.get('environment_config'))!=config_sha256(env) or config_sha256(state.get('algorithm_config'))!=config_sha256(config):
        raise RuntimeError('ACIRL embedded config fingerprint mismatch')
    for key in ('policy','critics','policy_optimizer','team_optimizer','rng_states','parameter_counts','initial_parameters',
                'evaluation_history','best_evaluation','training_runtime'):
        if key not in state:raise RuntimeError(f'ACIRL checkpoint missing {key}')
    gate_q=config['ablation']['mode'] not in ('no_intent','fixed_interval','no_gate_q')
    if gate_q and state.get('gate_optimizer') is None:raise RuntimeError('ACIRL missing gate critic optimizer')
    if state.get('formal_exact_resume_supported') is not False:raise RuntimeError('ACIRL does not support exact environment resume')
    for key in ('sampled_steps','vector_steps','update_count','policy_optimizer_steps','team_optimizer_steps','gate_optimizer_steps'):
        if type(state.get(key)) is not int or state[key]<0:raise RuntimeError(f'ACIRL invalid counter {key}')
    if state['sampled_steps']!=state['vector_steps']*config['training']['num_train_envs']:raise RuntimeError('sampled transition unit mismatch')
    if state.get('training_seed')!=config['training']['seed'] or state.get('extra',{}).get('training_seed')!=config['training']['seed']:
        raise RuntimeError('ACIRL checkpoint seed mismatch')
    for key in ('behavior','entropy','permutation','torch','cuda','numpy','python'):
        if key not in state['rng_states']:raise RuntimeError(f'ACIRL missing RNG {key}')
    for name,counter in (('policy_optimizer','policy_optimizer_steps'),('team_optimizer','team_optimizer_steps'),('gate_optimizer','gate_optimizer_steps')):
        optimizer=state.get(name)
        if optimizer is None:
            if state[counter]!=0:raise RuntimeError(f'ACIRL inactive {name} has optimizer steps')
            continue
        for item in optimizer['state'].values():
            if 'step' not in item or not 0<int(item['step'])<=state[counter]:
                raise RuntimeError(f'ACIRL {name} actual Adam step mismatch')
    return state
