"""Strict Li et al. MAAC shared-policy continuous baseline identity."""
from env.config import validate_config,environment_dimensions
from algorithm.common.protocol import config_sha256

IMPLEMENTATION_VERSION=3
VARIANT='madsac_paper_maac_continuous_shared_eq15_mc_v3'
DIMENSIONS={'3.29':(56,3,5),'3.30':(45,3,4),'3.31':(45,3,4),'3.32':(45,3,4),'3.33':(45,3,4),'3.34':(45,3,4),'3.35':(45,3,4),'3.36':(45,3,4),'3.37':(45,3,4),'3.38':(45,3,4),'3.39':(45,3,4),'3.41':(45,3,4),'3.44':(49,3,4),'3.46':(49,3,4),'3.47':(45,3,4)}


def validate_config_pair(env_config,config):
    validate_config(env_config)
    expected_dims=DIMENSIONS.get(env_config.get('environment_version'))
    if expected_dims is None or environment_dimensions(env_config)!=expected_dims:
        raise RuntimeError('MAAC requires a registered supported environment identity (environment_version) with exact dimensions: v3.29 uses 56/3/5; v3.30-v3.39, v3.41 and v3.47 use 45/3/4; v3.44 and v3.46 use 49/3/4')
    if config.get('algorithm')!='MAAC' or config.get('environment')!={'environment_version':env_config['environment_version'],'team_size':expected_dims[2]}:
        raise RuntimeError('MAAC algorithm/environment identity mismatch')
    n,t=config['network'],config['training']
    expected=dict(zip(('observation_dim','action_dim','num_agents'),expected_dims))
    expected.update(hidden_dim=128,attention_heads=4)
    if any(type(n.get(k)) is not int or n[k]!=v for k,v in expected.items()):
        raise RuntimeError(f'MAAC exact dimensions must be {expected_dims}, hidden=128, heads=4')
    if config.get('implementation',{}).get('variant')!=VARIANT:raise RuntimeError('MAAC variant identity mismatch')
    for k in ('num_train_envs','total_sampled_steps','minibatch_size','replay_capacity','learning_starts',
              'sampled_steps_per_update','critic_updates_per_cycle','policy_updates_per_cycle',
              'evaluation_episodes','evaluation_interval_sampled_steps','checkpoint_interval_sampled_steps',
              'console_interval_sampled_steps','recent_episode_window','baseline_samples'):
        if type(t[k]) is not int or t[k]<=0:raise RuntimeError(f'MAAC {k} must be a positive integer')
    if t['replay_capacity']<max(t['minibatch_size'],t['learning_starts']):raise RuntimeError('MAAC replay too small')
    if not 0<=t['gamma']<=1 or not 0<t['tau']<=1 or t['alpha']<0 or min(t['lr_actor'],t['lr_critic'])<=0:
        raise RuntimeError('invalid MAAC discount/temperature/update/learning rates')
    from env.v343_config import validate_shaping_gamma
    validate_shaping_gamma(env_config,config)


def identity(env_config,config,smoke=False):
    n=config['network'];t=config['training']
    return dict(environment_version=env_config['environment_version'],
        environment_config_sha256=config_sha256(env_config),algorithm_config_sha256=config_sha256(config),
        **dict(zip(('observation_dim','action_dim','num_agents'),environment_dimensions(env_config))),variant=VARIANT,training_smoke=bool(smoke),
        network_architecture=dict(hidden_dim=32 if smoke else n['hidden_dim'],attention_heads=n['attention_heads'],
            independent_actors=False,shared_actor=True,single_attention_critic=True,self_attention=False,
            own_embedding_shortcut=True,value_activation='leaky_relu',target_actors=1,target_critic=True,
            actor_distribution='tanh_gaussian',policy_gradient='eq15_score_function',baseline='independent_monte_carlo',
            delayed_policy_update=False,maximum_entropy=True),
        gamma=t['gamma'],alpha=t['alpha'],tau=t['tau'],replay_buffer_in_checkpoint=False,
        formal_exact_resume_supported=False)


def validate_checkpoint(state,env_config,config):
    validate_config_pair(env_config,config)
    if state.get('algorithm')!='MAAC' or state.get('implementation_version')!=IMPLEMENTATION_VERSION:
        raise RuntimeError('MAAC checkpoint algorithm/implementation identity mismatch (MAAC v1/v2 are incompatible with shared v3)')
    extra=state.get('extra',{})
    if type(extra.get('training_smoke')) is not bool:raise RuntimeError('MAAC checkpoint training_smoke missing/invalid')
    for k,v in identity(env_config,config,extra['training_smoke']).items():
        if extra.get(k)!=v:raise RuntimeError(f'MAAC checkpoint {k} mismatch')


def validate_checkpoint_for_resume(state,env_config,config):
    validate_checkpoint(state,env_config,config)
    raise RuntimeError('MAAC exact resume unsupported: replay/environment/full RNG state are not saved')
