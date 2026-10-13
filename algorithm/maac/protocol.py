"""Strict identity for paper MAAC's continuous score-gradient adaptation."""
from env.config import validate_config,environment_dimensions
from algorithm.common.protocol import config_sha256

IMPLEMENTATION_VERSION=2
VARIANT='paper_maac_continuous_eq8_mc_v2'


def validate_config_pair(env_config,config):
    validate_config(env_config)
    if env_config.get('environment_version')!='3.29' or environment_dimensions(env_config)!=(56,3,5):
        raise RuntimeError('MAAC requires exact environment_version 3.29 and 56/3/5')
    if config.get('algorithm')!='MAAC' or config.get('environment')!={'environment_version':'3.29','team_size':5}:
        raise RuntimeError('MAAC algorithm/environment identity mismatch')
    n,t=config['network'],config['training']
    expected=dict(observation_dim=56,action_dim=3,num_agents=5,hidden_dim=128,attention_heads=4)
    if any(type(n.get(k)) is not int or n[k]!=v for k,v in expected.items()):
        raise RuntimeError('MAAC exact dimensions/appendix architecture must be 56/3/5, hidden=128, heads=4')
    if config.get('implementation',{}).get('variant')!=VARIANT:raise RuntimeError('MAAC variant identity mismatch')
    for k in ('num_train_envs','total_sampled_steps','minibatch_size','replay_capacity','learning_starts',
              'sampled_steps_per_update','critic_updates_per_cycle','policy_updates_per_cycle',
              'evaluation_episodes','evaluation_interval_sampled_steps','checkpoint_interval_sampled_steps',
              'console_interval_sampled_steps','recent_episode_window','baseline_samples'):
        if type(t[k]) is not int or t[k]<=0:raise RuntimeError(f'MAAC {k} must be a positive integer')
    if t['replay_capacity']<max(t['minibatch_size'],t['learning_starts']):raise RuntimeError('MAAC replay too small')
    if not 0<=t['gamma']<=1 or not 0<t['tau']<=1 or t['alpha']<0 or min(t['lr_actor'],t['lr_critic'])<=0:
        raise RuntimeError('invalid MAAC discount/temperature/update/learning rates')


def identity(env_config,config,smoke=False):
    n=config['network'];t=config['training']
    return dict(environment_version=env_config['environment_version'],
        environment_config_sha256=config_sha256(env_config),algorithm_config_sha256=config_sha256(config),
        observation_dim=56,action_dim=3,num_agents=5,variant=VARIANT,training_smoke=bool(smoke),
        network_architecture=dict(hidden_dim=32 if smoke else n['hidden_dim'],attention_heads=n['attention_heads'],
            independent_actors=5,shared_actor=False,single_attention_critic=True,self_attention=False,
            own_embedding_shortcut=True,value_activation='leaky_relu',target_actors=5,target_critic=True,
            actor_distribution='tanh_gaussian',policy_gradient='eq8_score_function',baseline='independent_monte_carlo'),
        gamma=t['gamma'],alpha=t['alpha'],tau=t['tau'],replay_buffer_in_checkpoint=False,
        formal_exact_resume_supported=False)


def validate_checkpoint(state,env_config,config):
    validate_config_pair(env_config,config)
    if state.get('algorithm')!='MAAC' or state.get('implementation_version')!=IMPLEMENTATION_VERSION:
        raise RuntimeError('MAAC checkpoint algorithm/implementation identity mismatch (v1 draft is incompatible)')
    extra=state.get('extra',{})
    if type(extra.get('training_smoke')) is not bool:raise RuntimeError('MAAC checkpoint training_smoke missing/invalid')
    for k,v in identity(env_config,config,extra['training_smoke']).items():
        if extra.get(k)!=v:raise RuntimeError(f'MAAC checkpoint {k} mismatch')
