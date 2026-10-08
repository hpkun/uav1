"""Config validation and keyword mapping for formal 5v5 and 8v8 controls."""
from algorithm.common.critic_protocol import layer_width


def validate_control_config(config, algorithm):
    if config.get('algorithm') != algorithm:
        raise ValueError(f'algorithm must be {algorithm}')
    n,t,i = (config[key] for key in ('network','training','implementation'))
    if tuple(n[k] for k in ('observation_dim','action_dim','num_agents')) not in ((65,3,5),(104,3,8)):
        raise ValueError('formal controls require observation/action/agents 65/3/5 or 104/3/8')
    if n['critic_type'] != 'mlp' or layer_width(n,'critic_hidden_layers') != 256:
        raise ValueError('formal controls require CentralizedMLPCritic [256,256]')
    expected = ({'actor_type':'flat_recurrent','flat_encoder_dim':128,'gru_hidden_dim':128,
                 'gru_layers':1,'recurrent_sequence_length':32} if algorithm == 'RMAPPO' else
                {'actor_type':'entity_attention','entity_dim':64,'entity_attention_heads':2,'spatial_hidden_dim':128})
    if any(n.get(key) != value for key,value in expected.items()):
        raise ValueError(f'{algorithm} actor architecture must match the formal design')
    if algorithm == 'EA-MAPPO' and any(k in n for k in ('gru_hidden_dim','gru_layers','recurrent_sequence_length')):
        raise ValueError('EA-MAPPO has no recurrent configuration')
    if i['actor_activation'] != 'relu' or i['critic_activation'] != 'relu' or t['optimizer'] != 'Adam':
        raise ValueError('formal controls require ReLU and Adam')
    if i['policy_std_mode'] != 'state_independent' or (i['log_std_min'],i['log_std_max'],i['log_std_init'],i['mean_head_init_gain']) != (-5.,.5,-.5,.01):
        raise ValueError('formal controls require the current state-independent Gaussian protocol')
    if min(int(t[k]) for k in ('rollout_steps','minibatch_size','ppo_epochs','num_train_envs','total_sampled_steps')) <= 0:
        raise ValueError('training sizes must be positive')
    if algorithm == 'RMAPPO' and (int(t['rollout_steps']) % 32 or int(t['minibatch_size']) % 32):
        raise ValueError('RMAPPO rollout/minibatch must be divisible by sequence length32')


def trainer_kwargs(config,device,seed):
    n,t,i = (config[key] for key in ('network','training','implementation'))
    result = {key:n[key] for key in ('observation_dim','action_dim','num_agents','attention_heads','critic_type')}
    result.update({key:t[key] for key in ('actor_learning_rate','critic_learning_rate','gamma','gae_lambda',
        'clip_ratio','value_loss_coefficient','entropy_coefficient','max_grad_norm','ppo_epochs','minibatch_size','target_kl')})
    result.update({key:i[key] for key in ('actor_activation','critic_activation','log_std_min','log_std_max',
        'normalize_advantages','clip_value_loss','policy_std_mode','log_std_init','mean_head_init_gain')})
    result.update(hidden_dim=256,critic_hidden_dim=256,device=device,seed=int(t['seed'] if seed is None else seed))
    return result
