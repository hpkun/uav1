"""Implementation-v5 architecture and Algorithm-1 schedule; no weight migration."""
CRITIC_ARCHITECTURE = {
    'attention_self_inclusive': True,
    'attention_value_activation': 'leaky_relu',
    'attention_value_negative_slope': 0.01,
    'attention_logit_scaling': 'sqrt_head_dim',
    'critic_raw_self_shortcut': False,
}

UPDATE_CYCLE_CLOCK = 'vector_environment_steps_mod_policy_delay'
TARGET_UPDATE_SCHEDULE = 'once_per_delayed_update_cycle'


def schedule_contract(steps_per_update, num_updates, policy_delay):
    return dict(steps_per_update=int(steps_per_update), num_updates=int(num_updates),
                policy_delay=int(policy_delay), update_cycle_clock=UPDATE_CYCLE_CLOCK,
                policy_delay_clock='vector_steps',
                delayed_policy_condition='vector_steps % policy_delay == 0',
                target_update_schedule=TARGET_UPDATE_SCHEDULE,
                replay_batch_sampling='independent_sample_per_critic_and_actor_optimization',
                transitions_reset='zero_after_each_completed_update_cycle',
                warmup='replay_minibatch_readiness_only')


def validate_checkpoint_contract(state, hidden_dim, attention_heads):
    if state.get('algorithm') != 'madsac' or state.get('implementation_version') != 5:
        raise RuntimeError('MADSAC implementation-v5 required; legacy v3/v4 checkpoints are incompatible; no weight migration')
    architecture = state.get('network_architecture', {})
    expected = dict(CRITIC_ARCHITECTURE, hidden_dim=int(hidden_dim), attention_heads=int(attention_heads),
                    attention_head_dim=int(hidden_dim)//int(attention_heads))
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise RuntimeError(f'MADSAC v5 checkpoint architecture {key} mismatch')
        extra_architecture=state.get('extra',{}).get('network_architecture')
        if extra_architecture is not None and extra_architecture.get(key) != value:
            raise RuntimeError(f'MADSAC v5 checkpoint extra architecture {key} mismatch')
    schedule = state.get('training_schedule', {})
    for key in ('steps_per_update','num_updates','policy_delay'):
        if type(schedule.get(key)) is not int or schedule[key] <= 0:
            raise RuntimeError(f'MADSAC v5 checkpoint schedule {key} mismatch')
    expected_schedule = schedule_contract(schedule['steps_per_update'],schedule['num_updates'],schedule['policy_delay'])
    if schedule != expected_schedule:
        raise RuntimeError('MADSAC v5 checkpoint update-cycle schedule mismatch')
    for key in ('update_cycle_count','transitions_since_update','critic_update_count','actor_update_count',
                'target_update_count','replay_sample_count_critic','replay_sample_count_actor',
                'critic_optimizer_steps','actor_optimizer_steps'):
        if type(state.get(key)) is not int or state[key] < 0:
            raise RuntimeError(f'MADSAC v5 checkpoint counter {key} mismatch')
    if (state['critic_optimizer_steps'] != 2*state['critic_update_count'] or
            state['actor_optimizer_steps'] != state['actor_update_count']):
        raise RuntimeError('MADSAC v5 checkpoint optimizer counter mismatch')
    validate_critic_shapes(state,hidden_dim)


def validate_critic_shapes(state, hidden_dim):
    for name in ('critic1', 'critic2', 'target_critic1', 'target_critic2'):
        weight = state.get(name, {}).get('q_network.0.weight')
        if weight is None or tuple(weight.shape) != (hidden_dim, hidden_dim):
            raise RuntimeError(f'MADSAC checkpoint {name} context-only q_network architecture mismatch')
