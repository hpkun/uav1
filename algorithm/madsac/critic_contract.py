"""Implementation-v3 checkpoint contract; no weight migration."""
CRITIC_ARCHITECTURE = {
    'attention_self_inclusive': True,
    'attention_value_activation': 'leaky_relu',
    'attention_value_negative_slope': 0.01,
    'attention_logit_scaling': 'none',
    'critic_raw_self_shortcut': False,
}


def validate_critic_shapes(state, hidden_dim):
    for name in ('critic1', 'critic2', 'target_critic1', 'target_critic2'):
        weight = state.get(name, {}).get('q_network.0.weight')
        if weight is None or tuple(weight.shape) != (hidden_dim, hidden_dim):
            raise RuntimeError(f'MADSAC checkpoint {name} context-only q_network architecture mismatch')
