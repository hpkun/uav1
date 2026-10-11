"""Input-schema checks only; historical layouts keep their original defaults."""
def validate_entity_layout(network):
    layout = network.get('observation_layout', 'legacy')
    if layout == 'madsac56':
        if (network['observation_dim'], network['action_dim'], network['num_agents'],
            network.get('self_feature_dim', 7)) != (56, 3, 5, 7):
            raise ValueError('madsac56 requires 56/3/5 and self7')
    elif layout == 'legacy':
        self_dim = network.get('self_feature_dim', 7)
        if network['observation_dim'] != 13*network['num_agents'] + self_dim-7:
            raise ValueError('legacy entity dimension mismatch')
        if self_dim not in (7,8) or (self_dim == 8 and network['num_agents'] != 5):
            raise ValueError('legacy self feature layout mismatch')
    else:
        raise ValueError('unknown observation_layout')


def entity_feature_dims(layout, self_dim):
    if layout == 'madsac56' and self_dim == 7:
        return 7, 6, 5
    if layout == 'legacy' and self_dim in (7,8):
        return self_dim, 7, 6
    raise ValueError('invalid entity observation layout')
