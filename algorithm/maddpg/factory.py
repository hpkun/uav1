"""Construct only independent deterministic MADDPG networks and optimizers."""
from .trainer import MADDPGTrainer


def build_maddpg_trainer(config, device='cuda', seed=None):
    if config.get('algorithm') != 'maddpg':
        raise RuntimeError('algorithm config is not maddpg')
    n, t = config['network'], config['training']
    if n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]:
        raise RuntimeError('MADDPG factory requires 256/256 MLPs')
    return MADDPGTrainer(**{key: n[key] for key in ('observation_dim', 'action_dim', 'num_agents')},
        **{key: t[key] for key in ('actor_learning_rate', 'critic_learning_rate', 'gamma', 'tau')},
        device=device, seed=t['seed'] if seed is None else seed)
