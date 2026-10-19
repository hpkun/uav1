"""Construct only independent deterministic MATD3 networks and optimizers."""
from .trainer import MATD3Trainer


def build_matd3_trainer(config, device='cuda', seed=None):
    if config.get('algorithm') != 'matd3':
        raise RuntimeError('algorithm config is not matd3')
    n, t = config['network'], config['training']
    if n['actor_hidden_layers'] != [256, 256] or n['critic_hidden_layers'] != [256, 256]:
        raise RuntimeError('MATD3 factory requires 256/256 MLPs')
    return MATD3Trainer(**{key: n[key] for key in ('observation_dim', 'action_dim', 'num_agents')},
        **{key: t[key] for key in ('actor_learning_rate', 'critic_learning_rate', 'gamma', 'tau', 'policy_delay', 'target_policy_noise', 'target_noise_clip')},
        device=device, seed=t['seed'] if seed is None else seed,
        gradient_clip_norm=t['gradient_clip_norm'], diagnostics_interval_updates=t.get('diagnostics_interval_updates',100))
