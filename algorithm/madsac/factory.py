"""Construct the MADSAC baseline from its configuration."""
from .trainer import MADSACTrainer


def build_madsac_trainer(config: dict, device: str, hidden_dim: int | None = None,
                         seed: int | None = None) -> MADSACTrainer:
    n, t, i = config['network'], config['training'], config['implementation']
    return MADSACTrainer(
        n['observation_dim'], n['action_dim'], n['num_agents'],
        hidden_dim or n['actor_hidden_layers'][0], n['attention_heads'],
        t['actor_learning_rate'], t['critic_learning_rate'], t['gamma'],
        t['alpha'], t['tau'], t['policy_delay'], device,
        t['seed'] if seed is None else seed, i['actor_activation'],
        i['critic_activation'], i['log_std_min'], i['log_std_max'],
    )
