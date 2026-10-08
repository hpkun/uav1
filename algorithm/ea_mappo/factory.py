"""Validated formal EA-MAPPO construction."""
from algorithm.common.control_factory import validate_control_config, trainer_kwargs
from .trainer import EAMAPPOTrainer


def validate_config(config):
    validate_control_config(config,"EA-MAPPO")


def build_ea_mappo_trainer(config,device,*,seed=None,smoke=False):
    validate_config(config)
    kwargs = trainer_kwargs(config,device,seed)
    n = config["network"]
    kwargs.update({key:n[key] for key in ('entity_dim', 'entity_attention_heads', 'spatial_hidden_dim')})
    # Smoke retains formal actor/critic dimensions and PPO hyperparameters.
    return EAMAPPOTrainer(**kwargs)
