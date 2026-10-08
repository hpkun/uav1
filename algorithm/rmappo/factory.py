"""Validated formal RMAPPO construction."""
from algorithm.common.control_factory import validate_control_config, trainer_kwargs
from .trainer import RMAPPOTrainer


def validate_config(config):
    validate_control_config(config,"RMAPPO")


def build_rmappo_trainer(config,device,*,seed=None,smoke=False):
    validate_config(config)
    kwargs = trainer_kwargs(config,device,seed)
    n = config["network"]
    kwargs.update({key:n[key] for key in ('flat_encoder_dim', 'gru_hidden_dim', 'gru_layers', 'recurrent_sequence_length')})
    # Smoke retains formal actor/critic dimensions and PPO hyperparameters.
    return RMAPPOTrainer(**kwargs)
