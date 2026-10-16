"""Strict attention ablation builder using the unchanged baseline PPO arguments."""
from algorithm.common.control_factory import trainer_kwargs
from .trainer import EAAMAPPOTrainer
from .protocol import validate_config


def build_eaa_mappo_trainer(config, env_config, device='cuda', seed=None):
    validate_config(env_config, config)
    return EAAMAPPOTrainer(env_config, config, **trainer_kwargs(config,device,seed))
