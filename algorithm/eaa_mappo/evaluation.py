"""Load only the explicit EAA-family checkpoint/config and reuse combat evaluator."""
from algorithm.common.evaluator import evaluate
from .factory import build_eaa_mappo_trainer
from .protocol import require_cuda


def load_checkpoint(checkpoint, config, env_config, device='cuda'):
    require_cuda(device)
    trainer = build_eaa_mappo_trainer(config,env_config,device)
    trainer.load(checkpoint)
    return trainer


def evaluate_checkpoint(checkpoint, config, env_config, seeds, device='cuda'):
    return evaluate(load_checkpoint(checkpoint,config,env_config,device),env_config,seeds)
