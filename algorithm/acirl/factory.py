"""Independent ACIRL construction seam."""
from .trainer import ACIRLTrainer


def build_acirl_trainer(config,device='cuda',seed=None):
    if config.get('algorithm')!='acirl':raise RuntimeError('algorithm config is not acirl')
    return ACIRLTrainer(config,device,seed)
