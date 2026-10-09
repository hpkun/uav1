"""Independent deterministic, unshared, single-Q MADDPG baseline."""
from .trainer import MADDPGTrainer, MADDPG_IMPL_VERSION

__all__ = ['MADDPGTrainer', 'MADDPG_IMPL_VERSION']
