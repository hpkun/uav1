"""Fixed-scale v3.38 entity/agent attention ablations, independent of legacy EA."""
from .networks import EntityAttentionMAPPOActor, split_v338_entities
from .factory import build_eaa_mappo_trainer

__all__ = ['EntityAttentionMAPPOActor', 'split_v338_entities', 'build_eaa_mappo_trainer']
