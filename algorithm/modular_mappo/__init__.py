"""Independent capability-oriented MAPPO implementation."""
from .trainer import ModularMAPPOTrainer
from .runner import ModularMAPPOTrainingRunner

__all__ = ["ModularMAPPOTrainer", "ModularMAPPOTrainingRunner"]
