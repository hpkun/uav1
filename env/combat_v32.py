"""v3.1 physics and initialization with terminal-safe PBRS only."""
import numpy as np
from .combat_v31 import CombatEnvironmentV31

class CombatEnvironmentV32(CombatEnvironmentV31):
    environment_version = '3.2'

    def shaping_next_potential(self, actual_phi_next, terminated, truncated):
        return np.zeros_like(actual_phi_next) if terminated or truncated else actual_phi_next
