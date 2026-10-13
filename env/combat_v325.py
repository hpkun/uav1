"""v3.24 with symmetric boundary deaths and config-disabled Safety shaping."""
from .combat_v324 import CombatEnvironmentV324
from .combat_v30 import CombatEnvironmentV30


class CombatEnvironmentV325(CombatEnvironmentV324):
    environment_version = '3.25'

    def _resolve_noncombat_losses(self):
        # This existing resolver has exactly the frozen Red rules for both teams.
        return CombatEnvironmentV30._resolve_noncombat_losses(self)
