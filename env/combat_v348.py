"""v3.47 observation/reward ablation with the original v3.42 weapon path.

Use the common probabilistic ancestor rather than entering v3.44 dwell hooks.
All reward hooks below are the very same implementations inherited by v3.47.
"""
from .combat_v343 import CombatEnvironmentV343
from .combat_v346 import CombatEnvironmentV346
from .combat_v347 import CombatEnvironmentV347

EVENT_SEMANTICS = 'distance_body_gaussian_probability'


class CombatEnvironmentV348(CombatEnvironmentV343):
    environment_version = '3.48'
    event_semantics = EVENT_SEMANTICS
    _observations = CombatEnvironmentV347._observations
    dense_combat_reward = CombatEnvironmentV346.dense_combat_reward

    def _mechanism_info(self):
        return dict(event_semantics=EVENT_SEMANTICS, dwell_metrics_applicable=False,
                    legacy_fire_metrics_applicable=True, ammo_cooldown_applicable=True,
                    probability_angular_frame='signed_body_azimuth_elevation',
                    cooldown_steps=self.cooldown_steps)

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        info.update(self._mechanism_info())
        return obs, info

    def _info(self, *args, **kwargs):
        info = super()._info(*args, **kwargs)
        info.update(self._mechanism_info())
        return info
