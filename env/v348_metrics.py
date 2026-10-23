"""Version-local reporting adapter; probability events are never dwell events."""
from .combat_v348 import EVENT_SEMANTICS


def mechanism_metrics():
    row = dict(environment_version='3.48', event_semantics=EVENT_SEMANTICS,
               dwell_metrics_applicable=False, legacy_fire_metrics_applicable=True,
               ammo_cooldown_applicable=True)
    row.update(dwell_start_episode_rate=None, dwell_complete_episode_rate=None)
    for side in ('red', 'blue'):
        for field in ('dwell_started', 'dwell_interrupted', 'dwell_completed',
                      'dwell_neutralizations', 'dwell_start_episode_rate',
                      'dwell_complete_episode_rate'):
            row[f'{side}_{field}'] = None
    return row
