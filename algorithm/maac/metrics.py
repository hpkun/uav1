"""MAAC log aliases; dwell events never masquerade as probabilistic fire."""
DWELL_VERSIONS=('3.44','3.45')
EVENT_SEMANTICS='continuous_dwell_neutralization'


def evaluation_metrics(result,env_config):
    """Add legacy aliases without changing historical MAAC output meanings."""
    row=dict(result)
    row.update(fire_rate=None if env_config['environment_version'] in DWELL_VERSIONS
               else row['red_attempt_episode_rate'],kill_rate=row['red_kill_episode_rate'])
    return row


def empty_dwell_metrics():
    """No completed episodes means unavailable, rather than zero success."""
    keys=['mean_return','mean_episode_return','win_rate','kill_rate','red_win_rate','blue_win_rate',
          'draw_rate','mutual_annihilation_draw_rate','average_red_loss','average_blue_loss',
          'red_survivors','blue_survivors','dwell_start_episode_rate','dwell_complete_episode_rate']
    for side in ('red','blue'):
        keys.extend(f'{side}_{name}' for name in ('dwell_started','dwell_completed','dwell_neutralizations',
                    'dwell_start_episode_rate','dwell_complete_episode_rate','dwell_geometry_episode_rate'))
    keys.extend(f'average_episode_r{i}_total' for i in range(1,5))
    return dict.fromkeys(keys,None)|dict(event_semantics=EVENT_SEMANTICS,
        legacy_fire_metrics_applicable=False,fire_rate=None)
