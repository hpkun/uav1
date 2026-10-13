"""Compact combat-facing console formatting; detailed logs remain separate."""
def format_steps(value, total=False):
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M' if total else f'{value / 1_000_000:.2f}M'
    if value >= 1_000:
        return f'{value / 1_000:g}k'
    return str(int(value))


def combat_fields(metrics):
    fields = [('win', 'red_win_rate', 'rate'), ('Rloss', 'average_red_loss', 'number'),
              ('Bloss', 'average_blue_loss', 'number'), ('fire', 'red_attempt_episode_rate', 'rate'),
              ('kill', 'red_kill_episode_rate', 'rate'), ('timeout', 'timeout_rate', 'rate'),
              ('len', 'average_episode_length', 'length'), ('R', 'average_return', 'number')]
    parts = []
    for label, key, style in fields:
        value = metrics.get(key)
        display = 'n/a' if value is None else (f'{100 * value:.0f}%' if style == 'rate'
                   else f'{value:.0f}' if style == 'length' else f'{value:.2f}')
        parts.append(f'{label}={display}')
    return ' | '.join(parts)
