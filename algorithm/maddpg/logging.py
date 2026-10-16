"""Compact MADDPG console formatting; metric meanings come from common evaluator."""
import math


def format_steps(value):
    if value >= 1_000_000:
        return f'{value/1_000_000:.2f}M' if value % 1_000_000 else f'{value/1_000_000:.1f}M'
    if value >= 1000:
        return f'{value/1000:g}k'
    return str(value)


def number(value, digits=2, percent=False):
    if value is None or not math.isfinite(float(value)):
        return 'N/A'
    return f'{value:.0%}' if percent else f'{value:.{digits}f}'


def combat_fields(row):
    parts = [('win', 'win_rate', 0, True), ('Rloss', 'average_red_loss', 2, False),
        ('Bloss', 'average_blue_loss', 2, False), ('fire', 'red_attempt_episode_rate', 0, True),
        ('kill', 'red_kill_episode_rate', 0, True), ('timeout', 'timeout_rate', 0, True),
        ('len', 'episode_length', 0, False), ('R', 'average_return', 2, False)]
    return ' | '.join(f'{label}={number(row.get(key),digits,percent)}' for label,key,digits,percent in parts)
