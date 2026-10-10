"""Outcome-neutral statistics shared by training and evaluation writers."""
TIMEOUT_REASONS = frozenset({'red_failure_timeout', 'red_win_timeout_survivors',
                           'blue_win_timeout_survivors', 'draw_timeout_equal_survivors'})

def episode_is_timeout(record):
    """An explicit timeout flag is authoritative; support known legacy reasons."""
    if 'timeout' in record:
        return bool(record['timeout'])
    return str(record.get('termination_reason', '')) in TIMEOUT_REASONS
