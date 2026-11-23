"""Short-run diagnostics must account for the environment reward exactly."""

import numpy as np
import pytest

from env.interception import InterceptionConfig
from tools.diagnose import analyze_trace
from tools.replay import collect


class ZeroPolicy:
    def policy_actions(self, observations, active_masks, *, deterministic):
        return np.zeros((len(observations), 5, 2), dtype=np.float32)


def test_diagnostics_decompose_existing_approach_reward():
    config = InterceptionConfig(observation_layout='relative_assignment_target_velocity',
                                assignment_method='huang_boundary_distance',
                                defender_initial_speed=2, attacker_initial_speed=2)
    trace = collect(config, ZeroPolicy(), 'direct_rush', 600000)
    result = analyze_trace(trace)
    parts = result['approach']
    assert parts['defender_motion'] + parts['attacker_motion'] == pytest.approx(parts['total'])
    assert result['captures'] == trace['summary']['captured_count']
    assert len(result['defenders']) == 5
    assert all(row['assigned_visible_s'] <= row['visible_s'] + 1e-9 for row in result['defenders'])
