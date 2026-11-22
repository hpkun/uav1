"""Huang-style global matching without changing the existing controller."""

from dataclasses import replace

import numpy as np
import pytest

from assignment import HuangGlobalAssignment
from assignment.huang import minimum_travel_time
from env.interception import InterceptionConfig, InterceptionEnv
from experiment.interception import read_config


def allocator(method='huang_boundary_distance', factor=1.1):
    return HuangGlobalAssignment(20, method, [0, 0], 10, 2, 5, 2, 5, factor)


def test_travel_time_acceleration_and_cruise():
    assert minimum_travel_time(0, 0, 2, 5) == 0
    assert minimum_travel_time(1, 0, 2, 5) == pytest.approx(1)
    assert minimum_travel_time(10, 0, 2, 5) == pytest.approx(2.5 + (10 - 6.25) / 5)
    assert minimum_travel_time(5, -3, 2, 5) == minimum_travel_time(5, 0, 2, 5)


@pytest.mark.parametrize('method', ('huang_boundary_distance', 'huang_intercept_time'))
def test_global_matching_is_one_to_one_and_holds_until_trigger(method):
    matching = allocator(method)
    defenders = np.array([[-2., -20], [2, -20], [25, -20], [40, -20], [55, -20]])
    attackers = np.array([[-2., 60], [2, 60], [25, 60], [40, 60], [55, 60]])
    velocity_d = np.tile([0., 2], (5, 1))
    velocity_a = np.tile([0., -2], (5, 1))
    alive = np.ones(5, bool)
    def update(step, events=()):
        return matching.update(defenders, attackers, alive, alive, 100, step, events,
                               defender_velocities=velocity_d, attacker_velocities=velocity_a)
    targets, replanned = update(0)
    assert replanned and sorted(targets.tolist()) == list(range(5))
    attackers[0, 0] = 40
    held, replanned = update(1)
    assert not replanned
    np.testing.assert_array_equal(held, targets)
    assert update(20)[1]


def test_arrival_feasibility_rejects_a_near_but_late_defender():
    matching = allocator()
    defenders = np.array([[0., -40], [0, 0], [0, 0], [0, 0], [0, 0]])
    attackers = np.array([[0., 30], [200, 200], [210, 210], [220, 220], [230, 230]])
    defender_alive = np.array([True, False, False, False, False])
    attacker_alive = np.array([True, False, False, False, False])
    velocities = np.zeros((5, 2))
    attacker_velocities = np.zeros((5, 2))
    attacker_velocities[0] = [0, -5]
    targets, _ = matching.update(defenders, attackers, defender_alive, attacker_alive, 100, 0,
                                 defender_velocities=velocities,
                                 attacker_velocities=attacker_velocities)
    assert targets.tolist() == [-1] * 5


def test_resolution_event_replans_and_retires_captured_pair():
    matching = allocator()
    defenders = np.array([[0., -20], [20, -20], [40, -20], [60, -20], [80, -20]])
    attackers = np.array([[0., 60], [20, 60], [40, 60], [60, 60], [80, 60]])
    velocities_d = np.tile([0., 2], (5, 1))
    velocities_a = np.tile([0., -2], (5, 1))
    defender_alive = np.ones(5, bool)
    attacker_alive = np.ones(5, bool)
    def update(step, events=()):
        return matching.update(defenders, attackers, defender_alive, attacker_alive, 100,
                               step, events, defender_velocities=velocities_d,
                               attacker_velocities=velocities_a)
    targets, _ = update(0)
    assert targets[0] == 0
    defender_alive[0] = False
    attacker_alive[0] = False
    revised, changed = update(1, [{'type': 'capture'}])
    assert changed and revised[0] == -1 and 0 not in revised


def test_environment_and_checkpoint_contract_preserve_old_baseline():
    new, _, _ = read_config('configs/interception_huang_global.yaml')
    old, _, _ = read_config('configs/interception_assignment.yaml')
    assert old.assignment_method == 'distance_matching'
    env = InterceptionEnv(new)
    observations, info = env.reset(seed=7)
    assert observations.shape == (5, 54)
    assert info['central_state'].shape == (75,)
    assert env.contract['assignment_schema'] == 'huang_boundary_distance_period_20'
    assert env.config.sensing_radius == 100
    assert env.step(np.zeros((5, 2)))[0].shape == (5, 54)
    with pytest.raises(ValueError, match='direct_rush only'):
        replace(new, attack_mode='split_attack')
