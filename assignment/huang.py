"""Huang-inspired global one-to-one interception matching.

This uses Huang's buffer-boundary interception point and arrival-time feasibility.
It is a global assignment paired with the existing MADDPG controller, not MASAC.
"""

import numpy as np

from .distance import DistanceAssignment


def minimum_travel_time(distance, projected_speed, acceleration, speed_limit):
    """Continuous accelerate-then-cruise estimate along a fixed direction."""
    if distance <= 0:
        return 0.0
    speed = float(np.clip(projected_speed, -speed_limit, speed_limit))
    acceleration_time = (speed_limit - speed) / acceleration
    acceleration_distance = speed * acceleration_time + .5 * acceleration * acceleration_time**2
    if distance <= acceleration_distance:
        root = np.sqrt(speed * speed + 2 * acceleration * distance)
        return ((root - speed) / acceleration if speed < 0
                else 2 * distance / (root + speed))
    return acceleration_time + (distance - acceleration_distance) / speed_limit


def predict_boundary_point(attacker, base, breach_radius):
    """Nominal entry point for a direct flight towards the protected center."""
    toward_base = base - attacker
    range_to_base = np.linalg.norm(toward_base)
    if range_to_base <= breach_radius:
        return None
    direction = toward_base / range_to_base
    return base - breach_radius * direction, range_to_base - breach_radius, direction


class HuangGlobalAssignment(DistanceAssignment):
    """Periodic/event-triggered global assignment with feasible boundary intercepts."""

    def __init__(self, period, method, base, breach_radius, defender_acceleration,
                 defender_speed, attacker_acceleration, attacker_speed, safety_factor=1.1):
        if method not in ('huang_boundary_distance', 'huang_intercept_time'):
            raise ValueError('Unknown Huang global assignment method')
        super().__init__(period, 'distance_matching')
        self.method = method
        self.base = np.asarray(base, dtype=np.float64)
        self.breach_radius = breach_radius
        self.defender_acceleration = defender_acceleration
        self.defender_speed = defender_speed
        self.attacker_acceleration = attacker_acceleration
        self.attacker_speed = attacker_speed
        self.safety_factor = safety_factor

    def update(self, defenders, attackers, defender_alive, attacker_alive, radius, step,
               events=(), *, defender_velocities, attacker_velocities):
        distances = np.linalg.norm(defenders[:, None, :] - attackers[None, :, :], axis=-1)
        visible = (distances <= radius) & defender_alive[:, None] & attacker_alive[None, :]
        detected_now = visible.any(axis=0)
        new_target_detected = bool(np.any(detected_now & ~self.detected_targets))
        self.detected_targets |= detected_now
        replan = (step == 0 or step % self.period == 0 or new_target_detected or
                  any(event['type'] in ('capture', 'penetration') for event in events))
        if replan:
            costs = np.full((5, 5), np.inf, dtype=np.float64)
            for j in np.flatnonzero(attacker_alive & detected_now):
                prediction = predict_boundary_point(attackers[j], self.base, self.breach_radius)
                if prediction is None:
                    continue
                intercept, attacker_distance, direction = prediction
                attacker_time = minimum_travel_time(
                    attacker_distance, np.dot(attacker_velocities[j], direction),
                    self.attacker_acceleration, self.attacker_speed)
                for i in np.flatnonzero(visible[:, j]):
                    displacement = intercept - defenders[i]
                    defender_distance = np.linalg.norm(displacement)
                    heading = displacement / defender_distance if defender_distance > 0 else np.zeros(2)
                    defender_time = minimum_travel_time(
                        defender_distance, np.dot(defender_velocities[i], heading),
                        self.defender_acceleration, self.defender_speed)
                    if self.safety_factor * defender_time <= attacker_time:
                        costs[i, j] = (defender_distance if self.method == 'huang_boundary_distance'
                                       else defender_time)
            self.targets = self._match(costs, np.isfinite(costs))
        return self.targets.copy(), bool(replan)
