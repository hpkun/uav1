"""Observable-state encounter estimates for the two-dimensional interception task.

The forecast does not call the scripted attacker policy or inspect future environment
states. A reference controller estimates what one bounded-acceleration defender could
do; its result is not a certificate of reachability for every possible controller.
"""

from dataclasses import dataclass
import math

import numpy as np


def _limited(vector, maximum):
    length = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) * min(1.0, maximum / max(length, 1e-12))


def constant_velocity_eit(defender_position, attacker_position, attacker_velocity,
                          defender_cruise_speed):
    """Smallest nonnegative root of |r + v*t| = defender_cruise_speed*t."""
    r = np.asarray(attacker_position, dtype=np.float64) - np.asarray(defender_position, dtype=np.float64)
    v = np.asarray(attacker_velocity, dtype=np.float64)
    if defender_cruise_speed <= 0 or not math.isfinite(defender_cruise_speed):
        raise ValueError('Cruise speed must be finite and positive')
    c = float(r @ r)
    if c <= 1e-24:
        return 0.0
    a = float(v @ v - defender_cruise_speed**2)
    b = float(2 * r @ v)
    scale = max(float(v @ v), defender_cruise_speed**2)
    if abs(a) <= 1e-12 * scale:
        return -c / b if b < 0 else math.inf
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return math.inf
    root = math.sqrt(max(0.0, discriminant))
    q = -0.5 * (b + math.copysign(root, b))
    candidates = ([q / a, c / q] if q else [-b / (2 * a)])
    positive = [time for time in candidates if time > 0 and math.isfinite(time)]
    return min(positive) if positive else math.inf


@dataclass(frozen=True)
class AttackerForecast:
    positions: np.ndarray
    velocities: np.ndarray
    breach_time_s: float | None
    dt: float


@dataclass(frozen=True)
class EncounterPrediction:
    geometric_eit_s: float | None
    geometric_lead_xy: tuple[float, float] | None
    reference_capture_s: float | None
    reference_capture_xy: tuple[float, float] | None
    closest_s: float
    closest_distance_m: float
    closest_xy: tuple[float, float]
    forecast_breach_s: float | None


def forecast_direct_attacker(position, velocity, base, *, dt, speed_max,
                             acceleration_max, breach_radius, max_steps):
    """Forecast goal-seeking motion with bounded acceleration, independent of attacker script."""
    if (dt <= 0 or speed_max <= 0 or acceleration_max <= 0 or breach_radius <= 0 or
            max_steps < 1 or not all(math.isfinite(x) for x in
                                     (dt, speed_max, acceleration_max, breach_radius))):
        raise ValueError('Invalid forecast dynamics')
    p = np.asarray(position, dtype=np.float64).copy()
    v = np.asarray(velocity, dtype=np.float64).copy()
    goal = np.asarray(base, dtype=np.float64)
    if p.shape != (2,) or v.shape != (2,) or goal.shape != (2,) or not all(
            np.isfinite(x).all() for x in (p, v, goal)):
        raise ValueError('Forecast expects finite two-dimensional states')
    positions = [p.copy()]
    velocities = [v.copy()]
    breach = 0.0 if np.linalg.norm(p - goal) <= breach_radius else None
    for step in range(1, max_steps + 1):
        if breach is not None:
            break
        delta = goal - p
        desired = speed_max * delta / max(float(np.linalg.norm(delta)), 1e-12)
        acceleration = _limited((desired - v) / dt, acceleration_max)
        p = p + v * dt  # Same position-before-velocity order as InterceptionEnv.step().
        v = _limited(v + acceleration * dt, speed_max)
        positions.append(p.copy())
        velocities.append(v.copy())
        if np.linalg.norm(p - goal) <= breach_radius:
            breach = step * dt
    return AttackerForecast(np.asarray(positions), np.asarray(velocities), breach, dt)


def reference_acceleration(position, velocity, aim_point, *, speed_max,
                           acceleration_max, velocity_tracking_s=0.5):
    """Bounded acceleration toward a supplied aim point, for diagnostic guidance."""
    delta = np.asarray(aim_point, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    desired = (speed_max * delta / np.linalg.norm(delta)
               if np.linalg.norm(delta) > 1e-12 else np.zeros(2))
    return _limited((desired - velocity) / velocity_tracking_s, acceleration_max)


def predict_encounter(defender_position, defender_velocity, forecast, *,
                      defender_speed_max, defender_acceleration_max, capture_radius,
                      velocity_tracking_s=0.5):
    """Roll out Zuo-style lead guidance with this environment's discrete dynamics."""
    if (defender_speed_max <= 0 or defender_acceleration_max <= 0 or capture_radius <= 0 or
            velocity_tracking_s <= 0):
        raise ValueError('Invalid defender prediction parameters')
    p = np.asarray(defender_position, dtype=np.float64).copy()
    v = np.asarray(defender_velocity, dtype=np.float64).copy()
    if p.shape != (2,) or v.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('Prediction expects finite two-dimensional defender states')
    attacker_initial = forecast.positions[0]
    eit = constant_velocity_eit(p, attacker_initial, forecast.velocities[0], defender_speed_max)
    geometric_point = (attacker_initial + forecast.velocities[0] * eit if math.isfinite(eit) else None)
    closest_distance = float(np.linalg.norm(attacker_initial - p))
    closest_step = 0
    closest_point = attacker_initial
    capture_step = 0 if closest_distance <= capture_radius else None
    capture_point = attacker_initial if capture_step == 0 else None
    for step in range(1, len(forecast.positions)):
        attacker = forecast.positions[step - 1]
        attacker_v = forecast.velocities[step - 1]
        tau = constant_velocity_eit(p, attacker, attacker_v, defender_speed_max)
        if math.isfinite(tau):
            remaining = (len(forecast.positions) - step) * forecast.dt
            lead = attacker + attacker_v * min(tau, remaining)
        else:
            lead = attacker
        acceleration = reference_acceleration(p, v, lead, speed_max=defender_speed_max,
                                              acceleration_max=defender_acceleration_max,
                                              velocity_tracking_s=velocity_tracking_s)
        p = p + v * forecast.dt
        v = _limited(v + acceleration * forecast.dt, defender_speed_max)
        distance = float(np.linalg.norm(p - forecast.positions[step]))
        if distance < closest_distance:
            closest_distance = distance
            closest_step = step
            closest_point = forecast.positions[step]
        if distance <= capture_radius:
            capture_step = step
            capture_point = forecast.positions[step]
            break
    return EncounterPrediction(
        geometric_eit_s=float(eit) if math.isfinite(eit) else None,
        geometric_lead_xy=tuple(map(float, geometric_point)) if geometric_point is not None else None,
        reference_capture_s=capture_step * forecast.dt if capture_step is not None else None,
        reference_capture_xy=tuple(map(float, capture_point)) if capture_point is not None else None,
        closest_s=closest_step * forecast.dt,
        closest_distance_m=closest_distance,
        closest_xy=tuple(map(float, closest_point)),
        forecast_breach_s=forecast.breach_time_s)
