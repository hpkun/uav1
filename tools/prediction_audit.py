"""Compare observable-state attacker forecasts with short direct-rush episodes."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from assignment.distance import DistanceAssignment
from assignment.prediction import (constant_velocity_eit, forecast_direct_attacker,
                                   predict_encounter, reference_acceleration)
from env.interception import InterceptionEnv
from experiment.interception import read_config


def audit(config, seed_start, episodes, horizons=(1, 3, 5, 10)):
    if config.attack_mode != 'direct_rush':
        raise ValueError('The current forecast audit supports direct_rush only')
    steps = {seconds:round(seconds / config.dt) for seconds in horizons}
    if any(abs(count * config.dt - seconds) > 1e-9 for seconds, count in steps.items()):
        raise ValueError('Audit horizons must align with the environment timestep')
    errors = {seconds:dict(constant_velocity=[], bounded_acceleration=[]) for seconds in horizons}
    breach_errors = []
    for seed in range(seed_start, seed_start + episodes):
        env = InterceptionEnv(config)
        env.reset(seed=seed)
        # Prevent defender contact so that attacker trajectories can be compared
        # against both forecasts through their full approach to the protected zone.
        env.defender_alive[:] = False
        initial_positions = env.positions[1].copy()
        initial_velocities = env.velocities[1].copy()
        forecasts = [forecast_direct_attacker(initial_positions[j], initial_velocities[j],
                     env.base, dt=config.dt, speed_max=config.attacker_speed_max,
                     acceleration_max=config.attacker_acceleration_max,
                     breach_radius=config.breach_radius, max_steps=config.episode_steps)
                     for j in range(5)]
        actual_breach = [None] * 5
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros((5, 2), dtype=np.float32))
            step = env.step_count
            for seconds, at in steps.items():
                if step == at:
                    for j in range(5):
                        if not env.attacker_alive[j] or step >= len(forecasts[j].positions):
                            continue
                        actual = env.positions[1, j]
                        cv = initial_positions[j] + initial_velocities[j] * seconds
                        errors[seconds]['constant_velocity'].append(float(np.linalg.norm(actual - cv)))
                        errors[seconds]['bounded_acceleration'].append(
                            float(np.linalg.norm(actual - forecasts[j].positions[step])))
            for event in info['events']:
                if event['type'] == 'penetration':
                    actual_breach[event['attacker']] = event['time_s']
            if terminated or truncated:
                break
        for j, actual in enumerate(actual_breach):
            estimated = forecasts[j].breach_time_s
            if actual is not None and estimated is not None:
                breach_errors.append(float(estimated - actual))
        env.close()
    profile = {}
    for seconds, models in errors.items():
        profile[str(seconds)] = {name:dict(n=len(values), mean_m=float(np.mean(values)),
                                              p90_m=float(np.percentile(values, 90)))
                                 if values else dict(n=0, mean_m=None, p90_m=None)
                                 for name, values in models.items()}
    return dict(schema='InterceptionPredictionAudit/1', attack_mode=config.attack_mode,
                seed_start=seed_start, episodes=episodes, horizons_s=profile,
                breach_time_error_s=dict(n=len(breach_errors), mean=float(np.mean(breach_errors)),
                                         mean_absolute=float(np.mean(np.abs(breach_errors)))))


def reference_guidance_audit(config, seed_start, episodes):
    """Isolate predicted lead from pursuit under identical global matching and dynamics.

    Both controllers see globally known target states, including beyond sensing range.
    This is an information-rich feasibility diagnostic, not a decentralized RL score.
    """
    results = {}
    for method in ('pure_pursuit', 'velocity_lead'):
        captures = []
        all_captured = 0
        for seed in range(seed_start, seed_start + episodes):
            env = InterceptionEnv(config)
            env.reset(seed=seed)
            targets = np.full(5, -1, dtype=np.int64)
            while True:
                if (env.step_count % config.assignment_period == 0 or
                        any(target >= 0 and not env.attacker_alive[target] for target in targets)):
                    distances = np.linalg.norm(env.positions[0, :, None] -
                                               env.positions[1, None, :], axis=-1)
                    edges = env.defender_alive[:, None] & env.attacker_alive[None, :]
                    targets = DistanceAssignment._match(distances, edges)
                actions = np.zeros((5, 2), dtype=np.float64)
                for i, j in enumerate(targets):
                    if j < 0 or not env.defender_alive[i] or not env.attacker_alive[j]:
                        continue
                    point = env.positions[1, j]
                    if method == 'velocity_lead':
                        time = constant_velocity_eit(env.positions[0, i], point,
                                                     env.velocities[1, j], config.defender_speed_max)
                        if math.isfinite(time):
                            point = point + env.velocities[1, j] * min(time, config.episode_steps * config.dt)
                    actions[i] = reference_acceleration(
                        env.positions[0, i], env.velocities[0, i], point,
                        speed_max=config.defender_speed_max,
                        acceleration_max=config.defender_acceleration_max)
                _, _, terminated, truncated, info = env.step(actions)
                if terminated or truncated:
                    break
            row = info['episode_summary']
            captures.append(row['captured_count'])
            all_captured += row['outcome'] == 'all_captured'
            env.close()
        results[method] = dict(mean_captures=float(np.mean(captures)),
                               all_captured=int(all_captured), captures=captures)
    return dict(information='global_target_positions_and_velocities_even_outside_sensing',
                shared_matching='periodic_global_distance_one_to_one',
                episodes=episodes, methods=results)


def single_pair_audit(config, seed_start, episodes):
    """Compare predicted and actual capture time under the same reference controller."""
    predicted_hits = actual_hits = 0
    capture_time_errors = []
    for seed in range(seed_start, seed_start + episodes):
        env = InterceptionEnv(config)
        env.reset(seed=seed)
        env.defender_alive[1:] = False
        env.attacker_alive[1:] = False
        forecast = forecast_direct_attacker(
            env.positions[1, 0], env.velocities[1, 0], env.base,
            dt=config.dt, speed_max=config.attacker_speed_max,
            acceleration_max=config.attacker_acceleration_max,
            breach_radius=config.breach_radius, max_steps=config.episode_steps)
        prediction = predict_encounter(
            env.positions[0, 0], env.velocities[0, 0], forecast,
            defender_speed_max=config.defender_speed_max,
            defender_acceleration_max=config.defender_acceleration_max,
            capture_radius=config.capture_radius)
        predicted_hits += prediction.reference_capture_s is not None
        while True:
            point = env.positions[1, 0]
            time = constant_velocity_eit(env.positions[0, 0], point,
                                         env.velocities[1, 0], config.defender_speed_max)
            if math.isfinite(time):
                point = point + env.velocities[1, 0] * min(time, config.episode_steps * config.dt)
            actions = np.zeros((5, 2), dtype=np.float64)
            actions[0] = reference_acceleration(
                env.positions[0, 0], env.velocities[0, 0], point,
                speed_max=config.defender_speed_max,
                acceleration_max=config.defender_acceleration_max)
            _, _, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
        actual = info['episode_summary']['captured_count'] > 0
        actual_hits += actual
        if actual and prediction.reference_capture_s is not None:
            capture_time_errors.append(prediction.reference_capture_s - info['time_s'])
        env.close()
    return dict(episodes=episodes, predicted_captures=int(predicted_hits),
                actual_captures=int(actual_hits), paired_capture_times=len(capture_time_errors),
                mean_capture_time_error_s=(float(np.mean(capture_time_errors)) if capture_time_errors else None),
                mean_absolute_capture_time_error_s=(float(np.mean(np.abs(capture_time_errors)))
                                                    if capture_time_errors else None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', type=Path, required=True)
    parser.add_argument('--algorithm-config', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=600000)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.episodes < 1 or args.output.exists():
        raise ValueError('Invalid seed, episode count or existing output')
    config, _, _ = read_config(args.env_config, args.algorithm_config)
    result = audit(replace(config, attack_mode='direct_rush'), args.seed, args.episodes)
    result['reference_guidance'] = reference_guidance_audit(config, args.seed, args.episodes)
    result['single_pair'] = single_pair_audit(config, args.seed, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
