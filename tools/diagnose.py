"""Diagnose short checkpoint episodes without changing training or replay data."""

import argparse
import json
from pathlib import Path

import numpy as np

from assignment.huang import minimum_travel_time, predict_boundary_point
from experiment.interception import digest, load_run, require_matching_attack_mode
from .replay import collect, validate_trace


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _boundary_times(frame, config, i, j):
    base = np.array([config['base_x'], config['base_y']], dtype=float)
    defender = np.asarray(frame['positions'][0][i], dtype=float)
    attacker = np.asarray(frame['positions'][1][j], dtype=float)
    prediction = predict_boundary_point(attacker, base, config['breach_radius'])
    if prediction is None:
        return None
    point, distance_a, direction_a = prediction
    velocity_a = np.asarray(frame['velocities'][1][j], dtype=float)
    time_a = minimum_travel_time(distance_a, np.dot(velocity_a, direction_a),
                                 config['attacker_acceleration_max'], config['attacker_speed_max'])
    displacement = point - defender
    distance_d = float(np.linalg.norm(displacement))
    direction_d = displacement / distance_d if distance_d else np.zeros(2)
    velocity_d = np.asarray(frame['velocities'][0][i], dtype=float)
    time_d = minimum_travel_time(distance_d, np.dot(velocity_d, direction_d),
                                 config['defender_acceleration_max'], config['defender_speed_max'])
    return float(time_d), float(time_a)


def _loss_reason(before, after, events, config, i, target):
    if not after['defender_alive'][i]:
        return 'defender_retired'
    if not after['attacker_alive'][target]:
        return ('target_captured' if any(e['type'] == 'capture' and e['attacker'] == target for e in events)
                else 'target_penetrated')
    visible = [j for j, alive in enumerate(after['attacker_alive']) if alive and
               _distance(after['positions'][0][i], after['positions'][1][j]) <= config['sensing_radius']]
    if not visible:
        return 'no_visible_target'
    if config['assignment_method'].startswith('huang_'):
        feasible = False
        for j in visible:
            times = _boundary_times(after, config, i, j)
            if times is not None and config['assignment_safety_factor'] * times[0] <= times[1]:
                feasible = True
                break
        if not feasible:
            return 'no_feasible_boundary_pair'
    return 'unmatched_after_replan'


def _unassigned_reason(frame, config, i, visible):
    if config['assignment_method'].startswith('huang_'):
        if not any((times := _boundary_times(frame, config, i, j)) is not None and
                   config['assignment_safety_factor'] * times[0] <= times[1]
                   for j in visible):
            return 'no_feasible_boundary_pair'
    return 'feasible_pair_but_unassigned'


def analyze_trace(trace):
    """Return one episode's assignment, meeting-distance and reward diagnostics."""
    validate_trace(trace)
    config = trace['config']
    if any(frame['assigned_target_ids'] is None for frame in trace['frames']):
        raise ValueError('Diagnostics require an assignment-enabled interception checkpoint')
    dt = config['dt']
    events_by_step = {}
    for event in trace['events']:
        events_by_step.setdefault(event['step'], []).append(event)
    agents = [dict(defender=i, active_steps=0, visible_steps=0, assigned_steps=0,
                   assigned_visible_steps=0, min_enemy_distance_m=None,
                   min_assigned_distance_m=None, first_assignment_s=None,
                   last_assignment_s=None, losses=[], switches=[], unassigned_visible_steps_by_reason={},
                   boundary_defender_times_s=[], boundary_attacker_times_s=[])
              for i in range(5)]
    reward_defender = reward_attacker = reward_heading = 0.0
    transitions = trace['frames']
    for step, (before, after) in enumerate(zip(transitions, transitions[1:]), start=1):
        for i, row in enumerate(agents):
            if not before['defender_alive'][i]:
                continue
            row['active_steps'] += 1
            alive_targets = [j for j, alive in enumerate(before['attacker_alive']) if alive]
            visible_targets = []
            if alive_targets:
                closest = min(min(_distance(before['positions'][0][i], before['positions'][1][j]),
                                  _distance(after['positions'][0][i], after['positions'][1][j]))
                              for j in alive_targets)
                row['min_enemy_distance_m'] = min(closest, row['min_enemy_distance_m']
                                                  if row['min_enemy_distance_m'] is not None else closest)
                visible_targets = [j for j in alive_targets
                                   if _distance(before['positions'][0][i], before['positions'][1][j])
                                   <= config['sensing_radius']]
                if visible_targets:
                    row['visible_steps'] += 1
            target = before['assigned_target_ids'][i]
            if target < 0 and visible_targets:
                reason = _unassigned_reason(before, config, i, visible_targets)
                counts = row['unassigned_visible_steps_by_reason']
                counts[reason] = counts.get(reason, 0) + 1
            if target >= 0:
                row['assigned_steps'] += 1
                row['first_assignment_s'] = (before['time_s'] if row['first_assignment_s'] is None
                                             else row['first_assignment_s'])
                row['last_assignment_s'] = before['time_s']
                if _distance(before['positions'][0][i], before['positions'][1][target]) <= config['sensing_radius']:
                    row['assigned_visible_steps'] += 1
                closest_target = min(_distance(before['positions'][0][i], before['positions'][1][target]),
                                     _distance(after['positions'][0][i], after['positions'][1][target]))
                row['min_assigned_distance_m'] = min(closest_target, row['min_assigned_distance_m']
                                                     if row['min_assigned_distance_m'] is not None else closest_target)
                times = _boundary_times(before, config, i, target)
                if times is not None:
                    row['boundary_defender_times_s'].append(times[0])
                    row['boundary_attacker_times_s'].append(times[1])
                # Symmetric two-order decomposition: the components sum exactly to
                # the existing distance reduction, without privileging either side.
                d00 = _distance(before['positions'][0][i], before['positions'][1][target])
                d10 = _distance(after['positions'][0][i], before['positions'][1][target])
                d01 = _distance(before['positions'][0][i], after['positions'][1][target])
                d11 = _distance(after['positions'][0][i], after['positions'][1][target])
                retired_oob = (config.get('defender_oob_retire') and any(
                    event['type']=='defender_oob' and event['defender']==i
                    for event in events_by_step.get(step, [])))
                if not retired_oob:
                    reward_defender += 0.5 * ((d00 - d10) + (d01 - d11))
                    reward_attacker += 0.5 * ((d00 - d01) + (d10 - d11))
                    if config.get('heading_weight', 0) and before['attacker_alive'][target] and \
                            d00 <= config['sensing_radius'] and d00 > 1e-12:
                        velocity = (np.asarray(before['velocities'][0][i], dtype=float) +
                                    np.asarray(after['defender_actions'][i], dtype=float) * dt)
                        speed = float(np.linalg.norm(velocity))
                        if speed > config['defender_speed_max']:
                            velocity *= config['defender_speed_max'] / speed
                            speed = config['defender_speed_max']
                        if speed > 1e-12:
                            direction = np.asarray(before['positions'][1][target], dtype=float) - \
                                        np.asarray(before['positions'][0][i], dtype=float)
                            reward_heading += float(np.clip(np.dot(velocity, direction) /
                                                            (speed * d00), -1.0, 1.0))
            next_target = after['assigned_target_ids'][i]
            if target >= 0 and next_target < 0:
                row['losses'].append(dict(time_s=after['time_s'], target=target,
                                          reason=_loss_reason(before, after, events_by_step.get(step, []), config, i, target)))
            elif target >= 0 and next_target >= 0 and target != next_target:
                row['switches'].append(dict(time_s=after['time_s'], from_target=target,
                                            to_target=next_target))
    for row in agents:
        row['active_s'] = row.pop('active_steps') * dt
        row['visible_s'] = row.pop('visible_steps') * dt
        row['assigned_s'] = row.pop('assigned_steps') * dt
        row['assigned_visible_s'] = row.pop('assigned_visible_steps') * dt
        row['unassigned_visible_s_by_reason'] = {reason: steps * dt for reason, steps in
                                                 row.pop('unassigned_visible_steps_by_reason').items()}
        for name, output in (('boundary_defender_times_s', 'mean_boundary_defender_time_s'),
                             ('boundary_attacker_times_s', 'mean_boundary_attacker_time_s')):
            values = row.pop(name)
            row[output] = float(np.mean(values)) if values else None
    weight = config['approach_weight'] / 5
    approach_total = weight * (reward_defender + reward_attacker)
    heading_total = config.get('heading_weight', 0) * reward_heading / 5
    if config['episode_protocol'] == 'resolve_all_one_penalty_trade':
        summary = trace['summary']
        expected = (summary['team_return'] - config['capture_reward'] * summary['captured_count']
                    + config['penetration_penalty'] * bool(summary['penetrated_count'])
                    + config['boundary_penalty'] * summary['defender_oob_count'] / 5
                    - heading_total)
        if not np.isclose(approach_total, expected, rtol=0, atol=1e-6):
            raise AssertionError('Diagnostic approach decomposition disagrees with reward ledger')
    return dict(seed=trace['seed'], outcome=trace['summary']['outcome'],
                captures=trace['summary']['captured_count'], duration_s=trace['summary']['duration_s'],
                team_return=trace['summary']['team_return'],
                approach=dict(defender_motion=weight * reward_defender,
                              attacker_motion=weight * reward_attacker,
                              total=approach_total),
                defenders=agents)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', choices=('Best', 'Last'), default='Last')
    parser.add_argument('--attack-mode', choices=('direct_rush', 'split_attack', 'jia_straight'), required=True)
    parser.add_argument('--seed', type=int, default=600000)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.episodes < 1 or args.output.exists():
        raise ValueError('Invalid seed, episode count or existing output')
    config, policy, algorithm, model, settings = load_run(args.checkpoint, args.label, args.device)
    require_matching_attack_mode(settings, args.attack_mode)
    episodes = [analyze_trace(collect(config, policy, args.attack_mode, args.seed + offset))
                for offset in range(args.episodes)]
    loss_counts = {}
    for episode in episodes:
        for defender in episode['defenders']:
            for loss in defender['losses']:
                key = loss['reason']
                loss_counts[key] = loss_counts.get(key, 0) + 1
    all_defenders = [row for episode in episodes for row in episode['defenders']]
    visible = sum(row['visible_s'] for row in all_defenders)
    unassigned_times = {}
    for row in all_defenders:
        for reason, seconds in row['unassigned_visible_s_by_reason'].items():
            unassigned_times[reason] = unassigned_times.get(reason, 0.0) + seconds
    aggregate = dict(episodes=len(episodes), mean_captures=float(np.mean([e['captures'] for e in episodes])),
                     mean_team_return=float(np.mean([e['team_return'] for e in episodes])),
                     mean_approach={name:float(np.mean([e['approach'][name] for e in episodes]))
                                    for name in ('defender_motion', 'attacker_motion', 'total')},
                     assigned_fraction_of_visible_defender_time=(sum(row['assigned_visible_s'] for row in all_defenders)
                                                                 / visible if visible else None),
                     unassigned_visible_defender_time_s_by_reason=unassigned_times,
                     assignment_loss_reasons=loss_counts)
    result = dict(schema='InterceptionDiagnostics/1', algorithm=algorithm, attack_mode=args.attack_mode,
                  label=args.label, checkpoint_sha256=digest(model), seed_start=args.seed,
                  aggregate=aggregate, episodes=episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
