"""Bounded, checkpoint-free geometric validation of fixed v2.6 parameters."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter
import numpy as np
from env.config import load_config, aircraft_spec
from env.control import action_to_control
from env.dynamics import PointMassDynamics
from env.integrator import RK4Integrator
from env.models import AircraftState, ControlCommand
from env.fixed_policy import SternConversionPolicy, SternPhase
from env.geometry import engagement_geometry
from env.weapon import RearAspectWeaponEnvelope
from env.scenario import random_combat_states


def run_trial(config, seed=None, synthetic=False, trace=False):
    if synthetic:
        target = AircraftState(-4000, 0, -3000, 225, 0, 0)
        blue = AircraftState(4000, 0, -3000, 225, 0, np.pi)
    else:
        # Same opposing centers and perturbations as the environment, zero 1v1
        # formation offset; targets remain straight and constant-speed.
        scenario = dict(config['scenario'], team_size=1, formation_offsets=[0.0])
        red, blues, _ = random_combat_states(np.random.default_rng(seed), **scenario)
        target, blue = red[0], blues[0]
    policy = SternConversionPolicy(config['blue_policy'], config['action'], 1)
    integrator = RK4Integrator(config['simulation']['dt'])
    dynamics, spec = PointMassDynamics(), aircraft_spec(config)
    weapon = RearAspectWeaponEnvelope(**config['weapon'])
    phases = Counter()
    transitions, rows = [], []
    first_window, minimum_range, max_lateral = None, float('inf'), 0.0
    boundary = ground = blue_exit = target_exit = False
    previous = None
    for step in range(int(config['simulation']['max_steps']) + 1):
        geometry = engagement_geometry(blue, target)
        longitudinal, lateral = policy.target_coordinates(blue, target)
        minimum_range = min(minimum_range, geometry.distance)
        max_lateral = max(max_lateral, abs(lateral))
        if first_window is None and weapon.qualifies(geometry, blue.v, target.v):
            first_window = step
        # Stop when either aircraft leaves the unchanged combat arena/ground.
        blue_exit = bool(np.hypot(blue.x, blue.y) > config['arena']['radius'])
        target_exit = bool(np.hypot(target.x, target.y) > config['arena']['radius'])
        boundary = blue_exit or target_exit
        ground = blue.altitude <= 0 or target.altitude <= 0
        if boundary or ground or step == config['simulation']['max_steps']:
            break
        action = policy.action(blue, [target])
        state = policy.states[0]
        phases[state.phase.value] += 1
        row = {'step': step, 'phase': state.phase.value, 'range': geometry.distance,
               'longitudinal': longitudinal, 'lateral': lateral,
               'blue_heading': blue.psi, 'target_heading': target.psi,
               'stored_bearing': state.relative_bearing_heading,
               'blue_position': [blue.x, blue.y, blue.z],
               'target_position': [target.x, target.y, target.z],
               'fire_window': bool(weapon.qualifies(geometry, blue.v, target.v))}
        if state.phase != previous:
            transitions.append(row)
            previous = state.phase
        if trace:
            rows.append(row)
        blue = integrator.step(blue, action_to_control(blue, action, config['action']), dynamics, spec)
        # Zero tangential acceleration, level nz=1, bank=0 gives straight,
        # constant-speed flight under the actual point-mass dynamics.
        target = integrator.step(target, ControlCommand(0.0, 1.0, 0.0), dynamics, spec)
    result = {'seed': seed, 'synthetic': synthetic, 'fire_window_success': first_window is not None,
              'first_fire_window_step': first_window, 'phase_completion': len(transitions) == 4,
              'minimum_range': minimum_range, 'maximum_lateral_displacement': max_lateral,
              'ground_loss': bool(ground), 'arena_exit': boundary,
              'blue_arena_exit': blue_exit, 'target_arena_exit': target_exit,
              'boundary_before_fire_window': boundary and first_window is None,
              'ground_before_fire_window': bool(ground and first_window is None), 'steps': step,
              'phase_steps': dict(phases), 'transitions': transitions}
    if trace:
        result['trace'] = rows
    return result


def validate(config, trials=200, seed_base=260000):
    records = [run_trial(config, seed_base+i) for i in range(trials)]
    first = [r['first_fire_window_step'] for r in records if r['fire_window_success']]
    return {'trials': trials, 'seed_base': seed_base, 'parameters': config['blue_policy'],
            'fire_window_success_rate': float(np.mean([r['fire_window_success'] for r in records])),
            'median_first_window_step': float(np.median(first)) if first else None,
            'p90_first_window_step': float(np.percentile(first, 90)) if first else None,
            'first_window_percentiles_population': 'successful trials only',
            'phase_completion_rate': float(np.mean([r['phase_completion'] for r in records])),
            'boundary_failure_count': sum(r['boundary_before_fire_window'] for r in records),
            'ground_failure_count': sum(r['ground_before_fire_window'] for r in records),
            'boundary_failure_definition': 'either aircraft leaves before first legal fire window',
            'blue_arena_exit_count': sum(r['blue_arena_exit'] for r in records),
            'target_arena_exit_count': sum(r['target_arena_exit'] for r in records),
            'ground_loss_count': sum(r['ground_loss'] for r in records),
            'final_phase_counts': dict(Counter(r['transitions'][-1]['phase'] for r in records)),
            'synthetic_sanity_case': run_trial(config, synthetic=True, trace=True),
            'records': records, 'checkpoint_loaded': False, 'parameters_tuned': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', default=str(ROOT/'configs/combat_environment_v26.yaml'))
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--seed-base', type=int, default=260000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 200:
        raise ValueError('bounded validation requires 1..200 trials')
    config = load_config(args.env_config)
    if str(config['environment_version']) != '2.6':
        raise ValueError('Stern validation requires v2.6')
    result = validate(config, args.trials, args.seed_base)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'synthetic_sanity_case')}, indent=2))
    print('synthetic:', json.dumps({k:v for k,v in result['synthetic_sanity_case'].items() if k != 'trace'}))


if __name__ == '__main__':
    main()
