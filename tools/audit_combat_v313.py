"""Same-state/actions/RNG audit of the HA-only v3.12 -> v3.13 fork."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v36_geometry import combat_geometry
from env.v313_geometry import combat_geometry_v313
from tools._capture_v33_reference import canonical, states
from tools.audit_combat_v39 import check_decomposition


def run_audit():
    old = MultiUAVCombatEnv('configs/combat_environment_v312.yaml')
    new = MultiUAVCombatEnv('configs/combat_environment_v313.yaml')
    old.reset(313)
    new.reset(313)
    assert canonical(states(old)) == canonical(states(new))
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    cfg = deepcopy(new.config)
    cfg['environment_version'] = '3.12'
    assert cfg == old.config
    assert type(old.fixed_policy) is type(new.fixed_policy)
    # A legal, deliberately HA-sensitive test state, identical for both versions.
    # Others are dead slots. Both live aircraft retain initial six missiles.
    for e in (old, new):
        e.red = [AircraftState(0., 0., -3000., 225., np.pi/3, 0.)] + [s.copy() for s in e.red[1:]]
        e.blue = [AircraftState(4500., 0., -3000., 225., 0., np.pi)] + [s.copy() for s in e.blue[1:]]
        for s in e.red[1:] + e.blue[1:]:
            s.alive = False
    traces = [[], []]
    for e, trace, geometry in ((old, traces[0], combat_geometry), (new, traces[1], combat_geometry_v313)):
        original = e.dense_combat_reward
        def capture(e=e, trace=trace, geometry=geometry, original=original):
            g = geometry(e.red, e.blue)
            reward, diagnostics = original()
            trace.append(dict(step=e.steps, states=canonical(states(e)),
                rng=canonical(e.rng.bit_generator.state),
                geometry={key: float(value[0, 0]) for key, value in g.items()},
                red_eligible=bool(e.weapon.eligibility(e.red, e.blue, e.red_ammo)[0][0, 0]),
                blue_eligible=bool(e.weapon.eligibility(e.blue, e.red, e.blue_ammo)[0][0, 0]),
                reward_diagnostics=canonical(diagnostics)))
            return reward, diagnostics
        e.dense_combat_reward = capture
    totals = [np.zeros(5), np.zeros(5)]
    first_reward_difference = first_qualification_difference = first_state_difference = first_rng_difference = None
    done = [False, False]
    final = [None, None]
    actions = np.zeros((5, 3))
    for step in range(1, 1001):
        results = []
        for i, e in enumerate((old, new)):
            if done[i]:
                results.append(None)
                continue
            result = e.step(actions, actions)
            totals[i] += result[1]
            check_decomposition(result[1], result[4], totals[i])
            done[i] = result[2] or result[3]
            final[i] = result[4]
            results.append(result)
        if all(result is not None for result in results):
            a, b = traces[0][-1], traces[1][-1]
            if first_qualification_difference is None:
                # These snapshots are post-motion, immediately before sampling.
                assert a['states'] == b['states'] and a['rng'] == b['rng']
                for key in ('distance', 'ata', 'aa', 'los_elevation'):
                    assert a['geometry'][key] == b['geometry'][key]
                if a['red_eligible'] != b['red_eligible'] or a['blue_eligible'] != b['blue_eligible']:
                    first_qualification_difference = step
                    assert a['geometry']['ha'] != b['geometry']['ha']
            if first_reward_difference is None and not np.array_equal(results[0][1], results[1][1]):
                first_reward_difference = step
            if first_state_difference is None and canonical(states(old)) != canonical(states(new)):
                first_state_difference = step
            if first_rng_difference is None and old.rng.bit_generator.state != new.rng.bit_generator.state:
                first_rng_difference = step
        if all(done):
            break
    assert first_qualification_difference is not None
    assert first_rng_difference is None or first_rng_difference >= first_qualification_difference
    assert first_state_difference is None or first_state_difference >= first_qualification_difference
    a, b = [trace[first_qualification_difference-1] for trace in traces]
    return dict(initial_states_identical=True, config_only_version_differs=True,
        blue_policy_class_identical=True, external_red_and_blue_actions_identical=True,
        initial_rng_identical=True, pre_first_qualification_motion_and_rng_identical=True,
        ata_aa_distance_and_signed_los_identical_before_fork=True,
        first_reward_difference_step=first_reward_difference,
        first_qualification_difference_step=first_qualification_difference,
        first_postcombat_state_difference_step=first_state_difference,
        first_rng_difference_step=first_rng_difference,
        qualification_snapshot_v312=a, qualification_snapshot_v313=b,
        explanation='Only HA differs at the shared pre-combat state. R3/R4 can differ before firing; '
            'eligibility and hit inputs then cause natural combat/RNG trajectory divergence.',
        terminal_v312=canonical(final[0]), terminal_v313=canonical(final[1]),
        episode_return_v312=totals[0].tolist(), episode_return_v313=totals[1].tolist())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for audit; unavailable')
    result = run_audit()
    result['cuda_device'] = torch.cuda.get_device_name()
    Path('outputs/v313_validation/same_state_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k.endswith('_step')}, indent=2))


if __name__ == '__main__':
    main()
