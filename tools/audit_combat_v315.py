"""Model-free v3.15 target, controller, dynamics and observation audit."""
import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from env.combat_env import MultiUAVCombatEnv
from env.control import action_to_target, action_to_control
from env.models import AircraftState
from env.v315_control import action_to_target_v315, action_to_control_v315


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for repository audits')
    output.mkdir(parents=True, exist_ok=True)
    old, new = [MultiUAVCombatEnv(ROOT / f'configs/combat_environment_v{v}.yaml') for v in (314, 315)]
    old.reset(31)
    new.reset(31)
    cfg, spec = new.config['action'], new.spec
    targets = []
    interior = 0
    for theta, speed, ah, ap, av in product([0, 55, 60, -55, -60], [150, 160, 225, 290, 300], *([[-1, -.5, 0, .5, 1]] * 3)):
        state = AircraftState(0., 0., -3000., speed, np.deg2rad(theta), 3.)
        action = [ah, ap, av]
        raw = action_to_target(state, action, cfg['command'])
        target = action_to_target_v315(state, action, cfg, spec)
        assert target.heading == raw.heading
        assert spec.theta_min <= target.pitch <= spec.theta_max
        assert spec.v_min <= target.speed <= spec.v_max
        same = raw == target
        if same:
            assert action_to_control(state, action, cfg) == action_to_control_v315(state, action, cfg, spec)
            interior += 1
        targets.append(dict(theta_deg=theta, speed=speed, heading_action=ah, pitch_action=ap, speed_action=av,
                            raw_pitch_deg=np.rad2deg(raw.pitch), target_pitch_deg=np.rad2deg(target.pitch),
                            raw_speed=raw.speed, target_speed=target.speed, heading_exact=True, unchanged_target=same))
    differences = []
    cases = [(55, 225, [0, 1, 0]), (-55, 225, [0, -1, 0]), (60, 225, [0, .6, 0]),
             (-60, 225, [0, -.6, 0]), (0, 290, [0, 0, 1]), (0, 160, [0, 0, -1]),
             (0, 300, [0, 0, .8]), (0, 150, [0, 0, -.8]), (0, 225, [.3, .1, -.2])]
    responses = []
    for case, (theta, speed, action) in enumerate(cases):
        initial = AircraftState(0., 0., -3000., speed, np.deg2rad(theta), 0.)
        for version, env in [(314, old), (315, new)]:
            state = initial.copy()
            for step in range(20):
                target = action_to_target(state, action, cfg['command']) if version == 314 else action_to_target_v315(state, action, cfg, spec)
                control = action_to_control(state, action, cfg) if version == 314 else action_to_control_v315(state, action, cfg, spec)
                derivative = env.dynamics.derivatives(state, control)
                nxt = env.integrator.step(state, control, env.dynamics, spec)
                row = dict(case=case, version=version, step=step, theta_deg=np.rad2deg(state.theta), speed=state.v,
                           target_pitch_deg=np.rad2deg(target.pitch), target_speed=target.speed,
                           nx=control.nx, nz=control.nz, phi=control.phi, theta_rate=derivative[4], acceleration=derivative[3],
                           next_x=nxt.x, next_y=nxt.y, next_z=nxt.z, next_speed=nxt.v,
                           next_theta=nxt.theta, next_heading=nxt.psi)
                responses.append(row)
                if step == 0:
                    differences.append(row.copy())
                    if version == 315 and case in (2, 3):
                        assert abs(derivative[4]) < 1e-12
                    if version == 315 and case in (6, 7):
                        assert abs(derivative[3]) < 1e-12
                state = nxt
    # Same interior command must produce exactly the same RK4 state.
    normal = [r for r in responses if r['case'] == 8 and r['step'] == 0]
    assert {k: v for k, v in normal[0].items() if k != 'version'} == {k: v for k, v in normal[1].items() if k != 'version'}
    feedback = []
    action = np.tile([.5, -1, 0], (5, 1))
    for env in (old, new):
        env.red = [AircraftState(float(i), 0., -3000., 225., 0., 0.) for i in range(5)]
        env.blue = [s.copy() for s in env.red]
        env.red_last_executed_phi = env._advance(env.red, action)
        env.blue_last_executed_phi = env._advance(env.blue, action)
        assert np.array_equal(env.red_last_executed_phi, env.blue_last_executed_phi)
        for r, b in zip(env.red, env.blue):
            np.testing.assert_array_equal(r.as_array(), b.as_array())
        obs = env._observations()
        for i in range(5):
            feedback.append(dict(version=env.environment_version, agent=i, executed_phi=env.red_last_executed_phi[i], observation_slot4=obs[i, 4]))
        if env is old:
            oldobs = obs
        else:
            assert not obs[:, 4].any() and env.red_last_executed_phi.any()
            np.testing.assert_array_equal(np.delete(obs, 4, axis=1), np.delete(oldobs, 4, axis=1))
    for r, n in zip(old.red, new.red):
        np.testing.assert_array_equal(r.as_array(), n.as_array())
    checks = dict(cuda_device=torch.cuda.get_device_name(), model_loaded=False, target_cases=len(targets),
                  feasible_pitch=True, feasible_speed=True, heading_exact=True, interior_controls_exact=interior,
                  normal_control_and_rk4_exact=True, boundary_pitch_rate_zero=True, boundary_acceleration_zero=True,
                  response_rows=len(responses), observation_feedback_disconnected=True,
                  other_55_observation_slots_exact=True, red_blue_control_symmetric=True, executed_phi_retained=True)
    for name, rows in [('target_feasibility', targets), ('control_response', responses),
                       ('v314_v315_difference', differences), ('observation_feedback_check', feedback)]:
        numeric = [[v for v in row.values() if isinstance(v, (int, float, np.number))] for row in rows]
        assert all(np.isfinite(values).all() for values in numeric)
        write_csv(output / f'{name}.csv', rows)
    checks['all_numeric_outputs_finite'] = True
    (output / 'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/v315_validation')
    run(parser.parse_args().output)
