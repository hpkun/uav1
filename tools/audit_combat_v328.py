"""Lightweight trim-turn audit using unchanged controller, dynamics and RK4."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.math_utils import wrap_angle
from env.v320_control import action_to_control_v320


def turn_response(version, speed):
    e = MultiUAVCombatEnv(ROOT / f'configs/combat_environment_v{version}.yaml')
    state = AircraftState(0, 0, -3000, speed, 0, 0)
    first = action_to_control_v320(state, [1, 0, 0], e.config['action'], e.spec)
    yaw_rate = float(e.dynamics.derivatives(state, first)[5])
    angle = time = 0.
    points = [state.as_array()[:2]]
    crossing = {}
    while angle < math.pi:
        control = action_to_control_v320(state, [1, 0, 0], e.config['action'], e.spec)
        next_state = e.integrator.step(state, control, e.dynamics, e.spec)
        delta = float(wrap_angle(next_state.psi - state.psi))
        for name, threshold in (('90', math.pi/2), ('180', math.pi)):
            if name not in crossing and angle < threshold <= angle + delta:
                crossing[name] = time + e.dt * (threshold - angle) / delta
        angle += delta
        time += e.dt
        state = next_state
        points.append(state.as_array()[:2])
    xy = np.asarray(points)
    # Circle fit to actual RK4 positions, independent of the analytic radius.
    matrix = np.column_stack((2*xy[:,0], 2*xy[:,1], np.ones(len(xy))))
    center_x, center_y, constant = np.linalg.lstsq(matrix, np.sum(xy**2, axis=1), rcond=None)[0]
    radius_rk4 = float(np.sqrt(constant + center_x**2 + center_y**2))
    return dict(version=f'3.{version[1:]}', speed_m_s=speed, achieved_nz=first.nz,
        phi_deg=float(np.degrees(first.phi)), yaw_rate_deg_s=float(np.degrees(yaw_rate)),
        turn_90_s=crossing['90'], turn_180_s=crossing['180'],
        approximate_radius_m=speed/abs(yaw_rate), rk4_circle_radius_m=radius_rk4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v328_validation/turn_audit.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for repository audits; no CPU fallback')
    rows = [turn_response(v, speed) for v in ('326', '328') for speed in (150, 225, 250, 300)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
