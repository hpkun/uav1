"""Controller grid and CPU environment smoke, without training or checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import sys
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v329_control import action_to_control_v329


def grid_audit(e, speed, grid):
    state = AircraftState(0, 0, -3000, speed, 0, 0)
    rows = []
    for psi in grid:
        for theta in grid:
            c = action_to_control_v329(state, [psi, theta, 0], e.config['action'], e.spec)
            rates = e.dynamics.derivatives(state, c)
            rows.append(dict(v=speed, theta=0, a_psi=float(psi), a_theta=float(theta),
                nz=c.nz, phi_deg=math.degrees(c.phi), psi_dot_deg_s=math.degrees(rates[5]),
                theta_dot_deg_s=math.degrees(rates[4])))
    n = len(grid)
    yaw = np.array([r['psi_dot_deg_s'] for r in rows]).reshape(n, n)
    pitch = np.array([r['theta_dot_deg_s'] for r in rows]).reshape(n, n)
    raw_a = 1 + speed * np.asarray(grid) * math.pi / 6 / 9.81
    summary = dict(v=speed, grid_points=len(rows), nz_ge_9_9_fraction=sum(r['nz']>=9.9 for r in rows)/len(rows),
        a_raw_negative_fraction=float(np.mean(raw_a<0)),
        yaw_nonmonotonic_adjacent_count=int((np.diff(yaw,axis=0)<-1e-9).sum()),
        pitch_nonmonotonic_adjacent_count=int((np.diff(pitch,axis=1)<-1e-9).sum()),
        yaw_flat_adjacent_fraction=float(np.mean(np.abs(np.diff(yaw,axis=0))<1e-9)),
        pitch_flat_adjacent_fraction=float(np.mean(np.abs(np.diff(pitch,axis=1))<1e-9)),
        heading_sign_reversal_count=sum(r['a_psi']*r['psi_dot_deg_s'] < -1e-9 for r in rows),
        positive_pitch_negative_response_count=sum(r['a_theta']>0 and r['theta_dot_deg_s'] < -1e-9 for r in rows),
        zero_pitch_negative_response_count=sum(r['a_theta']==0 and r['theta_dot_deg_s'] < -1e-9 for r in rows),
        finite=all(np.isfinite(list(r.values())).all() for r in rows))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v329_validation/controller_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    e = MultiUAVCombatEnv(ROOT/'configs/combat_environment_v329.yaml')
    summaries, rows, fine, fine_rows = [], [], [], []
    for speed in (150,225,300):
        s, r = grid_audit(e, speed, np.linspace(-1,1,9)); summaries.append(s); rows.extend(r)
        s, r = grid_audit(e, speed, np.linspace(-1,1,81)); fine.append(s); fine_rows.extend(r)
    for name, data in (('grid.csv', rows), ('fine_grid.csv', fine_rows)):
        with (args.output/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    cpu = []
    for seed in (1,31,10000000):
        obs, _ = e.reset(seed); rng=np.random.default_rng(seed)
        for step in range(100):
            obs,reward,terminated,truncated,info=e.step(rng.uniform(-1,1,(5,3)))
            assert obs.shape==(5,56) and reward.shape==(5,)
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            assert all(np.isfinite(s.as_array()).all() for s in e.red+e.blue)
            assert all(150<=s.v<=300 and abs(s.theta)<=math.pi/3 for s in e.red+e.blue)
            if terminated or truncated:break
        cpu.append(dict(seed=seed,steps=e.steps,finite=True,termination_reason=info['termination_reason']))
    result=dict(coarse_grid=summaries,fine_grid=fine,cpu_environment_smoke=cpu,
                axis_responses=[r for r in rows if r['a_psi']==0 or r['a_theta']==0],
                simultaneous_large_commands=[r for r in rows if abs(r['a_psi'])==abs(r['a_theta'])==1],
                positive_pitch_negative_response_examples=[r for r in fine_rows
                    if r['a_psi']==1 and r['a_theta']>0 and r['theta_dot_deg_s'] < -1e-9][:6],
                thresholds=dict(load_diagnostic=9.9,flat_rate_tolerance_deg_s=1e-9),
                note='Diagnostic thresholds are descriptive, not pass/fail acceptance rates.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
