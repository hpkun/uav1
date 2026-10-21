"""Fixed-controller navigation precheck; no long RL training or combat imports."""
from pathlib import Path
import argparse
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.navigation3d import load_config
from algorithm.navigation3d.trainer import require_cuda
from algorithm.navigation3d.evaluation import run_episode, waypoint_controller, write_json
from tools.render_navigation3d import render


def controller(name):
    if name == 'waypoint':
        return waypoint_controller
    def action(env):
        result = np.zeros((env.num_agents, 3))
        if name == 'orbit':
            result[:, 0] = 1.
            result[:, 1] = [-s.theta/env.config['motion']['pitch_delta_max'] for s in env.states]
        elif name == 'slow_orbit':
            result[:, 0] = 1.
            result[:, 1] = [-s.theta/env.config['motion']['pitch_delta_max'] for s in env.states]
            result[:, 2] = -1.
        elif name == 'retreat':
            result = waypoint_controller(env)
            result[:, 0] = [((np.arctan2((env.goals[i]-env.positions[i])[1], (env.goals[i]-env.positions[i])[0])
                +np.pi-s.psi+np.pi) % (2*np.pi)-np.pi)/env.config['motion']['heading_delta_max'] for i, s in enumerate(env.states)]
            result[:, 1] = [-s.theta/env.config['motion']['pitch_delta_max'] for s in env.states]
        return np.clip(result, -1, 1)*env.valid_mask[:, None]
    return action


def validate(output_dir, seeds):
    require_cuda()
    config = load_config(ROOT/'configs/navigation3d_v1.yaml')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    records, summaries = [], []
    for name in ('waypoint', 'zero', 'orbit', 'slow_orbit', 'retreat'):
        rows = []
        for seed in seeds:
            info, trace = run_episode(config, seed, controller=controller(name))
            shaped = trace['reward_components'][:, :3].sum(axis=1)
            discounted = (.99**np.arange(len(shaped))) @ shaped
            # Terminal Phi=0 implies discounted shaping is nonpositive.
            if (discounted > 1e-9).any() or (shaped.sum(axis=0) > 1e-9).any():
                raise AssertionError('navigation shaping harvest detected')
            row = dict(controller=name, seed=seed, success=info['success'], steps=info['episode_steps'],
                duration_s=info['simulation_time_s'], termination_reason=info['termination_reason'],
                episode_return=info['episode_return'], component_totals=info['component_totals'],
                component_positive=info['component_positive'], component_negative=info['component_negative'],
                shaping_per_agent=shaped.sum(axis=0), discounted_shaping_per_agent=discounted,
                shaping_positive_agent_seconds=float((shaped > 0).sum()*config['dt']),
                shaping_positive_per_agent=trace['reward_components'][:, :3].clip(min=0).sum(axis=(0, 1)),
                completion_bonus_per_agent=config['reward']['completion_bonus'])
            records.append(row); rows.append(row)
            if seed == seeds[0]:
                folder = output/name; folder.mkdir()
                np.savez_compressed(folder/'trace.npz', **trace)
                render(folder/'trace.npz', folder)
        mean_component = {k: float(np.mean([r['component_totals'][k].mean() for r in rows])) for k in rows[0]['component_totals']}
        summaries.append(dict(controller=name, episodes=len(rows), successes=sum(r['success'] for r in rows),
            success_rate=float(np.mean([r['success'] for r in rows])),
            average_steps=float(np.mean([r['steps'] for r in rows])),
            average_return=float(np.mean([r['episode_return'] for r in rows])),
            average_shaping_per_agent=float(np.mean([r['shaping_per_agent'].mean() for r in rows])),
            max_positive_shaping_per_agent=float(max(r['shaping_positive_per_agent'].max() for r in rows)),
            average_component_per_agent=mean_component,
            termination_counts={reason: sum(r['termination_reason'] == reason for r in rows) for reason in sorted(set(r['termination_reason'] for r in rows))}))
        print(f'{name}: success={summaries[-1]["successes"]}/{len(rows)}, shaping/agent={summaries[-1]["average_shaping_per_agent"]:.6f}', flush=True)
    write_json(output/'summary.json', summaries)
    write_json(output/'episodes.json', records)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        columns = ['controller', 'episodes', 'successes', 'success_rate', 'average_steps', 'average_return', 'average_shaping_per_agent', 'max_positive_shaping_per_agent']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(summaries)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--episodes', type=int, default=20)
    args = p.parse_args()
    if not 1 <= args.episodes <= 100:
        p.error('precheck episodes must be in [1,100]')
    validate(args.output_dir, list(range(20_000_000, 20_000_000+args.episodes)))


if __name__ == '__main__':
    main()
