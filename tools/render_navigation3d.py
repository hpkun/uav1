"""Render saved navigation traces only: 3-D positions and navigation attitudes."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(trace_path, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    with np.load(trace_path, allow_pickle=False) as data:
        states, goals, dt = data['states'], data['goals'], float(data['dt'])
        masks, phases = data['valid_masks'], data['phases']
    if states.ndim != 3 or states.shape[-1] != 6 or not np.isfinite(states).all():
        raise ValueError('expected finite navigation pose trace [frame,agent,6]')
    time = np.arange(len(states))*dt
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(states.shape[1]):
        line, = ax.plot(states[:, i, 0], states[:, i, 1], -states[:, i, 2], label=f'UAV {i+1}')
        color = line.get_color()
        ax.scatter(*np.r_[states[0, i, :2], -states[0, i, 2]], color=color, marker='o')
        ax.scatter(*np.r_[states[-1, i, :2], -states[-1, i, 2]], color=color, marker='x')
        for phase in np.unique(phases):
            goal = goals[np.flatnonzero(phases == phase)[0], i]
            ax.scatter(goal[0], goal[1], -goal[2], color=color, marker='*', s=90)
        sample = np.unique(np.linspace(0, len(states)-1, 6, dtype=int))
        s = states[sample, i]
        ax.quiver(s[:, 0], s[:, 1], -s[:, 2], np.cos(s[:, 4])*np.cos(s[:, 5]),
            np.cos(s[:, 4])*np.sin(s[:, 5]), np.sin(s[:, 4]), length=90, color=color)
    ax.set(xlabel='North x (m)', ylabel='East y (m)', zlabel='Altitude (m)',
        title=f'Cooperative navigation | {time[-1]:.1f} s | circles: start, crosses: end, stars: waypoints')
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'trajectory.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i in range(states.shape[1]):
        axes[0].plot(time, states[:, i, 3], label=f'UAV {i+1}')
        axes[1].plot(time, np.rad2deg(states[:, i, 4]))
        axes[2].plot(time, np.rad2deg(np.unwrap(states[:, i, 5])))
    for axis, label in zip(axes, ['Speed (m/s)', 'Pitch (deg, up +)', 'Heading (deg, unwrapped)']):
        axis.set_ylabel(label); axis.grid(alpha=.3)
    axes[0].legend(); axes[2].set_xlabel('Simulation time (s)')
    fig.tight_layout(); fig.savefig(output/'attitudes.png', dpi=140); plt.close(fig)
    with (output/'states.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'uav', 'valid', 'waypoint_index', 'x_north_m', 'y_east_m', 'altitude_m', 'speed_m_s', 'pitch_deg', 'heading_deg'])
        for frame, stamp in enumerate(time):
            for i, s in enumerate(states[frame]):
                writer.writerow([stamp, i+1, int(masks[frame, i]), int(phases[frame]), s[0], s[1], -s[2], s[3], np.rad2deg(s[4]), np.rad2deg(s[5])])
    return dict(frames=len(states), agents=states.shape[1], duration_s=float(time[-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True); p.add_argument('--output-dir', required=True)
    args = p.parse_args(); print(render(args.trace, args.output_dir))


if __name__ == '__main__':
    main()
