"""Render a combat trace as a trajectory image or animated GIF/MP4."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.combat_visualization import read_trace


def render(trace_path: str | Path, metadata_path: str | Path, output: str | Path,
           stride: int = 4, fps: int = 20) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if stride <= 0 or fps <= 0:
        raise ValueError('stride and fps must be positive')
    trace = read_trace(trace_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    radius = float(metadata['arena_radius'])
    altitude = max(4000, max(float(-trace[f'{side}_kinematics'][:, :, 2].min()) for side in ('red', 'blue')))

    def draw(frame):
        ax.clear()
        angle = np.linspace(0, 2 * np.pi, 120)
        ax.plot(radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle), color='gray', alpha=.5)
        survivors = {}
        for side, color, label in (('red', '#df3348', 'R'), ('blue', '#2484d0', 'B')):
            alive = trace[f'{side}_alive']
            values = trace[f'{side}_kinematics']
            survivors[side] = int(alive[frame].sum())
            for agent in range(4):
                # End the path at destruction, keeping its terminal position visible.
                deaths = np.flatnonzero(~alive[:frame+1, agent])
                last = int(deaths[0]) if len(deaths) else frame
                path = values[:last+1, agent]
                ax.plot(path[:, 0], path[:, 1], -path[:, 2], color=color, alpha=.6)
                x, y, z = values[last, agent, :3]
                ax.scatter(x, y, -z, color=color, marker='o' if alive[frame, agent] else 'x')
                ax.text(x, y, -z, f'{label}{agent+1}', fontsize=8)
        for event in metadata.get('events', []):
            if event['type'] == 'fire_attempt' and event['step'] == int(trace['steps'][frame]):
                pair = np.asarray([event['start'], event['end']])
                ax.plot(pair[:, 0], pair[:, 1], -pair[:, 2], color='green' if event['hit'] else 'orange', linestyle='--')
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(0, altitude),
               xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)',
               title=f"Combat | t={trace['time_s'][frame]:.1f}s | step={trace['steps'][frame]} | Red {survivors['red']} / Blue {survivors['blue']}")

    try:
        if output.suffix.lower() in ('.gif', '.mp4'):
            frames = sorted(set(range(0, len(trace['steps']), stride)) | {len(trace['steps'])-1})
            animation = FuncAnimation(fig, draw, frames=frames, interval=1000/fps)
            animation.save(output, writer='pillow' if output.suffix.lower() == '.gif' else 'ffmpeg', fps=fps)
        else:
            draw(len(trace['steps'])-1)
            fig.savefig(output, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    print(render(args.trace, args.metadata, args.output, args.stride, args.fps))


if __name__ == '__main__':
    main()
