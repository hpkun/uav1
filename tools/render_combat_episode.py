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


class CombatScene:
    """Persistent artists keep long 8v8 MP4 rendering practical."""
    def __init__(self, ax, trace, metadata):
        from matplotlib.lines import Line2D
        self.ax, self.trace, self.metadata = ax, trace, metadata
        radius = float(metadata['arena_radius'])
        altitude = max(4000, max(float(-trace[f'{side}_kinematics'][:, :, 2].min()) for side in ('red', 'blue')))
        angle = np.linspace(0, 2*np.pi, 120)
        ax.plot(radius*np.cos(angle), radius*np.sin(angle), np.zeros_like(angle), color='gray', alpha=.5)
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(0, altitude),
               xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)')
        ax.view_init(elev=35, azim=-60)
        self.agents = []
        for side, color, label in (('red', '#df3348', 'R'), ('blue', '#2484d0', 'B')):
            for agent in range(trace[f'{side}_alive'].shape[1]):
                line, = ax.plot([], [], [], color=color, alpha=.35, linewidth=1.1)
                marker, = ax.plot([], [], [], color=color, marker='o', markersize=6, linestyle='')
                text = ax.annotate(f'{label}{agent+1}', xy=(0, 0), xytext=(0, 0),
                    fontsize=8, color=color, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=.8),
                    arrowprops=dict(arrowstyle='-', color=color, linewidth=.5, alpha=.55))
                self.agents.append((side, agent, line, marker, text))
        self.event_artists = []
        self.previous_frame = 0
        ax.legend(handles=[Line2D([], [], color='#df3348', marker='o', label='Red alive'),
            Line2D([], [], color='#2484d0', marker='o', label='Blue alive'),
            Line2D([], [], color='gray', marker='x', linestyle='', label='Destroyed'),
            Line2D([], [], color='green', linestyle='--', label='Hit'),
            Line2D([], [], color='orange', linestyle='--', label='Miss')], loc='upper left', fontsize=8)

    def draw(self, frame, all_events=False):
        from mpl_toolkits.mplot3d import proj3d
        trace = self.trace
        projected = {'red': [], 'blue': []}
        for side, agent, line, marker, text in self.agents:
            alive = trace[f'{side}_alive']
            values = trace[f'{side}_kinematics']
            deaths = np.flatnonzero(~alive[:frame+1, agent])
            last = int(deaths[0]) if len(deaths) else frame
            path = values[:last+1, agent]
            line.set_data_3d(path[:, 0], path[:, 1], -path[:, 2])
            x, y, z = values[last, agent, :3]
            marker.set_data_3d([x], [y], [-z])
            marker.set_marker('o' if alive[frame, agent] else 'x')
            px, py, _ = proj3d.proj_transform(x, y, -z, self.ax.get_proj())
            text.xy = (px, py)
            projected[side].append((py, px, agent, text))
        # Separate labels in two compact columns; leader lines identify clustered UAVs.
        for side, points in projected.items():
            points.sort(key=lambda p: (p[0], p[2]))
            center_x = float(np.mean([p[1] for p in points])) + (-.022 if side == 'red' else .022)
            center_y = float(np.mean([p[0] for p in points]))
            for i, (_, _, _, text) in enumerate(points):
                text.set_position((center_x, center_y+(i-(len(points)-1)/2)*.006))
        for artist in self.event_artists:
            artist.remove()
        self.event_artists = []
        # Include events in skipped simulation steps rather than losing them to stride.
        lower = -1 if all_events else int(trace['steps'][min(self.previous_frame, frame)])
        upper = int(trace['steps'][frame])
        for event in self.metadata.get('events', []):
            if event['type'] == 'fire_attempt' and lower < event['step'] <= upper:
                pair = np.asarray([event['start'], event['end']])
                artist, = self.ax.plot(pair[:, 0], pair[:, 1], -pair[:, 2],
                    color='green' if event['hit'] else 'orange', linestyle='--',
                    alpha=.35 if all_events else .95, linewidth=1 if all_events else 2)
                self.event_artists.append(artist)
        self.previous_frame = frame
        survivors = [int(trace[f'{side}_alive'][frame].sum()) for side in ('red', 'blue')]
        self.ax.set_title(f"{self.metadata.get('algorithm', 'Combat')} | seed {self.metadata['episode_seed']}\n"
            f"t={trace['time_s'][frame]:.1f}s | step={upper} | Red {survivors[0]} / Blue {survivors[1]}"
            f"{self.metadata.get('playback_label', '')}")


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
    if Path(output).suffix.lower() in ('.mp4', '.gif'):
        metadata['playback_label'] = f" | ~{fps*stride*metadata['dt']:g}x playback"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scene = CombatScene(ax, trace, metadata)

    try:
        if output.suffix.lower() in ('.gif', '.mp4'):
            frames = sorted(set(range(0, len(trace['steps']), stride)) | {len(trace['steps'])-1})
            animation = FuncAnimation(fig, scene.draw, frames=frames, interval=1000/fps, repeat=False)
            animation.save(output, writer='pillow' if output.suffix.lower() == '.gif' else 'ffmpeg', fps=fps)
        else:
            scene.draw(len(trace['steps'])-1, all_events=True)
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
