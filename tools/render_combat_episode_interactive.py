"""Create a standalone, interactive 3D replay of an ordinary combat episode."""
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
           stride: int = 4) -> Path:
    import plotly.graph_objects as go
    if stride <= 0:
        raise ValueError('stride must be positive')
    trace = read_trace(trace_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    radius = float(metadata['arena_radius'])
    indices = sorted(set(range(0, len(trace['steps']), stride)) | {len(trace['steps'])-1})
    # Include every event step even when the normal replay stride skips it.
    event_steps = {event['step'] for event in metadata.get('events', [])}
    indices = sorted(set(indices) | {i for i, step in enumerate(trace['steps']) if step in event_steps})

    def data_at(frame):
        data = []
        for side, color, label in (('red', '#df3348', 'R'), ('blue', '#2484d0', 'B')):
            alive = trace[f'{side}_alive']
            values = trace[f'{side}_kinematics']
            for agent in range(4):
                deaths = np.flatnonzero(~alive[:frame+1, agent])
                last = int(deaths[0]) if len(deaths) else frame
                path = values[:last+1, agent]
                data.append(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=-path[:, 2],
                    mode='lines', line=dict(color=color, width=2), showlegend=False,
                    name=f'{label}{agent+1} trajectory', hoverinfo='skip'))
                state = values[last, agent]
                data.append(go.Scatter3d(x=[state[0]], y=[state[1]], z=[-state[2]],
                    mode='markers+text', text=[f'{label}{agent+1}'], textposition='top center',
                    name=f'{label}{agent+1}', marker=dict(color=color, size=5,
                    symbol='circle' if alive[frame, agent] else 'x'),
                    customdata=[[bool(alive[frame, agent]), *state[3:].tolist()]],
                    hovertemplate='Alive: %{customdata[0]}<br>Speed: %{customdata[1]:.1f} m/s<br>Pitch: %{customdata[2]:.3f}<br>Heading: %{customdata[3]:.3f}<extra>%{fullData.name}</extra>'))
        for hit, color in ((True, '#20c75a'), (False, '#ed9924')):
            coordinates = []
            for event in metadata.get('events', []):
                if event['type'] == 'fire_attempt' and event['step'] == int(trace['steps'][frame]) and event['hit'] == hit:
                    coordinates.extend([event['start'], event['end'], [None, None, None]])
            data.append(go.Scatter3d(x=[p[0] for p in coordinates], y=[p[1] for p in coordinates],
                z=[None if p[2] is None else -p[2] for p in coordinates], mode='lines',
                line=dict(color=color, width=4), name='Weapon hit' if hit else 'Missed attempt'))
        return data

    def title(frame):
        red = int(trace['red_alive'][frame].sum()); blue = int(trace['blue_alive'][frame].sum())
        current = [event['type'] for event in metadata.get('events', []) if event['step'] == int(trace['steps'][frame])]
        ending = f" | {metadata['termination_reason']}" if frame == len(trace['steps'])-1 else ''
        return f"Combat | step {trace['steps'][frame]} | {trace['time_s'][frame]:.1f}s | Red {red} / Blue {blue} | {', '.join(current)}{ending}"

    fig = go.Figure(data=data_at(0))
    fig.frames = [go.Frame(name=str(frame), data=data_at(frame), traces=list(range(18)),
                           layout=go.Layout(title=title(frame))) for frame in indices]
    angle = np.linspace(0, 2*np.pi, 120)
    fig.add_trace(go.Scatter3d(x=radius*np.cos(angle), y=radius*np.sin(angle), z=np.zeros_like(angle),
                             mode='lines', line=dict(color='gray', width=2), name='Arena'))
    altitude = max(4000, max(float(-trace[f'{side}_kinematics'][:, :, 2].min()) for side in ('red', 'blue')))
    animation = dict(frame=dict(duration=80, redraw=True), transition=dict(duration=0), fromcurrent=True)
    fig.update_layout(title=title(0), uirevision='combat-replay',
        scene=dict(xaxis=dict(title='X (m)', range=[-radius, radius]),
                   yaxis=dict(title='Y (m)', range=[-radius, radius]),
                   zaxis=dict(title='Altitude (m)', range=[0, altitude]), aspectmode='cube', dragmode='orbit'),
        updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, animation]),
            dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
        sliders=[dict(active=0, steps=[dict(label=f"{trace['time_s'][frame]:.1f}s", method='animate',
            args=[[str(frame)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for frame in indices])])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(scrollZoom=True, displaylogo=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stride', type=int, default=4)
    args = parser.parse_args()
    print(render(args.trace, args.metadata, args.output, args.stride))


if __name__ == '__main__':
    main()
