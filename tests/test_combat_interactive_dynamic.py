"""Interactive replay includes every UAV and preserves the user-controlled camera."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import pytest
import yaml
from tools.record_combat_episode import record
from tools.render_combat_episode_interactive import render

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('config_name,n', [('combat_environment', 4), ('combat_environment_v24', 8)])
def test_interactive_all_aircraft_frame_mapping_and_playback(tmp_path, monkeypatch, config_name, n):
    class ZeroActor:
        def act(self, observations, alive, deterministic=True):
            return np.zeros((*observations.shape[:-1], 3), np.float32)
    config = yaml.safe_load((ROOT/f'configs/{config_name}.yaml').read_text())
    config['simulation']['max_steps'] = 3
    directory = tmp_path/'episode'; record(ZeroActor(), config, 101, directory)
    figures = []
    original = go.Figure.write_html
    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(go.Figure, 'write_html', capture)
    output = render(directory/'episode_trace.npz', directory/'metadata.json', tmp_path/'replay.html', stride=1)
    figure = figures[0]
    assert {t.name for t in figure.data if t.mode == 'markers+text'} == {
        f'{side}{i}' for side in ('R', 'B') for i in range(1, n+1)}
    assert len(figure.data) == 4*n+3  # history+marker for each UAV, hit/miss, static arena
    assert all(tuple(frame.traces) == tuple(range(4*n+2)) for frame in figure.frames)
    assert len(figure.frames) == 4
    assert figure.layout.scene.uirevision == 'combat-camera'
    buttons = {b.label: b for b in figure.layout.updatemenus[0].buttons}
    for speed in (.25, .5, 1., 2.):
        assert buttons[f'Play ({speed:g}x)'].args[1]['frame']['duration'] == pytest.approx(100/speed)
    text = output.read_text()
    assert '<script src=' not in text.lower() and 'Pause' in text and 'scrollZoom' in text
    with pytest.raises(ValueError, match='playback speed'):
        render(directory/'episode_trace.npz', directory/'metadata.json', tmp_path/'bad.html', playback_speed=0)
