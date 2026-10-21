"""Position and attitude rendering uses a saved deterministic navigation trace."""
from pathlib import Path
import numpy as np
from env.navigation3d import load_config
from algorithm.navigation3d.evaluation import run_episode, waypoint_controller
from tools.render_navigation3d import render


def test_saved_trace_geometry_and_attitude_render(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1]/'configs/navigation3d_v1.yaml')
    info, trace = run_episode(config, 123, controller=waypoint_controller)
    assert info['success']
    original = trace['states'].copy()
    path = tmp_path/'trace.npz'; np.savez_compressed(path, **trace)
    result = render(path, tmp_path)
    assert result['frames'] == info['episode_steps']+1
    assert result['duration_s'] == info['episode_steps']*config['dt']
    assert (tmp_path/'trajectory.png').stat().st_size > 10000
    assert (tmp_path/'attitudes.png').stat().st_size > 10000
    assert (tmp_path/'states.csv').stat().st_size > 1000
    assert np.array_equal(original, trace['states'])
