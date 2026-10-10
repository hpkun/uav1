import json
from pathlib import Path
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.models import AircraftState
from tools.combat_visualization import (RecordingCombatEnv, FRAME_FIELDS, append_frame,
    read_trace, write_trace, ensure_fresh_output, states_array)
from tools.record_combat_episode import record
from tools.render_combat_episode import render
from tools.render_combat_episode_interactive import render as render_interactive

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('version', ['33', '34', '35'])
def test_versioned_recorder_preserves_full_episode_and_info(version):
    from tools._capture_v33_reference import canonical
    config = load_config(ROOT / f'configs/combat_environment_v{version}.yaml')
    base, observed = MultiUAVCombatEnv(config), RecordingCombatEnv(config)
    assert observed.environment_version == base.environment_version
    assert canonical(base.reset(10000000)) == canonical(observed.reset(10000000))
    rng = np.random.default_rng(44)
    while True:
        action = rng.uniform(-1, 1, (5, 3)).astype(np.float32)
        expected, actual = base.step(action), observed.step(action)
        assert canonical(expected) == canonical(actual)
        np.testing.assert_array_equal(states_array(base.red + base.blue), states_array(observed.red + observed.blue))
        assert base.rng.bit_generator.state == observed.rng.bit_generator.state
        if version == '35':
            assert base.fixed_policy.rng.bit_generator.state == observed.fixed_policy.rng.bit_generator.state
        if actual[2] or actual[3]:
            break


def test_recorder_preserves_transition_and_rng_semantics():
    base, observed = MultiUAVCombatEnv(), RecordingCombatEnv()
    assert np.array_equal(base.reset(701)[0], observed.reset(701)[0])
    rng = np.random.default_rng(44)
    for _ in range(100):
        action = rng.uniform(-1, 1, (4, 3)).astype(np.float32)
        expected, actual = base.step(action), observed.step(action)
        for first, second in zip(expected[:4], actual[:4]):
            assert np.array_equal(first, second)
        assert np.array_equal(states_array(base.red + base.blue), states_array(observed.red + observed.blue))
        assert base.rng.bit_generator.state == observed.rng.bit_generator.state
        if actual[2] or actual[3]:
            break


def test_recording_captures_exact_attempts_and_shared_kills(monkeypatch):
    env = RecordingCombatEnv()
    env.reset(2)
    env.red = [AircraftState(0, i*20, -3000, 225, 0, 0) for i in range(4)]
    env.blue = [AircraftState(1000, 0, -3000, 225, 0, np.pi)] + [AircraftState(0, 0, -3000, 225, 0, 0, False) for _ in range(3)]
    monkeypatch.setattr(type(env.weapon), 'attempt_hit', lambda self, geometry, rng: True)
    _, _, _, _, info = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    attempts = [event for event in env.events if event['type'] == 'fire_attempt']
    kills = [event for event in env.events if event['type'] == 'attack_kill']
    assert len(attempts) == info['red_step_fire_attempts'] + info['blue_step_fire_attempts']
    red_kill = next(event for event in kills if event['side'] == 'red')
    assert red_kill['target'] == 0 and red_kill['attackers'] == [0, 1, 2, 3]
    assert sum(event['hit'] for event in attempts) == info['red_step_weapon_hits'] + info['blue_step_weapon_hits']


def test_trace_shape_and_schema_validation(tmp_path):
    env = RecordingCombatEnv(); env.reset(7)
    frames = {key: [] for key in FRAME_FIELDS}; append_frame(frames, env)
    path = tmp_path / 'trace.npz'; write_trace(path, frames, {})
    trace = read_trace(path)
    assert trace['red_kinematics'].shape == trace['blue_kinematics'].shape == (1, 4, 6)
    assert trace['red_alive'].shape == trace['blue_alive'].shape == (1, 4)
    np.savez(path, trace_schema_version=1)
    with pytest.raises(ValueError, match='schema'):
        read_trace(path)


def test_record_and_render_complete_timeout_episode(tmp_path):
    class ZeroActor:
        def act(self, observation, alive, deterministic=True):
            return np.zeros((*observation.shape[:-1], 3), np.float32)
    config = load_config(ROOT / 'configs/combat_environment.yaml')
    config['simulation']['max_steps'] = 3
    output = tmp_path / 'record'
    result = record(ZeroActor(), config, 101, output)
    trace = read_trace(output / 'episode_trace.npz')
    assert result['termination_reason'] == 'red_failure_timeout'
    assert trace['local_rewards'].shape == (3, 4)
    assert trace['reward_components'].shape == (3, 4, 4)
    assert bool(trace['truncated'][-1])
    assert len(trace['steps']) == 4
    image = render(output / 'episode_trace.npz', output / 'metadata.json', tmp_path / 'combat.png')
    assert image.stat().st_size > 10000
    html = render_interactive(output / 'episode_trace.npz', output / 'metadata.json', tmp_path / 'combat.html')
    text = html.read_text(encoding='utf-8')
    assert html.stat().st_size > 1000000
    assert '<script src=' not in text.lower()
    assert all(token in text for token in ('Play', 'Pause', 'Altitude (m)', 'scatter3d', 'red_failure_timeout'))
    assert json.loads((output / 'metadata.json').read_text())['red_survivors'] == 4
    with pytest.raises(FileExistsError):
        ensure_fresh_output(output)
