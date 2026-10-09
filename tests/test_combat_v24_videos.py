"""Focused recording lifecycle, official checkpoint and representative-selection tests."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from algorithm.common.protocol import config_sha256
from tools.combat_visualization import RecordingCombatEnv, states_array, read_trace, checkpoint_sha256
from tools.record_combat_episode import record, load_recording_policy
from tools.render_combat_episode import CombatScene, render
from tools.create_combat_v24_videos import analyze_episode, select_representative_seeds, probe_video
from env.combat_env import MultiUAVCombatEnv

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())


class ZeroActor:
    def act(self, observations, alive, deterministic=True):
        assert deterministic
        return np.zeros((*observations.shape[:-1], 3), np.float32)


@pytest.mark.parametrize('environment,n', [('combat_environment', 4), ('combat_environment_v24', 8)])
def test_dynamic_trace_markers_and_labels(tmp_path, environment, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env = config(environment); env['simulation']['max_steps'] = 3
    directory = tmp_path / 'record'
    metadata = record(ZeroActor(), env, 202, directory)
    trace = read_trace(directory / 'episode_trace.npz')
    assert trace['red_kinematics'].shape == (4, n, 6)
    fig = plt.figure(); ax = fig.add_subplot(projection='3d')
    try:
        scene = CombatScene(ax, trace, metadata); scene.draw(3)
        assert len(scene.agents) == 2*n
        assert {text.get_text() for *_, text in scene.agents} == {f'{side}{i}' for side in ('R', 'B') for i in range(1, n+1)}
    finally:
        plt.close(fig)


def test_v24_recording_preserves_rng_and_transitions():
    env = config('combat_environment_v24')
    expected, observed = MultiUAVCombatEnv(env), RecordingCombatEnv(env)
    np.testing.assert_array_equal(expected.reset(54)[0], observed.reset(54)[0])
    rng = np.random.default_rng(5)
    for _ in range(80):
        actions = rng.uniform(-1, 1, (8, 3)).astype(np.float32)
        a, b = expected.step(actions), observed.step(actions)
        for x, y in zip(a[:4], b[:4]):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(states_array(expected.red+expected.blue), states_array(observed.red+observed.blue))
        assert expected.rng.bit_generator.state == observed.rng.bit_generator.state
        if a[2] or a[3]:
            break


@pytest.mark.parametrize('identity', ['RMAPPO', 'STEA-MAPPO'])
def test_recurrent_retention_death_and_episode_reset(tmp_path, monkeypatch, identity):
    import tools.record_combat_episode as recorder
    class DeathEnv(RecordingCombatEnv):
        def step(self, action):
            if self.steps == 0:
                self.red[0].alive = False
            return super().step(action)
    monkeypatch.setattr(recorder, 'RecordingCombatEnv', DeathEnv)
    class Recurrent:
        actor = SimpleNamespace(gru_hidden_dim=128)
        def __init__(self):
            self.inputs, self.outputs = [], []
        def act(self, obs, alive, hidden, start, deterministic=True):
            assert deterministic
            self.inputs.append((hidden.copy(), float(start)))
            next_hidden = hidden+1
            self.outputs.append(next_hidden)
            actions = np.zeros((8, 3), np.float32)
            return actions, actions, np.zeros(8), next_hidden
    trainer = Recurrent()
    env = config('combat_environment_v24'); env['simulation']['max_steps'] = 3
    for i in range(2):
        record(trainer, env, 202+i, tmp_path / str(i), {'algorithm': identity})
    for offset in (0, 3):
        assert trainer.inputs[offset][1] == 1 and not trainer.inputs[offset][0].any()
        for step in (1, 2):
            hidden, start = trainer.inputs[offset+step]
            assert start == 0 and not hidden[0].any()
            assert np.all(hidden[1:] == step)
        assert not trainer.outputs[offset+2].any()  # terminal hidden explicitly reset


@pytest.mark.parametrize('name,config_name,environment', [
    ('mappo', 'mappo', 'combat_environment'),
    ('madsac', 'madsac', 'combat_environment'),
    ('rmappo', 'rmappo_8v8', 'combat_environment_v24'),
    ('ea_mappo', 'ea_mappo_8v8', 'combat_environment_v24'),
    ('stea_mappo', 'stea_mappo_8v8_formal', 'combat_environment_v24')])
def test_official_checkpoint_record_and_evaluation_parity(tmp_path, name, config_name, environment):
    if not torch.cuda.is_available():
        pytest.fail('CUDA is mandatory for checkpoint recording tests')
    algorithm, env = config(config_name), config(environment)
    env['simulation']['max_steps'] = 3  # isolated test fixture, never writes formal configs
    factory = importlib.import_module(f'algorithm.{name}.factory')
    build = getattr(factory, f'build_{name}_trainer')
    trainer = build(algorithm, 'cuda')
    n = algorithm['network']; architecture = getattr(trainer, 'network_architecture', {})
    if name == 'madsac':
        architecture = {'hidden_dim': n['actor_hidden_layers'][0], 'attention_heads': n['attention_heads']}
    extra = {'environment_version': env['environment_version'],
        'environment_config_sha256': config_sha256(env), 'algorithm_config_sha256': config_sha256(algorithm),
        'observation_dim': n['observation_dim'], 'action_dim': 3, 'num_agents': n['num_agents'],
        'training_seed': algorithm['training']['seed'], 'training_gamma': trainer.gamma,
        'training_num_envs': 1, 'training_total_sampled_steps': 1, 'training_smoke': False,
        'effective_hidden_dim': n['critic_hidden_layers'][0], 'network_architecture': architecture}
    if name in ('rmappo', 'ea_mappo', 'stea_mappo'):
        extra[f'{name}_impl_version'] = 1
        for label, module in (('actor', trainer.actor), ('critic', trainer.critic)):
            extra[f'{label}_parameter_count'] = sum(p.numel() for p in module.parameters())
        extra['total_parameter_count'] = extra['actor_parameter_count']+extra['critic_parameter_count']
    checkpoint = tmp_path / 'policy.pt'; trainer.save(checkpoint, extra)
    original_sha = checkpoint_sha256(checkpoint)
    loaded, state, provenance = load_recording_policy(checkpoint, env, algorithm, 'cuda')
    assert state['algorithm'].lower().replace('-', '_') == name
    output = tmp_path / 'episode'
    result = record(loaded, env, 202, output, provenance)
    trace = read_trace(output / 'episode_trace.npz')
    assert result['episode_steps'] == 3 and trace['red_actions'].shape == (3, n['num_agents'], 3)
    assert checkpoint_sha256(checkpoint) == original_sha
    if name in ('rmappo', 'ea_mappo', 'stea_mappo'):
        evaluation = importlib.import_module(f'algorithm.{name}.evaluation').evaluate(loaded, env, [202])
        assert evaluation['average_return'] == pytest.approx(result['episode_return'], abs=1e-6)
        assert evaluation['red_survivors'] == result['red_survivors']
    # Repeating complete recording must preserve every state/action/event.
    repeated = tmp_path / 'repeat'; again = record(loaded, env, 202, repeated, provenance)
    for key, value in read_trace(repeated / 'episode_trace.npz').items():
        np.testing.assert_array_equal(value, trace[key])
    assert again == result
    # Snapshot contract must reject tampered config before loading.
    altered = dict(algorithm); altered['diagnostic_tamper'] = True
    with pytest.raises(RuntimeError, match='(fingerprint|sha256)'):
        load_recording_policy(checkpoint, env, altered, 'cuda')


def sample_records():
    rows = []
    for seed in range(66000000, 66000005):
        for i, name in enumerate(('first', 'second')):
            rows.append({'algorithm': name, 'episode_seed': seed, 'episode_steps': 200+20*(seed-66000000)+i*10,
                'red_survivors': 8-i*(seed-66000000), 'episode_return': 70-10*i,
                'red_win': True, 'blue_win': False, 'draw': False, 'termination_reason': 'blue_eliminated'})
    return rows


def test_representative_seeds_reproducible_order_and_name_independent():
    rows = sample_records(); expected = select_representative_seeds(rows)
    assert len({expected[f'{mode}_seed'] for mode in ('typical', 'hard', 'fast')}) == 3
    renamed = [dict(row, algorithm={'first': 'Zebra', 'second': 'Alpha'}[row['algorithm']]) for row in reversed(rows)]
    assert select_representative_seeds(renamed) == expected == select_representative_seeds(rows)
    assert expected['typical_seed'] == 66000002
    assert expected['hard_seed'] == 66000004
    assert expected['fast_seed'] == 66000000
    with pytest.raises(ValueError, match='every algorithm'):
        select_representative_seeds(rows[:-1])


def test_mp4_probe_and_stride_fire_event_visible(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env = config('combat_environment_v24'); env['simulation']['max_steps'] = 3
    output = tmp_path / 'episode'; metadata = record(ZeroActor(), env, 202, output)
    trace = read_trace(output / 'episode_trace.npz')
    metadata['events'] = [{'type': 'fire_attempt', 'step': 2, 'start': [0, 0, -3000],
        'end': [1000, 0, -3000], 'hit': True}]
    fig = plt.figure(); ax = fig.add_subplot(projection='3d')
    try:
        scene = CombatScene(ax, trace, metadata); scene.draw(0); scene.draw(3)
        assert len(scene.event_artists) == 1
    finally:
        plt.close(fig)
    movie = render(output / 'episode_trace.npz', output / 'metadata.json', tmp_path / 'video.mp4')
    probe = probe_video(movie)
    assert probe['frame_count'] == 2 and probe['duration_s'] > 0


def test_illegal_attack_detection_and_per_agent_shared_credit(tmp_path):
    env = config('combat_environment_v24'); env['simulation']['max_steps'] = 3
    output = tmp_path / 'episode'; metadata = record(ZeroActor(), env, 202, output)
    metadata['events'] = [
        {'type': 'fire_attempt', 'side': 'red', 'step': 1, 'attacker': 0, 'target': 0, 'hit': True,
         'attacker_state': [0, 0, -3000, 200, 0, 0], 'target_state': [1000, 0, -3000, 250, 0, 0],
         'start': [0, 0, -3000], 'end': [1000, 0, -3000], 'distance': 1000., 'off_boresight': 0., 'target_aspect': 0.},
        {'type': 'attack_kill', 'side': 'red', 'step': 1, 'target': 0, 'attackers': [0, 1]}]
    trace = read_trace(output / 'episode_trace.npz')
    analysis = analyze_episode(trace, metadata, env)
    assert analysis['illegal_red_fire_count'] == 0  # slower attacker is geometrically legal
    assert analysis['per_agent']['red'][0]['fractional_kill_credit'] == .5
    assert sum(r['fractional_kill_credit'] for r in analysis['per_agent']['red']) == 1
    metadata['events'][0]['target_state'][5] = np.pi
    assert analyze_episode(trace, metadata, env)['illegal_red_fire_count'] == 1
