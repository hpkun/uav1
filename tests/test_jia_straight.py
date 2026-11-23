"""Isolated straight intruder routes for the ideal-vision experiment."""

from dataclasses import replace

import numpy as np

from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv
from experiment.interception import load_run, read_config, train
from tools.replay import collect, validate_trace


CONFIG = 'configs/interception_ideal_jia_straight_heading_approach005.yaml'


def test_straight_lanes_keep_the_old_ideal_contract_unchanged():
    new, _, _ = read_config(CONFIG)
    old, _, _ = read_config('configs/interception_ideal_global_initial_heading_approach005.yaml')
    assert new == replace(old, attack_mode='jia_straight')
    assert InterceptionEnv(new).contract['version'] == 'interception_6'
    assert InterceptionEnv(old).contract['version'] == 'interception_5'


def test_intruders_follow_separated_fixed_headings_and_breach():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=900001)
    starting = env.positions[1].copy()
    directions = env.attacker_course_directions.copy()
    assert np.ptp(starting[:, 0]) > 90
    assert min(np.linalg.norm(starting[i]-starting[j]) for i in range(5) for j in range(i+1, 5)) > 8
    np.testing.assert_allclose(np.linalg.norm(env.velocities[1], axis=1), 2)
    env.defender_alive[:] = False
    for _ in range(150):
        env.step(np.zeros((5, 2)))
    displacement = env.positions[1]-starting
    cross = displacement[:,0]*directions[:,1]-displacement[:,1]*directions[:,0]
    np.testing.assert_allclose(cross, 0, atol=1e-9)
    assert np.ptp(env.positions[1,:,0]) > 30
    np.testing.assert_allclose(np.linalg.norm(env.velocities[1], axis=1), 5, atol=1e-9)
    while not env.finished:
        env.step(np.zeros((5, 2)))
    assert env.penetrated_count == 5


def test_jia_mode_short_training_checkpoint_and_replay(tmp_path):
    config, _, _ = read_config(CONFIG)
    short = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / 'jia'
    train(short, parameters, 2, 'maddpg', steps=4, num_envs=2, seed=7,
          device='cpu', attack_mode='jia_straight', output=output,
          eval_interval=4, eval_episodes=1, log_every_steps=4)
    restored, policy, _, _, settings = load_run(output, 'Last', 'cpu')
    assert settings['attack_mode'] == restored.attack_mode == 'jia_straight'
    assert validate_trace(collect(restored, policy, 'jia_straight', 900001))['summary']['steps'] == 2
