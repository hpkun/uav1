"""Tests for the isolated MATD3 (twin-critic + target smoothing + delayed policy) learner."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from algorithm.maddpg import BoxActor, DiskActor
from algorithm.maddpg_config import MADDPGConfig, NormalizedMADDPGConfig
from algorithm.matd3 import MATD3
from algorithm.matd3_config import MATD3Config, NormalizedMATD3Config
from env import make_environment
from experiment.cli import main as cli_main
from experiment.interception import (
    evaluate_policy,
    load_run,
    make_policy,
    read_config,
    read_update_log,
    read_update_summary,
    train,
    validate_warm_start,
)
from tools.replay import collect, validate_trace

JIA_ENV = 'configs/interception_radial_5v5_jia_heading_time01.yaml'
JIA_MATD3 = 'configs/matd3_jia_heading_interval10.yaml'
JIA_MADDPG = 'configs/maddpg_jia_heading_interval10.yaml'
DISK_ENV = 'configs/interception_assignment.yaml'
DISK_MATD3 = 'configs/matd3_interception.yaml'


def _synthetic_batch(obs_dim, state_dim, batch_size=4, *, normalized=False):
    rng = np.random.default_rng(123)
    actions = rng.uniform(-0.8, 0.8, size=(batch_size, 5, 2)).astype(np.float32)
    if not normalized:
        actions *= 1.0  # Well inside radius-2 disk.
    active = np.ones((batch_size, 5), dtype=np.float32)
    active[:, 4] = 0.0
    actions[:, 4, :] = 0.0
    next_active = active.copy()
    next_active[0, 3] = 0.0
    return dict(
        observations=rng.normal(0, 0.5, size=(batch_size, 5, obs_dim)).astype(np.float32),
        next_observations=rng.normal(0, 0.5, size=(batch_size, 5, obs_dim)).astype(np.float32),
        states=rng.normal(0, 0.5, size=(batch_size, state_dim)).astype(np.float32),
        next_states=rng.normal(0, 0.5, size=(batch_size, state_dim)).astype(np.float32),
        actions=actions,
        rewards=rng.normal(0, 1.0, size=(batch_size, 5)).astype(np.float32),
        dones=np.array([0.0, 1.0] + [0.0] * (batch_size - 2), dtype=np.float32),
        active_masks=active,
        next_active_masks=next_active,
    )


def test_matd3_config_validation():
    assert MATD3Config().policy_delay == 2
    assert NormalizedMATD3Config().target_noise_clip_speed == 0.5
    with pytest.raises(ValueError, match='policy_delay'):
        MATD3Config(policy_delay=0)
    with pytest.raises(ValueError, match='policy_delay'):
        MATD3Config(policy_delay=True)
    with pytest.raises(ValueError, match='target_noise_std_mps2'):
        MATD3Config(target_noise_std_mps2=-0.1)
    with pytest.raises(ValueError, match='target_noise_clip_turn'):
        NormalizedMATD3Config(target_noise_clip_turn=-0.1)


def test_twin_critics_min_target_delayed_policy_and_smoothing_bounds():
    config, parameters, _ = read_config(JIA_ENV, JIA_MATD3)
    params = replace(
        parameters,
        hidden_dim=16,
        batch_size=4,
        buffer_capacity=16,
        learning_starts=4,
        policy_delay=2,
        target_noise_std_speed=0.0,
        target_noise_std_turn=0.0,
    )
    policy = make_policy('matd3', config, params, 'cpu', 7)
    assert isinstance(policy, MATD3)
    assert isinstance(policy.actors[0], BoxActor)

    # Twin critics start with distinct weights.
    assert any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(policy.critics_1.parameters(), policy.critics_2.parameters())
    )

    batch = _synthetic_batch(26, 85, batch_size=4, normalized=True)
    # Manually compute clipped double-Q target with zero smoothing noise.
    with torch.no_grad():
        b = policy._batch(batch)
        next_act = policy._joint(b['next_observations'], b['next_active_masks'], target=True)
        nq1 = torch.stack([c(b['next_states'], next_act) for c in policy.target_critics_1], 1)
        nq2 = torch.stack([c(b['next_states'], next_act) for c in policy.target_critics_2], 1)
        expected_targets = b['rewards'] + params.gamma * (1.0 - b['dones'])[:, None] * b['next_active_masks'] * torch.minimum(nq1, nq2)
        expected_target_mean = float(expected_targets[b['active_masks'].bool()].mean())

    actor_before = [p.detach().clone() for p in policy.actors.parameters()]
    target_actor_before = [p.detach().clone() for p in policy.target_actors.parameters()]
    target_c1_before = [p.detach().clone() for p in policy.target_critics_1.parameters()]
    c1_before = [p.detach().clone() for p in policy.critics_1.parameters()]
    c2_before = [p.detach().clone() for p in policy.critics_2.parameters()]

    # Update 1 (odd): critics step, actors & targets stay frozen.
    m1 = policy.update(batch)
    assert m1['critic_updates'] == 1
    assert m1['actor_updates'] == 0 and m1['target_updates'] == 0
    assert m1['actor_loss'] is None and m1['actor_grad_norm'] is None and m1['actor_clip_rate'] is None
    assert m1['target_q_mean'] == pytest.approx(expected_target_mean, rel=1e-6, abs=1e-6)
    assert m1['active_critic_count'] == 4  # Agent 4 is masked out.
    assert any(not torch.allclose(a, b_) for a, b_ in zip(policy.critics_1.parameters(), c1_before))
    assert any(not torch.allclose(a, b_) for a, b_ in zip(policy.critics_2.parameters(), c2_before))
    for a, b_ in zip(policy.actors.parameters(), actor_before):
        torch.testing.assert_close(a, b_, rtol=0, atol=0)
    for a, b_ in zip(policy.target_actors.parameters(), target_actor_before):
        torch.testing.assert_close(a, b_, rtol=0, atol=0)
    for a, b_ in zip(policy.target_critics_1.parameters(), target_c1_before):
        torch.testing.assert_close(a, b_, rtol=0, atol=0)

    # Update 2 (even): critics, actors and all target networks step.
    m2 = policy.update(batch)
    assert m2['critic_updates'] == 2
    assert m2['actor_updates'] == 1 and m2['target_updates'] == 1
    assert m2['actor_loss'] is not None and np.isfinite(m2['actor_loss'])
    assert m2['actor_optimizer_steps'] == 4
    assert any(not torch.allclose(a, b_) for a, b_ in zip(policy.actors.parameters(), actor_before))
    assert any(not torch.allclose(a, b_) for a, b_ in zip(policy.target_actors.parameters(), target_actor_before))
    assert any(not torch.allclose(a, b_) for a, b_ in zip(policy.target_critics_1.parameters(), target_c1_before))

    # Target smoothing respects box bounds and inactive zero masks.
    noisy_policy = make_policy('matd3', config, replace(params, target_noise_std_speed=2.0, target_noise_std_turn=2.0), 'cpu', 7)
    b_t = noisy_policy._batch(batch)
    smoothed = noisy_policy._smoothed_target_actions(b_t['next_observations'], b_t['next_active_masks'])
    assert float(smoothed.abs().max()) <= 1.0 + 1e-6
    assert torch.all(smoothed[b_t['next_active_masks'] == 0] == 0)

    # Disk-action smoothing respects physical acceleration disk radius and masks.
    disk_cfg, disk_params, _ = read_config(DISK_ENV, DISK_MATD3)
    disk_policy = make_policy(
        'matd3',
        disk_cfg,
        replace(disk_params, hidden_dim=16, target_noise_std_mps2=5.0, target_noise_clip_mps2=3.0),
        'cpu',
        7,
    )
    assert isinstance(disk_policy.actors[0], DiskActor)
    disk_batch = disk_policy._batch(_synthetic_batch(54, 75, batch_size=4, normalized=False))
    disk_smoothed = disk_policy._smoothed_target_actions(disk_batch['next_observations'], disk_batch['next_active_masks'])
    assert float(disk_smoothed.norm(dim=-1).max()) <= disk_cfg.defender_acceleration_max + 1e-5
    assert torch.all(disk_smoothed[disk_batch['next_active_masks'] == 0] == 0)


def test_matd3_and_maddpg_strict_version_isolation(tmp_path):
    config, matd3_params, _ = read_config(JIA_ENV, JIA_MATD3)
    _, maddpg_params, _ = read_config(JIA_ENV, JIA_MADDPG)
    short_cfg = replace(config, episode_steps=2)
    small_matd3 = replace(matd3_params, hidden_dim=16, batch_size=2, buffer_capacity=8, learning_starts=2, update_interval_steps=2)
    small_maddpg = replace(maddpg_params, hidden_dim=16, batch_size=2, buffer_capacity=8, learning_starts=2, update_interval_steps=2)

    matd3_dir = tmp_path / 'matd3_run'
    maddpg_dir = tmp_path / 'maddpg_run'
    train(short_cfg, small_matd3, 2, 'matd3', steps=4, num_envs=1, seed=7, device='cpu',
          attack_mode='radial_straight', output=matd3_dir, eval_interval=2, eval_episodes=1, log_every_steps=4)
    train(short_cfg, small_maddpg, 2, 'maddpg', steps=4, num_envs=1, seed=7, device='cpu',
          attack_mode='radial_straight', output=maddpg_dir, eval_interval=2, eval_episodes=1, log_every_steps=4)

    # Checkpoints cannot cross-load between MADDPG and MATD3.
    matd3_learner = make_policy('matd3', short_cfg, small_matd3, 'cpu', 11)
    maddpg_learner = make_policy('maddpg', short_cfg, small_maddpg, 'cpu', 11)
    with pytest.raises(ValueError, match='Incompatible MATD3'):
        matd3_learner.load(maddpg_dir / 'checkpoints' / 'last.pt', load_optimizers=False)
    with pytest.raises(ValueError, match='Incompatible MADDPG'):
        maddpg_learner.load(matd3_dir / 'checkpoints' / 'last.pt', load_optimizers=False)

    # Full optimizer roundtrip succeeds for MATD3 and preserves counters.
    matd3_learner.load(matd3_dir / 'checkpoints' / 'last.pt', load_optimizers=True)
    assert matd3_learner.critic_update_count == 2
    assert matd3_learner.actor_update_count == 1

    # Warm-start rejects cross-algorithm source runs and accepts matching MATD3 runs.
    with pytest.raises(ValueError, match='does not match'):
        validate_warm_start(maddpg_dir, short_cfg, small_matd3, 'matd3', 'radial_straight', 'cpu')
    with pytest.raises(ValueError, match='does not match'):
        validate_warm_start(matd3_dir, short_cfg, small_maddpg, 'maddpg', 'radial_straight', 'cpu')
    model_path, _ = validate_warm_start(matd3_dir, short_cfg, small_matd3, 'matd3', 'radial_straight', 'cpu')
    assert model_path == (matd3_dir / 'checkpoints' / 'last.pt').resolve()

    # CLI rejects mismatched --algorithm vs YAML algorithm.
    with pytest.raises(ValueError, match='Algorithm config mismatch'):
        cli_main([
            'preflight',
            '--env-config', JIA_ENV,
            '--algorithm-config', JIA_MADDPG,
            '--algorithm', 'matd3',
            '--attack-mode', 'radial_straight',
            '--total-sampled-steps', '4',
            '--output-dir', str(tmp_path / 'bad_plan'),
        ])


def test_matd3_end_to_end_train_eval_logs_checkpoints_and_replay(tmp_path):
    plan = cli_main([
        'preflight',
        '--env-config', JIA_ENV,
        '--algorithm-config', JIA_MATD3,
        '--algorithm', 'matd3',
        '--attack-mode', 'radial_straight',
        '--total-sampled-steps', '10',
        '--num-envs', '1',
        '--output-dir', str(tmp_path / 'plan_ok'),
    ])
    assert plan['algorithm'] == 'matd3' and plan['save_eval_checkpoints'] is True

    config, parameters, _ = read_config(JIA_ENV, JIA_MATD3)
    small = replace(parameters, hidden_dim=16, batch_size=2, buffer_capacity=16,
                    learning_starts=2, update_interval_steps=2, updates_per_interval=1, policy_delay=2)
    output = tmp_path / 'matd3_jia'
    summary = train(
        replace(config, episode_steps=2),
        small,
        2,
        'matd3',
        steps=10,
        num_envs=1,
        seed=7,
        device='cpu',
        attack_mode='radial_straight',
        output=output,
        eval_interval=5,
        eval_episodes=1,
        log_every_steps=10,
        update_log_interval=4,
    )
    assert summary['updates'] == 5
    raw_updates = read_update_log(output)
    summary_updates = read_update_summary(output)
    assert len(raw_updates) == 5
    assert [row['count'] for row in summary_updates] == [4, 1]
    # policy_delay=2: updates 2 and 4 update actors in the first window of 4 -> valid actor_loss count == 2.
    assert summary_updates[0]['metrics']['actor_loss']['count'] == 2
    assert summary_updates[0]['metrics']['critic_loss']['count'] == 4
    assert summary_updates[0]['counters']['critic_updates'] == 4
    assert summary_updates[0]['counters']['actor_updates'] == 2
    # Final 1-update window (update 5, odd) has actor_loss=None -> count == 0, valid JSON.
    assert summary_updates[1]['metrics']['actor_loss']['count'] == 0

    # Checkpoints (Best, Last, step_5, step_10) load cleanly and replay/evaluate work.
    restored, policy, name, _, settings = load_run(output, 'step_5', 'cpu')
    assert name == 'matd3' and settings['algorithm'] == 'matd3'
    load_run(output, 'Best', 'cpu')
    load_run(output, 'Last', 'cpu')
    rows = evaluate_policy(restored, policy, 'radial_straight', 1, 900001)
    assert len(rows) == 1
    trace = validate_trace(collect(restored, policy, 'radial_straight', 900001))
    assert trace['schema'] == 'InterceptionReplay/3'

    # Also verify standard disk-action environment works cleanly with MATD3.
    disk_cfg, disk_params, _ = read_config(DISK_ENV, DISK_MATD3)
    disk_out = tmp_path / 'matd3_disk'
    train(
        replace(disk_cfg, episode_steps=2),
        replace(disk_params, hidden_dim=16, batch_size=2, buffer_capacity=8,
                learning_starts=2, update_interval_steps=2, updates_per_interval=1),
        2,
        'matd3',
        steps=4,
        num_envs=2,
        seed=7,
        device='cpu',
        attack_mode='direct_rush',
        output=disk_out,
        eval_interval=4,
        eval_episodes=1,
        log_every_steps=4,
    )
    _, disk_policy, disk_name, _, _ = load_run(disk_out, 'Last', 'cpu')
    assert disk_name == 'matd3' and isinstance(disk_policy, MATD3)
