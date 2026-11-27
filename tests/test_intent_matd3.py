"""Tests for IntentMATD3: enemy trajectory intent prediction and single-layer prediction-guided focus."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from algorithm.intent_matd3 import FutureTargetDelayQueue, IntentActor, IntentMATD3, annotate_future_enemy_targets
from algorithm.intent_matd3_config import IntentMATD3Config
from algorithm.matd3 import MATD3
from algorithm.matd3_config import NormalizedMATD3Config
from env import make_environment
from experiment.cli import main as cli_main
from experiment.interception import evaluate_policy, load_run, make_policy, read_config, train, validate_warm_start
from tools.replay import collect, validate_trace

ENV_5V3 = 'configs/interception_radial_5v3_jia_heading_nearest_d08_h003_time0015.yaml'
ENV_5V5 = 'configs/interception_radial_5v5_jia_heading_nearest_d08_h003_time0015.yaml'
ENV_13 = 'configs/interception_radial_5v5_jia_heading_time01.yaml'
ALG_INTENT = 'configs/intent_matd3_jia_heading_interval10.yaml'


def test_intent_matd3_config_contracts_and_historical_preservation():
    for env_path, expected_obs, expected_state, expected_att in (
        (ENV_5V3, 41, 50, 3),
        (ENV_5V5, 51, 60, 5),
    ):
        cfg, params, rollout = read_config(env_path, ALG_INTENT)
        assert isinstance(params, IntentMATD3Config)
        assert rollout == 128
        assert params.prediction_horizons == [5, 10, 20]
        assert params.intent_embed_dim == 64
        assert params.prediction_loss_weight == 1.0
        # Base hyperparams stay identical to MATD3 baseline for single-factor comparison.
        assert (params.actor_learning_rate, params.critic_learning_rate) == (0.0003, 0.001)
        assert (params.exploration_std_speed, params.exploration_std_turn) == (0.3, 0.3)
        assert params.policy_delay == 2

        policy = make_policy('intent_matd3', cfg, params, 'cpu', 7)
        assert isinstance(policy, IntentMATD3)
        assert isinstance(policy.actors[0], IntentActor)
        assert policy.actors[0].observation_dim == expected_obs
        assert policy.actors[0].n_attackers == expected_att
        assert policy.critics_1[0].net[0].in_features == expected_state + 10
        assert policy.critics_2[0].net[0].in_features == expected_state + 10

    with pytest.raises(ValueError, match='Interception/14 and Interception/15'):
        read_config(ENV_13, ALG_INTENT)
    with pytest.raises(ValueError, match='strictly increasing'):
        IntentMATD3Config(prediction_horizons=[10, 5])
    with pytest.raises(ValueError, match='positive integers'):
        IntentMATD3Config(prediction_horizons=[0, 5])

    # Historical environment digests remain byte-for-byte untouched.
    assert make_environment(read_config(ENV_13)[0]).contract['implementation_sha256'] == (
        '46230752bf44cca962b0061c62f56c8bd70ae0e13cd21ced226b2a6ee0dfe21d'
    )
    assert make_environment(read_config(ENV_5V5)[0]).contract['implementation_sha256'] == (
        'e71343f91623db71ad9172d753adae4391a4e4ac8ca3c2f7529b7839b3c065a0'
    )
    assert make_environment(read_config(ENV_5V3)[0]).contract['implementation_sha256'] == (
        'c8e946c58aeb55d9ff8b7157a027deb2f0234eb7850e55db2814d1c6a60666d1'
    )


def test_future_enemy_target_annotation_exactness_and_masking():
    cfg, _, _ = read_config(ENV_5V3)
    env = make_environment(cfg)
    obs, info = env.reset(seed=7)
    def_positions_hist = [env.defender_positions.copy()]
    att_positions_hist = [env.attacker_positions.copy()]
    att_alive_hist = [env.attacker_alive.copy()]
    entries = []

    # Step 8 steps; force attacker 1 capture at step 4 so horizons reaching >=4 mask attacker 1 out.
    for step_idx in range(8):
        if step_idx == 3:
            env.defender_positions[0] = env.attacker_positions[1].copy()
        state = info['central_state'].copy()
        mask = info['active_masks'].copy()
        next_obs, ind, term, trunc, next_info = env.step(np.zeros((5, 2), dtype=np.float32))
        entries.append((dict(
            observations=obs.copy(),
            states=state,
            next_states=next_info['central_state'].copy(),
            active_masks=mask,
        ), None))
        obs, info = next_obs, next_info
        def_positions_hist.append(env.defender_positions.copy())
        att_positions_hist.append(env.attacker_positions.copy())
        att_alive_hist.append(env.attacker_alive.copy())

    horizons = [1, 3, 5]
    annotate_future_enemy_targets(entries, n_attackers=3, horizons=horizons)

    # At t=0, h=1 (t_fut=1): all 3 attackers alive; compare exact normalized relative position to physical coords.
    item0 = entries[0][0]
    for i in range(5):
        for j in range(3):
            expected = (att_positions_hist[1][j] - def_positions_hist[0][i]) / (2.0 * cfg.arena_half_extent)
            assert item0['future_enemy_valid'][i, j, 0] == 1.0
            np.testing.assert_allclose(item0['future_enemy_positions'][i, j, 0], expected, atol=1e-5)

    # At t=0, h=5 (t_fut=5): attacker 1 was captured at step 4 -> must be masked 0, attackers 0 & 2 remain valid!
    assert np.all(item0['future_enemy_valid'][:, 1, 2] == 0.0)
    np.testing.assert_array_equal(item0['future_enemy_positions'][:, 1, 2], 0.0)
    assert np.all(item0['future_enemy_valid'][:, 0, 2] == 1.0)
    assert np.all(item0['future_enemy_valid'][:, 2, 2] == 1.0)

    # At t=5, h=5 (t_fut=10 > 8 steps): out of episode horizon -> must be masked 0.
    assert np.all(entries[5][0]['future_enemy_valid'][:, :, 2] == 0.0)
    # Defender 0 retired at step 4 -> at t=5, all its targets are masked 0.
    assert np.all(entries[5][0]['future_enemy_valid'][0] == 0.0)

    # Rolling delay queue must emit step-by-step after max_horizon steps, match full-episode labels bit-for-bit,
    # and never leak states across episode boundaries.
    queue = FutureTargetDelayQueue(n_attackers=3, horizons=horizons)
    streamed = []
    for t, (item, _) in enumerate(entries):
        raw_copy = {k: v.copy() for k, v in item.items() if not k.startswith('future_enemy_')}
        ready = queue.push(raw_copy, None, done=(t == len(entries) - 1))
        if t < max(horizons) - 1:
            assert ready == []
        elif t < len(entries) - 1:
            assert len(ready) == 1
        streamed.extend(ready)
    assert len(queue) == 0 and len(streamed) == len(entries)
    for (ref_item, _), (q_item, _) in zip(entries, streamed):
        np.testing.assert_array_equal(q_item['future_enemy_positions'], ref_item['future_enemy_positions'])
        np.testing.assert_array_equal(q_item['future_enemy_valid'], ref_item['future_enemy_valid'])

    # Second short episode pushed into the same queue must never see Episode 1's states.
    raw0 = {k: v.copy() for k, v in entries[0][0].items() if not k.startswith('future_enemy_')}
    short_ready = queue.push(raw0, None, done=True)
    assert len(short_ready) == 1 and len(queue) == 0
    assert np.all(short_ready[0][0]['future_enemy_valid'][:, :, 0] == 1.0)  # h=1 uses next_states of that 1 step
    assert np.all(short_ready[0][0]['future_enemy_valid'][:, :, 1:] == 0.0)  # h=3,5 exceed 1-step episode


def test_intent_matd3_optimizer_decoupling_and_target_soft_update():
    cfg, _, _ = read_config(ENV_5V3, ALG_INTENT)
    env = make_environment(cfg)
    obs, info = env.reset(seed=7)
    next_obs, ind, term, trunc, next_info = env.step(np.zeros((5, 2), dtype=np.float32))
    batch = dict(
        observations=np.repeat(obs[None], 4, axis=0),
        next_observations=np.repeat(next_obs[None], 4, axis=0),
        states=np.repeat(info['central_state'][None], 4, axis=0),
        next_states=np.repeat(next_info['central_state'][None], 4, axis=0),
        actions=np.zeros((4, 5, 2), dtype=np.float32),
        rewards=np.repeat(ind[None], 4, axis=0),
        dones=np.zeros((4,), dtype=np.float32),
        active_masks=np.ones((4, 5), dtype=np.float32),
        next_active_masks=np.ones((4, 5), dtype=np.float32),
        future_enemy_positions=np.full((4, 5, 3, 2, 2), 0.25, dtype=np.float32),
        future_enemy_valid=np.ones((4, 5, 3, 2), dtype=np.float32),
    )

    # With prediction_loss_weight == 0.0, predictor parameters must remain strictly frozen across actor updates.
    params_zero = IntentMATD3Config(
        hidden_dim=16, intent_embed_dim=16, prediction_horizons=[1, 2], prediction_loss_weight=0.0,
        batch_size=4, buffer_capacity=16, learning_starts=4, policy_delay=1, exploration_std_mps2=0.0,
    )
    p0 = make_policy('intent_matd3', cfg, params_zero, 'cpu', 7)
    pred_before = [p.detach().clone() for p in p0.actors[0].predictor.parameters()]
    pol_before = [p.detach().clone() for p in p0.actors[0].policy_parameters()]
    m0 = p0.update(batch)
    assert m0['pred_valid_h1'] == pytest.approx(1.0) and m0['pred_valid_h2'] == pytest.approx(1.0)
    for before, after in zip(pred_before, p0.actors[0].predictor.parameters()):
        torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)
    assert any(not torch.equal(b, a) for b, a in zip(pol_before, p0.actors[0].policy_parameters()))

    # With prediction_loss_weight == 1.0, predictor weights update and target_actors soft-update as well.
    params_one = IntentMATD3Config(
        hidden_dim=16, intent_embed_dim=16, prediction_horizons=[1, 2], prediction_loss_weight=1.0,
        batch_size=4, buffer_capacity=16, learning_starts=4, policy_delay=1, exploration_std_mps2=0.0,
    )
    p1 = make_policy('intent_matd3', cfg, params_one, 'cpu', 7)
    pred_before_1 = [p.detach().clone() for p in p1.actors[0].predictor.parameters()]
    tgt_pred_before_1 = [p.detach().clone() for p in p1.target_actors[0].predictor.parameters()]
    p1.update(batch)
    assert any(not torch.equal(b, a) for b, a in zip(pred_before_1, p1.actors[0].predictor.parameters()))
    assert any(not torch.equal(b, a) for b, a in zip(tgt_pred_before_1, p1.target_actors[0].predictor.parameters()))


def test_intent_actor_null_fallback_and_strict_gradient_decoupling():
    torch.manual_seed(7)
    actor = IntentActor(observation_dim=41, n_attackers=3, hidden_dim=32, intent_embed_dim=16, n_horizons=3)

    # All-zero observation (0 visible enemies / retired agent) must produce finite outputs and 100% weight on null slot 0.
    zero_obs = torch.zeros((4, 41), dtype=torch.float32)
    act_z, pred_z, w_z = actor.forward_with_intent(zero_obs)
    assert torch.isfinite(act_z).all() and torch.isfinite(pred_z).all() and torch.isfinite(w_z).all()
    np.testing.assert_allclose(w_z[:, 0].detach().numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(w_z[:, 1:].detach().numpy(), 0.0, atol=1e-6)

    # Gradient isolation 1: policy action loss must NOT send any gradient into self.predictor.
    obs = torch.randn((8, 41), dtype=torch.float32)
    obs[:, 26 + 4::5] = 1.0  # all 3 enemies valid
    actor.zero_grad()
    act, pred, w = actor.forward_with_intent(obs)
    assert w.shape == (8, 4) and np.allclose(w.sum(dim=-1).detach().numpy(), 1.0, atol=1e-6)
    act.square().mean().backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in actor.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in actor.policy_parameters())

    # Gradient isolation 2: trajectory prediction loss must NOT send any gradient into policy_parameters().
    actor.zero_grad()
    _, pred2, _ = actor.forward_with_intent(obs)
    pred2.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in actor.predictor.parameters())
    assert all(p.grad is None or torch.all(p.grad == 0) for p in actor.policy_parameters())


@pytest.mark.parametrize('env_path,expected_ver,expected_obs,expected_state,expected_att', [
    (ENV_5V3, 'interception_15', 41, 50, 3),
    (ENV_5V5, 'interception_14', 51, 60, 5),
])
def test_intent_matd3_preflight_train_checkpoint_eval_replay_and_isolation(
    tmp_path, env_path, expected_ver, expected_obs, expected_state, expected_att
):
    cfg, _, _ = read_config(env_path, ALG_INTENT)
    out = tmp_path / f'intent_{expected_ver}'
    preflight = cli_main([
        'preflight',
        '--env-config', env_path,
        '--algorithm-config', ALG_INTENT,
        '--algorithm', 'intent_matd3',
        '--attack-mode', 'radial_straight',
        '--total-sampled-steps', '16',
        '--num-envs', '1',
        '--seed', '7',
        '--device', 'cpu',
        '--output-dir', str(out),
    ])
    assert preflight['status'] == 'supported'
    assert preflight['algorithm'] == 'intent_matd3'
    assert preflight['environment_contract']['version'] == expected_ver

    params = IntentMATD3Config(
        hidden_dim=16,
        intent_embed_dim=16,
        prediction_horizons=[1, 2],
        prediction_loss_weight=1.0,
        batch_size=2,
        buffer_capacity=32,
        learning_starts=4,
        update_interval_steps=2,
        updates_per_interval=1,
        exploration_std_mps2=0.0,
    )
    train(
        replace(cfg, episode_steps=4),
        params,
        4,
        'intent_matd3',
        steps=16,
        num_envs=1,
        seed=7,
        device='cpu',
        attack_mode='radial_straight',
        output=out,
        eval_interval=8,
        eval_episodes=1,
        log_every_steps=8,
        update_log_interval=2,
    )
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['updates'] == 7
    upd_rows = [json.loads(l) for l in (out / 'updates_summary.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    assert any(r['metrics'].get('pred_loss', {}).get('count', 0) > 0 for r in upd_rows)
    assert any(r['metrics'].get('pred_valid_h1', {}).get('count', 0) > 0 for r in upd_rows)
    assert any(r['metrics'].get('pred_valid_h2', {}).get('count', 0) > 0 for r in upd_rows)

    restored, policy, name, ckpt_path, settings = load_run(out, 'Best', 'cpu')
    assert name == 'intent_matd3'
    assert settings['environment_contract']['version'] == expected_ver
    assert settings['environment_contract']['observation_dim'] == expected_obs
    assert settings['environment_contract']['state_dim'] == expected_state

    # Checkpoint must never cross-load into MATD3.
    matd3_policy = MATD3(
        NormalizedMATD3Config(hidden_dim=16, batch_size=2, buffer_capacity=32, learning_starts=4, exploration_std_mps2=0.0),
        'cpu',
        7,
        environment_config=settings['environment'],
        environment_contract=settings['environment_contract'],
    )
    with pytest.raises(ValueError, match='Incompatible MATD3'):
        matd3_policy.load(ckpt_path, load_optimizers=False)

    # Warm-start validation must succeed with matching config/params (verifies list serialization parity).
    ws_model, _ = validate_warm_start(out, replace(cfg, episode_steps=4), params, 'intent_matd3', 'radial_straight', 'cpu')
    assert ws_model.exists()

    # Check deterministic inference + intent output shapes + eval + replay.
    env = make_environment(restored)
    obs, info = env.reset(seed=900001)
    actions, pred, weights = policy.policy_actions(obs[None], info['active_masks'][None], deterministic=True, return_intent=True)
    assert actions.shape == (1, 5, 2)
    assert pred.shape == (1, 5, expected_att, 2, 2)
    assert weights.shape == (1, 5, 1 + expected_att)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    rows = evaluate_policy(restored, policy, 'radial_straight', 1, 900001)
    assert np.isclose(sum(rows[0]['reward_components'].values()), rows[0]['team_return'], atol=1e-6)
    trace = validate_trace(collect(restored, policy, 'radial_straight', 900001))
    assert trace['schema'] == 'InterceptionReplay/3'
