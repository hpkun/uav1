from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.mappo.trainer import compute_gae
from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import (
    MilestoneAwareRetentionCreditModule,
    compute_local_gae,
    continuation_coefficients,
    successful_wave_from_transition,
    tempered_wave_weights,
)
from algorithm.train_modular_mappo import load_config

ROOT = Path(__file__).resolve().parents[1]


def tensors(transition=False):
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[.2]], [[.4]], [[.6]]])
    next_values = torch.tensor([[[.4]], [[.6]], [[.8]]])
    dones = torch.zeros(3, 1)
    alive = torch.ones(3, 1, 1)
    transitions = torch.tensor([[0.0], [float(transition)], [0.0]])
    return rewards, values, next_values, dones, alive, alive.clone(), transitions


def marc_config(hidden=16):
    config = load_config(ROOT / "configs/dev_marc_mappo_v1_1m.yaml")
    config = deepcopy(config)
    config["network"]["actor_hidden_layers"] = [hidden, hidden]
    config["network"]["critic_hidden_layers"] = [hidden, hidden]
    config["training"].update({"ppo_epochs": 1, "minibatch_size": 8})
    return config


def test_local_gae_equals_global_without_transition():
    r, v, nv, d, a, na, tr = tensors(False)
    global_adv, _ = compute_gae(r, v, nv, d, a, na, .99, .95)
    local_adv, _ = compute_local_gae(r, v, nv, d, a, na, tr, .99, .95)
    assert torch.allclose(local_adv, global_adv)


def test_local_trace_stops_at_transition():
    r, v, nv, d, a, na, tr = tensors(True)
    local, _ = compute_local_gae(r, v, nv, d, a, na, tr, .99, .95)
    delta = r[1] + .99 * nv[1] - v[1]
    assert torch.allclose(local[1], delta)


def test_transition_td_still_bootstraps_next_value():
    zeros = torch.zeros(1, 1, 1); next_values = torch.full_like(zeros, 2.0)
    local, _ = compute_local_gae(zeros, zeros, next_values, torch.zeros(1, 1),
                                 torch.ones_like(zeros), torch.ones_like(zeros),
                                 torch.ones(1, 1), .9, .95)
    assert local.item() == pytest.approx(1.8)


def test_continuation_advantage_identity():
    r, v, nv, d, a, na, tr = tensors(True)
    global_adv, _ = compute_gae(r, v, nv, d, a, na, .99, .95)
    local, _ = compute_local_gae(r, v, nv, d, a, na, tr, .99, .95)
    assert torch.allclose(local + (global_adv - local), global_adv)


def test_eta_values_are_one_third_one_half_one():
    eta = continuation_coefficients(torch.tensor([1, 2, 3]), 3, 1.0)
    assert eta.tolist() == pytest.approx([1 / 3, 1 / 2, 1])


def test_actor_advantage_formula():
    local = torch.tensor([3.0]); continuation = torch.tensor([6.0]); eta = torch.tensor([.5])
    assert (local + eta * continuation).item() == 6.0


def test_rare_wave_receives_larger_tempered_weight():
    waves = torch.tensor([[1], [1], [1], [3]])
    alive = torch.ones(4, 1, 4)
    _, metrics = tempered_wave_weights(waves, alive)
    assert metrics["marc_weight_wave3"] > metrics["marc_weight_wave1"]


def test_tempered_weights_are_bounded_and_alive_mean_one():
    waves = torch.tensor([[1], [1], [1], [1], [2], [3]])
    alive = torch.ones(6, 1, 4); alive[-1, 0, 2:] = 0
    weights, metrics = tempered_wave_weights(waves, alive)
    expanded = weights.unsqueeze(-1).expand_as(alive)
    assert expanded[alive > 0].min() >= .5 and expanded[alive > 0].max() <= 2
    assert metrics["marc_effective_weight_mean"] == pytest.approx(1.0, abs=1e-7)


def test_single_present_wave_has_unit_weight():
    weights, metrics = tempered_wave_weights(torch.ones(3, 2, dtype=torch.long), torch.ones(3, 2, 4))
    assert torch.equal(weights, torch.ones_like(weights))
    assert metrics["marc_effective_weight_mean"] == 1.0


@pytest.mark.parametrize("wave", [1, 2])
def test_wave_transition_is_success_for_source_wave(wave):
    assert successful_wave_from_transition(wave, spawned_next_wave=True,
                                           episode_done=False, red_success=False) == wave


def test_wave3_only_commits_on_final_red_success():
    assert successful_wave_from_transition(3, spawned_next_wave=False,
                                           episode_done=True, red_success=True) == 3
    assert successful_wave_from_transition(3, spawned_next_wave=False,
                                           episode_done=True, red_success=False) is None


def test_failure_never_counts_as_success():
    assert successful_wave_from_transition(1, spawned_next_wave=False,
                                           episode_done=True, red_success=False) is None


def test_runner_pending_candidates_cross_rollout_and_commit():
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    module = MilestoneAwareRetentionCreditModule({"enabled": True, "retention_stride": 8})
    runner.trainer = type("Trainer", (), {"milestone_aware_retention_credit": module})()
    runner.num_envs = 1; runner.episode_steps = np.array([0])
    runner.marc_pending_episode = [{1: [], 2: [], 3: []}]
    runner._marc_collect_candidates(np.ones((1, 4, 52), np.float32),
                                    np.array([[1, 0, 1, 0]], np.float32), np.array([1]))
    assert len(runner.marc_pending_episode[0][1]) == 2
    done = np.array([False])
    completed = runner._marc_finalize_step(np.array([1]), done,
                                           [{"spawned_next_wave": True, "red_success": False}])
    assert completed[0]["observations"].shape == (2, 52)
    assert runner.marc_pending_episode[0][1] == []


def test_runner_discards_failed_episode_candidates():
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    module = MilestoneAwareRetentionCreditModule({"enabled": True})
    runner.trainer = type("Trainer", (), {"milestone_aware_retention_credit": module})()
    runner.marc_pending_episode = [{1: [np.ones(52)], 2: [], 3: []}]
    completed = runner._marc_finalize_step(np.array([1]), np.array([True]),
                                           [{"spawned_next_wave": False, "red_success": False}])
    assert completed == [] and all(not rows for rows in runner.marc_pending_episode[0].values())


def test_success_ingest_records_detached_references():
    trainer = build_modular_mappo_trainer(marc_config(), "cpu", total_sampled_steps=100)
    module = trainer.milestone_aware_retention_credit
    module.ingest_success_segments([{"wave": 1, "observations": np.ones((4, 52), np.float32)}],
                                   trainer.actor, trainer.device)
    row = module.banks[1][0]
    assert all(isinstance(value, np.ndarray) for value in row.values())
    assert all(value.dtype == np.float16 for value in row.values())


def test_balanced_retention_sampling_uses_equal_counts():
    trainer = build_modular_mappo_trainer(marc_config(), "cpu", total_sampled_steps=100)
    module = trainer.milestone_aware_retention_credit
    for wave in (1, 2):
        module.ingest_success_segments([{"wave": wave, "observations": np.ones((40, 52), np.float32)}],
                                       trainer.actor, trainer.device)
    sampled = module.sample_balanced()
    assert {wave: len(rows) for wave, rows in sampled.items()} == {1: 32, 2: 32}


def test_retention_kl_is_exactly_zero_at_snapshot_policy():
    trainer = build_modular_mappo_trainer(marc_config(), "cpu", total_sampled_steps=100)
    module = trainer.milestone_aware_retention_credit
    module.ingest_success_segments([{"wave": 1, "observations": np.ones((40, 52), np.float32)}],
                                   trainer.actor, trainer.device)
    loss, metrics = module.retention_loss(trainer.actor, trainer.device)
    assert loss.item() == pytest.approx(0.0, abs=2e-6)
    assert metrics["marc_retention_kl_wave1"] == pytest.approx(0.0, abs=2e-4)


def test_retention_kl_increases_after_actor_change():
    trainer = build_modular_mappo_trainer(marc_config(), "cpu", total_sampled_steps=100)
    module = trainer.milestone_aware_retention_credit
    module.ingest_success_segments([{"wave": 1, "observations": np.ones((40, 52), np.float32)}],
                                   trainer.actor, trainer.device)
    with torch.no_grad(): trainer.actor.mean.bias.add_(.2)
    loss, _ = module.retention_loss(trainer.actor, trainer.device)
    assert loss.item() > 0


def test_marc_checkpoint_round_trip_preserves_banks_rng_and_counts():
    config = marc_config(); trainer = build_modular_mappo_trainer(config, "cpu", total_sampled_steps=100)
    module = trainer.milestone_aware_retention_credit
    module.ingest_success_segments([{"wave": 2, "observations": np.ones((40, 52), np.float32)}],
                                   trainer.actor, trainer.device)
    state = module.state_dict(); restored = MilestoneAwareRetentionCreditModule(module.config, 999)
    restored.load_state_dict(state)
    assert len(restored.banks[2]) == 40 and restored.ingested_success_count[2] == 1
    assert restored.rng.bit_generator.state == module.rng.bit_generator.state


def test_strict_checkpoint_rejects_missing_marc_state(tmp_path):
    config = marc_config(); trainer = build_modular_mappo_trainer(config, "cpu", total_sampled_steps=100)
    state = trainer.checkpoint_state(); state["milestone_aware_retention_credit_state"] = None
    path = tmp_path / "bad.pt"; torch.save(state, path)
    with pytest.raises(RuntimeError, match="MARC checkpoint state is missing"):
        trainer.load(path)


def test_config_enables_only_actor_decay_and_marc():
    config = load_config(ROOT / "configs/dev_marc_mappo_v1_1m.yaml")
    enabled = sorted(k for k, v in config["modules"].items() if v.get("enabled", False))
    assert enabled == ["actor_lr_decay", "milestone_aware_retention_credit"]


def test_actor_topology_remains_plain_52d_feedforward():
    trainer = build_modular_mappo_trainer(marc_config(), "cpu", total_sampled_steps=100)
    architecture = checkpoint_architecture(trainer)
    assert architecture["actor_input_dim"] == 52
    assert architecture["actor_context_dim"] == 0 and architecture["actor_gru_hidden_dim"] == 0
    assert architecture["actor_wave_input"] is False


def test_invalid_extra_enabled_module_is_rejected():
    config = marc_config(); config["modules"]["wave_balancing"]["enabled"] = True
    with pytest.raises(ValueError, match="exact enabled modules"):
        build_modular_mappo_trainer(config, "cpu", total_sampled_steps=100)


def test_synthetic_marc_update_is_finite_and_reports_metrics():
    config = marc_config(); trainer = build_modular_mappo_trainer(config, "cpu", total_sampled_steps=100)
    rng = np.random.default_rng(4); T, E, A = 2, 1, 4
    obs = rng.normal(size=(T, E, A, 52)).astype(np.float32)
    raw = rng.normal(size=(T, E, A, 3)).astype(np.float32); actions = np.tanh(raw)
    alive = np.ones((T, E, A), np.float32); waves = np.array([[1], [2]], np.int64)
    with torch.no_grad():
        dist = trainer.actor.distribution(torch.as_tensor(obs))
        oldlog = trainer.actor._squashed_log_prob(dist, torch.as_tensor(raw), torch.as_tensor(actions)).numpy()
    rewards = rng.normal(size=(T, E, A)).astype(np.float32)
    rollout = ModularRolloutBatch(
        observations=obs, actions=actions, raw_actions=raw, old_log_probs=oldlog,
        rewards=rewards, raw_environment_rewards=rewards.copy(), dones=np.zeros((T, E), np.float32),
        alive_masks=alive, next_observations=obs.copy(), next_alive_masks=alive.copy(),
        wave_indices=waves, total_waves=np.full((T, E), 3),
        contexts=np.zeros((T, E, 0), np.float32), next_contexts=np.zeros((T, E, 0), np.float32),
        episode_masks=np.ones((T, E), np.float32), wave_transition_flags=np.array([[1], [0]], np.float32),
        marc_success_segments=[])
    metrics = trainer.update(rollout)
    for key in ("actor_loss", "value_loss", "approx_kl", "entropy", "marc_global_adv_mean",
                "marc_local_adv_mean", "marc_cont_adv_mean", "marc_actor_adv_mean",
                "marc_effective_weight_mean", "marc_retention_loss"):
        assert np.isfinite(metrics[key]), key


def test_critic_target_remains_global_full_horizon_returns():
    from types import MethodType
    config = marc_config(); trainer = build_modular_mappo_trainer(config, "cpu", total_sampled_steps=100)
    T, E, A = 3, 1, 4
    obs = np.zeros((T, E, A, 52), np.float32); raw = np.zeros((T, E, A, 3), np.float32)
    actions = np.zeros_like(raw); alive = np.ones((T, E, A), np.float32)
    rewards = np.broadcast_to(np.array([1., 2., 3.], np.float32)[:, None, None], (T, E, A)).copy()
    values = torch.zeros(T, E, A); next_values = torch.ones(T, E, A)
    trainer._value_rollout = MethodType(lambda self, r, o, n: (values, next_values), trainer)
    captured = {}
    def capture(self, obs, act, raw, oldlog, alive, adv, oldvalue, target, weights, ctx, waves, marc_weights):
        captured["target"] = target.detach().clone()
        return {}
    trainer._update_flat_marc = MethodType(capture, trainer)
    rollout = ModularRolloutBatch(
        observations=obs, actions=actions, raw_actions=raw, old_log_probs=np.zeros((T, E, A), np.float32),
        rewards=rewards, raw_environment_rewards=rewards.copy(), dones=np.zeros((T, E), np.float32),
        alive_masks=alive, next_observations=obs.copy(), next_alive_masks=alive.copy(),
        wave_indices=np.array([[1], [2], [2]]), total_waves=np.full((T, E), 3),
        contexts=np.zeros((T, E, 0), np.float32), next_contexts=np.zeros((T, E, 0), np.float32),
        episode_masks=np.ones((T, E), np.float32), wave_transition_flags=np.array([[1], [0], [0]], np.float32),
        marc_success_segments=[])
    trainer.update(rollout)
    _, expected_returns = compute_gae(torch.as_tensor(rewards), values, next_values,
                                      torch.zeros(T, E), torch.ones(T, E, A),
                                      torch.ones(T, E, A), trainer.gamma, trainer.gae_lambda)
    assert torch.allclose(captured["target"], expected_returns)


def test_frozen_environment_reward_blue_and_weapon_hashes():
    env = load_config(ROOT / "configs/persistent_wave_v2_environment.yaml")
    expected = {
        "reward": "ca32c924d8e85f7946fb785212691a460aebc175cea05a8edf19c7e3312a2ac8",
        "blue_policy": "b70d8df8ef5ad3a9f049983d06c8d45e2ec3f3ddcd4e40991bd41e7340c248c9",
        "weapon": "e4926808f99ff7a0f00a008d4c300ce8d3b335e4123055b2e7ec6f8cdf4fcafc",
    }
    assert {name: config_sha256(env[name]) for name in expected} == expected


def test_disabled_baseline_does_not_expose_marc_architecture():
    baseline = load_config(ROOT / "configs/diag_mappo_learnability_common_3m.yaml")
    trainer = build_modular_mappo_trainer(baseline, "cpu", 16, 100)
    assert not trainer.milestone_aware_retention_credit.enabled
    assert "milestone_aware_retention_credit_enabled" not in checkpoint_architecture(trainer)
