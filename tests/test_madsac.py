from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithm.common.vector_env import VectorStep
from algorithm.madsac.evaluation import (
    evaluate_madsac,
    evaluation_episode_policy_seed,
    evaluation_generator,
)
from algorithm.madsac.networks import CentralizedAttentionQCritic, SharedMADSACActor
from algorithm.madsac.replay_buffer import JointReplayBuffer, ReplayBatch
from algorithm.madsac.runner import MADSACTrainingRunner, advance_sampled_steps
from algorithm.madsac.trainer import (
    MADSACTrainer,
    actor_objective,
    batch_mean_agent_sum,
    critic_target,
    joint_actions_with_own_gradient,
    masked_mean,
    polyak_update,
)


def tiny_batch(batch_size=8):
    rng = np.random.default_rng(12)
    return ReplayBatch(
        rng.normal(size=(batch_size, 4, 52)).astype(np.float32),
        rng.uniform(-1, 1, size=(batch_size, 4, 3)).astype(np.float32),
        rng.normal(size=(batch_size, 4)).astype(np.float32),
        rng.normal(size=(batch_size, 4, 52)).astype(np.float32),
        np.zeros(batch_size, dtype=np.bool_),
        np.ones((batch_size, 4), dtype=np.float32),
        np.ones((batch_size, 4), dtype=np.float32),
    )


def tiny_trainer(seed=7, policy_delay=2):
    return MADSACTrainer(hidden_dim=16, attention_heads=2, policy_delay=policy_delay, seed=seed)


def parameters(module):
    return [value.detach().clone() for value in module.parameters()]


def any_changed(before, module):
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


def test_actor_shapes_bounds_finite_log_prob_and_deterministic_identity():
    actor = SharedMADSACActor(hidden_dim=16)
    observations = torch.randn(3, 4, 52)
    generator = torch.Generator().manual_seed(101)
    actions, raw, log_prob = actor.sample(observations, generator)
    assert actions.shape == raw.shape == (3, 4, 3)
    assert log_prob.shape == (3, 4)
    assert torch.isfinite(log_prob).all()
    assert torch.all(actions >= -1) and torch.all(actions <= 1)
    assert torch.equal(actor.deterministic(observations), torch.tanh(actor.distribution(observations).mean))


def test_actor_generator_reproducibility_and_flat_inputs():
    actor = SharedMADSACActor(hidden_dim=16)
    observations = torch.randn(5, 52)
    first = actor.sample(observations, torch.Generator().manual_seed(9))[0]
    second = actor.sample(observations, torch.Generator().manual_seed(9))[0]
    assert first.shape == (5, 3)
    assert torch.equal(first, second)


def test_critics_independent_shape_dead_masks_and_no_dead_key_value_influence():
    torch.manual_seed(3)
    critic1 = CentralizedAttentionQCritic(hidden_dim=16, attention_heads=2)
    critic2 = CentralizedAttentionQCritic(hidden_dim=16, attention_heads=2)
    assert not any(a.data_ptr() == b.data_ptr() for a, b in zip(critic1.parameters(), critic2.parameters()))
    observations = torch.randn(2, 4, 52)
    actions = torch.randn(2, 4, 3)
    alive = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.float32)
    baseline, attention = critic1(observations, actions, alive, return_attention=True)
    changed_obs, changed_actions = observations.clone(), actions.clone()
    changed_obs[alive == 0] = 1e6
    changed_actions[alive == 0] = -1e6
    changed = critic1(changed_obs, changed_actions, alive)
    assert baseline.shape == (2, 4)
    assert attention.shape == (2, 2, 4, 4)
    assert torch.equal(baseline[alive == 0], torch.zeros_like(baseline[alive == 0]))
    assert torch.allclose(baseline[alive > 0], changed[alive > 0], atol=1e-6)


def test_target_networks_are_hard_copied_frozen_and_have_no_optimizer():
    trainer = tiny_trainer(seed=13)
    for online, target in (
        (trainer.actor, trainer.target_actor),
        (trainer.critic1, trainer.target_critic1),
        (trainer.critic2, trainer.target_critic2),
    ):
        assert all(torch.equal(a, b) for a, b in zip(online.state_dict().values(), target.state_dict().values()))
        assert not any(parameter.requires_grad for parameter in target.parameters())
    optimizer_ids = {
        id(parameter)
        for optimizer in (trainer.actor_optimizer, trainer.critic1_optimizer, trainer.critic2_optimizer)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not any(id(parameter) in optimizer_ids for module in (
        trainer.target_actor, trainer.target_critic1, trainer.target_critic2,
    ) for parameter in module.parameters())


def test_replay_stores_joint_transition_and_true_terminal_next_observation():
    replay = JointReplayBuffer(capacity=4, seed=2)
    original = np.zeros((1, 4, 52), np.float32)
    transition_next = np.full((1, 4, 52), 7, np.float32)
    auto_reset = np.full((1, 4, 52), 99, np.float32)
    result = VectorStep(
        auto_reset, transition_next, np.ones((1, 4), np.float32),
        np.array([True]), np.array([False]), [{}],
        np.ones((1, 4), np.float32), np.zeros((1, 4), np.float32),
    )
    runner = object.__new__(MADSACTrainingRunner)
    runner.replay = replay
    runner.store_vector_step(original, np.zeros((1, 4, 3), np.float32), result)
    assert replay.observations.shape == (4, 4, 52)
    assert np.all(replay.next_observations[0] == 7)
    assert not np.any(replay.next_observations[0] == 99)
    assert bool(replay.dones[0])


def test_terminal_next_death_twin_min_and_entropy_target_formula():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    q1 = torch.tensor([[8.0, 7.0, 6.0, 5.0]])
    q2 = torch.tensor([[6.0, 8.0, 4.0, 9.0]])
    log_prob = torch.tensor([[-2.0, -1.0, -3.0, -4.0]])
    next_alive = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    alive_target = critic_target(rewards, torch.tensor([0.0]), next_alive, q1, q2, log_prob, .5, .1)
    assert alive_target[0, 0] == pytest.approx(1 + .5 * (6 - .1 * -2))
    assert alive_target[0, 1] == 2  # next_alive=0 stops bootstrap
    terminal_target = critic_target(rewards, torch.tensor([1.0]), next_alive, q1, q2, log_prob, .5, .1)
    assert torch.equal(terminal_target, rewards)


def test_death_transition_current_alive_remains_in_critic_loss():
    predicted = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[2.0, 100.0]])
    current_alive = torch.tensor([[1.0, 0.0]])
    next_alive = torch.tensor([[0.0, 0.0]])
    assert masked_mean((predicted - target).square(), current_alive).item() == 4.0
    assert critic_target(target, torch.tensor([0.0]), next_alive, predicted, predicted, predicted, .99, .1)[0, 0] == 2


def test_paper_objective_reduction_sums_agents_then_means_batch():
    ones = torch.ones(3, 4)
    alive = torch.ones(3, 4)
    assert masked_mean(ones, alive).item() == 1.0
    assert batch_mean_agent_sum(ones, alive).item() == 4.0
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    mixed = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    assert batch_mean_agent_sum(values, mixed).item() == pytest.approx((4.0 + 13.0) / 2.0)


def test_actor_objective_sign():
    loss = actor_objective(
        torch.tensor([[-2.0, -4.0]]), torch.tensor([[3.0, 1.0]]),
        torch.tensor([[2.0, 5.0]]), torch.ones(1, 2), alpha=.1,
    )
    assert loss.item() == pytest.approx((-.2 - 2.0) + (-.4 - 1.0))


def test_q_i_backward_has_only_own_action_gradient():
    torch.manual_seed(14)
    critic = CentralizedAttentionQCritic(hidden_dim=16, attention_heads=2)
    observations = torch.randn(3, 4, 52)
    actions = torch.randn(3, 4, 3, requires_grad=True)
    alive = torch.ones(3, 4)
    agent_index = 2
    own_only = joint_actions_with_own_gradient(actions, agent_index)
    critic(observations, own_only, alive)[:, agent_index].sum().backward()
    assert actions.grad[:, agent_index].abs().sum().item() > 0
    other = torch.cat((actions.grad[:, :agent_index], actions.grad[:, agent_index + 1:]), dim=1)
    assert torch.count_nonzero(other).item() == 0


def test_delayed_actor_and_targets_update_only_on_delay_and_critics_not_actor_optimized():
    trainer = tiny_trainer(policy_delay=2)
    actor_before = parameters(trainer.actor)
    target_before = parameters(trainer.target_actor)
    first = trainer.update(tiny_batch())
    assert first["actor_updated"] == 0
    assert not any_changed(actor_before, trainer.actor)
    assert not any_changed(target_before, trainer.target_actor)
    critic_after_first = parameters(trainer.critic1)
    second = trainer.update(tiny_batch())
    assert second["actor_updated"] == 1
    assert any_changed(actor_before, trainer.actor)
    assert any_changed(target_before, trainer.target_actor)
    assert any_changed(critic_after_first, trainer.critic1)  # critic optimizer still runs once only
    assert second["actor_gradient_norm"] > 0  # Q(action) gradient reaches shared actor


def test_actor_step_has_q_gradient_without_changing_frozen_critic_parameters():
    trainer = MADSACTrainer(
        hidden_dim=16, attention_heads=2, policy_delay=1, seed=8,
        critic_learning_rate=0.0,
    )
    critic1_before, critic2_before = parameters(trainer.critic1), parameters(trainer.critic2)
    actor_before = parameters(trainer.actor)
    metrics = trainer.update(tiny_batch())
    assert metrics["actor_updated"] == 1
    assert metrics["actor_gradient_norm"] > 0
    assert not any_changed(critic1_before, trainer.critic1)
    assert not any_changed(critic2_before, trainer.critic2)
    assert any_changed(actor_before, trainer.actor)


def test_polyak_formula_exact():
    online = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    online.weight.data.fill_(4)
    target.weight.data.fill_(2)
    polyak_update(target, online, .25)
    assert torch.equal(target.weight, torch.full_like(target.weight, 2.5))


def test_checkpoint_reload_actor_for_evaluation(tmp_path):
    trainer = tiny_trainer(seed=19)
    observations = np.ones((1, 4, 52), np.float32)
    before = trainer.act(observations, deterministic=True)
    checkpoint = tmp_path / "madsac.pt"
    trainer.save(checkpoint)
    restored = tiny_trainer(seed=20)
    restored.load_for_evaluation(checkpoint)
    assert np.array_equal(before, restored.act(observations, deterministic=True))
    with pytest.raises(RuntimeError, match="exact resume unsupported"):
        restored.resume(checkpoint)


def test_evaluation_modes_and_rng_isolation(monkeypatch):
    trainer = tiny_trainer(seed=31)
    training_rng_before = trainer.policy_generator.get_state().clone()

    def fake_episode(local_trainer, _config, seed, mode, generator):
        obs = np.full((1, 4, 52), float(seed % 3), np.float32)
        action = local_trainer.act(obs, np.ones((1, 4), np.float32), mode == "deterministic", generator)[0]
        return {
            "episode_return": float(action.sum()), "mean_agent_episode_return": float(action.mean()),
            "red_success": 0.0, "blue_win": 0.0, "draw": 1.0,
            "termination_reason": "red_failure_timeout", "red_losses": 0,
            "blue_losses": 0, "red_boundary_exits": 0, "red_ground_losses": 0,
            "episode_length": 1, "waves_cleared": 0, "total_waves": 3,
            "per_wave_metrics": [],
        }

    monkeypatch.setattr("algorithm.madsac.evaluation.evaluate_madsac_episode", fake_episode)
    stochastic1 = evaluate_madsac(trainer, {}, (44_000_000, 44_000_001), "stochastic", 770001)
    stochastic2 = evaluate_madsac(trainer, {}, (44_000_000, 44_000_001), "stochastic", 770001)
    deterministic = evaluate_madsac(trainer, {}, (44_000_000, 44_000_001), "deterministic", 770001)
    assert stochastic1 == stochastic2
    assert stochastic1["average_return"] != deterministic["average_return"]
    assert torch.equal(training_rng_before, trainer.policy_generator.get_state())


def test_per_scenario_policy_rng_is_stable_distinct_and_order_independent(monkeypatch):
    trainer = tiny_trainer(seed=32)
    seed_a, seed_b = 44_000_017, 44_000_018
    derived_a = evaluation_episode_policy_seed(770001, seed_a)
    derived_b = evaluation_episode_policy_seed(770001, seed_b)
    assert derived_a == evaluation_episode_policy_seed(770001, seed_a)
    assert derived_a != derived_b
    sequence_a = torch.rand(8, generator=evaluation_generator("cpu", derived_a))
    sequence_b = torch.rand(8, generator=evaluation_generator("cpu", derived_b))
    assert not torch.equal(sequence_a, sequence_b)

    first_draws = {}

    def fake_episode(_trainer, _config, seed, _mode, generator):
        first_draws.setdefault(seed, []).append(float(torch.rand((), generator=generator)))
        if seed == seed_a:
            torch.rand(137, generator=generator)  # scenario A length/consumption cannot shift B
        return {
            "episode_return": 0.0, "mean_agent_episode_return": 0.0,
            "red_success": 0.0, "blue_win": 0.0, "draw": 1.0,
            "termination_reason": "red_failure_timeout", "red_losses": 0,
            "blue_losses": 0, "red_boundary_exits": 0, "red_ground_losses": 0,
            "episode_length": 1, "waves_cleared": 0, "total_waves": 3,
            "per_wave_metrics": [],
        }

    monkeypatch.setattr("algorithm.madsac.evaluation.evaluate_madsac_episode", fake_episode)
    training_before = trainer.policy_generator.get_state().clone()
    evaluate_madsac(trainer, {}, (seed_a, seed_b), "stochastic", 770001)
    evaluate_madsac(trainer, {}, (seed_b,), "stochastic", 770001)
    assert first_draws[seed_b][0] == first_draws[seed_b][1]
    assert torch.equal(training_before, trainer.policy_generator.get_state())


def test_wave_transition_not_done_but_terminated_or_truncated_are_done():
    replay = JointReplayBuffer(capacity=4)
    runner = object.__new__(MADSACTrainingRunner)
    runner.replay = replay
    zeros_obs = np.zeros((2, 4, 52), np.float32)
    result = VectorStep(
        zeros_obs, zeros_obs, np.zeros((2, 4), np.float32),
        np.array([False, False]), np.array([False, True]),
        [{"wave_transition": True}, {}], np.ones((2, 4), np.float32), np.ones((2, 4), np.float32),
    )
    runner.store_vector_step(zeros_obs, np.zeros((2, 4, 3), np.float32), result)
    assert replay.dones[:2].tolist() == [False, True]


def test_sampled_steps_count_environment_transitions_not_agents():
    assert advance_sampled_steps(0, 24) == 24
    assert advance_sampled_steps(24, 24) == 48
