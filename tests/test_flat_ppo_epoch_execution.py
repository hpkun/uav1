import math

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from tools.preflight_actor_grad_clip_fixed10_300k import assert_no_epoch_early_return


def rollout(sample_count: int, seed: int = 73) -> ModularRolloutBatch:
    rng = np.random.default_rng(seed)
    t, e, a, f = sample_count, 1, 4, 52
    obs = rng.normal(size=(t, e, a, f)).astype("f")
    raw = rng.normal(size=(t, e, a, 3)).astype("f")
    actions = np.tanh(raw).astype("f")
    alive = np.ones((t, e, a), dtype="f")
    rewards = rng.normal(size=(t, e, a)).astype("f")
    dones = np.zeros((t, e), dtype="f")
    context = np.zeros((t, e, 0), dtype="f")
    return ModularRolloutBatch(
        obs, actions, raw, np.zeros((t, e, a), dtype="f"), rewards,
        rewards.copy(), dones, alive, obs.copy(), alive.copy(),
        np.ones((t, e), dtype=int), np.full((t, e), 3, dtype=int),
        context, context, episode_masks=np.ones((t, e), dtype="f"),
    )


@pytest.mark.parametrize("epochs", [1, 3, 10])
@pytest.mark.parametrize("sample_count,minibatch_size", [(6, 3), (5, 3)])
def test_flat_ppo_executes_every_epoch_and_minibatch(epochs, sample_count, minibatch_size):
    trainer = ModularMAPPOTrainer(
        hidden_dim=8, ppo_epochs=epochs, minibatch_size=minibatch_size,
        seed=19, modules_config={},
    )
    before = (trainer.ppo_update_count, trainer.actor_update_count, trainer.critic_update_count)
    metrics = trainer.update(rollout(sample_count))
    expected_per_epoch = math.ceil(sample_count / minibatch_size)
    expected_steps = epochs * expected_per_epoch
    assert trainer.ppo_update_count - before[0] == 1
    assert trainer.actor_update_count - before[1] == expected_steps
    assert trainer.critic_update_count - before[2] == expected_steps
    assert metrics["ppo_epochs_configured"] == epochs
    assert metrics["ppo_epochs_executed"] == epochs
    assert metrics["ppo_minibatches_per_epoch"] == expected_per_epoch
    assert metrics["ppo_minibatches_executed"] == expected_steps
    assert metrics["actor_optimizer_steps_this_update"] == expected_steps
    assert metrics["critic_optimizer_steps_this_update"] == expected_steps


def test_fixed10_continuation_update_count_arithmetic():
    full_rollouts, full_samples, partial_samples, minibatch, epochs = 48, 6144, 5088, 512, 10
    assert math.ceil(full_samples / minibatch) == 12
    assert math.ceil(partial_samples / minibatch) == 10
    assert full_rollouts + 1 == 49
    assert epochs * (full_rollouts * 12 + 10) == 5860


def test_flat_update_ast_has_no_return_inside_epoch_loop():
    assert_no_epoch_early_return()
