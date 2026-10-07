from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import HTAWorkerConsolidationModule, ManagerTransitionBatch


def build_pair(seed=5301, hidden=16):
    v1_config = load_config("configs/dev_hta_mappo_v1_3m.yaml")
    v2_config = load_config("configs/dev_hta_mappo_v2_3m.yaml")
    for config in (v1_config, v2_config):
        config["training"]["seed"] = seed
        config["network"]["actor_hidden_layers"] = [hidden, hidden]
        config["network"]["critic_hidden_layers"] = [hidden, hidden]
        config["training"]["ppo_epochs"] = 2
        config["training"]["minibatch_size"] = 64
    return (
        build_modular_mappo_trainer(v1_config, "cpu", hidden, 3_000_000),
        build_modular_mappo_trainer(v2_config, "cpu", hidden, 3_000_000),
    )


def identical(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def synthetic_batch(trainer, time_steps=4, envs=2):
    rng = np.random.default_rng(7231)
    agents = 4
    observations = rng.normal(size=(time_steps, envs, agents, 52)).astype(np.float32)
    next_observations = rng.normal(size=(time_steps, envs, agents, 52)).astype(np.float32)
    alive = np.ones((time_steps, envs, agents), np.float32)
    options = rng.integers(0, 4, size=(time_steps, envs, agents), dtype=np.int64)
    next_options = np.roll(options, -1, axis=0)
    actions, raw_actions, old_log_probs = [], [], []
    for step in range(time_steps):
        action, raw, log_prob, _ = trainer.act(
            observations[step], alive[step], False, True, option_ids=options[step]
        )
        actions.append(action)
        raw_actions.append(raw)
        old_log_probs.append(log_prob)
    rewards = rng.normal(scale=0.1, size=(time_steps, envs, agents)).astype(np.float32)
    contexts = np.zeros((time_steps, envs, 0), np.float32)
    manager = ManagerTransitionBatch(
        observations=np.asarray([observations[0, 0], observations[0, 1], observations[2, 0], observations[2, 1]]),
        alive_masks=np.ones((4, agents), np.float32),
        options=np.asarray([options[0, 0], options[0, 1], options[2, 0], options[2, 1]]),
        old_log_probs=np.full((4, agents), -np.log(4), np.float32),
        rewards=rng.normal(scale=0.2, size=(4, agents)).astype(np.float32),
        next_observations=np.asarray([next_observations[1, 0], next_observations[1, 1], next_observations[3, 0], next_observations[3, 1]]),
        next_alive_masks=np.ones((4, agents), np.float32),
        durations=np.asarray([2, 2, 2, 2]),
        bootstrap_masks=np.ones((4, agents), np.float32),
        trace_masks=np.asarray([[1] * agents, [1] * agents, [0] * agents, [0] * agents], np.float32),
        env_ids=np.asarray([0, 1, 0, 1]),
        end_reasons=np.asarray(["periodic", "wave_transition", "rollout_truncation", "rollout_truncation"]),
    )
    return ModularRolloutBatch(
        observations, np.asarray(actions), np.asarray(raw_actions), np.asarray(old_log_probs),
        rewards, rewards.copy(), np.zeros((time_steps, envs), np.float32), alive,
        next_observations, alive.copy(), np.ones((time_steps, envs), np.int64),
        np.full((time_steps, envs), 3, np.int64), contexts, contexts.copy(),
        hta_options=options, hta_next_options=next_options,
        hta_manager_transitions=manager,
    )


def apply_v2_learning_rates(trainer, step):
    baseline = trainer.actor_lr_decay.apply(
        trainer.actor_optimizer, step, trainer.base_actor_learning_rate
    )
    trainer.actor_lr_decay.apply(
        trainer.manager_actor_optimizer, step, trainer.base_actor_learning_rate
    )
    return trainer.hta_worker_consolidation.apply(trainer.actor_optimizer, step, baseline)


def test_schedule_and_effective_worker_lr_are_exact():
    module = HTAWorkerConsolidationModule({
        "enabled": True, "schedule": "progressive_linear",
        "start_step": 1_500_000, "end_step": 2_500_000,
        "final_lr_multiplier": 0.25,
    })
    steps = (0, 600_000, 900_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000)
    assert [module.multiplier(step) for step in steps] == [1, 1, 1, 1, 0.625, 0.25, 0.25]
    baseline = (3e-4, 3e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    assert [rate * module.multiplier(step) for rate, step in zip(baseline, steps)] == [3e-4, 3e-4, 1e-4, 1e-4, 6.25e-5, 2.5e-5, 2.5e-5]


@pytest.mark.parametrize("field,value", [
    ("schedule", "other"), ("start_step", 1_400_000),
    ("end_step", 2_600_000), ("final_lr_multiplier", 0.5),
])
def test_schedule_configuration_fails_closed(field, value):
    config = {"enabled": True, "schedule": "progressive_linear", "start_step": 1_500_000, "end_step": 2_500_000, "final_lr_multiplier": 0.25}
    config[field] = value
    with pytest.raises(ValueError):
        HTAWorkerConsolidationModule(config)


def test_v1_v2_same_seed_initial_states_and_rng_are_exact():
    v1, v2 = build_pair(seed=91)
    assert identical(v1.actor.state_dict(), v2.actor.state_dict())
    assert identical(v1.critic.state_dict(), v2.critic.state_dict())
    assert identical(v1.manager_actor.state_dict(), v2.manager_actor.state_dict())
    assert identical(v1.manager_critic.state_dict(), v2.manager_critic.state_dict())
    assert not isinstance(v2.hta_worker_consolidation, torch.nn.Module)


def test_v2_metadata_is_opt_in_and_v1_identity_is_unchanged():
    v1, v2 = build_pair(seed=96)
    v1_architecture = checkpoint_architecture(v1)
    v2_architecture = checkpoint_architecture(v2)
    assert "hta_worker_consolidation_enabled" not in v1_architecture
    assert v2_architecture["hta_version"] == 1
    assert v2_architecture["hta_worker_consolidation_enabled"] is True
    assert v2_architecture["hta_worker_consolidation_version"] == 1
    assert v2_architecture["worker_consolidation_start_step"] == 1_500_000
    assert v2_architecture["worker_consolidation_end_step"] == 2_500_000
    assert v2_architecture["worker_final_lr_multiplier"] == 0.25
    identities = []
    for trainer, method in ((v1, "hta_mappo_v1"), (v2, "hta_mappo_v2")):
        runner = object.__new__(ModularMAPPOTrainingRunner)
        runner.trainer = trainer
        runner.algorithm_config = {"development_method": method}
        identities.append(runner.method_identity())
    assert "hta_worker_consolidation_enabled" not in identities[0]
    assert identities[1]["hta_worker_consolidation_enabled"] is True
    assert identities[1]["manager_learning_schedule"] == "unchanged_actor_lr_decay"
    assert identities[1]["tactical_critic_learning_schedule"] == "unchanged_constant"


def test_manager_and_critics_are_not_modified_by_consolidation_lr():
    _, trainer = build_pair(seed=92)
    critic_before = [group["lr"] for group in trainer.critic_optimizer.param_groups]
    manager_critic_before = [group["lr"] for group in trainer.manager_critic_optimizer.param_groups]
    result = apply_v2_learning_rates(trainer, 2_000_000)
    assert result == {"effective_lr": 6.25e-5, "multiplier": 0.625}
    assert trainer.actor_optimizer.param_groups[0]["lr"] == 6.25e-5
    assert trainer.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4
    assert [group["lr"] for group in trainer.critic_optimizer.param_groups] == critic_before == [3e-4]
    assert [group["lr"] for group in trainer.manager_critic_optimizer.param_groups] == manager_critic_before == [3e-4]
    apply_v2_learning_rates(trainer, 2_500_000)
    assert trainer.actor_optimizer.param_groups[0]["lr"] == 2.5e-5
    assert trainer.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4


def test_pre_1p5m_full_update_is_numerically_identical_and_displacement_is_finite():
    v1, v2 = build_pair(seed=93)
    batch = synthetic_batch(v1)
    v1.sampled_steps = v2.sampled_steps = 1_200_000
    torch_state = torch.get_rng_state().clone()
    metrics_v1 = v1.update(deepcopy(batch))
    torch.set_rng_state(torch_state)
    metrics_v2 = v2.update(deepcopy(batch))
    for first, second in (
        (v1.actor.state_dict(), v2.actor.state_dict()),
        (v1.critic.state_dict(), v2.critic.state_dict()),
        (v1.manager_actor.state_dict(), v2.manager_actor.state_dict()),
        (v1.manager_critic.state_dict(), v2.manager_critic.state_dict()),
    ):
        assert first.keys() == second.keys()
        assert all(torch.allclose(first[key], second[key], rtol=0, atol=1e-12) for key in first)
    assert metrics_v1["actor_learning_rate"] == metrics_v2["actor_learning_rate"] == 1e-4
    for key in (
        "hta_worker_parameter_norm_before", "hta_worker_parameter_norm_after",
        "hta_worker_parameter_delta_l2", "hta_worker_relative_parameter_update",
    ):
        assert np.isfinite(metrics_v2[key]) and metrics_v2[key] >= 0
    assert metrics_v2["hta_worker_consolidation_multiplier"] == 1.0
    assert metrics_v2["hta_manager_worker_lr_ratio"] == 1.0


def test_lr_split_is_applied_by_real_update_at_2m():
    _, trainer = build_pair(seed=94)
    trainer.sampled_steps = 2_000_000
    metrics = trainer.update(synthetic_batch(trainer))
    assert metrics["actor_learning_rate"] == metrics["hta_worker_actor_lr"] == 6.25e-5
    assert metrics["hta_manager_actor_lr"] == 1e-4
    assert metrics["critic_learning_rate"] == 3e-4
    assert metrics["hta_manager_worker_lr_ratio"] == 1.6


def test_manager_worker_optimizer_separation_and_checkpoint_roundtrip(tmp_path: Path):
    _, trainer = build_pair(seed=95)
    observations = torch.randn(2, 4, 52)
    options = torch.zeros(2, 4, dtype=torch.long)
    worker_before = {key: value.clone() for key, value in trainer.actor.state_dict().items()}
    loss = -trainer.manager_actor.distribution(observations).log_prob(options).mean()
    trainer.manager_actor_optimizer.zero_grad(); loss.backward(); trainer.manager_actor_optimizer.step()
    assert all(torch.equal(value, trainer.actor.state_dict()[key]) for key, value in worker_before.items())
    manager_before = {key: value.clone() for key, value in trainer.manager_actor.state_dict().items()}
    loss = trainer.actor.distribution(observations, option_ids=options).mean.sum()
    trainer.actor_optimizer.zero_grad(); loss.backward(); trainer.actor_optimizer.step()
    assert all(torch.equal(value, trainer.manager_actor.state_dict()[key]) for key, value in manager_before.items())
    trainer.sampled_steps = 2_600_000
    apply_v2_learning_rates(trainer, trainer.sampled_steps)
    path = tmp_path / "hta_v2.pt"
    trainer.save(path)
    _, restored = build_pair(seed=95)
    restored.load(path)
    assert identical(trainer.actor.state_dict(), restored.actor.state_dict())
    assert identical(trainer.critic.state_dict(), restored.critic.state_dict())
    assert identical(trainer.manager_actor.state_dict(), restored.manager_actor.state_dict())
    assert identical(trainer.manager_critic.state_dict(), restored.manager_critic.state_dict())
    assert restored.actor_optimizer.param_groups[0]["lr"] == 2.5e-5
    assert restored.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4
    architecture = restored.checkpoint_state()["development_feature_versions"]
    assert architecture["hierarchical_temporal_abstraction"] == 1
    assert architecture["hta_worker_consolidation"] == 1
