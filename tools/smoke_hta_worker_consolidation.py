"""Synthetic HTA V2 consolidation smoke; no environment rollout."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules import ManagerTransitionBatch


def build(config_path, seed=88_040_002):
    config = load_config(config_path)
    config["training"]["seed"] = seed
    config["training"]["ppo_epochs"] = 1
    config["training"]["minibatch_size"] = 64
    config["network"]["actor_hidden_layers"] = [16, 16]
    config["network"]["critic_hidden_layers"] = [16, 16]
    return build_modular_mappo_trainer(config, "cpu", 16, 3_000_000)


def same(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def batch(trainer):
    rng = np.random.default_rng(7202); time_steps, envs, agents = 4, 2, 4
    obs = rng.normal(size=(time_steps, envs, agents, 52)).astype(np.float32)
    next_obs = rng.normal(size=(time_steps, envs, agents, 52)).astype(np.float32)
    alive = np.ones((time_steps, envs, agents), np.float32)
    options = rng.integers(0, 4, size=(time_steps, envs, agents), dtype=np.int64)
    actions, raw, logs = [], [], []
    for step in range(time_steps):
        values = trainer.act(obs[step], alive[step], False, True, option_ids=options[step])
        actions.append(values[0]); raw.append(values[1]); logs.append(values[2])
    rewards = rng.normal(scale=0.1, size=(time_steps, envs, agents)).astype(np.float32)
    context = np.zeros((time_steps, envs, 0), np.float32)
    manager = ManagerTransitionBatch(
        observations=np.asarray([obs[0, 0], obs[0, 1], obs[2, 0], obs[2, 1]]),
        alive_masks=np.ones((4, agents), np.float32),
        options=np.asarray([options[0, 0], options[0, 1], options[2, 0], options[2, 1]]),
        old_log_probs=np.full((4, agents), -np.log(4), np.float32),
        rewards=rng.normal(scale=0.2, size=(4, agents)).astype(np.float32),
        next_observations=np.asarray([next_obs[1, 0], next_obs[1, 1], next_obs[3, 0], next_obs[3, 1]]),
        next_alive_masks=np.ones((4, agents), np.float32), durations=np.asarray([2, 2, 2, 2]),
        bootstrap_masks=np.ones((4, agents), np.float32),
        trace_masks=np.asarray([[1] * agents, [1] * agents, [0] * agents, [0] * agents], np.float32),
        env_ids=np.asarray([0, 1, 0, 1]),
        end_reasons=np.asarray(["periodic", "wave_transition", "rollout_truncation", "rollout_truncation"]),
    )
    return ModularRolloutBatch(
        obs, np.asarray(actions), np.asarray(raw), np.asarray(logs), rewards, rewards.copy(),
        np.zeros((time_steps, envs), np.float32), alive, next_obs, alive.copy(),
        np.ones((time_steps, envs), np.int64), np.full((time_steps, envs), 3, np.int64),
        context, context.copy(), hta_options=options, hta_next_options=np.roll(options, -1, axis=0),
        hta_manager_transitions=manager,
    )


def main():
    v1 = build("configs/dev_hta_mappo_v1_3m.yaml")
    v2 = build("configs/dev_hta_mappo_v2_3m.yaml")
    checks = {
        "initial_actor_identity": same(v1.actor.state_dict(), v2.actor.state_dict()),
        "initial_critic_identity": same(v1.critic.state_dict(), v2.critic.state_dict()),
        "initial_manager_actor_identity": same(v1.manager_actor.state_dict(), v2.manager_actor.state_dict()),
        "initial_manager_critic_identity": same(v1.manager_critic.state_dict(), v2.manager_critic.state_dict()),
    }
    steps = (0, 600_000, 900_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000)
    checks["schedule_exact"] = [v2.hta_worker_consolidation.multiplier(step) for step in steps] == [1, 1, 1, 1, 0.625, 0.25, 0.25]
    rollout = batch(v1); v1.sampled_steps = v2.sampled_steps = 1_200_000
    rng = torch.get_rng_state().clone(); v1.update(deepcopy(rollout)); torch.set_rng_state(rng)
    metrics = v2.update(deepcopy(rollout))
    checks["pre_1p5m_actor_identity"] = same(v1.actor.state_dict(), v2.actor.state_dict())
    checks["pre_1p5m_critic_identity"] = same(v1.critic.state_dict(), v2.critic.state_dict())
    checks["pre_1p5m_manager_identity"] = same(v1.manager_actor.state_dict(), v2.manager_actor.state_dict()) and same(v1.manager_critic.state_dict(), v2.manager_critic.state_dict())
    checks["parameter_displacement_finite"] = all(np.isfinite(metrics[key]) and metrics[key] >= 0 for key in (
        "hta_worker_parameter_norm_before", "hta_worker_parameter_norm_after",
        "hta_worker_parameter_delta_l2", "hta_worker_relative_parameter_update"))
    baseline = v2.actor_lr_decay.learning_rate(2_000_000, v2.base_actor_learning_rate)
    split_2m = v2.hta_worker_consolidation.apply(v2.actor_optimizer, 2_000_000, baseline)
    v2.actor_lr_decay.apply(v2.manager_actor_optimizer, 2_000_000, v2.base_actor_learning_rate)
    checks["lr_split_2m"] = split_2m["effective_lr"] == 6.25e-5 and v2.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4
    baseline = v2.actor_lr_decay.learning_rate(2_500_000, v2.base_actor_learning_rate)
    split_2p5m = v2.hta_worker_consolidation.apply(v2.actor_optimizer, 2_500_000, baseline)
    checks["lr_split_2p5m"] = split_2p5m["effective_lr"] == 2.5e-5 and v2.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hta_v2.pt"; v2.sampled_steps = 2_500_000; v2.save(path)
        restored = build("configs/dev_hta_mappo_v2_3m.yaml"); restored.load(path)
        checks["checkpoint_roundtrip"] = same(v2.actor.state_dict(), restored.actor.state_dict()) and restored.actor_optimizer.param_groups[0]["lr"] == 2.5e-5 and restored.manager_actor_optimizer.param_groups[0]["lr"] == 1e-4
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError("HTA V2 consolidation smoke failed: " + ", ".join(failed))
    print("HTA_V2_CONSOLIDATION_SMOKE_PASS")


if __name__ == "__main__":
    main()
