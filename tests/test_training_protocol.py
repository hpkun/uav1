from pathlib import Path
import copy

import numpy as np
import pytest
import torch
import yaml

from env.combat_env import MultiUAVCombatEnv
from algorithm.common.evaluator import episode_return_metrics, evaluate
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.vector_env import ParallelVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def configs():
    environment = yaml.safe_load(
        (ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8")
    )
    algorithm = yaml.safe_load(
        (ROOT / "configs/mappo.yaml").read_text(encoding="utf-8")
    )
    return environment, algorithm


def test_protocol_and_interface_dimensions():
    _, algorithm = configs()
    assert algorithm["network"] == {
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
        "actor_hidden_layers": [256, 256],
        "critic_hidden_layers": [256, 256],
        "attention_heads": 2,
    }
    assert algorithm["training"]["num_train_envs"] == 24
    assert algorithm["training"]["evaluation_episodes"] == 20


def test_vector_seed_formula_no_reuse_and_52d_observations():
    environment, _ = configs()
    environment = copy.deepcopy(environment)
    environment["simulation"]["max_steps"] = 1
    vector = ParallelVectorEnv(
        3, environment, base_seed=10,
        forbidden_seeds=range(10_000_000, 10_000_020),
    )
    observations = vector.reset()
    assert observations.shape == (3, 4, 52)
    assert vector.last_reset_seeds.tolist() == [10, 11, 12]
    vector.step_batch(np.zeros((3, 4, 3), dtype=np.float32))
    assert vector.last_reset_seeds.tolist() == [13, 14, 15]
    assert len(vector.used_training_seeds) == 6
    assert vector.num_workers == 3
    assert len(set(vector.worker_pids)) == 3
    assert all(pid != 0 for pid in vector.worker_pids)
    vector.close()
    assert not any(process.is_alive() for process in vector._processes)


def test_parallel_workers_match_direct_environment_step_semantics():
    environment, _ = configs()
    seeds = [701, 702, 703]
    direct = [MultiUAVCombatEnv(environment) for _ in seeds]
    expected_observations = np.stack([
        env.reset(seed)[0] for env, seed in zip(direct, seeds)
    ])
    vector = ParallelVectorEnv(3, environment, base_seed=seeds[0])
    actual_observations = vector.reset()
    assert np.array_equal(actual_observations, expected_observations)
    rng = np.random.default_rng(44)
    for _ in range(5):
        actions = rng.uniform(-1.0, 1.0, size=(3, 4, 3)).astype(np.float32)
        expected = [env.step(actions[index]) for index, env in enumerate(direct)]
        actual = vector.step_batch(actions, auto_reset=False)
        assert np.array_equal(actual.transition_next_observations, np.stack([
            row[0] for row in expected
        ]))
        assert np.array_equal(actual.rewards, np.stack([row[1] for row in expected]))
        assert np.array_equal(actual.terminated, [row[2] for row in expected])
        assert np.array_equal(actual.truncated, [row[3] for row in expected])
    vector.close()


def test_mappo_startup_summary_reports_effective_network_and_rollout(tmp_path):
    environment, algorithm = configs()
    runner = MAPPOTrainingRunner(
        environment, algorithm, num_envs=2, total_sampled_steps=4,
        output_dir=tmp_path, smoke=True,
    )
    try:
        summary = runner.startup_summary()
        assert summary["algorithm"] == "MAPPO"
        assert summary["mode"] == "smoke"
        assert (summary["observation_dim"], summary["action_dim"], summary["num_agents"]) == (52, 3, 4)
        assert summary["effective_hidden_dim"] == 64
        assert summary["rollout_steps"] == 4
        assert summary["ppo_epochs"] == 2
        assert summary["environment_backend"] == "multiprocess_spawn"
        assert summary["environment_workers"] == 2
    finally:
        runner.vector.close()


@pytest.mark.parametrize("field,value", [
    ("observation_dim", 54), ("action_dim", 4), ("num_agents", 3),
])
def test_runner_fails_fast_on_network_environment_dimension_mismatch(
    tmp_path, field, value
):
    environment, algorithm = configs()
    algorithm = copy.deepcopy(algorithm)
    algorithm["network"][field] = value
    with pytest.raises(ValueError, match="network/environment dimension mismatch"):
        MAPPOTrainingRunner(
            environment, algorithm, total_sampled_steps=4, output_dir=tmp_path,
            smoke=True,
        )


def test_evaluator_exposes_split_combat_and_required_episode_metrics():
    environment, _ = configs()
    environment = copy.deepcopy(environment)
    environment["simulation"]["max_steps"] = 1

    class ZeroActor:
        @staticmethod
        def act(observation, alive_mask, deterministic=True):
            assert deterministic
            return np.zeros((4, 3), dtype=np.float32)

    result = evaluate(ZeroActor(), environment, seeds=[10_000_000])
    expected = {
        "average_return", "average_agent_return", "win_rate", "loss_rate",
        "draw_rate", "timeout_rate", "average_red_loss", "average_blue_loss",
        "average_red_attack_kills", "average_blue_attack_kills",
        "average_red_boundary_exits", "average_blue_boundary_exits",
        "average_red_ground_losses", "average_blue_ground_losses",
        "average_episode_length",
    } | {
        f"{side}_{event}_episode_rate"
        for side in ("red", "blue")
        for event in ("fire_window", "attempt", "hit", "kill")
    }
    expected |= {
        f"average_episode_{name}_total" for name in ("r1", "r2", "r3", "r4")
    }
    assert expected <= result.keys()


def test_episode_return_metric_is_team_sum_and_agent_mean():
    team, agent = episode_return_metrics(np.array([1.0, 2.0, 3.0, 4.0]))
    assert team == 10.0
    assert agent == 2.5


def test_checkpoint_version_is_saved_and_incompatible_semantics_fail_fast(tmp_path):
    environment, algorithm = configs()
    runner = MAPPOTrainingRunner(
        environment, algorithm, num_envs=1, total_sampled_steps=1,
        output_dir=tmp_path, smoke=True,
    )
    try:
        valid = tmp_path / "valid.pt"
        invalid = tmp_path / "invalid.pt"
        runner.save_checkpoint(valid)
        state = torch.load(valid, map_location="cpu", weights_only=False)
        assert state["extra"]["environment_version"] == "2.3"
        state["extra"]["environment_version"] = "2.0"
        torch.save(state, invalid)
        with pytest.raises(RuntimeError, match="environment semantics"):
            runner.resume(invalid)
        assert runner.trainer.sampled_steps == 0
    finally:
        runner.vector.close()
