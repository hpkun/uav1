"""Actor-only deterministic or stochastic MADSAC evaluation."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from algorithm.common.evaluator import episode_return_metrics, persistent_mission_metrics
from env.factory import make_combat_environment


def evaluation_generator(device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def evaluation_episode_policy_seed(base_policy_seed: int, environment_seed: int) -> int:
    """Stable per-scenario policy-noise seed, independent of episode ordering/length."""
    payload = f"madsac:{int(base_policy_seed)}:{int(environment_seed)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def evaluate_madsac_episode(trainer, env_config, seed: int, mode="stochastic",
                            generator: torch.Generator | None = None):
    if mode not in {"stochastic", "deterministic"}:
        raise ValueError("mode must be stochastic or deterministic")
    env = make_combat_environment(env_config)
    observation, _ = env.reset(int(seed))
    returns = np.zeros(4, dtype=np.float64)
    while True:
        actions = trainer.act(
            observation[None], env.red_alive_mask[None],
            deterministic=mode == "deterministic", generator=generator,
        )[0]
        observation, reward, terminated, truncated, info = env.step(actions)
        returns += reward
        if terminated or truncated:
            team, agent = episode_return_metrics(returns)
            return {"episode_return": team, "mean_agent_episode_return": agent, **info}


def aggregate_madsac_records(records):
    mean = lambda key: float(np.mean([row[key] for row in records]))
    result = {
        "average_return": mean("episode_return"),
        "average_agent_return": mean("mean_agent_episode_return"),
        "win_rate": mean("red_success"), "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"),
        "timeout_rate": float(np.mean([row["termination_reason"] == "red_failure_timeout" for row in records])),
        "average_red_loss": mean("red_losses"), "average_blue_loss": mean("blue_losses"),
        "average_red_boundary_exits": mean("red_boundary_exits"),
        "average_red_ground_losses": mean("red_ground_losses"),
        "average_episode_length": mean("episode_length"),
        "evaluation_episodes": len(records), **persistent_mission_metrics(records),
    }
    for wave in (1, 2, 3):
        current = result[f"clear_wave_{wave}_probability"]
        previous = result[f"clear_wave_{wave-1}_probability"] if wave > 1 else None
        if wave == 2:
            result["Q2"] = None if previous == 0 else current / previous
        elif wave == 3:
            result["Q3"] = None if previous == 0 else current / previous
    return result


def evaluate_madsac(trainer, env_config, seeds, mode="stochastic", policy_seed=770001,
                    progress=None):
    before = trainer.policy_generator.get_state().clone()
    records = []
    for index, seed in enumerate(seeds, 1):
        generator = None
        if mode == "stochastic":
            generator = evaluation_generator(
                trainer.device,
                evaluation_episode_policy_seed(policy_seed, int(seed)),
            )
        records.append(evaluate_madsac_episode(trainer, env_config, seed, mode, generator))
        if progress is not None:
            progress(index, len(seeds))
    if not torch.equal(before, trainer.policy_generator.get_state()):
        raise RuntimeError("MADSAC evaluation polluted training policy RNG")
    return aggregate_madsac_records(records)


__all__ = [
    "aggregate_madsac_records", "evaluate_madsac", "evaluate_madsac_episode",
    "evaluation_episode_policy_seed", "evaluation_generator",
]
