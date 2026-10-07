"""Actor-only deterministic or stochastic MADSAC evaluation."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from algorithm.common.evaluator import episode_return_metrics, aggregate_combat_records
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
    return aggregate_combat_records(records)



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
