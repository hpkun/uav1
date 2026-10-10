"""Deterministic evaluation on seeds disjoint from training."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Any
from env.factory import make_combat_environment
from .metrics import episode_is_timeout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMBAT_CONFIG = PROJECT_ROOT / "configs/combat_environment.yaml"


def episode_return_metrics(agent_returns: np.ndarray) -> tuple[float, float]:
    """Team sum and per-agent mean return diagnostics."""
    values = np.asarray(agent_returns, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("agent_returns must be a non-empty one-dimensional array")
    return float(values.sum()), float(values.mean())


def aggregate_combat_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate completed episodes with equal weight per episode."""
    if not records:
        raise ValueError("evaluation records must not be empty")
    mean = lambda key: float(np.mean([row[key] for row in records]))
    result = {
        "average_return": mean("episode_return"),
        "average_agent_return": mean("mean_agent_episode_return"),
        "win_rate": mean("red_success"), "red_win_rate": mean("red_success"),
        "loss_rate": mean("blue_win"), "blue_win_rate": mean("blue_win"),
        "draw_rate": mean("draw"),
        "timeout_rate": float(np.mean([episode_is_timeout(row) for row in records])),
        "episode_return": mean("episode_return"), "episode_length": mean("episode_length"),
        "average_episode_length": mean("episode_length"),
        "average_red_loss": mean("red_losses"), "average_blue_loss": mean("blue_losses"),
        "red_losses": mean("red_losses"), "blue_losses": mean("blue_losses"),
        "red_survivors": mean("red_survivors"), "blue_survivors": mean("blue_survivors"),
        "evaluation_episodes": len(records),
        "evaluation_boundary_exit_rate": float(np.mean([row["red_boundary_exits"] > 0 for row in records])),
    }
    for side in ("red", "blue"):
        for event in ("fire_attempts", "weapon_hits", "attack_kills", "boundary_exits", "ground_losses"):
            result[f"{side}_{event}"] = mean(f"{side}_{event}")
            result[f"average_{side}_{event}"] = result[f"{side}_{event}"]
        for event in ("fire_window", "attempt", "hit", "kill"):
            result[f"{side}_{event}_episode_rate"] = float(np.mean([
                row[f"{side}_first_{event}_step"] is not None for row in records]))
    for event in ("fire_attempts", "weapon_hits", "attack_kills", "boundary_exits", "ground_losses"):
        result[event] = result[f"red_{event}"] + result[f"blue_{event}"]
    for name in ("r1", "r2", "r3", "r4"):
        result[f"average_episode_{name}_total"] = mean(f"episode_{name}_total")
    if all(row.get('environment_version') in ('3.0', '3.1', '3.2', '3.3') for row in records):
        for name in ('event','outcome','adv','safe'):
            result[f'average_episode_{name}_total'] = mean(f'episode_{name}_total')
        for side in ('red','blue'):
            for event in ('ceiling_losses','ammo_used'):
                result[f'{side}_{event}'] = mean(f'{side}_{event}')
            result[f'{side}_noncombat_loss_episode_rate'] = float(np.mean([
                row[f'{side}_boundary_exits']+row[f'{side}_ground_losses']>0 for row in records]))
    if all(row.get('environment_version') == '3.3' for row in records):
        for name in ('individual_event','team_casualty'):
            result[f'average_episode_{name}_total'] = mean(f'episode_{name}_total')
    return result



def evaluate(actor, config=DEFAULT_COMBAT_CONFIG, seeds=range(10_000_000, 10_000_020)) -> dict[str, float]:
    records = []
    policy_rows: list[dict[str, float]] = []
    for seed in seeds:
        env = make_combat_environment(config)
        observation, _ = env.reset(int(seed))
        agent_returns = np.zeros(env.team_size, dtype=float)
        while True:
            if hasattr(actor, "policy_statistics"):
                policy_rows.append(actor.policy_statistics(
                    observation, env.red_alive_mask
                ))
            actions = actor.act(observation, env.red_alive_mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(actions)
            agent_returns += reward
            if terminated or truncated:
                team_return, mean_agent_return = episode_return_metrics(agent_returns)
                records.append({
                    "episode_return": team_return,
                    "mean_agent_episode_return": mean_agent_return,
                    **info,
                })
                break
    result = aggregate_combat_records(records)
    if policy_rows:
        result.update({
            key: float(np.mean([row[key] for row in policy_rows]))
            for key in policy_rows[0]
        })
    return result


__all__ = [
    "episode_return_metrics", "evaluate", "aggregate_combat_records",
]
