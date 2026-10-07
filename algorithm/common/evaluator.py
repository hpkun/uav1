"""Deterministic evaluation on seeds disjoint from training."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Any
from env.factory import make_combat_environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMBAT_CONFIG = PROJECT_ROOT / "configs/combat_environment.yaml"


def episode_return_metrics(agent_returns: np.ndarray) -> tuple[float, float]:
    """Team sum and per-agent mean return diagnostics."""
    values = np.asarray(agent_returns, dtype=float)
    if values.shape != (4,):
        raise ValueError("agent_returns must have shape (4,)")
    return float(values.sum()), float(values.mean())


def persistent_mission_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate mission/wave metrics only when records use the wave variant."""
    if not records or "waves_cleared" not in records[0]:
        return {}
    total_waves = int(records[0]["total_waves"])
    result: dict[str, float] = {
        "average_waves_cleared": float(np.mean([
            row["waves_cleared"] for row in records
        ])),
    }
    for wave_index in range(1, total_waves + 1):
        result[f"clear_wave_{wave_index}_probability"] = float(np.mean([
            row["waves_cleared"] >= wave_index for row in records
        ]))
        survivor_values = [
            wave["red_survivors_end"]
            for row in records
            for wave in row.get("per_wave_metrics", [])
            if wave["wave_index"] == wave_index
            and wave.get("wave_cleared", True)
        ]
        result[f"average_red_survivors_after_wave_{wave_index}"] = (
            float(np.mean(survivor_values)) if survivor_values else 0.0
        )
    total_blue_losses = float(sum(row["blue_losses"] for row in records))
    total_red_losses = float(sum(row["red_losses"] for row in records))
    result.update({
        "total_blue_losses": total_blue_losses,
        "total_red_losses": total_red_losses,
        "kill_loss_ratio": total_blue_losses / max(total_red_losses, 1.0),
    })
    return result


def evaluate(actor, config=DEFAULT_COMBAT_CONFIG, seeds=range(10_000_000, 10_000_020)) -> dict[str, float]:
    records = []
    policy_rows: list[dict[str, float]] = []
    for seed in seeds:
        env = make_combat_environment(config)
        observation, _ = env.reset(int(seed))
        agent_returns = np.zeros(4, dtype=float)
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
    mean = lambda key: float(np.mean([record[key] for record in records]))
    result = {
        "average_return": mean("episode_return"),
        "average_agent_return": mean("mean_agent_episode_return"),
        "win_rate": mean("red_success"),
        "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"),
        "timeout_rate": float(np.mean([
            record["termination_reason"] == "red_failure_timeout" for record in records
        ])),
        "average_red_loss": mean("red_losses"),
        "average_blue_loss": mean("blue_losses"),
        **{
            f"{side}_{event}_episode_rate": float(np.mean([
                record[f"{side}_first_{event}_step"] is not None for record in records
            ]))
            for side in ("red", "blue")
            for event in ("fire_window", "attempt", "hit", "kill")
        },
        "average_red_attack_kills": mean("red_attack_kills"),
        "average_blue_attack_kills": mean("blue_attack_kills"),
        "average_red_boundary_exits": mean("red_boundary_exits"),
        "evaluation_boundary_exit_rate": float(np.mean([
            record["red_boundary_exits"] > 0 for record in records
        ])),
        "average_blue_boundary_exits": mean("blue_boundary_exits"),
        "average_red_ground_losses": mean("red_ground_losses"),
        "average_blue_ground_losses": mean("blue_ground_losses"),
        "average_episode_length": mean("episode_length"),
        **{
            f"average_episode_{name}_total": mean(f"episode_{name}_total")
            for name in ("r1", "r2", "r3", "r4")
        },
        "evaluation_episodes": len(records),
        **persistent_mission_metrics(records),
    }
    if policy_rows:
        result.update({
            key: float(np.mean([row[key] for row in policy_rows]))
            for key in policy_rows[0]
        })
    return result


__all__ = [
    "episode_return_metrics", "evaluate", "persistent_mission_metrics",
]
