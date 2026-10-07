"""On-policy MAPPO runner using long-lived parallel combat environments."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch

from env.config import ENVIRONMENT_VERSION
from env.combat_env import MultiUAVCombatEnv
from .trainer import MAPPO_IMPL_VERSION, MAPPOTrainer, RolloutBatch
from algorithm.common.evaluator import (
    episode_return_metrics,
    evaluate,
    aggregate_combat_records,
)
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import (
    evaluation_selection_key,
    validate_checkpoint_for_resume,
)
from algorithm.common.protocol import config_sha256


class MAPPOTrainingRunner:
    def __init__(self, env_config: dict, algorithm_config: dict,
                 num_envs: int | None = None,
                 total_sampled_steps: int | None = None,
                 device: str | None = None, seed: int | None = None,
                 output_dir: str | Path | None = None,
                 smoke: bool = False) -> None:
        self.env_config, self.algorithm_config = env_config, algorithm_config
        self.smoke = bool(smoke)
        training, network = algorithm_config["training"], algorithm_config["network"]
        implementation = algorithm_config["implementation"]
        configured = (int(network["observation_dim"]), int(network["action_dim"]),
                      int(network["num_agents"]))
        expected = (MultiUAVCombatEnv.observation_dim, MultiUAVCombatEnv.action_dim,
                    MultiUAVCombatEnv.team_size)
        if configured != expected:
            raise ValueError(f"network/environment dimension mismatch: configured obs/action/agents={configured}, environment={expected}")
        self.observation_dim, self.action_dim, self.num_agents = configured
        self.num_envs = int(num_envs or training["num_train_envs"])
        self.total_sampled_steps = int(total_sampled_steps or training["total_sampled_steps"])
        self.device = str(device or training["device"])
        self.seed = int(training["seed"] if seed is None else seed)
        if output_dir is None:
            raise ValueError("output_dir must be the final run directory")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_steps = 4 if smoke else int(training["rollout_steps"])
        hidden_dim = 64 if smoke else int(network["actor_hidden_layers"][0])
        self.effective_hidden_dim = hidden_dim
        self.trainer = MAPPOTrainer(
            self.observation_dim, self.action_dim, self.num_agents, hidden_dim,
            int(network["attention_heads"]), float(training["actor_learning_rate"]),
            float(training["critic_learning_rate"]), float(training["gamma"]),
            float(training["gae_lambda"]), float(training["clip_ratio"]),
            float(training["value_loss_coefficient"]),
            float(training["entropy_coefficient"]), float(training["max_grad_norm"]),
            2 if smoke else int(training["ppo_epochs"]),
            32 if smoke else int(training["minibatch_size"]),
            bool(implementation["normalize_advantages"]),
            bool(implementation["clip_value_loss"]), self.device, self.seed,
            str(implementation["actor_activation"]),
            str(implementation["critic_activation"]),
            float(implementation["log_std_min"]), float(implementation["log_std_max"]),
            critic_type=str(network.get("critic_type", "attention")),
        )
        evaluation_base = int(implementation["evaluation_seed_base"])
        self.evaluation_seeds = list(range(
            evaluation_base, evaluation_base + int(training["evaluation_episodes"])
        ))
        if self.seed + self.total_sampled_steps + self.num_envs >= evaluation_base:
            raise ValueError("configured training seed range can overlap evaluation seeds")
        self.vector = ParallelVectorEnv(self.num_envs, env_config, self.seed,
                                        self.evaluation_seeds)
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.agent_episode_returns = np.zeros((self.num_envs, self.num_agents), dtype=float)
        self.completed_records: list[dict[str, Any]] = []
        self.last_metrics: dict[str, float] = {}
        self.evaluation_history: list[dict[str, float]] = []
        self.best_evaluation: dict[str, float] | None = None
        logging = algorithm_config["runtime_logging"]
        self.console_interval = int(logging["console_interval_sampled_steps"])
        self.recent_episode_window = int(logging["recent_episode_window"])
        self.evaluation_interval = int(training["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = int(implementation["checkpoint_interval_sampled_steps"])
        if min(self.rollout_steps, self.console_interval, self.recent_episode_window,
               self.evaluation_interval, self.checkpoint_interval) <= 0:
            raise ValueError("MAPPO rollout and reporting intervals must be positive")
        self.next_console_log = self.console_interval
        self.next_evaluation = self.evaluation_interval
        self.next_checkpoint = self.checkpoint_interval

    def startup_summary(self) -> dict[str, Any]:
        return {
            "algorithm": "MAPPO", "mode": "smoke" if self.smoke else "formal",
            "device": self.device, "observation_dim": self.observation_dim,
            "action_dim": self.action_dim, "num_agents": self.num_agents,
            "effective_hidden_dim": self.effective_hidden_dim,
            "critic_type": self.trainer.critic_type,
            "attention_heads": self.trainer.attention_heads,
            "actor_parameter_count":sum(p.numel() for p in self.trainer.actor.parameters()),
            "critic_parameter_count":sum(p.numel() for p in self.trainer.critic.parameters()),
            "total_parameter_count":sum(p.numel() for p in self.trainer.actor.parameters())+sum(p.numel() for p in self.trainer.critic.parameters()),
            "num_envs_M": self.num_envs, "environment_backend": self.vector.backend,
            "environment_workers": self.vector.num_workers, "seed": self.seed,
            "total_sampled_steps": self.total_sampled_steps,
            "rollout_steps": self.rollout_steps, "ppo_epochs": self.trainer.ppo_epochs,
            "minibatch_size": self.trainer.minibatch_size,
            "gamma": self.trainer.gamma, "gae_lambda": self.trainer.gae_lambda,
            "clip_ratio": self.trainer.clip_ratio,
            "entropy_coefficient": self.trainer.entropy_coefficient,
            "environment_version": ENVIRONMENT_VERSION,
            "max_steps": int(self.env_config["simulation"]["max_steps"]),
        }

    def start_log_line(self) -> str:
        s = self.startup_summary()
        return (f"[START] algorithm=MAPPO | mode={s['mode']} | device={s['device']} "
                f"| obs={s['observation_dim']} | act={s['action_dim']} | agents={s['num_agents']} "
                f"| hidden={s['effective_hidden_dim']} | critic={s['critic_type']} | heads={s['attention_heads']} "
                f"| envs={s['num_envs_M']} | workers={s['environment_workers']} "
                f"| backend={s['environment_backend']} | seed={s['seed']} "
                f"| total={s['total_sampled_steps']} | rollout={s['rollout_steps']} "
                f"| epochs={s['ppo_epochs']} | minibatch={s['minibatch_size']} "
                f"| gamma={s['gamma']} | lambda={s['gae_lambda']} | clip={s['clip_ratio']} "
                f"| max_steps={s['max_steps']}")

    @staticmethod
    def _display(value: float | None, digits: int) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    def recent_episode_metrics(self) -> dict[str, float | None]:
        rows = self.completed_records[-self.recent_episode_window:]
        if not rows:
            return {key: None for key in ("return", "win", "red_loss", "fire", "kill")}
        return {
            "return": float(np.mean([r["team_episode_return"] for r in rows])),
            "win": float(np.mean([r["red_success"] for r in rows])),
            "red_loss": float(np.mean([r["red_losses"] for r in rows])),
            "fire": float(np.mean([r["red_first_fire_window_step"] is not None for r in rows])),
            "kill": float(np.mean([r["red_first_kill_step"] is not None for r in rows])),
        }

    def train_log_line(self) -> str:
        r = self.recent_episode_metrics()
        percent = 100.0 * self.trainer.sampled_steps / self.total_sampled_steps
        return (f"[TRAIN] steps={self.trainer.sampled_steps}/{self.total_sampled_steps} ({percent:.1f}%) "
                f"| eps={len(self.completed_records)} | return={self._display(r['return'], 2)} "
                f"| win={self._display(r['win'], 2)} | red_loss={self._display(r['red_loss'], 2)} "
                f"| fire={self._display(r['fire'], 2)} | kill={self._display(r['kill'], 2)} "
                f"| policy={self._display(self.last_metrics.get('actor_loss'), 4)} "
                f"| value={self._display(self.last_metrics.get('value_loss'), 4)} "
                f"| H={self._display(self.last_metrics.get('entropy'), 2)} "
                f"| KL={self._display(self.last_metrics.get('approx_kl'), 5)} "
                f"| clipfrac={self._display(self.last_metrics.get('clip_fraction'), 3)}")

    @staticmethod
    def evaluation_log_line(r: dict[str, float]) -> str:
        return (f"[EVAL] steps={int(r['sampled_steps'])} | return={r['average_return']:.2f} "
                f"| agent_return={r['average_agent_return']:.2f} | win={r['win_rate']:.2f} "
                f"| red_loss={r['average_red_loss']:.2f} | ep_len={r['average_episode_length']:.1f}")

    @staticmethod
    def checkpoint_log_line(steps: int, path: str | Path) -> str:
        return f"[CKPT] steps={steps} | saved={Path(path).name}"

    @staticmethod
    def done_log_line(s: dict[str, Any]) -> str:
        return (f"[DONE] algorithm=MAPPO | steps={s['sampled_steps']} "
                f"| episodes={s['completed_episodes']} | return={s['average_return']:.2f} "
                f"| win={s['win_rate']:.2f} | red_loss={s['average_red_loss']:.2f}")

    def _completed(self, result) -> list[dict[str, Any]]:
        self.agent_episode_returns += result.rewards
        rows = []
        for index, done in enumerate(result.terminated | result.truncated):
            if done:
                per_agent = self.agent_episode_returns[index].copy()
                team_return, agent_return = episode_return_metrics(per_agent)
                row = {"episode_return": team_return, "team_episode_return": team_return,
                       "mean_agent_episode_return": agent_return,
                       "per_agent_episode_returns": per_agent, **result.infos[index]}
                self.completed_records.append(row); rows.append(row)
                self.agent_episode_returns[index].fill(0.0)
        return rows

    def _write_step_metrics(self, result, rows: list[dict[str, Any]]) -> None:
        mean = lambda key: float(np.mean([r[key] for r in rows])) if rows else None
        rate = lambda predicate: float(np.mean([predicate(r) for r in rows])) if rows else None
        record = {
            "sampled_steps": self.trainer.sampled_steps,
            "mean_step_reward": float(np.mean(result.rewards)),
            **{f"mean_{name}_reward": float(np.mean(np.stack([
                info[f"{name}_rewards"] for info in result.infos
            ]))) for name in ("r1", "r2", "r3", "r4")},
            **{f"{side}_fire_window_pairs": float(np.mean([
                info[f"{side}_fire_window_pairs"] for info in result.infos
            ])) for side in ("red", "blue")},
            **{f"{side}_step_{event}": float(np.mean([
                info[f"{side}_step_{event}"] for info in result.infos
            ])) for side in ("red", "blue") for event in ("fire_attempts", "weapon_hits")},
            "team_episode_return": mean("team_episode_return"),
            "mean_agent_episode_return": mean("mean_agent_episode_return"),
            "win_rate": mean("red_success"), "loss_rate": mean("blue_win"),
            "draw_rate": mean("draw"),
            "timeout_rate": rate(lambda r: r["termination_reason"] == "red_failure_timeout"),
            **{f"{side}_{event}_episode_rate": rate(
                lambda r, field=f"{side}_first_{event}_step": r[field] is not None
            ) for side in ("red", "blue") for event in ("fire_window", "attempt", "hit", "kill")},
            **{key: mean(source) for key, source in {
                "red_uav_losses": "red_losses", "blue_uav_losses": "blue_losses",
                "red_attack_kills": "red_attack_kills", "blue_attack_kills": "blue_attack_kills",
                "red_boundary_exits": "red_boundary_exits", "blue_boundary_exits": "blue_boundary_exits",
                "red_ground_losses": "red_ground_losses", "blue_ground_losses": "blue_ground_losses",
                "episode_length": "episode_length",
            }.items()},
            **{f"episode_{name}_total": mean(f"episode_{name}_total")
               for name in ("r1", "r2", "r3", "r4")},
            **{key: self.last_metrics.get(key) for key in (
                "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                "value", "explained_variance", "actor_grad_norm", "critic_grad_norm")},
        }
        if rows:
            record.update(aggregate_combat_records(rows))
        with (self.output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def collect_rollout(self, vector_steps: int | None = None) -> RolloutBatch:
        storage = {key: [] for key in ("observations", "actions", "raw_actions", "old_log_probs",
                   "rewards", "dones", "alive_masks", "next_observations",
                   "next_alive_masks")}
        for _ in range(int(vector_steps or self.rollout_steps)):
            observations, alive = self.observations.copy(), self.alive_masks.copy()
            actions, raw_actions, log_probs = self.trainer.act(
                observations, alive, return_policy_data=True
            )
            result = self.vector.step_batch(actions)
            storage["observations"].append(observations)
            storage["actions"].append(actions)
            storage["raw_actions"].append(raw_actions)
            storage["old_log_probs"].append(log_probs)
            storage["rewards"].append(result.rewards)
            storage["dones"].append((result.terminated | result.truncated).astype(np.float32))
            storage["alive_masks"].append(result.alive_masks)
            storage["next_observations"].append(result.transition_next_observations)
            storage["next_alive_masks"].append(result.next_alive_masks)
            rows = self._completed(result)
            self.observations, self.alive_masks = result.observations, self.vector.current_alive_masks.copy()
            self.trainer.sampled_steps += self.num_envs; self.trainer.vector_steps += 1
            self._write_step_metrics(result, rows)
        return RolloutBatch(**{key: np.stack(value).astype(np.float32)
                               for key, value in storage.items()})

    def run(self) -> dict[str, Any]:
        try:
            while self.trainer.sampled_steps < self.total_sampled_steps:
                remaining = math.ceil((self.total_sampled_steps - self.trainer.sampled_steps) / self.num_envs)
                self.last_metrics = self.trainer.update(
                    self.collect_rollout(min(self.rollout_steps, remaining))
                )
                update_record = {
                    "sampled_steps": self.trainer.sampled_steps,
                    "rollout_update": self.trainer.ppo_update_count,
                    "actor_learning_rate": float(
                        self.trainer.actor_optimizer.param_groups[0]["lr"]
                    ),
                    "critic_learning_rate": float(
                        self.trainer.critic_optimizer.param_groups[0]["lr"]
                    ),
                    **self.last_metrics,
                }
                with (self.output_dir / "optimization_metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(update_record) + "\n")
                if self.trainer.sampled_steps >= self.next_console_log:
                    print(self.train_log_line(), flush=True)
                    while self.next_console_log <= self.trainer.sampled_steps:
                        self.next_console_log += self.console_interval
                if self.trainer.sampled_steps >= self.next_evaluation:
                    row = self._evaluation_record()
                    self.evaluation_history.append(row); self._write_evaluation()
                    self._consider_best_evaluation(row)
                    print(self.evaluation_log_line(row), flush=True)
                    while self.next_evaluation <= self.trainer.sampled_steps:
                        self.next_evaluation += self.evaluation_interval
                if self.trainer.sampled_steps >= self.next_checkpoint:
                    path = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt"
                    self.save_checkpoint(path); print(self.checkpoint_log_line(self.trainer.sampled_steps, path), flush=True)
                    while self.next_checkpoint <= self.trainer.sampled_steps:
                        self.next_checkpoint += self.checkpoint_interval
            self.save_checkpoint(self.output_dir / "latest.pt")
            return self.summary()
        finally:
            self.vector.close()

    def _evaluation_record(self) -> dict[str, Any]:
        """Evaluate and persist the exact scenario-seed provenance in the row."""
        if not self.evaluation_seeds:
            raise RuntimeError("evaluation seed protocol is empty")
        return {
            "sampled_steps": self.trainer.sampled_steps,
            "evaluation_seed_base": int(self.evaluation_seeds[0]),
            "evaluation_seed_end": int(self.evaluation_seeds[-1]),
            **evaluate(self.trainer, self.env_config, self.evaluation_seeds),
        }

    def _write_evaluation(self) -> None:
        if self.evaluation_history:
            with (self.output_dir / "evaluation_history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.evaluation_history[0]))
                writer.writeheader(); writer.writerows(self.evaluation_history)

    def _evaluation_key(self, record: dict[str, float]) -> tuple[float, ...]:
        return evaluation_selection_key(record)

    def _consider_best_evaluation(self, record: dict[str, float]) -> bool:
        if (
            self.best_evaluation is not None
            and self._evaluation_key(record) <= self._evaluation_key(self.best_evaluation)
        ):
            return False
        self.best_evaluation = dict(record)
        self.save_checkpoint(self.output_dir / "best_eval.pt")
        return True

    def save_checkpoint(self, path: str | Path) -> None:
        self.trainer.save(path, {"environment_version": ENVIRONMENT_VERSION,
            "mappo_impl_version": MAPPO_IMPL_VERSION,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "training_seed": self.seed,
            "training_gamma": self.trainer.gamma,
            "training_num_envs": self.num_envs,
            "training_total_sampled_steps": self.total_sampled_steps,
            "training_smoke": self.smoke,
            "effective_hidden_dim": self.effective_hidden_dim,
            "critic_type":self.trainer.critic_type,
            "network_architecture": {
                "hidden_dim": self.effective_hidden_dim,
                "attention_heads": self.trainer.attention_heads,
                "critic_type": self.trainer.critic_type,
            },
            "actor_parameter_count":sum(p.numel() for p in self.trainer.actor.parameters()),
            "critic_parameter_count":sum(p.numel() for p in self.trainer.critic.parameters()),
            "total_parameter_count":sum(p.numel() for p in self.trainer.actor.parameters())+sum(p.numel() for p in self.trainer.critic.parameters()),
            "environment_config_sha256": config_sha256(self.env_config),
            "algorithm_config_sha256": config_sha256(self.algorithm_config),
            "episode_indices": self.vector.episode_indices.tolist(),
            "evaluation_history": self.evaluation_history,
            "best_evaluation": self.best_evaluation})

    def resume(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_for_resume(
            state, self.env_config, self.algorithm_config
        )
        extra = self.trainer.load(path)
        if extra.get("training_seed") is not None and int(extra["training_seed"]) != self.seed:
            raise RuntimeError("checkpoint training_seed mismatch")
        previous = np.asarray(extra.get("episode_indices", [0] * self.num_envs), dtype=np.int64)
        if previous.shape != (self.num_envs,):
            raise RuntimeError("checkpoint environment count mismatch")
        self.vector.episode_indices = previous + 1
        self.observations = self.vector.reset(); self.alive_masks = self.vector.current_alive_masks.copy()
        self.evaluation_history = list(extra.get("evaluation_history", []))
        best = extra.get("best_evaluation")
        self.best_evaluation = dict(best) if best is not None else None
        self.next_console_log = (self.trainer.sampled_steps // self.console_interval + 1) * self.console_interval
        self.next_evaluation = (self.trainer.sampled_steps // self.evaluation_interval + 1) * self.evaluation_interval
        self.next_checkpoint = (self.trainer.sampled_steps // self.checkpoint_interval + 1) * self.checkpoint_interval

    def summary(self) -> dict[str, Any]:
        mean = lambda key: float(np.mean([r[key] for r in self.completed_records])) if self.completed_records else 0.0
        best = self.best_evaluation or {}
        return {**self.startup_summary(), "sampled_steps": self.trainer.sampled_steps,
            "vector_steps": self.trainer.vector_steps, "rollout_updates": self.trainer.ppo_update_count,
            "actor_updates": self.trainer.actor_update_count, "critic_updates": self.trainer.critic_update_count,
            "completed_episodes": len(self.completed_records), "average_return": mean("episode_return"),
            "average_agent_return": mean("mean_agent_episode_return"), "win_rate": mean("red_success"),
            "loss_rate": mean("blue_win"), "draw_rate": mean("draw"),
            "timeout_rate": float(np.mean([r["termination_reason"] == "red_failure_timeout" for r in self.completed_records])) if self.completed_records else 0.0,
            "average_red_loss": mean("red_losses"), "average_blue_loss": mean("blue_losses"),
            "average_red_attack_kills": mean("red_attack_kills"), "average_blue_attack_kills": mean("blue_attack_kills"),
            "total_red_attack_kills": int(sum(r["red_attack_kills"] for r in self.completed_records)),
            "total_blue_attack_kills": int(sum(r["blue_attack_kills"] for r in self.completed_records)),
            "average_red_boundary_exits": mean("red_boundary_exits"), "average_blue_boundary_exits": mean("blue_boundary_exits"),
            "average_red_ground_losses": mean("red_ground_losses"), "average_blue_ground_losses": mean("blue_ground_losses"),
            "average_episode_length": mean("episode_length"),
            **{f"{side}_{event}_episode_rate": float(np.mean([r[f"{side}_first_{event}_step"] is not None for r in self.completed_records])) if self.completed_records else 0.0
               for side in ("red", "blue") for event in ("fire_window", "attempt", "hit", "kill")},
            **{f"average_episode_{name}_total": mean(f"episode_{name}_total") for name in ("r1", "r2", "r3", "r4")},
            "last_update_metrics": self.last_metrics, "evaluation_history": self.evaluation_history,
            "best_evaluation_steps": best.get("sampled_steps"),
            "best_evaluation_win_rate": best.get("win_rate"),
            "best_evaluation_return": best.get("average_return"),
            "best_evaluation_red_loss": best.get("average_red_loss")}


__all__ = ["MAPPOTrainingRunner"]
