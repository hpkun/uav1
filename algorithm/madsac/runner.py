"""Independent off-policy MADSAC training runner."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.evaluator import episode_return_metrics
from algorithm.common.protocol import config_sha256
from algorithm.common.vector_env import ParallelVectorEnv, VectorStep
from env.config import ENVIRONMENT_VERSION
from .evaluation import evaluate_madsac
from .protocol import validate_madsac_config
from .replay_buffer import JointReplayBuffer
from .trainer import MADSACTrainer, MADSAC_IMPL_VERSION


FUTURE_FINAL_45M = tuple(range(45_000_000, 45_000_200))


def advance_sampled_steps(current: int, num_envs: int) -> int:
    """Count joint environment transitions, never per-agent transitions."""
    return int(current) + int(num_envs)


class MADSACTrainingRunner:
    def __init__(self, env_config: dict, algorithm_config: dict,
                 num_envs=None, total_sampled_steps=None, device=None, seed=None,
                 output_dir=None, smoke=False):
        validate_madsac_config(env_config, algorithm_config)
        self.env_config = deepcopy(env_config)
        self.algorithm_config = deepcopy(algorithm_config)
        self.smoke = bool(smoke)
        network, training, implementation = (
            algorithm_config["network"], algorithm_config["training"],
            algorithm_config["implementation"],
        )
        self.num_envs = int(num_envs or training["num_train_envs"])
        self.total_sampled_steps = int(total_sampled_steps or training["total_sampled_steps"])
        self.device = str(device or training["device"])
        self.seed = int(training["seed"] if seed is None else seed)
        if self.total_sampled_steps % self.num_envs:
            raise ValueError("total_sampled_steps must be divisible by num_train_envs")
        if output_dir is None:
            raise ValueError("output_dir is required")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dim = int(network["observation_dim"])
        self.action_dim = int(network["action_dim"])
        self.num_agents = int(network["num_agents"])
        hidden_dim = 32 if self.smoke else int(network["actor_hidden_layers"][0])
        self.effective_hidden_dim = hidden_dim
        self.trainer = MADSACTrainer(
            self.observation_dim, self.action_dim, self.num_agents, hidden_dim,
            int(network["attention_heads"]), float(training["actor_learning_rate"]),
            float(training["critic_learning_rate"]), float(training["gamma"]),
            float(training["alpha"]), float(training["tau"]),
            int(training["policy_delay"]), self.device, self.seed,
            implementation["actor_activation"], implementation["critic_activation"],
            float(implementation["log_std_min"]), float(implementation["log_std_max"]),
        )
        replay_capacity = min(256, int(training["replay_capacity"])) if self.smoke else int(training["replay_capacity"])
        self.minibatch_size = min(16, int(training["minibatch_size"])) if self.smoke else int(training["minibatch_size"])
        self.learning_starts = min(self.num_envs * 2, int(training["learning_starts"])) if self.smoke else int(training["learning_starts"])
        self.gradient_steps = int(training["gradient_steps_per_vector_step"])
        self.replay = JointReplayBuffer(
            replay_capacity, self.num_agents, self.observation_dim,
            self.action_dim, seed=self.seed ^ 0x31A5,
        )
        evaluation_base = int(implementation["evaluation_seed_base"])
        evaluation_count = 2 if self.smoke else int(training["evaluation_episodes"])
        self.evaluation_seeds = tuple(range(evaluation_base, evaluation_base + evaluation_count))
        self.evaluation_policy_seed = int(implementation["evaluation_policy_seed"])
        self.evaluation_mode = str(implementation["evaluation_mode"])
        forbidden = (*range(44_000_000, 44_000_050), *FUTURE_FINAL_45M)
        if any(self.seed <= value <= self.seed + self.total_sampled_steps for value in forbidden):
            raise ValueError("configured training seed interval overlaps development/future-final seeds")
        self.vector = ParallelVectorEnv(self.num_envs, self.env_config, self.seed, forbidden)
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.agent_episode_returns = np.zeros((self.num_envs, self.num_agents), np.float64)
        self.completed_records: list[dict[str, Any]] = []
        self.last_metrics: dict[str, float] = {}
        self.evaluation_history: list[dict[str, float]] = []
        self.best_evaluation = None
        self.best_sampled_steps = None
        logging = algorithm_config["runtime_logging"]
        self.console_interval = int(logging["console_interval_sampled_steps"])
        self.recent_episode_window = int(logging["recent_episode_window"])
        self.evaluation_interval = int(training["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = int(implementation["checkpoint_interval_sampled_steps"])
        if self.smoke:
            self.evaluation_interval = min(self.evaluation_interval, self.total_sampled_steps)
            self.checkpoint_interval = min(self.checkpoint_interval, self.total_sampled_steps)
        self.next_console = self.console_interval
        self.next_evaluation = self.evaluation_interval
        self.next_checkpoint = self.checkpoint_interval

    def startup_summary(self):
        return {
            "algorithm": "MADSAC", "mode": "smoke" if self.smoke else "formal",
            "device": self.device, "seed": self.seed, "num_envs": self.num_envs,
            "total_sampled_steps": self.total_sampled_steps,
            "observation_dim": self.observation_dim, "action_dim": self.action_dim,
            "num_agents": self.num_agents, "hidden_dim": self.effective_hidden_dim,
            "attention_heads": self.trainer.critic1.attention_heads,
            "gamma": self.trainer.gamma, "alpha": self.trainer.alpha,
            "tau": self.trainer.tau, "policy_delay": self.trainer.policy_delay,
            "learning_starts": self.learning_starts,
            "gradient_steps_per_vector_step": self.gradient_steps,
            "replay_capacity": self.replay.capacity,
            "minibatch_size": self.minibatch_size,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_policy_seed": self.evaluation_policy_seed,
            "environment_variant": self.env_config["environment_variant"],
            "total_waves": int(self.env_config["persistent_waves"]["total_waves"]),
            "max_steps": int(self.env_config["simulation"]["max_steps"]),
            "sampled_steps_unit": "environment_transitions_not_agent_transitions",
            "formal_exact_resume_supported": False,
        }

    def start_log_line(self):
        s = self.startup_summary()
        return (f"[START] algorithm=MADSAC | mode={s['mode']} | device={s['device']} "
                f"| seed={s['seed']} | envs={s['num_envs']} | total={s['total_sampled_steps']} "
                f"| obs={s['observation_dim']} | act={s['action_dim']} | agents={s['num_agents']} "
                f"| hidden={s['hidden_dim']} | heads={s['attention_heads']} | gamma={s['gamma']} "
                f"| alpha={s['alpha']} | tau={s['tau']} | delay={s['policy_delay']} "
                f"| replay={s['replay_capacity']} | batch={s['minibatch_size']} "
                f"| learning_starts={s['learning_starts']} | eval={s['evaluation_mode']}")

    def _completed(self, result: VectorStep):
        self.agent_episode_returns += result.rewards
        rows = []
        for env_id, done in enumerate(result.terminated | result.truncated):
            if done:
                team, agent = episode_return_metrics(self.agent_episode_returns[env_id])
                row = {"episode_return": team, "team_episode_return": team,
                       "mean_agent_episode_return": agent, **result.infos[env_id]}
                self.completed_records.append(row); rows.append(row)
                self.agent_episode_returns[env_id].fill(0.0)
        return rows

    def store_vector_step(self, observations, actions, result: VectorStep):
        done = result.terminated | result.truncated
        self.replay.add_batch(
            observations, actions, result.rewards,
            result.transition_next_observations, done,
            result.alive_masks, result.next_alive_masks,
        )

    def _write_training_metrics(self, result, rows):
        mean = lambda key: None if not rows else float(np.mean([row[key] for row in rows]))
        record = {
            "sampled_steps": self.trainer.sampled_steps,
            "vector_steps": self.trainer.vector_steps,
            "mean_step_reward": float(np.mean(result.rewards)),
            "completed_episodes_this_step": len(rows),
            "team_episode_return": mean("team_episode_return"),
            "average_waves_cleared": mean("waves_cleared"),
            "red_loss": mean("red_losses"), "blue_loss": mean("blue_losses"),
            "red_boundary_exits": mean("red_boundary_exits"),
            "red_ground_losses": mean("red_ground_losses"),
            "replay_size": len(self.replay), **self.last_metrics,
        }
        with (self.output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def _write_optimization(self, metrics):
        record = {"sampled_steps": self.trainer.sampled_steps,
                  "vector_steps": self.trainer.vector_steps,
                  "replay_size": len(self.replay), **metrics}
        with (self.output_dir / "optimization_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def _write_evaluation(self):
        with (self.output_dir / "evaluation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.evaluation_history[0]))
            writer.writeheader(); writer.writerows(self.evaluation_history)

    def _evaluation_key(self, row):
        return evaluation_selection_key(row, self.env_config["environment_variant"])

    def evaluate(self):
        row = {"sampled_steps": self.trainer.sampled_steps,
               "evaluation_mode": self.evaluation_mode,
               "evaluation_policy_seed": self.evaluation_policy_seed,
               **evaluate_madsac(
                   self.trainer, self.env_config, self.evaluation_seeds,
                   self.evaluation_mode, self.evaluation_policy_seed,
               )}
        self.evaluation_history.append(row); self._write_evaluation()
        if self.best_evaluation is None or self._evaluation_key(row) > self._evaluation_key(self.best_evaluation):
            self.best_evaluation = dict(row)
            self.best_sampled_steps = self.trainer.sampled_steps
            self.save_checkpoint(self.output_dir / "best_eval.pt")
        print(f"[EVAL] steps={self.trainer.sampled_steps} | mode={self.evaluation_mode} "
              f"| W1/W2/W3={row['clear_wave_1_probability']:.2f}/"
              f"{row['clear_wave_2_probability']:.2f}/{row['clear_wave_3_probability']:.2f} "
              f"| waves={row['average_waves_cleared']:.2f} | R={row['average_return']:.2f}", flush=True)
        return row

    def checkpoint_extra(self):
        training = self.algorithm_config["training"]
        return {
            "environment_version": ENVIRONMENT_VERSION,
            "environment_variant": self.env_config["environment_variant"],
            "environment_config": deepcopy(self.env_config),
            "algorithm_config": deepcopy(self.algorithm_config),
            "environment_config_sha256": config_sha256(self.env_config),
            "algorithm_config_sha256": config_sha256(self.algorithm_config),
            "observation_dim": self.observation_dim, "action_dim": self.action_dim,
            "num_agents": self.num_agents, "training_seed": self.seed,
            "training_num_envs": self.num_envs,
            "training_total_sampled_steps": self.total_sampled_steps,
            "training_smoke": self.smoke, "gamma": self.trainer.gamma,
            "alpha": self.trainer.alpha, "tau": self.trainer.tau,
            "policy_delay": self.trainer.policy_delay,
            "policy_delay_source": "project_implementation_choice_not_paper_reported",
            "learning_starts": self.learning_starts,
            "gradient_steps_per_vector_step": self.gradient_steps,
            "replay_capacity": self.replay.capacity,
            "minibatch_size": self.minibatch_size,
            "network_architecture": {
                "actor_class": type(self.trainer.actor).__name__,
                "critic_class": type(self.trainer.critic1).__name__,
                "shared_actor": True, "twin_independent_critics": True,
                "target_actor": True, "target_critics": 2,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim, "num_agents": self.num_agents,
                "hidden_dim": self.effective_hidden_dim,
                "attention_heads": self.trainer.critic1.attention_heads,
            },
            "evaluation_mode": self.evaluation_mode,
            "evaluation_seed_base": self.evaluation_seeds[0],
            "evaluation_seed_end": self.evaluation_seeds[-1],
            "evaluation_policy_seed": self.evaluation_policy_seed,
            "actor_policy_gradient": "own_action_only_other_joint_actions_detached",
            "objective_reduction": "per_transition_agent_sum_then_replay_batch_mean",
            "evaluation_policy_rng": "independent_deterministic_stream_per_environment_seed",
            "evaluation_history": deepcopy(self.evaluation_history),
            "best_evaluation": deepcopy(self.best_evaluation),
            "best_sampled_steps": self.best_sampled_steps,
            "episode_indices": self.vector.episode_indices.tolist(),
            "replay_sampling_rng_state": deepcopy(self.replay.rng.bit_generator.state),
            "sampled_steps_unit": "environment_transitions_not_agent_transitions",
            "paper_reported": deepcopy(self.algorithm_config["metadata"]["paper_reported"]),
            "project_implementation_choice": deepcopy(self.algorithm_config["metadata"]["project_implementation_choice"]),
            "future_final_45m_untouched": True,
            "formal_exact_resume_supported": False,
            "replay_buffer_in_checkpoint": False,
        }

    def save_checkpoint(self, path):
        self.trainer.save(path, self.checkpoint_extra())

    def recent_metrics(self):
        rows = self.completed_records[-self.recent_episode_window:]
        return {
            "return": None if not rows else float(np.mean([r["team_episode_return"] for r in rows])),
            "waves": None if not rows else float(np.mean([r.get("waves_cleared", 0) for r in rows])),
        }

    def run(self):
        print(self.start_log_line(), flush=True)
        try:
            while self.trainer.sampled_steps < self.total_sampled_steps:
                observations, alive = self.observations.copy(), self.alive_masks.copy()
                actions = self.trainer.act(observations, alive, deterministic=False)
                result = self.vector.step_batch(actions)
                self.store_vector_step(observations, actions, result)
                rows = self._completed(result)
                self.observations = result.observations
                self.alive_masks = self.vector.current_alive_masks.copy()
                self.trainer.sampled_steps = advance_sampled_steps(
                    self.trainer.sampled_steps, self.num_envs
                )
                self.trainer.vector_steps += 1
                if self.trainer.sampled_steps >= self.learning_starts and len(self.replay) >= self.minibatch_size:
                    for _ in range(self.gradient_steps):
                        self.last_metrics = self.trainer.update(self.replay.sample(self.minibatch_size))
                        self._write_optimization(self.last_metrics)
                self._write_training_metrics(result, rows)
                if self.trainer.sampled_steps >= self.next_console:
                    recent = self.recent_metrics()
                    print(f"[TRAIN] steps={self.trainer.sampled_steps}/{self.total_sampled_steps} "
                          f"| episodes={len(self.completed_records)} | recent_return={recent['return']} "
                          f"| recent_waves={recent['waves']} | replay={len(self.replay)} "
                          f"| critic_updates={self.trainer.critic_update_count} "
                          f"| actor_updates={self.trainer.actor_update_count}", flush=True)
                    while self.next_console <= self.trainer.sampled_steps:
                        self.next_console += self.console_interval
                if self.trainer.sampled_steps >= self.next_evaluation:
                    self.evaluate()
                    while self.next_evaluation <= self.trainer.sampled_steps:
                        self.next_evaluation += self.evaluation_interval
                if self.trainer.sampled_steps >= self.next_checkpoint:
                    path = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt"
                    self.save_checkpoint(path)
                    print(f"[CKPT] steps={self.trainer.sampled_steps} | saved={path.name}", flush=True)
                    while self.next_checkpoint <= self.trainer.sampled_steps:
                        self.next_checkpoint += self.checkpoint_interval
            exact = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt"
            if not exact.exists():
                self.save_checkpoint(exact)
            self.save_checkpoint(self.output_dir / "latest.pt")
            self.save_checkpoint(self.output_dir / "final.pt")
            return self.summary()
        finally:
            self.vector.close()

    def summary(self):
        latest = self.evaluation_history[-1] if self.evaluation_history else None
        return {
            "algorithm": "madsac", "implementation_version": MADSAC_IMPL_VERSION,
            "sampled_steps": self.trainer.sampled_steps,
            "vector_steps": self.trainer.vector_steps,
            "completed_episodes": len(self.completed_records),
            "critic_update_count": self.trainer.critic_update_count,
            "actor_update_count": self.trainer.actor_update_count,
            "replay_size": len(self.replay), "best_checkpoint_step": self.best_sampled_steps,
            "best_evaluation": self.best_evaluation, "latest_evaluation": latest,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_policy_seed": self.evaluation_policy_seed,
            "sampled_steps_unit": "environment_transitions_not_agent_transitions",
            "formal_exact_resume_supported": False,
            "replay_buffer_in_checkpoint": False,
            "future_final_45m_untouched": True,
            "protocol": self.startup_summary(),
            "final_optimization_metrics": deepcopy(self.last_metrics),
        }


__all__ = ["FUTURE_FINAL_45M", "MADSACTrainingRunner", "advance_sampled_steps"]
