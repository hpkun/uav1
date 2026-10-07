"""Synchronous batch API backed by persistent parallel environment workers."""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any
import numpy as np

from env.process_worker import combat_environment_worker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMBAT_CONFIG = PROJECT_ROOT / "configs/combat_environment.yaml"


@dataclass
class VectorStep:
    observations: np.ndarray
    transition_next_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


class ParallelVectorEnv:
    """One persistent spawn subprocess per environment with synchronous batches."""

    backend = "multiprocess_spawn"

    def __init__(
        self,
        num_envs: int,
        config: Any = DEFAULT_COMBAT_CONFIG,
        base_seed: int = 0,
        forbidden_seeds=(),
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.forbidden_seeds = set(map(int, forbidden_seeds))
        self.used_training_seeds: set[int] = set()
        self.episode_indices = np.zeros(num_envs, dtype=np.int64)
        self.current_alive_masks = np.ones((num_envs, 4), dtype=np.float32)
        self.last_reset_seeds = np.zeros(num_envs, dtype=np.int64)
        self._closed = False
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        try:
            for env_id in range(self.num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=combat_environment_worker,
                    args=(child, config),
                    name=f"uav-env-{env_id}",
                    daemon=True,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self.worker_metadata = [
                dict(self._receive(connection, "startup"))
                for connection in self._connections
            ]
            self.worker_pids = [int(row["pid"]) for row in self.worker_metadata]
            self.worker_environment_classes = [
                str(row["environment_class"]) for row in self.worker_metadata
            ]
            self.worker_environment_variants = [
                str(row["environment_variant"]) for row in self.worker_metadata
            ]
            self.worker_fixed_policy_classes = [
                str(row["fixed_policy_class"]) for row in self.worker_metadata
            ]
            dimensions = {
                (int(row["observation_dim"]), int(row["action_dim"]), int(row["team_size"]))
                for row in self.worker_metadata
            }
            if len(dimensions) != 1:
                raise RuntimeError(f"environment workers disagree on dimensions: {sorted(dimensions)}")
            self.observation_dim, self.action_dim, self.team_size = dimensions.pop()
            self.current_observations = np.zeros(
                (num_envs, self.team_size, self.observation_dim), dtype=np.float32
            )
        except BaseException:
            self.close()
            raise

    @property
    def num_workers(self) -> int:
        return len(self._processes)

    @staticmethod
    def _receive(connection: Connection, operation: str) -> Any:
        try:
            status, payload = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"environment worker exited during {operation}") from error
        if status == "error":
            raise RuntimeError(f"environment worker failed during {operation}:\n{payload}")
        if operation == "startup" and status != "ready":
            raise RuntimeError(f"invalid environment-worker startup response: {status}")
        if operation != "startup" and status != "ok":
            raise RuntimeError(f"invalid environment-worker response: {status}")
        return payload

    def seed_for(self, env_id: int, episode_index: int) -> int:
        return self.base_seed + episode_index * self.num_envs + env_id

    def _reset_seed(self, env_id: int) -> int:
        seed = self.seed_for(env_id, int(self.episode_indices[env_id]))
        if seed in self.forbidden_seeds:
            raise RuntimeError(f"training seed overlaps evaluation set: {seed}")
        if seed in self.used_training_seeds:
            raise RuntimeError(f"training seed reused: {seed}")
        self.used_training_seeds.add(seed)
        self.last_reset_seeds[env_id] = seed
        return seed

    def _reset_many(self, env_ids: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        for env_id in env_ids:
            self._connections[env_id].send(("reset", self._reset_seed(env_id)))
        return [
            self._receive(self._connections[env_id], "reset") for env_id in env_ids
        ]

    def reset(self) -> np.ndarray:
        self._ensure_open()
        results = self._reset_many(list(range(self.num_envs)))
        self.current_observations = self._validate_observations(
            np.stack([result[0] for result in results]), "reset"
        )
        self.current_alive_masks = np.stack([result[1] for result in results])
        return self.current_observations.copy()

    def step_batch(self, actions: np.ndarray, auto_reset: bool = True) -> VectorStep:
        self._ensure_open()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.team_size, self.action_dim):
            raise ValueError(
                f"vector actions must be [env, {self.team_size}, {self.action_dim}]"
            )
        alive_before = self.current_alive_masks.copy()
        for env_id, connection in enumerate(self._connections):
            connection.send(("step", actions[env_id]))
        results = [
            self._receive(connection, "step") for connection in self._connections
        ]
        transition_next = self._validate_observations(
            np.stack([result[0] for result in results]), "step"
        )
        rewards = np.stack([result[1] for result in results])
        terminated = np.asarray([result[2] for result in results])
        truncated = np.asarray([result[3] for result in results])
        infos = [result[4] for result in results]
        next_alive = np.stack([info["red_alive_mask"] for info in infos])
        current, current_alive = transition_next.copy(), next_alive.copy()
        if auto_reset:
            done_ids = np.flatnonzero(terminated | truncated).tolist()
            for env_id in done_ids:
                self.episode_indices[env_id] += 1
            for env_id, reset_result in zip(done_ids, self._reset_many(done_ids)):
                current[env_id], current_alive[env_id] = reset_result
        self.current_observations = current
        self.current_alive_masks = current_alive
        return VectorStep(
            current.copy(), transition_next, rewards, terminated, truncated,
            infos, alive_before, next_alive,
        )

    def _validate_observations(self, values: np.ndarray, operation: str) -> np.ndarray:
        expected = (self.num_envs, self.team_size, self.observation_dim)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"invalid {operation} observations: expected finite {expected}, got {values.shape}"
            )
        return values

    def step(self, actions: np.ndarray):
        result = self.step_batch(actions)
        return result.observations, result.rewards, result.terminated, result.truncated, result.infos

    def export_curriculum_states(self, env_ids) -> dict[int, dict[str, Any]]:
        """Export explicit dynamic-state snapshots from selected live workers."""
        self._ensure_open()
        ids = [int(value) for value in env_ids]
        if len(set(ids)) != len(ids) or any(value < 0 or value >= self.num_envs for value in ids):
            raise ValueError("invalid or duplicate curriculum export env id")
        for env_id in ids:
            self._connections[env_id].send(("export_curriculum_state", None))
        return {
            env_id: self._receive(self._connections[env_id], "export_curriculum_state")
            for env_id in ids
        }

    def restore_curriculum_states(
        self, snapshots: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Restore selected workers and synchronize vector-side live state."""
        self._ensure_open()
        ids = [int(value) for value in snapshots]
        if len(set(ids)) != len(ids) or any(value < 0 or value >= self.num_envs for value in ids):
            raise ValueError("invalid or duplicate curriculum restore env id")
        for env_id in ids:
            self._connections[env_id].send(
                ("restore_curriculum_state", snapshots[env_id])
            )
        results = {
            env_id: dict(self._receive(
                self._connections[env_id], "restore_curriculum_state"
            )) for env_id in ids
        }
        for env_id, result in results.items():
            observation = np.asarray(result["observation"], dtype=np.float32)
            expected = (self.team_size, self.observation_dim)
            if observation.shape != expected or not np.all(np.isfinite(observation)):
                raise RuntimeError("worker restored an invalid curriculum observation")
            self.current_observations[env_id] = observation
            self.current_alive_masks[env_id] = np.asarray(
                result["red_alive_mask"], dtype=np.float32
            )
        return results

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("parallel vector environment is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    self._receive(connection, "close")
                except RuntimeError:
                    pass
            connection.close()
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def __enter__(self) -> "ParallelVectorEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


# Compatibility import for external callers; the implementation is parallel.
SyncVectorEnv = ParallelVectorEnv

__all__ = ["ParallelVectorEnv", "SyncVectorEnv", "VectorStep"]
