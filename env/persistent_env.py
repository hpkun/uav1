"""Minimal persistent-wave variant of the frozen V2.3 combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy
import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState
from .combat_env import MultiUAVCombatEnv
from .fixed_policy import GroundAwareNearestTargetPursuitPolicy
from .weapon import FireState


PERSISTENT_WAVE_VARIANT = "persistent_wave_v1"
PERSISTENT_WAVE_V2_VARIANT = "persistent_wave_v2"
PERSISTENT_WAVE_VARIANTS = frozenset({
    PERSISTENT_WAVE_VARIANT, PERSISTENT_WAVE_V2_VARIANT,
})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSISTENT_CONFIG = PROJECT_ROOT / "configs/persistent_wave_environment.yaml"
CURRICULUM_SNAPSHOT_VERSION = 1


class PersistentWaveCombatEnv(MultiUAVCombatEnv):
    """Keep Red state and immediately replace each defeated Blue wave."""

    environment_variant = PERSISTENT_WAVE_VARIANT

    def __init__(self, config: Any = DEFAULT_PERSISTENT_CONFIG) -> None:
        super().__init__(config)
        variant = self.config.get("environment_variant")
        if variant not in PERSISTENT_WAVE_VARIANTS:
            raise ValueError(
                f"environment_variant must be one of {sorted(PERSISTENT_WAVE_VARIANTS)!r}"
            )
        self.environment_variant = str(variant)
        if self.environment_variant == PERSISTENT_WAVE_V2_VARIANT:
            self.fixed_policy = GroundAwareNearestTargetPursuitPolicy(
                self.config["blue_policy"], self.config["action"],
                self.config["aircraft"],
            )
        wave_config = self.config.get("persistent_waves")
        if not isinstance(wave_config, dict):
            raise ValueError("persistent_waves configuration is required")
        self.total_waves = int(wave_config["total_waves"])
        self.spawn_radius = float(wave_config["spawn_radius"])
        self.spawn_direction_count = int(wave_config["spawn_direction_count"])
        if self.total_waves < 1:
            raise ValueError("total_waves must be positive")
        if not 0.0 < self.spawn_radius < self.arena_radius:
            raise ValueError("spawn_radius must be inside the arena")
        if self.spawn_direction_count < 1:
            raise ValueError("spawn_direction_count must be positive")
        configured_blue_counts = wave_config.get("blue_units_per_wave")
        if configured_blue_counts is None:
            self.blue_units_per_wave = [self.team_size] * self.total_waves
        else:
            if not isinstance(configured_blue_counts, (list, tuple)):
                raise ValueError("blue_units_per_wave must be a sequence")
            self.blue_units_per_wave = [int(value) for value in configured_blue_counts]
        if len(self.blue_units_per_wave) != self.total_waves:
            raise ValueError("blue_units_per_wave length must equal total_waves")
        if any(count < 1 or count > self.team_size for count in self.blue_units_per_wave):
            raise ValueError("blue_units_per_wave values must be in [1, team_size]")
        if self.blue_units_per_wave[0] != self.team_size:
            raise ValueError("the first persistent wave must use all Blue slots")
        maximum_offset = max(map(abs, self.config["scenario"]["formation_offsets"]))
        if np.hypot(self.spawn_radius, maximum_offset) >= self.arena_radius:
            raise ValueError("configured Blue formation does not fit inside arena")
        self.wave_index = 1
        self.waves_cleared = 0
        self.last_spawn_candidate_index: int | None = None
        self.last_minimum_spawn_distance: float | None = None
        self.wave_records: list[dict[str, Any]] = []
        self._wave_start_step = 0
        self._wave_start_red_survivors = self.team_size
        self._wave_start_blue_survivors = self.team_size
        self._wave_start_counts: dict[str, dict[str, int]] = {}
        self._wave_start_rewards: dict[str, float] = {}
        self._wave_record_open = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed)
        if isinstance(self.fixed_policy, GroundAwareNearestTargetPursuitPolicy):
            self.fixed_policy.reset_diagnostics()
        self.wave_index = 1
        self.waves_cleared = 0
        self.last_spawn_candidate_index = None
        self.last_minimum_spawn_distance = None
        self.wave_records = []
        self._begin_wave_record()
        info.update(self._wave_info(False, False, None))
        return observation, info

    def _curriculum_metadata(self) -> dict[str, Any]:
        return {
            "snapshot_version": CURRICULUM_SNAPSHOT_VERSION,
            "environment_version": self.environment_version,
            "environment_variant": self.environment_variant,
            "wave_index": int(self.wave_index),
            "waves_cleared": int(self.waves_cleared),
            "steps": int(self.steps),
            "total_waves": int(self.total_waves),
            "max_steps": int(self.max_steps),
            "observation_dim": int(self.observation_dim),
            "action_dim": int(self.action_dim),
            "team_size": int(self.team_size),
        }

    def export_curriculum_state(self) -> dict[str, Any]:
        """Export one exact, legal post-spawn W2/W3 entry state."""
        if self.wave_index not in (2, 3):
            raise RuntimeError("curriculum snapshots are only legal at W2/W3 entry")
        if self.waves_cleared != self.wave_index - 1:
            raise RuntimeError("curriculum snapshot has inconsistent wave progress")
        if self.steps != self._wave_start_step:
            raise RuntimeError("curriculum snapshot must be exported immediately post-spawn")
        if self.steps >= self.max_steps or not self.red_alive_mask.any():
            raise RuntimeError("curriculum snapshot cannot be terminal")
        if int(self.blue_alive_mask.sum()) != self.blue_units_per_wave[self.wave_index - 1]:
            raise RuntimeError("curriculum snapshot has the wrong active Blue force size")
        fixed_policy_state = None
        if isinstance(self.fixed_policy, GroundAwareNearestTargetPursuitPolicy):
            fixed_policy_state = {
                "total_decision_steps": int(self.fixed_policy.total_decision_steps),
                "override_steps": int(self.fixed_policy.override_steps),
                "activation_count": int(self.fixed_policy.activation_count),
                "maximum_activation_duration_steps": int(
                    self.fixed_policy.maximum_activation_duration_steps
                ),
                "_previous_override": self.fixed_policy._previous_override.copy(),
                "_current_duration": self.fixed_policy._current_duration.copy(),
                "last_override_mask": self.fixed_policy.last_override_mask.copy(),
            }
        return deepcopy({
            "metadata": self._curriculum_metadata(),
            "red": [state.copy() for state in self.red],
            "blue": [state.copy() for state in self.blue],
            "red_fire_states": [FireState(state.armed) for state in self.red_fire_states],
            "blue_fire_states": [FireState(state.armed) for state in self.blue_fire_states],
            "red_last_executed_phi": self.red_last_executed_phi.copy(),
            "blue_last_executed_phi": self.blue_last_executed_phi.copy(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "steps": int(self.steps),
            "combat_counts": deepcopy(self.combat_counts),
            "first_steps": deepcopy(self.first_steps),
            "episode_reward_components": {
                key: value.copy() for key, value in self.episode_reward_components.items()
            },
            "wave_index": int(self.wave_index),
            "waves_cleared": int(self.waves_cleared),
            "last_spawn_candidate_index": self.last_spawn_candidate_index,
            "last_minimum_spawn_distance": self.last_minimum_spawn_distance,
            "wave_records": deepcopy(self.wave_records),
            "_wave_start_step": int(self._wave_start_step),
            "_wave_start_red_survivors": int(self._wave_start_red_survivors),
            "_wave_start_blue_survivors": int(self._wave_start_blue_survivors),
            "_wave_start_counts": deepcopy(self._wave_start_counts),
            "_wave_start_rewards": deepcopy(self._wave_start_rewards),
            "_wave_record_open": bool(self._wave_record_open),
            "fixed_policy_state": fixed_policy_state,
        })

    def restore_curriculum_state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Restore a validated dynamic state without changing task semantics."""
        state = deepcopy(snapshot)
        metadata = state.get("metadata", {})
        expected = self._curriculum_metadata()
        for key in (
            "snapshot_version", "environment_version", "environment_variant",
            "total_waves", "max_steps", "observation_dim", "action_dim", "team_size",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"curriculum snapshot {key} mismatch: "
                    f"expected {expected[key]!r}, got {metadata.get(key)!r}"
                )
        wave = int(metadata.get("wave_index", 0))
        cleared = int(metadata.get("waves_cleared", -1))
        steps = int(metadata.get("steps", -1))
        if wave not in (2, 3) or cleared != wave - 1:
            raise ValueError("curriculum snapshot is not a legal W2/W3 entry")
        if not 0 <= steps < self.max_steps:
            raise ValueError("curriculum snapshot has invalid global step")
        if len(state.get("red", [])) != self.team_size or len(state.get("blue", [])) != self.team_size:
            raise ValueError("curriculum snapshot team size mismatch")
        self.red = [item.copy() for item in state["red"]]
        self.blue = [item.copy() for item in state["blue"]]
        self.red_fire_states = [FireState(item.armed) for item in state["red_fire_states"]]
        self.blue_fire_states = [FireState(item.armed) for item in state["blue_fire_states"]]
        self.red_last_executed_phi = np.asarray(
            state["red_last_executed_phi"], dtype=np.float32
        ).copy()
        self.blue_last_executed_phi = np.asarray(
            state["blue_last_executed_phi"], dtype=np.float32
        ).copy()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self.steps = int(state["steps"])
        self.combat_counts = deepcopy(state["combat_counts"])
        self.first_steps = deepcopy(state["first_steps"])
        self.episode_reward_components = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in state["episode_reward_components"].items()
        }
        self.wave_index = int(state["wave_index"])
        self.waves_cleared = int(state["waves_cleared"])
        self.last_spawn_candidate_index = state["last_spawn_candidate_index"]
        self.last_minimum_spawn_distance = state["last_minimum_spawn_distance"]
        self.wave_records = deepcopy(state["wave_records"])
        self._wave_start_step = int(state["_wave_start_step"])
        self._wave_start_red_survivors = int(state["_wave_start_red_survivors"])
        self._wave_start_blue_survivors = int(
            state.get("_wave_start_blue_survivors", self.blue_units_per_wave[self.wave_index - 1])
        )
        self._wave_start_counts = deepcopy(state["_wave_start_counts"])
        self._wave_start_rewards = deepcopy(state["_wave_start_rewards"])
        self._wave_record_open = bool(state["_wave_record_open"])
        fixed = state.get("fixed_policy_state")
        if isinstance(self.fixed_policy, GroundAwareNearestTargetPursuitPolicy):
            if not isinstance(fixed, dict):
                raise ValueError("curriculum snapshot is missing fixed-policy state")
            for key in (
                "total_decision_steps", "override_steps", "activation_count",
                "maximum_activation_duration_steps",
            ):
                setattr(self.fixed_policy, key, int(fixed[key]))
            for key in ("_previous_override", "_current_duration", "last_override_mask"):
                setattr(self.fixed_policy, key, np.asarray(fixed[key]).copy())
        if self.steps != self._wave_start_step or not self.red_alive_mask.any():
            raise ValueError("restored curriculum state is not a live post-spawn entry")
        if int(self.blue_alive_mask.sum()) != self.blue_units_per_wave[self.wave_index - 1]:
            raise ValueError("restored curriculum state has the wrong active Blue force size")
        return {
            "observation": self._observations().copy(),
            "red_alive_mask": self.red_alive_mask.copy(),
            "blue_alive_mask": self.blue_alive_mask.copy(),
            "wave_index": int(self.wave_index),
            "waves_cleared": int(self.waves_cleared),
            "total_waves": int(self.total_waves),
            "steps": int(self.steps),
            "combat_counts": deepcopy(self.combat_counts),
        }

    def _begin_wave_record(self) -> None:
        self._wave_record_open = True
        self._wave_start_step = self.steps
        self._wave_start_red_survivors = int(self.red_alive_mask.sum())
        self._wave_start_blue_survivors = int(self.blue_alive_mask.sum())
        self._wave_start_counts = {
            side: dict(counts) for side, counts in self.combat_counts.items()
        }
        self._wave_start_rewards = {
            name: float(values.sum())
            for name, values in self.episode_reward_components.items()
        }

    def _finish_wave_record(
        self, wave_cleared: bool, termination_reason: str
    ) -> None:
        if not self._wave_record_open:
            return
        record: dict[str, Any] = {
            "wave_index": self.wave_index,
            "start_step": self._wave_start_step,
            "end_step": self.steps,
            "duration_steps": self.steps - self._wave_start_step,
            "red_survivors_start": self._wave_start_red_survivors,
            "red_survivors_end": int(self.red_alive_mask.sum()),
            "blue_survivors_start": self._wave_start_blue_survivors,
            "blue_survivors_end": int(self.blue_alive_mask.sum()),
            "wave_completed": True,
            "wave_cleared": bool(wave_cleared),
            "termination_reason": termination_reason,
        }
        for side in ("red", "blue"):
            for event in (
                "fire_attempts", "weapon_hits", "attack_kills",
                "boundary_exits", "ground_losses",
            ):
                record[f"{side}_{event}"] = (
                    self.combat_counts[side][event]
                    - self._wave_start_counts[side][event]
                )
        team_return = 0.0
        for name, values in self.episode_reward_components.items():
            value = float(values.sum()) - self._wave_start_rewards[name]
            record[f"{name}_total"] = value
            team_return += value
        record["team_return"] = team_return
        self.wave_records.append(record)
        self._wave_record_open = False

    def _candidate_blue_wave(self, radial_angle: float) -> list[AircraftState]:
        scenario = self.config["scenario"]
        radial = np.array([np.cos(radial_angle), np.sin(radial_angle)])
        lateral = np.array([-radial[1], radial[0]])
        center = self.spawn_radius * radial
        alive_red = [state for state in self.red if state.alive]
        red_center = np.mean([[state.x, state.y] for state in alive_red], axis=0)
        nominal_heading = float(np.arctan2(
            red_center[1] - center[1], red_center[0] - center[0]
        ))
        states = []
        for offset in scenario["formation_offsets"]:
            position = center + float(offset) * lateral
            states.append(AircraftState(
                x=float(position[0]),
                y=float(position[1]),
                z=-float(scenario["altitude_center"] + self.rng.uniform(
                    -scenario["altitude_perturbation_max"],
                    scenario["altitude_perturbation_max"],
                )),
                v=float(scenario["speed_center"] + self.rng.uniform(
                    -scenario["speed_perturbation_max"],
                    scenario["speed_perturbation_max"],
                )),
                theta=0.0,
                psi=float(wrap_angle(nominal_heading + self.rng.uniform(
                    -scenario["heading_perturbation_max"],
                    scenario["heading_perturbation_max"],
                ))),
            ))
        return states

    def _blue_wave_inside_arena(self, states: list[AircraftState]) -> bool:
        return bool(
            len(states) == self.team_size
            and all(np.hypot(state.x, state.y) < self.arena_radius for state in states)
        )

    def _minimum_red_blue_distance(self, states: list[AircraftState]) -> float:
        alive_red = [state for state in self.red if state.alive]
        if len(states) != self.team_size or not alive_red:
            raise RuntimeError("minimum spawn distance requires surviving Red")
        return min(
            float(np.linalg.norm(np.array([
                blue.x - red.x, blue.y - red.y, blue.z - red.z,
            ])))
            for blue in states for red in alive_red
        )

    def _spawn_next_wave(self) -> float:
        alive_red = [state for state in self.red if state.alive]
        if not alive_red:
            raise RuntimeError("cannot spawn a wave after Red elimination")
        best_index = 0
        best_angle = -np.pi
        best_distance = -np.inf
        best_candidate: list[AircraftState] | None = None
        for index, radial_angle in enumerate(np.linspace(
            -np.pi, np.pi, self.spawn_direction_count, endpoint=False
        )):
            radial_angle = float(radial_angle)
            candidate = self._candidate_blue_wave(radial_angle)
            distance = self._minimum_red_blue_distance(candidate)
            if distance > best_distance:
                best_index = index
                best_angle = radial_angle
                best_distance = distance
                best_candidate = candidate
        # __init__ proves every enumerated formation fits inside the arena.
        assert best_candidate is not None and self._blue_wave_inside_arena(best_candidate)
        # Candidate generation/search is always performed with the legacy full
        # four-slot geometry.  Only after the winning candidate is fixed do we
        # deterministically deactivate placeholder slots, consuming no RNG.
        next_wave_active_count = self.blue_units_per_wave[self.wave_index]
        if next_wave_active_count < self.team_size:
            inactive_count = self.team_size - next_wave_active_count
            for offset in range(inactive_count):
                best_candidate[(best_index + offset) % self.team_size].alive = False
        self.blue = best_candidate
        self.red_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_last_executed_phi.fill(0.0)
        if isinstance(self.fixed_policy, GroundAwareNearestTargetPursuitPolicy):
            self.fixed_policy.reset_transient_state()
        self.last_spawn_candidate_index = best_index
        self.last_minimum_spawn_distance = best_distance
        return best_angle

    def _wave_info(
        self,
        wave_cleared: bool,
        spawned_next_wave: bool,
        spawn_radial_angle: float | None,
    ) -> dict[str, Any]:
        result = {
            "environment_variant": self.environment_variant,
            "wave_index": self.wave_index,
            "total_waves": self.total_waves,
            "blue_units_per_wave": list(self.blue_units_per_wave),
            "waves_cleared": self.waves_cleared,
            "wave_cleared_this_step": wave_cleared,
            "spawned_next_wave": spawned_next_wave,
            "wave_spawn_radial_angle": spawn_radial_angle,
            "wave_spawn_candidate_index": (
                self.last_spawn_candidate_index if spawned_next_wave else None
            ),
            "minimum_spawn_distance": (
                self.last_minimum_spawn_distance if spawned_next_wave else None
            ),
            "per_wave_metrics": [dict(record) for record in self.wave_records],
        }
        if isinstance(self.fixed_policy, GroundAwareNearestTargetPursuitPolicy):
            result.update(self.fixed_policy.diagnostics())
            result["blue_ground_guard_override_mask"] = (
                self.fixed_policy.last_override_mask.copy()
            )
        return result

    def step(
        self, red_actions: np.ndarray, blue_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(
            red_actions, blue_actions
        )
        red_survivors = int(self.red_alive_mask.sum())
        wave_cleared = bool(red_survivors > 0 and self.blue_alive_mask.sum() == 0)
        spawned_next_wave = False
        spawn_radial_angle = None

        if wave_cleared:
            if self.wave_index < self.total_waves and self.steps >= self.max_steps:
                clear_reason = "red_failure_timeout"
            elif self.wave_index == self.total_waves:
                clear_reason = str(info["termination_reason"])
            else:
                clear_reason = "wave_cleared"
            self._finish_wave_record(True, clear_reason)
            self.waves_cleared += 1
            if self.wave_index < self.total_waves:
                if self.steps >= self.max_steps:
                    terminated = False
                    truncated = True
                    info.update({
                        "red_success": False,
                        "red_win": False,
                        "blue_win": False,
                        "draw": False,
                        "termination_reason": "red_failure_timeout",
                    })
                else:
                    spawn_radial_angle = self._spawn_next_wave()
                    self.wave_index += 1
                    self._begin_wave_record()
                    spawned_next_wave = True
                    terminated = False
                    truncated = False
                    observation = self._observations()
                    info.update({
                        "red_success": False,
                        "red_win": False,
                        "blue_win": False,
                        "draw": False,
                        "termination_reason": "ongoing",
                    })

        if (terminated or truncated) and self._wave_record_open:
            self._finish_wave_record(False, str(info["termination_reason"]))

        completed_blue_losses = sum(self.blue_units_per_wave[:self.waves_cleared])
        current_active_count = self.blue_units_per_wave[self.wave_index - 1]
        current_blue_losses = current_active_count - int(self.blue_alive_mask.sum())
        info["blue_losses"] = (
            completed_blue_losses
            if wave_cleared
            else completed_blue_losses + current_blue_losses
        )
        # super().step() built info before an intermediate replacement. Always
        # overwrite state-derived fields so returned masks match observation.
        info.update({
            "red_survivors": int(self.red_alive_mask.sum()),
            "blue_survivors": int(self.blue_alive_mask.sum()),
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        })
        info.update(self._wave_info(
            wave_cleared, spawned_next_wave, spawn_radial_angle
        ))
        return observation, reward, terminated, truncated, info


__all__ = [
    "CURRICULUM_SNAPSHOT_VERSION",
    "PERSISTENT_WAVE_VARIANT", "PERSISTENT_WAVE_V2_VARIANT",
    "PERSISTENT_WAVE_VARIANTS", "PersistentWaveCombatEnv",
]
