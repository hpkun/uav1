"""Paper-Constrained Direct 4v4 Combat Environment V2.3."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .config import ENVIRONMENT_VERSION, aircraft_spec, load_config
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .models import AircraftState
from .control import action_to_control
from .fixed_policy import NearestTargetPursuitPolicy
from .geometry import engagement_geometry
from .observation import OBSERVATION_DIM, build_team_observations, observation_dim_from_config
from .reward import paper_state_reward_components
from .scenario import random_combat_states
from .weapon import FireState, WeaponEnvelope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBAT_CONFIG = PROJECT_ROOT / "configs/combat_environment.yaml"


class MultiUAVCombatEnv:
    """Four learned Red UAVs versus deterministic nearest-target Blue UAVs."""

    team_size, observation_dim, action_dim = 4, OBSERVATION_DIM, 3
    environment_version = ENVIRONMENT_VERSION

    def __init__(
        self, config: str | Path | dict[str, Any] = DEFAULT_COMBAT_CONFIG
    ) -> None:
        self.config = load_config(config) if not isinstance(config, dict) else config
        if str(self.config.get("environment_version")) != ENVIRONMENT_VERSION:
            raise ValueError(f"environment_version must be {ENVIRONMENT_VERSION}")
        self.spec = aircraft_spec(self.config)
        self.observation_dim = observation_dim_from_config(self.config["observation"])
        self.dt = float(self.config["simulation"]["dt"])
        self.max_steps = int(self.config["simulation"]["max_steps"])
        self.arena_radius = float(self.config["arena"]["radius"])
        self.dynamics = PointMassDynamics()
        self.integrator = RK4Integrator(self.dt)
        self.fixed_policy = NearestTargetPursuitPolicy(
            self.config["blue_policy"], self.config["action"]
        )
        self.weapon = WeaponEnvelope(**self.config["weapon"])
        self.rng = np.random.default_rng()
        self.red: list[AircraftState] = []
        self.blue: list[AircraftState] = []
        self.red_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_fire_states = [FireState() for _ in range(self.team_size)]
        self.red_last_executed_phi = np.zeros(self.team_size, dtype=np.float32)
        self.blue_last_executed_phi = np.zeros(self.team_size, dtype=np.float32)
        self.steps = 0
        self.combat_counts: dict[str, dict[str, int]] = {}
        self.first_steps: dict[str, dict[str, int | None]] = {}
        self.episode_reward_components: dict[str, np.ndarray] = {}

    @property
    def red_alive_mask(self) -> np.ndarray:
        return np.asarray([state.alive for state in self.red], dtype=np.float32)

    @property
    def blue_alive_mask(self) -> np.ndarray:
        return np.asarray([state.alive for state in self.blue], dtype=np.float32)

    def _reset_metrics(self) -> None:
        events = (
            "fire_window_steps", "fire_window_pair_steps", "fire_attempts",
            "weapon_hits", "attack_kills", "boundary_exits", "ground_losses",
        )
        self.combat_counts = {
            side: {event: 0 for event in events} for side in ("red", "blue")
        }
        self.first_steps = {
            side: {event: None for event in ("fire_window", "attempt", "hit", "kill")}
            for side in ("red", "blue")
        }
        self.episode_reward_components = {
            name: np.zeros(self.team_size, dtype=np.float64)
            for name in ("r1", "r2", "r3", "r4")
        }

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, radial_angle = random_combat_states(
            self.rng, **self.config["scenario"]
        )
        self.red_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_fire_states = [FireState() for _ in range(self.team_size)]
        self.red_last_executed_phi.fill(0.0)
        self.blue_last_executed_phi.fill(0.0)
        self.steps = 0
        self._reset_metrics()
        return self._observations(), {
            "environment_version": ENVIRONMENT_VERSION,
            "radial_angle": radial_angle,
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }

    def _observations(self) -> np.ndarray:
        return build_team_observations(
            self.red, self.blue, self.config["observation"],
            self.red_last_executed_phi,
            np.asarray([state.armed for state in self.red_fire_states], dtype=np.float32)
            if self.observation_dim > OBSERVATION_DIM else None,
        )

    @staticmethod
    def _snapshot(states: list[AircraftState]) -> list[AircraftState]:
        return [state.copy() for state in states]

    def _advance(self, states: list[AircraftState], actions: np.ndarray) -> np.ndarray:
        phis = np.zeros(self.team_size, dtype=np.float32)
        for index, state in enumerate(states):
            if state.alive:
                control = action_to_control(state, actions[index], self.config["action"])
                phis[index] = control.phi
                states[index] = self.integrator.step(
                    state, control, self.dynamics, self.spec
                )
        return phis

    def _resolve_noncombat_losses(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        results: list[list[int]] = [[], [], [], []]
        for side_index, (side, states) in enumerate((
            ("red", self.red), ("blue", self.blue)
        )):
            exits = results[side_index]
            ground = results[side_index + 2]
            for index, state in enumerate(states):
                if state.alive and np.hypot(state.x, state.y) > self.arena_radius:
                    state.alive = False
                    exits.append(index)
                elif state.alive and state.altitude <= 0.0:
                    state.alive = False
                    ground.append(index)
            self.combat_counts[side]["boundary_exits"] += len(exits)
            self.combat_counts[side]["ground_losses"] += len(ground)
        return results[0], results[1], results[2], results[3]

    def _in_fire_window(self, attacker: AircraftState, target: AircraftState) -> bool:
        return bool(
            attacker.alive and target.alive
            and self.weapon.in_fire_window(engagement_geometry(attacker, target))
        )

    def _window_pair_count(
        self, attackers: list[AircraftState], targets: list[AircraftState]
    ) -> int:
        return sum(
            self._in_fire_window(attacker, target)
            for attacker in attackers for target in targets
        )

    def _select_target(
        self, attacker: AircraftState, targets: list[AircraftState]
    ) -> int | None:
        candidates = [
            (engagement_geometry(attacker, target).distance, target_index)
            for target_index, target in enumerate(targets)
            if self._in_fire_window(attacker, target)
        ]
        return min(candidates)[1] if candidates else None

    def _entry_attempts(
        self,
        attackers: list[AircraftState],
        targets: list[AircraftState],
        fire_states: list[FireState],
        side: str,
    ) -> list[tuple[int, int, bool]]:
        attempts: list[tuple[int, int, bool]] = []
        for attacker_index, attacker in enumerate(attackers):
            fire_state = fire_states[attacker_index]
            target_index = self._select_target(attacker, targets) if attacker.alive else None
            if target_index is None:
                fire_state.armed = True
                continue
            if not fire_state.armed:
                continue
            fire_state.armed = False
            geometry = engagement_geometry(attacker, targets[target_index])
            attempts.append((
                attacker_index,
                target_index,
                self.weapon.attempt_hit(geometry, self.rng),
            ))
        attempt_count = len(attempts)
        hit_count = sum(hit for _, _, hit in attempts)
        self.combat_counts[side]["fire_attempts"] += attempt_count
        self.combat_counts[side]["weapon_hits"] += hit_count
        if attempt_count and self.first_steps[side]["attempt"] is None:
            self.first_steps[side]["attempt"] = self.steps
        if hit_count and self.first_steps[side]["hit"] is None:
            self.first_steps[side]["hit"] = self.steps
        return attempts

    def _resolve_combat(
        self,
        red_attempts: list[tuple[int, int, bool]],
        blue_attempts: list[tuple[int, int, bool]],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        credited_by_side: list[dict[int, list[int]]] = [{}, {}]
        for side_index, (side, targets, attempts) in enumerate((
            ("red", self.blue, red_attempts),
            ("blue", self.red, blue_attempts),
        )):
            by_target: dict[int, list[int]] = {}
            for attacker_index, target_index, hit in attempts:
                if hit:
                    by_target.setdefault(target_index, []).append(attacker_index)
            for target_index, attacker_indices in sorted(by_target.items()):
                if targets[target_index].alive:
                    targets[target_index].alive = False
                    credited_by_side[side_index][target_index] = sorted(
                        set(attacker_indices)
                    )
            kills = len(credited_by_side[side_index])
            self.combat_counts[side]["attack_kills"] += kills
            if kills and self.first_steps[side]["kill"] is None:
                self.first_steps[side]["kill"] = self.steps
        return credited_by_side[0], credited_by_side[1]

    def _event_reward_components(
        self,
        red_exits: list[int],
        red_ground: list[int],
        red_kills: dict[int, list[int]],
        blue_kills: dict[int, list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.config["reward"]
        r1 = np.zeros(self.team_size, dtype=np.float32)
        r2 = np.zeros(self.team_size, dtype=np.float32)
        for attackers in red_kills.values():
            share = float(config["kill_reward"]) / len(attackers)
            r1[attackers] += share
        for red_index in set(red_ground) | set(blue_kills):
            r1[red_index] += float(config["loss_penalty"])
        for red_index in red_exits:
            r2[red_index] += float(config["boundary_penalty"])
        return r1, r2

    def _update_fire_window_metrics(
        self, red_pairs: int, blue_pairs: int
    ) -> None:
        for side, pairs in (("red", red_pairs), ("blue", blue_pairs)):
            self.combat_counts[side]["fire_window_pair_steps"] += pairs
            if pairs:
                self.combat_counts[side]["fire_window_steps"] += 1
                if self.first_steps[side]["fire_window"] is None:
                    self.first_steps[side]["fire_window"] = self.steps

    def _outcome(self, truncated: bool) -> tuple[bool, bool, bool, str]:
        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        if red_survivors == 0 and blue_survivors == 0:
            return False, False, True, "draw_mutual_destruction"
        if blue_survivors == 0:
            return True, False, False, "red_win"
        if red_survivors == 0:
            return False, True, False, "blue_win"
        if truncated:
            return False, False, False, "red_failure_timeout"
        return False, False, False, "ongoing"

    def _info(
        self,
        rewards: np.ndarray,
        components: dict[str, np.ndarray],
        actions: np.ndarray,
        executed_phi: np.ndarray,
        truncated: bool,
        red_fire_pairs: int,
        blue_fire_pairs: int,
        red_step_attempts: int,
        blue_step_attempts: int,
        red_step_hits: int,
        blue_step_hits: int,
        red_step_kills: int,
        blue_step_kills: int,
    ) -> dict[str, Any]:
        red_win, blue_win, draw, reason = self._outcome(truncated)
        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        info: dict[str, Any] = {
            "environment_version": ENVIRONMENT_VERSION,
            "red_success": red_win,
            "red_win": red_win,
            "blue_win": blue_win,
            "draw": draw,
            "termination_reason": reason,
            "red_losses": self.team_size - red_survivors,
            "blue_losses": self.team_size - blue_survivors,
            "red_survivors": red_survivors,
            "blue_survivors": blue_survivors,
            "episode_length": self.steps,
            "red_fire_window_pairs": red_fire_pairs,
            "blue_fire_window_pairs": blue_fire_pairs,
            "red_step_fire_attempts": red_step_attempts,
            "blue_step_fire_attempts": blue_step_attempts,
            "red_step_weapon_hits": red_step_hits,
            "blue_step_weapon_hits": blue_step_hits,
            "red_step_attack_kills": red_step_kills,
            "blue_step_attack_kills": blue_step_kills,
            "local_rewards": rewards.copy(),
            "r1_rewards": components["r1"].copy(),
            "r2_rewards": components["r2"].copy(),
            "r3_rewards": components["r3"].copy(),
            "r4_rewards": components["r4"].copy(),
            "executed_red_actions": actions.copy(),
            "executed_red_phi": executed_phi.copy(),
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }
        for side in ("red", "blue"):
            info.update({
                f"{side}_{key}": value
                for key, value in self.combat_counts[side].items()
            })
            info.update({
                f"{side}_first_{key}_step": value
                for key, value in self.first_steps[side].items()
            })
        for name, values in self.episode_reward_components.items():
            info[f"episode_{name}_rewards"] = values.copy()
            info[f"episode_{name}_total"] = float(values.sum())
        return info

    def step(
        self, red_actions: np.ndarray, blue_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        red_actions = np.asarray(red_actions, dtype=np.float32)
        if red_actions.shape != (self.team_size, self.action_dim) or not np.all(
            np.isfinite(red_actions)
        ):
            raise ValueError("red_actions must be finite with shape (4, 3)")
        if blue_actions is None:
            blue_actions = self.fixed_policy.team_actions(self.blue, self.red)
        blue_actions = np.asarray(blue_actions, dtype=np.float32)
        if blue_actions.shape != (self.team_size, self.action_dim) or not np.all(
            np.isfinite(blue_actions)
        ):
            raise ValueError("blue_actions must be finite with shape (4, 3)")

        executed_red = np.clip(red_actions, -1.0, 1.0) * self.red_alive_mask[:, None]
        executed_blue = np.clip(blue_actions, -1.0, 1.0) * self.blue_alive_mask[:, None]
        self.red_last_executed_phi = self._advance(self.red, executed_red)
        self.blue_last_executed_phi = self._advance(self.blue, executed_blue)
        self.steps += 1

        red_exits, _, red_ground, _ = self._resolve_noncombat_losses()
        post_red = self._snapshot(self.red)
        post_blue = self._snapshot(self.blue)
        state_components = paper_state_reward_components(
            post_red, post_blue, self.config["reward"]
        )
        red_fire_pairs = self._window_pair_count(post_red, post_blue)
        blue_fire_pairs = self._window_pair_count(post_blue, post_red)
        self._update_fire_window_metrics(red_fire_pairs, blue_fire_pairs)

        red_attempts_before = self.combat_counts["red"]["fire_attempts"]
        blue_attempts_before = self.combat_counts["blue"]["fire_attempts"]
        red_hits_before = self.combat_counts["red"]["weapon_hits"]
        blue_hits_before = self.combat_counts["blue"]["weapon_hits"]
        red_kills_before = self.combat_counts["red"]["attack_kills"]
        blue_kills_before = self.combat_counts["blue"]["attack_kills"]
        red_attempts = self._entry_attempts(
            post_red, post_blue, self.red_fire_states, "red"
        )
        blue_attempts = self._entry_attempts(
            post_blue, post_red, self.blue_fire_states, "blue"
        )
        red_kills, blue_kills = self._resolve_combat(red_attempts, blue_attempts)
        r1, r2 = self._event_reward_components(
            red_exits, red_ground, red_kills, blue_kills
        )
        components = {"r1": r1, "r2": r2, **state_components}
        rewards = sum(components.values(), np.zeros(self.team_size, dtype=np.float32))
        for name, values in components.items():
            self.episode_reward_components[name] += values

        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        terminated = bool(red_survivors == 0 or blue_survivors == 0)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        return (
            self._observations(),
            rewards.astype(np.float32),
            terminated,
            truncated,
            self._info(
                rewards, components, executed_red, self.red_last_executed_phi,
                truncated, red_fire_pairs, blue_fire_pairs,
                self.combat_counts["red"]["fire_attempts"] - red_attempts_before,
                self.combat_counts["blue"]["fire_attempts"] - blue_attempts_before,
                self.combat_counts["red"]["weapon_hits"] - red_hits_before,
                self.combat_counts["blue"]["weapon_hits"] - blue_hits_before,
                self.combat_counts["red"]["attack_kills"] - red_kills_before,
                self.combat_counts["blue"]["attack_kills"] - blue_kills_before,
            ),
        )


__all__ = ["MultiUAVCombatEnv"]
