"""Training-only wave-survival potential-based reward shaping."""
from __future__ import annotations

from typing import Any
import numpy as np

from .base import CapabilityModule


def mission_progress_from_blue_losses(
    blue_losses: np.ndarray, total_waves: np.ndarray, team_size: int
) -> np.ndarray:
    """Canonical cumulative mission progress used by PBRS and critic context."""
    losses = np.asarray(blue_losses, dtype=np.float32)
    waves = np.asarray(total_waves, dtype=np.float32)
    losses, waves = np.broadcast_arrays(losses, waves)
    denominator = np.maximum(waves * int(team_size), 1.0)
    progress = np.clip(losses / denominator, 0.0, 1.0).astype(np.float32)
    if not np.all(np.isfinite(progress)):
        raise FloatingPointError("mission progress is non-finite")
    return progress


def mission_progress_from_wave_state(
    wave_index: np.ndarray, blue_alive: np.ndarray, total_waves: np.ndarray
) -> np.ndarray:
    """Convert current wave/alive state to the same cumulative PBRS progress."""
    blue = np.asarray(blue_alive, dtype=np.float32)
    if blue.ndim < 1:
        raise ValueError("blue_alive must include an agent dimension")
    team_size = int(blue.shape[-1])
    wave = np.asarray(wave_index, dtype=np.float32)
    losses = (wave - 1.0) * team_size + (team_size - blue.sum(axis=-1))
    return mission_progress_from_blue_losses(losses, total_waves, team_size)


def mission_context_numpy(
    trainer, wave_index: np.ndarray, total_waves: np.ndarray,
    blue_alive: np.ndarray, episode_step: np.ndarray, max_steps: int | np.ndarray,
) -> np.ndarray:
    """Build the canonical policy/value mission context at an environment state.

    Keeping this adapter shared prevents training, evaluation, holdout and replay
    tools from silently using different definitions for mission progress/horizon.
    It is harmless for configurations with wave context disabled.
    """
    progress = mission_progress_from_wave_state(wave_index, blue_alive, total_waves)
    return trainer.context_numpy(
        wave_index, total_waves, mission_progress=progress,
        episode_step=episode_step, max_steps=max_steps,
    )


class WaveSurvivalPotentialShapingModule(CapabilityModule):
    name = "wave_survival_pbrs"

    def __init__(self, config: dict[str, Any] | None = None, gamma: float = 0.999) -> None:
        super().__init__(config)
        self.coefficient = float(self.config.get("coefficient", 1.0))
        self.potential = str(self.config.get("potential", "progress_times_survival"))
        self.terminal_zero = bool(self.config.get("terminal_zero", True))
        self.gamma = float(gamma)
        if self.enabled and (self.coefficient != 1.0 or self.potential != "progress_times_survival" or not self.terminal_zero):
            raise ValueError("WS-PBRS v1 requires coefficient=1, progress_times_survival, terminal_zero=true")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("PBRS gamma must be in [0,1]")
        self.last_transition: dict[str, np.ndarray] = {}

    @staticmethod
    def global_potential(progress: np.ndarray, red_alive: np.ndarray) -> np.ndarray:
        survival = np.asarray(red_alive, dtype=np.float32).mean(axis=-1)
        phi = np.asarray(progress, dtype=np.float32) * survival
        if not np.all(np.isfinite(phi)) or np.any(phi < -1e-7) or np.any(phi > 1.0 + 1e-7):
            raise FloatingPointError("WS-PBRS potential is non-finite or outside [0,1]")
        return phi

    def adapt(self, rewards: np.ndarray, infos: list[dict[str, Any]], pre_wave: np.ndarray,
              red_alive_before: np.ndarray, blue_alive_before: np.ndarray,
              next_red_alive: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        raw = np.asarray(rewards, dtype=np.float32)
        red_pre = np.asarray(red_alive_before, dtype=np.float32)
        blue_pre = np.asarray(blue_alive_before, dtype=np.float32)
        red_next = np.asarray(next_red_alive, dtype=np.float32)
        done = np.asarray(dones, dtype=bool)
        if raw.shape != red_pre.shape or raw.shape != blue_pre.shape or raw.shape != red_next.shape:
            raise ValueError("WS-PBRS alive masks must match reward shape")
        team_size = raw.shape[1]
        total_waves = np.asarray([max(1, int(info.get("total_waves", 1))) for info in infos], dtype=np.float32)
        removed_pre = (np.asarray(pre_wave, dtype=np.float32) - 1.0) * team_size + (team_size - blue_pre.sum(axis=1))
        removed_next = np.asarray([int(info.get("blue_losses", 0)) for info in infos], dtype=np.float32)
        progress_pre = mission_progress_from_blue_losses(removed_pre, total_waves, team_size)
        progress_next = mission_progress_from_blue_losses(removed_next, total_waves, team_size)
        global_pre = self.global_potential(progress_pre, red_pre)
        global_next = self.global_potential(progress_next, red_next)
        phi_pre = red_pre * global_pre[:, None]
        phi_next = red_next * global_next[:, None]
        phi_next[done] = 0.0
        if not self.enabled:
            shaping = np.zeros_like(raw)
            self.last_transition = {"phi_pre": phi_pre, "phi_next": phi_next, "shaping": shaping,
                                    "mission_progress": progress_next, "red_survival_ratio": red_next.mean(axis=1)}
            metrics = self._metrics(phi_pre, phi_next, shaping, pre_wave)
            metrics.update({"mission_progress": float(progress_next.mean()), "red_survival_ratio": float(red_next.mean())})
            return raw.copy(), metrics
        shaping = self.gamma * phi_next - phi_pre
        if not np.all(np.isfinite(shaping)) or np.any(np.abs(shaping) > 1.0 + 1e-6):
            raise FloatingPointError("WS-PBRS shaping is non-finite or outside its theoretical bound")
        self.last_transition = {"phi_pre": phi_pre, "phi_next": phi_next, "shaping": shaping,
                                "mission_progress": progress_next, "red_survival_ratio": red_next.mean(axis=1)}
        metrics = self._metrics(phi_pre, phi_next, shaping, pre_wave)
        metrics.update({"mission_progress": float(progress_next.mean()), "red_survival_ratio": float(red_next.mean())})
        return raw + self.coefficient * shaping, metrics

    @staticmethod
    def _metrics(phi_pre, phi_next, shaping, waves):
        wave = np.asarray(waves, dtype=np.int64)
        return {"pbrs_phi_pre_mean": float(phi_pre.mean()), "pbrs_phi_next_mean": float(phi_next.mean()),
                "pbrs_shaping_sum": float(shaping.sum()), "pbrs_shaping_abs_sum": float(np.abs(shaping).sum()),
                **{f"pbrs_shaping_wave{k}": float(shaping[wave == k].sum()) for k in (1, 2, 3)}}


__all__ = [
    "WaveSurvivalPotentialShapingModule", "mission_progress_from_blue_losses",
    "mission_progress_from_wave_state", "mission_context_numpy",
]
