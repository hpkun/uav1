"""Recent natural-entry state curriculum for persistent multi-wave training."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any
import numpy as np

from .base import CapabilityModule

WAVE_ENTRY_CURRICULUM_VERSION = 1
WAVE_ENTRY_CURRICULUM_SEED_XOR = 0x57454331


class WaveEntryCurriculumModule(CapabilityModule):
    """Exposure-deficit controller backed only by natural W1-start entries."""

    name = "wave_entry_curriculum"

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 0) -> None:
        super().__init__(config)
        self.version = WAVE_ENTRY_CURRICULUM_VERSION
        self.mode = str(self.config.get("mode", "adaptive_deficit"))
        self.recent_natural_window = int(self.config.get("recent_natural_window", 200))
        self.min_natural_episodes = int(self.config.get("min_natural_episodes_for_adaptation", 50))
        self.target_reach_w2 = float(self.config.get("target_reach_w2", 0.75))
        self.target_reach_w3 = float(self.config.get("target_reach_w3", 0.50))
        self.max_curriculum_fraction = float(self.config.get("max_curriculum_fraction", 0.50))
        self.min_bank_entries = int(self.config.get("min_bank_entries", 8))
        self.capacity_per_wave = int(self.config.get("capacity_per_wave", 256))
        self.sampling = str(self.config.get("sampling", "uniform"))
        if self.mode != "adaptive_deficit": raise ValueError("wave_entry_curriculum only supports adaptive_deficit")
        if self.sampling != "uniform": raise ValueError("wave_entry_curriculum only supports uniform sampling")
        if self.recent_natural_window <= 0 or self.min_natural_episodes <= 0: raise ValueError("wave-entry curriculum windows must be positive")
        if not 0.0 <= self.max_curriculum_fraction <= 0.5: raise ValueError("max_curriculum_fraction must be in [0, 0.5]")
        if not 0.0 <= self.target_reach_w3 <= self.target_reach_w2 <= 1.0: raise ValueError("invalid wave-entry reach targets")
        if self.min_bank_entries <= 0 or self.capacity_per_wave < self.min_bank_entries: raise ValueError("invalid wave-entry bank capacity")
        self.rng = np.random.default_rng(int(seed) ^ WAVE_ENTRY_CURRICULUM_SEED_XOR)
        self.bank = {2: deque(maxlen=self.capacity_per_wave), 3: deque(maxlen=self.capacity_per_wave)}
        self.recent_natural_outcomes: deque[dict[str, float]] = deque(maxlen=self.recent_natural_window)
        self.natural_episode_count = 0
        self.bank_insertions = {2: 0, 3: 0}
        self.reset_counts = {1: 0, 2: 0, 3: 0}
        self.snapshot_id_counter = 0
        self.current_probabilities = (1.0, 0.0, 0.0)

    def add_natural_entry(self, entry_wave: int, snapshot: dict[str, Any], *,
                          source_sampled_steps: int, source_training_seed: int,
                          source_env_id: int, source_episode_reset_seed: int,
                          red_survivors: int) -> str | None:
        if not self.enabled: return None
        wave = int(entry_wave)
        if wave not in (2, 3): raise ValueError("entry_wave must be 2 or 3")
        if int(snapshot.get("metadata", {}).get("wave_index", 0)) != wave: raise ValueError("snapshot wave does not match bank")
        self.snapshot_id_counter += 1
        snapshot_id = f"wec-{int(source_training_seed)}-{self.snapshot_id_counter}"
        self.bank[wave].append({"snapshot": deepcopy(snapshot), "source_wave": wave - 1,
            "entry_wave": wave, "source_sampled_steps": int(source_sampled_steps),
            "source_training_seed": int(source_training_seed), "source_env_id": int(source_env_id),
            "source_episode_reset_seed": int(source_episode_reset_seed), "red_survivors": int(red_survivors),
            "snapshot_id": snapshot_id, "source_id": snapshot_id})
        self.bank_insertions[wave] += 1
        return snapshot_id

    def record_natural_episode(self, waves_cleared: int) -> None:
        if not self.enabled: return
        waves = int(waves_cleared); self.natural_episode_count += 1
        self.recent_natural_outcomes.append({"reach_w2": float(waves >= 1),
            "reach_w3": float(waves >= 2), "waves_cleared": float(waves)})

    def probabilities(self) -> tuple[float, float, float]:
        rows = self.recent_natural_outcomes
        if not self.enabled or len(rows) < self.min_natural_episodes:
            self.current_probabilities = (1.0, 0.0, 0.0); return self.current_probabilities
        n = len(rows)
        r2 = (sum(row["reach_w2"] for row in rows) + 1.0) / (n + 2.0)
        r3 = (sum(row["reach_w3"] for row in rows) + 1.0) / (n + 2.0)
        d2 = max(self.target_reach_w2 - r2, 0.0) if len(self.bank[2]) >= self.min_bank_entries else 0.0
        d3 = max(self.target_reach_w3 - r3, 0.0) if len(self.bank[3]) >= self.min_bank_entries else 0.0
        total = d2 + d3
        if total <= 0.0: self.current_probabilities = (1.0, 0.0, 0.0)
        else:
            fraction = min(self.max_curriculum_fraction, total)
            self.current_probabilities = (1.0 - fraction, fraction * d2 / total, fraction * d3 / total)
        return self.current_probabilities

    def sample_reset(self) -> tuple[int, dict[str, Any] | None]:
        wave = int(self.rng.choice(np.asarray([1, 2, 3]), p=self.probabilities()))
        if wave != 1 and not self.bank[wave]: wave = 1
        self.reset_counts[wave] += 1
        if wave == 1: return 1, None
        return wave, deepcopy(self.bank[wave][int(self.rng.integers(0, len(self.bank[wave])))])

    def diagnostics(self) -> dict[str, float]:
        rows = list(self.recent_natural_outcomes); n = len(rows)
        r2 = (sum(row["reach_w2"] for row in rows) + 1.0) / (n + 2.0) if n else 0.5
        r3 = (sum(row["reach_w3"] for row in rows) + 1.0) / (n + 2.0) if n else 0.5
        p1, p2, p3 = self.probabilities(); resets = max(sum(self.reset_counts.values()), 1)
        return {"wec_natural_episode_count": float(self.natural_episode_count),
            "wec_recent_reach_w2": float(r2), "wec_recent_reach_w3": float(r3),
            "wec_recent_natural_aw": float(np.mean([row["waves_cleared"] for row in rows])) if rows else 0.0,
            "wec_deficit_w2": float(max(self.target_reach_w2-r2, 0.0)),
            "wec_deficit_w3": float(max(self.target_reach_w3-r3, 0.0)),
            "wec_probability_w1": p1, "wec_probability_w2": p2, "wec_probability_w3": p3,
            "wec_bank_size_w2": float(len(self.bank[2])), "wec_bank_size_w3": float(len(self.bank[3])),
            "wec_bank_insertions_w2": float(self.bank_insertions[2]), "wec_bank_insertions_w3": float(self.bank_insertions[3]),
            **{f"wec_reset_count_w{w}": float(self.reset_counts[w]) for w in (1,2,3)},
            **{f"wec_reset_fraction_w{w}": float(self.reset_counts[w]/resets) for w in (1,2,3)}}

    def state_dict(self) -> dict[str, Any]:
        return deepcopy({"version": self.version, "config": self.config,
            "bank": {wave: list(rows) for wave, rows in self.bank.items()},
            "rng_state": self.rng.bit_generator.state,
            "recent_natural_outcomes": list(self.recent_natural_outcomes),
            "natural_episode_count": self.natural_episode_count,
            "bank_insertions": self.bank_insertions, "reset_counts": self.reset_counts,
            "current_probabilities": self.current_probabilities,
            "snapshot_id_counter": self.snapshot_id_counter})

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self.enabled: return
        if int(state.get("version", -1)) != self.version: raise RuntimeError("wave-entry curriculum checkpoint version mismatch")
        if state.get("config") != self.config: raise RuntimeError("wave-entry curriculum checkpoint config mismatch")
        source = state["bank"]
        self.bank = {wave: deque((deepcopy(row) for row in source.get(wave, source.get(str(wave), []))), maxlen=self.capacity_per_wave) for wave in (2,3)}
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self.recent_natural_outcomes = deque((deepcopy(row) for row in state.get("recent_natural_outcomes", [])), maxlen=self.recent_natural_window)
        self.natural_episode_count = int(state.get("natural_episode_count", 0))
        insertions, resets = state.get("bank_insertions", {}), state.get("reset_counts", {})
        self.bank_insertions = {w: int(insertions.get(w, insertions.get(str(w), 0))) for w in (2,3)}
        self.reset_counts = {w: int(resets.get(w, resets.get(str(w), 0))) for w in (1,2,3)}
        self.current_probabilities = tuple(map(float, state.get("current_probabilities", (1,0,0))))
        self.snapshot_id_counter = int(state.get("snapshot_id_counter", 0))


__all__ = ["WAVE_ENTRY_CURRICULUM_SEED_XOR", "WAVE_ENTRY_CURRICULUM_VERSION", "WaveEntryCurriculumModule"]
