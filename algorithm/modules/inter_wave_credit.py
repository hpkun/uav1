"""Inter-wave state-quality credit support (IWSC-MAPPO v1)."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import numpy as np

from .base import CapabilityModule

IWSC_MAPPO_VERSION = 1


class InterWaveCreditModule(CapabilityModule):
    name = "inter_wave_credit"

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        self.version = IWSC_MAPPO_VERSION
        self.max_waves = int(c.get("max_waves", 3))
        self.quality_critic_learning_rate = float(c.get("quality_critic_learning_rate", 3e-4))
        self.actor_credit_coefficient = float(c.get("actor_credit_coefficient", 1.0))
        self.replay_segments_per_wave = int(c.get("replay_segments_per_wave", 128))
        self.max_states_per_segment = int(c.get("max_states_per_segment", 64))
        self.critic_minibatch_size = int(c.get("critic_minibatch_size", 512))
        self.critic_updates_per_rollout = int(c.get("critic_updates_per_rollout", 4))
        self.min_completed_segments_per_wave = int(c.get("min_completed_segments_per_wave", 16))
        self.min_target_std = float(c.get("min_target_std", .05))
        self.huber_delta = float(c.get("huber_delta", 1.0))
        self.wave_balanced_supervision = bool(c.get("wave_balanced_supervision", True))
        self.wave_balanced_actor_loss = bool(c.get("wave_balanced_actor_loss", True))
        self.gradient_projection = str(c.get("gradient_projection", "asymmetric_tactical_preserving"))
        if self.enabled:
            if self.max_waves != 3: raise ValueError("IWSC v1 requires max_waves=3")
            if self.gradient_projection != "asymmetric_tactical_preserving": raise ValueError("unsupported IWSC gradient projection")
            for key, value in (("replay_segments_per_wave", self.replay_segments_per_wave),
                               ("max_states_per_segment", self.max_states_per_segment),
                               ("critic_minibatch_size", self.critic_minibatch_size),
                               ("critic_updates_per_rollout", self.critic_updates_per_rollout),
                               ("min_completed_segments_per_wave", self.min_completed_segments_per_wave)):
                if value <= 0: raise ValueError(f"{key} must be positive")
            if self.max_states_per_segment < 3: raise ValueError("max_states_per_segment must preserve first/last/boundary")
        self.replay = {wave: deque(maxlen=self.replay_segments_per_wave) for wave in (1, 2)}
        self.valid_update_count = 0
        self.next_segment_id = 0

    def target(self, source_wave: int, waves_cleared: int) -> float:
        if source_wave == 1: return (float(waves_cleared >= 2) + float(waves_cleared >= 3)) / 2.0
        if source_wave == 2: return float(waves_cleared >= 3)
        raise ValueError("only source waves 1 and 2 receive IW targets")

    def cap_segment(self, states: list[dict], target: float, wave: int) -> dict:
        if not states: raise ValueError("empty IW segment")
        count = len(states)
        indices = list(range(count)) if count <= self.max_states_per_segment else np.linspace(0, count - 1, self.max_states_per_segment, dtype=np.int64).tolist()
        boundaries = [i for i, row in enumerate(states) if row.get("boundary", False)]
        for boundary in boundaries:
            if boundary not in indices:
                indices[-2] = boundary
        indices = sorted(set(indices))
        chosen = [states[i] for i in indices]
        segment = {
            "segment_id": self.next_segment_id, "credit_wave": int(wave), "target": float(target),
            "original_state_count": count, "sample_indices": np.asarray(indices,dtype=np.int64),
            "observations": np.asarray([x["observation"] for x in chosen], dtype=np.float16),
            "alive_masks": np.asarray([x["alive_mask"] for x in chosen], dtype=np.float16),
            "horizons": np.asarray([x["horizon"] for x in chosen], dtype=np.float32),
            "boundary_flags": np.asarray([x.get("boundary", False) for x in chosen], dtype=bool),
        }
        self.next_segment_id += 1
        return segment

    def ingest(self, segments) -> int:
        if not self.enabled or segments is None: return 0
        count = 0
        for segment in segments:
            wave = int(segment["credit_wave"])
            if wave in self.replay:
                self.replay[wave].append(deepcopy(segment)); count += 1
                self.next_segment_id = max(self.next_segment_id, int(segment.get("segment_id", -1)) + 1)
        return count

    def target_std(self, wave: int) -> float:
        targets = np.asarray([x["target"] for x in self.replay[wave]], dtype=np.float64)
        return float(targets.std()) if targets.size else 0.0

    def ready(self, wave: int) -> bool:
        return (len(self.replay[wave]) >= self.min_completed_segments_per_wave and
                self.target_std(wave) >= self.min_target_std and self.valid_update_count > 0)

    def sample_states(self, wave: int, count: int, rng: np.random.Generator) -> dict | None:
        segments = self.replay[wave]
        if not segments: return None
        si = rng.integers(0, len(segments), size=count)
        rows = []
        for index in si:
            segment = segments[int(index)]
            state_index = int(rng.integers(0, len(segment["horizons"])))
            rows.append((segment, state_index))
        return {
            "observations": np.asarray([s["observations"][i] for s, i in rows], dtype=np.float32),
            "alive_masks": np.asarray([s["alive_masks"][i] for s, i in rows], dtype=np.float32),
            "horizons": np.asarray([s["horizons"][i] for s, i in rows], dtype=np.float32),
            "credit_waves": np.full(count, wave, dtype=np.int64),
            "targets": np.asarray([s["target"] for s, _ in rows], dtype=np.float32),
        }

    def state_dict(self):
        return {"version": self.version, "replay": {k: list(v) for k, v in self.replay.items()},
                "valid_update_count": self.valid_update_count, "next_segment_id": self.next_segment_id}

    def load_state_dict(self, state):
        if not state: return
        if int(state.get("version", -1)) != self.version: raise RuntimeError("IWSC replay version mismatch")
        self.replay = {wave: deque(state.get("replay", {}).get(wave, state.get("replay", {}).get(str(wave), [])),
                                   maxlen=self.replay_segments_per_wave) for wave in (1, 2)}
        self.valid_update_count = int(state.get("valid_update_count", 0))
        self.next_segment_id = int(state.get("next_segment_id", 0))


__all__ = ["IWSC_MAPPO_VERSION", "InterWaveCreditModule"]
