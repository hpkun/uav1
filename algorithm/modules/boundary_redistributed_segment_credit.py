"""BRSC-MAPPO V1 boundary-state supervision and local credit redistribution."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import numpy as np

from .base import CapabilityModule

BRSC_MAPPO_VERSION = 1
W1_BOUNDARY_TO_W2 = "w1_to_w2"
W1_BOUNDARY_TO_W3 = "w1_to_w3"
W2_BOUNDARY_TO_W3 = "w2_to_w3"
BRSC_TASKS = (W1_BOUNDARY_TO_W2, W1_BOUNDARY_TO_W3, W2_BOUNDARY_TO_W3)
BRSC_TASK_SOURCE_WAVE = {W1_BOUNDARY_TO_W2: 1, W1_BOUNDARY_TO_W3: 1, W2_BOUNDARY_TO_W3: 2}
BRSC_TASK_HEAD = {W1_BOUNDARY_TO_W2: 0, W1_BOUNDARY_TO_W3: 1, W2_BOUNDARY_TO_W3: 1}


class BoundaryRedistributedSegmentCreditModule(CapabilityModule):
    """Recent-policy boundary predictor state, with held-out episode isolation."""
    name = "boundary_redistributed_segment_credit"

    def __init__(self, config=None):
        super().__init__(config); c = self.config; self.version = BRSC_MAPPO_VERSION
        self.max_waves = int(c.get("max_waves", 3))
        self.outcome_critic_learning_rate = float(c.get("outcome_critic_learning_rate", 3e-4))
        self.train_boundaries_per_class = int(c.get("train_boundaries_per_class", 256))
        self.validation_boundaries_per_wave = int(c.get("validation_boundaries_per_wave", 256))
        self.prior_window_boundaries_per_wave = int(c.get("prior_window_boundaries_per_wave", 512))
        self.critic_samples_per_class_per_task = int(c.get("critic_samples_per_class_per_task", 64))
        self.critic_updates_per_rollout = int(c.get("critic_updates_per_rollout", 4))
        self.min_train_boundaries_per_class = int(c.get("min_train_boundaries_per_class", 8))
        self.validation_interval_updates = int(c.get("validation_interval_updates", 10))
        self.min_validation_boundaries = int(c.get("min_validation_boundaries", 64))
        self.min_validation_positive = int(c.get("min_validation_positive", 16))
        self.min_validation_negative = int(c.get("min_validation_negative", 16))
        self.min_validation_auroc = float(c.get("min_validation_auroc", .60))
        self.min_validation_brier_skill = float(c.get("min_validation_brier_skill", 0.))
        self.readiness_consecutive_passes = int(c.get("readiness_consecutive_passes", 3))
        self.auxiliary_gradient_ratio_cap = float(c.get("auxiliary_gradient_ratio_cap", .25))
        self.prior_probability_epsilon = float(c.get("prior_probability_epsilon", 1e-4))
        if self.enabled:
            if self.max_waves != 3: raise ValueError("BRSC V1 requires three waves")
            if not 0 < self.auxiliary_gradient_ratio_cap <= 1: raise ValueError("invalid BRSC trust cap")
        self.train_replay = {1: {key: deque(maxlen=self.train_boundaries_per_class) for key in ("00", "10", "11")},
                             2: {key: deque(maxlen=self.train_boundaries_per_class) for key in ("0", "1")}}
        self.validation = {w: deque(maxlen=self.validation_boundaries_per_wave) for w in (1, 2)}
        self.prior_window = {w: deque(maxlen=self.prior_window_boundaries_per_wave) for w in (1, 2)}
        self.validation_pass_streak = {task: 0 for task in BRSC_TASKS}
        self.task_ready = {task: False for task in BRSC_TASKS}
        self.first_ready_sampled_steps = {task: None for task in BRSC_TASKS}
        self.ready_activation_count = {task: 0 for task in BRSC_TASKS}
        self.ready_deactivation_count = {task: 0 for task in BRSC_TASKS}
        self.validation_check_count = 0; self.next_boundary_id = 0

    @staticmethod
    def labels(waves_cleared):
        cleared = int(waves_cleared); return {"c2": int(cleared >= 2), "c3": int(cleared >= 3)}

    @staticmethod
    def episode_split(episode_group_id): return "validation" if int(episode_group_id) % 5 == 0 else "train"

    def complete_boundary(self, pending, waves_cleared, episode_group_id, source_env_id):
        wave = int(pending["source_wave"]); labels = self.labels(waves_cleared)
        if wave not in (1, 2): raise ValueError("BRSC source wave must be 1 or 2")
        if labels["c3"] and not labels["c2"]: raise RuntimeError("invalid BRSC outcome class 01")
        result = deepcopy(pending)
        result.update({"boundary_id": self.next_boundary_id, "episode_group_id": int(episode_group_id),
                       "source_env_id": int(source_env_id), "split": self.episode_split(episode_group_id),
                       "label_c2": labels["c2"], "label_c3": labels["c3"],
                       "episode_waves_cleared": int(waves_cleared)})
        self.next_boundary_id += 1; return result

    def ingest(self, boundaries):
        for boundary in boundaries or ():
            wave = int(boundary["source_wave"]); c2 = int(boundary["label_c2"]); c3 = int(boundary["label_c3"])
            if c3 and not c2: raise RuntimeError("invalid BRSC outcome class 01")
            if boundary["split"] == "validation": self.validation[wave].append(deepcopy(boundary))
            elif boundary["split"] == "train":
                key = f"{c2}{c3}" if wave == 1 else str(c3)
                self.train_replay[wave][key].append(deepcopy(boundary))
                self.prior_window[wave].append({"c2": c2, "c3": c3, "boundary_id": int(boundary["boundary_id"])})
            else: raise ValueError("unknown BRSC split")
            self.next_boundary_id = max(self.next_boundary_id, int(boundary["boundary_id"]) + 1)

    def task_classes(self, task):
        if task == W1_BOUNDARY_TO_W2: return [*self.train_replay[1]["10"], *self.train_replay[1]["11"]], [*self.train_replay[1]["00"]]
        if task == W1_BOUNDARY_TO_W3: return [*self.train_replay[1]["11"]], [*self.train_replay[1]["00"], *self.train_replay[1]["10"]]
        if task == W2_BOUNDARY_TO_W3: return [*self.train_replay[2]["1"]], [*self.train_replay[2]["0"]]
        raise KeyError(task)

    @staticmethod
    def task_label(task, boundary): return int(boundary["label_c2"] if task == W1_BOUNDARY_TO_W2 else boundary["label_c3"])

    def recent_prior(self, task):
        rows = self.prior_window[BRSC_TASK_SOURCE_WAVE[task]]
        if not rows: return .5
        key = "c2" if task == W1_BOUNDARY_TO_W2 else "c3"; return float(np.mean([row[key] for row in rows]))

    def apply_validation(self, task, metrics, sampled_steps):
        passed = (metrics["validation_segments"] >= self.min_validation_boundaries and
                  metrics["validation_positive"] >= self.min_validation_positive and
                  metrics["validation_negative"] >= self.min_validation_negative and
                  metrics["validation_auroc"] is not None and metrics["validation_auroc"] >= self.min_validation_auroc and
                  metrics["validation_brier_skill"] is not None and metrics["validation_brier_skill"] > self.min_validation_brier_skill)
        old = self.task_ready[task]
        self.validation_pass_streak[task] = self.validation_pass_streak[task] + 1 if passed else 0
        self.task_ready[task] = bool(passed and self.validation_pass_streak[task] >= self.readiness_consecutive_passes)
        if self.task_ready[task] and not old:
            self.ready_activation_count[task] += 1
            if self.first_ready_sampled_steps[task] is None: self.first_ready_sampled_steps[task] = int(sampled_steps)
        if old and not self.task_ready[task]: self.ready_deactivation_count[task] += 1
        return passed

    def state_dict(self):
        return {"version": self.version, "train_replay": {w: {k: list(v) for k, v in rows.items()} for w, rows in self.train_replay.items()},
                "validation": {w: list(v) for w, v in self.validation.items()}, "prior_window": {w: list(v) for w, v in self.prior_window.items()},
                "validation_pass_streak": deepcopy(self.validation_pass_streak), "task_ready": deepcopy(self.task_ready),
                "first_ready_sampled_steps": deepcopy(self.first_ready_sampled_steps), "ready_activation_count": deepcopy(self.ready_activation_count),
                "ready_deactivation_count": deepcopy(self.ready_deactivation_count), "validation_check_count": self.validation_check_count,
                "next_boundary_id": self.next_boundary_id}

    def load_state_dict(self, state):
        if not state: return
        if int(state.get("version", -1)) != self.version: raise RuntimeError("BRSC replay version mismatch")
        tr = state["train_replay"]
        self.train_replay = {w: {k: deque(tr.get(w, tr.get(str(w)))[k], maxlen=self.train_boundaries_per_class)
                                 for k in (("00", "10", "11") if w == 1 else ("0", "1"))} for w in (1, 2)}
        self.validation = {w: deque(state["validation"].get(w, state["validation"].get(str(w))), maxlen=self.validation_boundaries_per_wave) for w in (1, 2)}
        self.prior_window = {w: deque(state["prior_window"].get(w, state["prior_window"].get(str(w))), maxlen=self.prior_window_boundaries_per_wave) for w in (1, 2)}
        for name in ("validation_pass_streak", "task_ready", "first_ready_sampled_steps", "ready_activation_count", "ready_deactivation_count"):
            setattr(self, name, deepcopy(state[name]))
        self.validation_check_count = int(state["validation_check_count"]); self.next_boundary_id = int(state["next_boundary_id"])


def redistribute_boundary_credit(wave_indices, dones, boundary_flags, boundary_credit, decay):
    """Redistribute each event backward inside its current-rollout episode/wave block."""
    waves = np.asarray(wave_indices); done = np.asarray(dones) > .5; flags = np.asarray(boundary_flags) > .5
    credit = np.asarray(boundary_credit, dtype=np.float32)
    if waves.shape != flags.shape or waves.shape != credit.shape or waves.shape != done.shape: raise ValueError("BRSC arrays must share [T,E]")
    advantage = np.zeros_like(credit, dtype=np.float32); active = np.zeros_like(flags); segments = []
    for b, e in np.argwhere(flags):
        z = float(credit[b, e]); wave = int(waves[b, e]); distance = 0; length = 0
        if wave not in (1, 2): continue
        for t in range(int(b), -1, -1):
            if int(waves[t, e]) != wave: break
            if t < b and done[t, e]: break
            advantage[t, e] += z * float(decay) ** distance; active[t, e] = True
            length += 1; distance += 1
            if t > 0 and done[t - 1, e]: break
        segments.append(length)
    return advantage, active, segments


__all__ = ["BRSC_MAPPO_VERSION", "W1_BOUNDARY_TO_W2", "W1_BOUNDARY_TO_W3", "W2_BOUNDARY_TO_W3",
           "BRSC_TASKS", "BRSC_TASK_SOURCE_WAVE", "BRSC_TASK_HEAD", "BoundaryRedistributedSegmentCreditModule",
           "redistribute_boundary_credit"]
