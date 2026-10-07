"""Focused regressions for Direct/Persistent wave-safe trajectory capture."""
from __future__ import annotations

import csv
import numpy as np

from env.models import AircraftState
from tools.plot_best_model_trajectories import (
    POINT_FIELDS,
    rollout,
    select_representative_cases,
    trajectory_rows,
)
from tools.plot_persistent_wave_audit import trajectory_groups


class Actor:
    def act(self, observation, alive_mask, deterministic=True):
        return np.zeros((4, 3), dtype=float)


def states(offset: float = 0.0):
    return [AircraftState(offset + index, 0.0, -3000.0, 225.0, 0.0, 0.0)
            for index in range(4)]


class FakeWaveEnv:
    def __init__(self, waves: int):
        self.total_waves = waves
        self.wave_index = 1
        self.steps = 0
        self.red = states()
        self.blue = states(100.0)
        self.red_alive_mask = np.ones(4, dtype=bool)

    def reset(self, seed):
        return np.zeros((4, 52)), {"environment_variant": (
            "direct_v2_3" if self.total_waves == 1 else "persistent_wave_v2"
        )}

    def step(self, actions):
        self.steps += 1
        spawned = self.steps < self.total_waves
        old_wave = self.wave_index
        for state in self.blue:
            state.alive = False
        if spawned:
            self.wave_index += 1
            self.blue = states(100.0 * self.wave_index)
        done = not spawned
        info = {
            "environment_variant": "direct_v2_3" if self.total_waves == 1 else "persistent_wave_v2",
            "wave_index": self.wave_index, "total_waves": self.total_waves,
            "waves_cleared": self.steps, "wave_cleared_this_step": True,
            "spawned_next_wave": spawned, "wave_spawn_radial_angle": 0.1 * old_wave,
            "wave_spawn_candidate_index": old_wave, "minimum_spawn_distance": 1000.0,
            "termination_reason": "red_win" if done else "ongoing",
            "episode_length": self.steps, "red_success": done,
            "red_losses": 0, "blue_losses": 4 * self.steps,
            "red_attack_kills": 4 * self.steps, "blue_attack_kills": 0,
            "red_boundary_exits": 0, "blue_boundary_exits": 0,
            "red_ground_losses": 0, "blue_ground_losses": 0,
            "episode_r1_total": 10.0, "episode_r2_total": 0.0,
            "episode_r3_total": 0.0, "episode_r4_total": 0.0,
            "per_wave_metrics": [{"wave_index": index} for index in range(1, self.steps + 1)],
        }
        return np.zeros((4, 52)), np.zeros(4), done, False, info


def test_persistent_blue_tracks_are_wave_local(monkeypatch):
    monkeypatch.setattr(
        "tools.plot_best_model_trajectories.make_combat_environment",
        lambda config: FakeWaveEnv(3),
    )
    summary, tracks = rollout(Actor(), {}, 7, capture=True)
    assert summary["waves_cleared"] == 3
    assert {key[1] for key in tracks if key[0] == "blue"} == {1, 2, 3}
    assert len([key for key in tracks if key[0] == "blue"]) == 12
    assert all({point["wave_index"] for point in points} == {key[1]}
               for key, points in tracks.items() if key[0] == "blue")
    assert len([key for key in tracks if key[0] == "red"]) == 4


def test_direct_trajectory_and_csv_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.plot_best_model_trajectories.make_combat_environment",
        lambda config: FakeWaveEnv(1),
    )
    summary, tracks = rollout(Actor(), {}, 8, capture=True)
    assert summary["total_waves"] == 1
    assert {key[1] for key in tracks if key[0] == "blue"} == {1}
    rows = trajectory_rows(tracks)
    path = tmp_path / "trajectory.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=POINT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with path.open(newline="", encoding="utf-8") as stream:
        loaded = list(csv.DictReader(stream))
    assert loaded and {"wave_index", "side", "aircraft"} <= set(loaded[0])


def test_representative_selection_uses_waves_cleared():
    best = [
        {"seed": 1, "total_waves": 3, "waves_cleared": 3, "team_return": 80.0},
        {"seed": 2, "total_waves": 3, "waves_cleared": 2, "team_return": 40.0},
        {"seed": 3, "total_waves": 3, "waves_cleared": 3, "team_return": 100.0},
    ]
    latest = [
        {"seed": 1, "total_waves": 3, "waves_cleared": 1, "team_return": 0.0},
        {"seed": 2, "total_waves": 3, "waves_cleared": 2, "team_return": 42.0},
        {"seed": 3, "total_waves": 3, "waves_cleared": 3, "team_return": 90.0},
    ]
    selected = select_representative_cases(best, latest)
    assert selected["best_partial"] == 2
    assert selected["drift_pair"] == 1
    assert selected["latest_success"] == 3


def test_renderer_keeps_red_continuous_but_blue_wave_local():
    rows = [
        {"side": "red", "wave_index": str(wave), "aircraft": "1", "step": str(wave)}
        for wave in (1, 2, 3)
    ] + [
        {"side": "blue", "wave_index": str(wave), "aircraft": "1", "step": str(wave)}
        for wave in (1, 2, 3)
    ]
    groups = dict(trajectory_groups(rows))
    assert len(groups[("red", 0, 1)]) == 3
    assert all(len(groups[("blue", wave, 1)]) == 1 for wave in (1, 2, 3))
