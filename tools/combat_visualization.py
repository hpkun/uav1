"""Semantics-neutral observation and storage of ordinary 4v4 combat traces."""
from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.weapon import FireState

TRACE_SCHEMA_VERSION = 2
FEATURE_NAMES = ['x', 'y', 'z', 'v', 'theta', 'psi']
FRAME_FIELDS = ('red_kinematics', 'blue_kinematics', 'red_alive', 'blue_alive', 'steps', 'time_s')
TRANSITION_FIELDS = ('red_actions', 'local_rewards', 'reward_components', 'terminated', 'truncated')


def states_array(states: list[AircraftState]) -> np.ndarray:
    return np.asarray([state.as_array() for state in states], dtype=np.float64)


class RecordingCombatEnv(MultiUAVCombatEnv):
    """Observe original event results without adding RNG draws or transitions."""
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.events = []
        return super().reset(seed)

    def _entry_attempts(self, attackers: list[AircraftState], targets: list[AircraftState],
                        fire_states: list[FireState], side: str) -> list[tuple[int, int, bool]]:
        attempts = super()._entry_attempts(attackers, targets, fire_states, side)
        for attacker, target, hit in attempts:
            self.events.append({'type': 'fire_attempt', 'step': self.steps, 'side': side,
                                'attacker': attacker, 'target': target, 'hit': bool(hit),
                                'start': attackers[attacker].as_array()[:3].tolist(),
                                'end': targets[target].as_array()[:3].tolist()})
        return attempts

    def _resolve_combat(self, red_attempts, blue_attempts):
        result = super()._resolve_combat(red_attempts, blue_attempts)
        for side, kills in zip(('red', 'blue'), result):
            for target, attackers in kills.items():
                self.events.append({'type': 'attack_kill', 'step': self.steps,
                                    'side': side, 'target': target, 'attackers': attackers})
        return result

    def _resolve_noncombat_losses(self):
        result = super()._resolve_noncombat_losses()
        for side, kind, indices in zip(('red', 'blue', 'red', 'blue'),
                ('boundary_exit', 'boundary_exit', 'ground_loss', 'ground_loss'), result):
            for index in indices:
                self.events.append({'type': kind, 'step': self.steps, 'side': side, 'agent': index})
        return result


def append_frame(frames: dict[str, list[Any]], env: RecordingCombatEnv) -> None:
    values = {'red_kinematics': states_array(env.red), 'blue_kinematics': states_array(env.blue),
              'red_alive': env.red_alive_mask.astype(bool), 'blue_alive': env.blue_alive_mask.astype(bool),
              'steps': env.steps, 'time_s': env.steps * env.dt}
    for key in FRAME_FIELDS:
        frames[key].append(values[key])


def write_trace(path: str | Path, frames: dict[str, list[Any]],
                transitions: dict[str, list[Any]]) -> dict[str, list[int]]:
    arrays = {key: np.asarray(value) for key, value in {**frames, **transitions}.items()}
    arrays['trace_schema_version'] = np.asarray(TRACE_SCHEMA_VERSION)
    np.savez_compressed(path, **arrays)
    return {key: list(value.shape) for key, value in arrays.items()}


def read_trace(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        trace = {key: stored[key] for key in stored.files}
    if int(trace.get('trace_schema_version', -1)) != TRACE_SCHEMA_VERSION:
        raise ValueError('unsupported combat trace schema')
    count = len(trace['steps'])
    for side in ('red', 'blue'):
        if trace[f'{side}_kinematics'].shape != (count, 4, 6) or trace[f'{side}_alive'].shape != (count, 4):
            raise ValueError('invalid combat trace shape')
    return trace


def ensure_fresh_output(path: str | Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'output directory is non-empty: {path}')
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
