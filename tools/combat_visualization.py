"""Semantics-neutral observation and storage of ordinary 4v4 combat traces."""
from pathlib import Path
from typing import Any
from dataclasses import asdict
import hashlib
import json
import numpy as np
from env.combat_env import MultiUAVCombatEnv, DEFAULT_COMBAT_CONFIG
from env.models import AircraftState
from env.weapon import FireState
from env.geometry import engagement_geometry

from tools.combat_trace import (TRACE_SCHEMA_VERSION, FEATURE_NAMES, FRAME_FIELDS,
    TRANSITION_FIELDS, COOLDOWN_FIELDS, write_trace, read_trace)
from env.v36_geometry import angle_within
from env.v336_weapon import PaperATAHAWeapon
from env.v317_weapon import BodyFrameGaussianWeapon
from env.v342_weapon import ConeWeapon


def states_array(states: list[AircraftState]) -> np.ndarray:
    return np.asarray([state.as_array() for state in states], dtype=np.float64)


class RecordingCombatEnv(MultiUAVCombatEnv):
    """Observe original event results without adding RNG draws or transitions."""
    def __new__(cls, config=DEFAULT_COMBAT_CONFIG):
        if issubclass(cls, RecordingCombatEnv):
            base = type(MultiUAVCombatEnv.__new__(MultiUAVCombatEnv, config))
            if base is not MultiUAVCombatEnv and not issubclass(cls, base):
                # Keep recording hooks first while resolving initialization and
                # combat/reset through the actual versioned environment class.
                recording_class = type(f'Recording{base.__name__}', (cls, base), {})
                return object.__new__(recording_class)
        return object.__new__(cls)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.events = []
        self.observed_actions = {}
        return super().reset(seed)

    def step(self, red_actions, blue_actions=None):
        self.observed_actions = {}
        return super().step(red_actions, blue_actions)

    def _advance(self, states, actions):
        # Observe the exact already-clipped input; never call the policy again.
        side = 'red' if states is self.red else 'blue'
        self.observed_actions[side] = np.asarray(actions).copy()
        return super()._advance(states, actions)

    def _entry_attempts(self, attackers: list[AircraftState], targets: list[AircraftState],
                        fire_states: list[FireState], side: str) -> list[tuple[int, int, bool]]:
        attempts = super()._entry_attempts(attackers, targets, fire_states, side)
        for attacker, target, hit in attempts:
            # Capture the actual firing state before simultaneous kills mutate alive flags.
            geometry = engagement_geometry(attackers[attacker], targets[target])
            audit = {'range_pass': bool(self.config['weapon']['range_min'] <= geometry.distance <= self.config['weapon']['range_max'])}
            if isinstance(self.weapon, ConeWeapon):
                audit.update(cone_pass=bool(angle_within(geometry.off_boresight,self.weapon.off_boresight_max)),
                             probability_angular_frame='signed_body_azimuth_elevation')
            elif isinstance(self.weapon, PaperATAHAWeapon):
                audit.update(ata_pass=bool(angle_within(abs(geometry.ata),self.weapon.ata_max)),
                             ha_pass=bool(angle_within(abs(geometry.ha),self.weapon.ha_max)))
            elif isinstance(self.weapon, BodyFrameGaussianWeapon):
                audit.update(body_azimuth_pass=bool(angle_within(abs(geometry.boresight_azimuth_error),self.weapon.boresight_azimuth_max)),
                             body_elevation_pass=bool(angle_within(abs(geometry.boresight_elevation_error),self.weapon.boresight_elevation_max)))
            self.events.append({'type': 'fire_attempt', 'step': self.steps, 'time_s': self.steps*self.dt, 'side': side,
                'attacker': int(attacker), 'target': int(target), 'hit': bool(hit),
                'attacker_state': attackers[attacker].as_array().tolist(),
                'target_state': targets[target].as_array().tolist(),
                **asdict(geometry), 'target_aspect': geometry.target_aspect, **audit,
                'start': attackers[attacker].as_array()[:3].tolist(),
                'end': targets[target].as_array()[:3].tolist()})
        return attempts

    def _resolve_combat(self, red_attempts, blue_attempts):
        result = super()._resolve_combat(red_attempts, blue_attempts)
        for side, kills in zip(('red', 'blue'), result):
            for target, attackers in kills.items():
                self.events.append({'type': 'attack_kill', 'step': self.steps, 'time_s': self.steps*self.dt,
                                    'side': side, 'target': target, 'attackers': attackers})
        return result

    def _resolve_noncombat_losses(self):
        before = {}
        ceiling_before = {}
        for side in ('red','blue'):
            ceiling_before[side] = self.combat_counts[side].get('ceiling_losses',0)
            for i,state in enumerate(getattr(self,side)):
                if state.alive:
                    before[side,i] = dict(state=state.as_array().tolist(),
                        horizontal_exceeded=bool(np.hypot(state.x,state.y)>self.arena_radius),
                        ceiling_exceeded=bool(state.altitude>self.config['arena'].get('altitude_max',float('inf'))),
                        ground_exceeded=bool(state.altitude<=self.config['arena'].get('altitude_min',0.)))
        result = super()._resolve_noncombat_losses()
        # The returned buckets, not a new boundary resolver, decide what was lost.
        # Current v3.37: ground first; any ceiling exit also increments ceiling_losses.
        # Historical base: horizontal may take priority over simultaneous ground.
        for side, exits, ground in (('red',result[0],result[2]),('blue',result[1],result[3])):
            ceiling_delta = self.combat_counts[side].get('ceiling_losses',0)-ceiling_before[side]
            for i in exits:
                flags=before[side,i]
                cause='ceiling' if ceiling_delta and flags['ceiling_exceeded'] else 'horizontal_boundary' if flags['horizontal_exceeded'] else 'unclassified_boundary'
                self.events.append(dict(type='noncombat_loss',side=side,agent=int(i),cause=cause,
                    step=self.steps,time_s=self.steps*self.dt,**flags))
            for i in ground:
                self.events.append(dict(type='noncombat_loss',side=side,agent=int(i),cause='ground',
                    step=self.steps,time_s=self.steps*self.dt,**before[side,i]))
        return result


def append_frame(frames: dict[str, list[Any]], env: RecordingCombatEnv) -> None:
    values = {'red_kinematics': states_array(env.red), 'blue_kinematics': states_array(env.blue),
              'red_alive': env.red_alive_mask.astype(bool), 'blue_alive': env.blue_alive_mask.astype(bool),
              'steps': env.steps, 'time_s': env.steps * env.dt}
    for key in FRAME_FIELDS:
        frames[key].append(values[key])


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
