"""Canonical numerical trace schema; no environment or policy decisions."""
from pathlib import Path
import numpy as np

TRACE_SCHEMA_VERSION = 3
FEATURE_NAMES = ['x', 'y', 'z', 'v', 'theta', 'psi']
FRAME_FIELDS = ('red_kinematics', 'blue_kinematics', 'red_alive', 'blue_alive', 'steps', 'time_s')
TRANSITION_FIELDS = ('red_actions', 'blue_actions', 'local_rewards', 'reward_components',
    'terminated', 'truncated', 'blue_pursuit_target_indices', 'red_ammo', 'blue_ammo')
COOLDOWN_FIELDS = ('red_cooldown', 'blue_cooldown')


def validate_trace(trace, metadata=None):
    version = int(trace.get('trace_schema_version', -1))
    if version not in (2, 3):
        raise ValueError('unsupported combat trace schema')
    required = set(FRAME_FIELDS) | {'red_actions', 'local_rewards', 'reward_components', 'terminated', 'truncated'}
    if version == 3:
        required |= set(TRANSITION_FIELDS) | {'dt'}
    if required - trace.keys():
        raise ValueError(f'missing combat trace fields: {sorted(required-trace.keys())}')
    if np.asarray(trace['trace_schema_version']).shape != () or (version==3 and np.asarray(trace['dt']).shape != ()):
        raise ValueError('schema version and dt must be scalar')
    for key, value in trace.items():
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError(f'non-finite or non-numerical trace field: {key}')
    steps = trace['steps']
    if steps.ndim != 1 or len(steps) < 1 or steps.dtype.kind not in 'iu':
        raise ValueError('steps must be a nonempty integer vector')
    f = len(steps); t = f-1
    if steps[0] != 0 or not np.array_equal(steps, np.arange(f)):
        raise ValueError('steps must increase by one from reset frame zero')
    n = None
    for side in ('red', 'blue'):
        a = trace[side+'_kinematics']; alive = trace[side+'_alive']
        if a.ndim != 3 or a.shape[0] != f or a.shape[2] != 6 or a.shape[1] < 1:
            raise ValueError('invalid kinematics shape')
        if n is not None and a.shape[1] != n:
            raise ValueError('Red/Blue team sizes differ')
        n = a.shape[1]
        if alive.shape != (f,n) or alive.dtype.kind != 'b':
            raise ValueError('invalid alive shape/dtype')
    dt = float(trace['dt']) if version == 3 else float((metadata or {}).get('dt',
        trace['time_s'][1]-trace['time_s'][0] if f>1 else .1))
    if not np.isfinite(dt) or dt <= 0 or trace['time_s'].shape != (f,):
        raise ValueError('invalid dt/time_s')
    if not np.allclose(trace['time_s'], steps*dt, rtol=0, atol=1e-9):
        raise ValueError('time_s must equal steps * dt')
    if metadata is not None and (not np.isclose(metadata['dt'],dt,rtol=0,atol=1e-12)
                                or metadata['episode_steps'] != t):
        raise ValueError('trace/metadata timing mismatch')
    if metadata is not None and version==3:
        if (metadata.get('reward_component_names') != ['R1','R2','R3','R4']
                or metadata.get('reward_component_axis')!=1 or metadata.get('agent_axis')!=2):
            raise ValueError('metadata reward component axes/order mismatch')
    shapes = dict(red_actions=(t,n,3),local_rewards=(t,n),reward_components=(t,4,n),
                  terminated=(t,),truncated=(t,))
    if version == 3:
        shapes.update(blue_actions=(t,n,3),blue_pursuit_target_indices=(t,n),red_ammo=(t,n),blue_ammo=(t,n))
    for name in COOLDOWN_FIELDS:
        if name in trace: shapes[name]=(t,n)
    for key, shape in shapes.items():
        if trace[key].shape != shape:
            raise ValueError(f'{key} shape must be {shape}, got {trace[key].shape}')
    for key in ('terminated','truncated'):
        if trace[key].dtype.kind != 'b':raise ValueError(f'{key} must be boolean')
    done = trace['terminated'] | trace['truncated']
    if t and (not done[-1] or done[:-1].any() or (trace['terminated'] & trace['truncated']).any()):
        raise ValueError('only the final transition must end the episode')
    if version == 3:
        targets=trace['blue_pursuit_target_indices']
        if targets.dtype.kind not in 'iu' or ((targets < -1)|(targets>=n)).any():
            raise ValueError('invalid Blue pursuit target indices')
        for key in ('red_ammo','blue_ammo',*COOLDOWN_FIELDS):
            if key in trace and (trace[key].dtype.kind not in 'iu' or (trace[key]<-1).any()):
                raise ValueError(f'invalid {key}; -1 means unavailable')
    return trace


def write_trace(path, frames, transitions, dt=None):
    arrays = {key:np.asarray(value) for key,value in {**frames,**transitions}.items()}
    t=len(arrays['steps'])-1;n=arrays['red_kinematics'].shape[1]
    # A reset-only snapshot is useful to historical unit fixtures, not a rollout.
    if t == 0:
        for key in TRANSITION_FIELDS:
            shape=(0,n,3) if key.endswith('actions') else (0,4,n) if key=='reward_components' else (0,) if key in ('terminated','truncated') else (0,n)
            dtype=bool if key in ('terminated','truncated') else int if key in ('blue_pursuit_target_indices','red_ammo','blue_ammo') else float
            arrays[key]=np.empty(shape,dtype=dtype)
    arrays['dt']=np.asarray(dt if dt is not None else (arrays['time_s'][1]-arrays['time_s'][0] if t else .1))
    arrays['trace_schema_version']=np.asarray(TRACE_SCHEMA_VERSION)
    validate_trace(arrays)
    np.savez_compressed(path,**arrays)
    return {key:list(value.shape) for key,value in arrays.items()}


def read_trace(path, metadata=None):
    """Schema 2 remains renderable; missing historical fields are never invented."""
    with np.load(Path(path),allow_pickle=False) as stored:
        trace={key:stored[key] for key in stored.files}
    return validate_trace(trace,metadata)
