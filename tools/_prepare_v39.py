"""Capture unchanged v3.8 reference and source hashes before extension."""
import hashlib
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical, states


def capture38():
    result = {}
    for mode in ('ZERO', 'MIRROR'):
        for seed in (1, 31, 10000000, 40000000):
            e = MultiUAVCombatEnv('configs/combat_environment_v38.yaml')
            obs, info = e.reset(seed)
            h = hashlib.sha256()
            h.update(json.dumps(canonical((states(e), obs, info)), sort_keys=True).encode())
            while True:
                action = np.zeros((5, 3)) if mode == 'ZERO' else e.fixed_policy.team_actions(e.red, e.blue)
                obs, r, t, tr, info = e.step(action)
                h.update(json.dumps(canonical((states(e), obs, r, t, tr, info,
                    e.rng.bit_generator.state, e.fixed_policy.rng.bit_generator.state)), sort_keys=True).encode())
                if t or tr:
                    break
            result[f'{mode}_{seed}'] = h.hexdigest()
    return result


def main():
    out = Path('outputs/v39_validation')
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'before_hashes.json'
    if target.exists():
        raise RuntimeError('pre-extension baseline already exists; do not overwrite')
    target.write_text(json.dumps({p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env', 'algorithm', 'configs') for p in Path(folder).rglob('*')
        if p.suffix in ('.py', '.yaml')}, indent=2))
    Path('tests/v39_v38_reference.json').write_text(json.dumps(capture38(), indent=2))


if __name__ == '__main__':
    main()
