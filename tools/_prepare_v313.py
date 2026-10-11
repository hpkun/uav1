"""Capture v3.12 full trajectories and source hashes before version extension."""
import hashlib
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical, states


def capture312():
    result = {}
    for mode in ('ZERO', 'PURSUIT'):
        for seed in (1, 31, 10000000):
            e = MultiUAVCombatEnv('configs/combat_environment_v312.yaml')
            obs, info = e.reset(seed)
            h = hashlib.sha256(json.dumps(canonical((states(e), obs, info)), sort_keys=True).encode())
            while True:
                action = np.zeros((5, 3)) if mode == 'ZERO' else e.fixed_policy.team_actions(e.red, e.blue)
                obs, reward, term, trunc, info = e.step(action)
                h.update(json.dumps(canonical((states(e), obs, reward, term, trunc, info,
                    e.rng.bit_generator.state)), sort_keys=True).encode())
                if term or trunc:
                    break
            result[f'{mode}_{seed}'] = h.hexdigest()
    return result


def main():
    out = Path('outputs/v313_validation')
    out.mkdir(parents=True, exist_ok=True)
    baseline = out / 'before_hashes.json'
    if baseline.exists():
        raise RuntimeError('baseline already exists')
    baseline.write_text(json.dumps({p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env', 'algorithm', 'configs') for p in Path(folder).rglob('*')
        if p.suffix in ('.py', '.yaml')}, indent=2))
    Path('tests/v313_v312_reference.json').write_text(json.dumps(capture312(), indent=2))


if __name__ == '__main__':
    main()
