"""Capture historical v3.9 traces before adding the isolated Blue policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical, states


def capture39():
    result = {}
    for mode in ('ZERO', 'MIRROR'):
        for seed in (1, 31, 10000000, 40000000):
            e = MultiUAVCombatEnv('configs/combat_environment_v39.yaml')
            obs, info = e.reset(seed)
            h = hashlib.sha256()
            h.update(json.dumps(canonical((states(e), obs, info)), sort_keys=True).encode())
            while True:
                action = np.zeros((5, 3)) if mode == 'ZERO' else e.fixed_policy.team_actions(e.red, e.blue)
                obs, reward, term, trunc, info = e.step(action)
                h.update(json.dumps(canonical((states(e), obs, reward, term, trunc, info,
                    e.rng.bit_generator.state, e.fixed_policy.rng.bit_generator.state)), sort_keys=True).encode())
                if term or trunc:
                    break
            result[f'{mode}_{seed}'] = h.hexdigest()
    return result


def main():
    out = Path('outputs/v310_validation')
    out.mkdir(parents=True, exist_ok=True)
    baseline = out / 'before_hashes.json'
    if baseline.exists():
        raise RuntimeError('historical baseline already exists; do not overwrite')
    baseline.write_text(json.dumps({p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env', 'algorithm', 'configs') for p in Path(folder).rglob('*')
        if p.suffix in ('.py', '.yaml')}, indent=2))
    Path('tests/v310_v39_reference.json').write_text(json.dumps(capture39(), indent=2))


if __name__ == '__main__':
    main()
