"""Capture complete v3.11 trajectories and historical entity actors before edits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical, states
from algorithm.ea_mappo.networks import EntityAttentionActor
from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor


def capture311():
    results = {}
    for mode in ('ZERO', 'PURSUIT'):
        for seed in (1, 31, 10000000):
            e = MultiUAVCombatEnv('configs/combat_environment_v311.yaml')
            obs, info = e.reset(seed)
            h = hashlib.sha256(json.dumps(canonical((states(e), obs, info)), sort_keys=True).encode())
            while True:
                action = np.zeros((5,3)) if mode == 'ZERO' else e.fixed_policy.team_actions(e.red, e.blue)
                obs, reward, term, trunc, info = e.step(action)
                h.update(json.dumps(canonical((states(e), obs, reward, term, trunc, info,
                    e.rng.bit_generator.state)), sort_keys=True).encode())
                if term or trunc:
                    break
            results[f'{mode}_{seed}'] = h.hexdigest()
    return results


def actor_reference():
    result = {}
    for kind, cls in (('ea', EntityAttentionActor), ('stea', SpatioTemporalEntityAttentionActor)):
        for dimension, self_dim in ((52,7),(65,7),(66,8),(104,7)):
            torch.manual_seed(71)
            actor = cls(self_feature_dim=self_dim)
            obs = torch.rand(2, dimension)
            alive = torch.ones(2)
            own, allies, enemies = __import__('algorithm.stea_mappo.networks', fromlist=['decompose_observations']).decompose_observations(obs)
            allies[..., -1] = 1.
            enemies[..., -1] = 1.
            allies[0,0,-1] = 0.
            with torch.no_grad():
                if kind == 'ea':
                    output = actor(obs)
                else:
                    output = actor(obs, torch.ones(2,128), alive, torch.tensor(0.))[0]
            h = hashlib.sha256()
            for key, value in actor.state_dict().items():
                h.update(key.encode())
                h.update(value.detach().numpy().tobytes())
            h.update(output.numpy().tobytes())
            result[f'{kind}_{dimension}'] = h.hexdigest()
    return result


def main():
    out = Path('outputs/v312_validation')
    out.mkdir(parents=True, exist_ok=True)
    baseline = out/'before_hashes.json'
    if baseline.exists():
        raise RuntimeError('baseline already exists')
    baseline.write_text(json.dumps({p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*')
        if p.suffix in ('.py','.yaml')}, indent=2))
    Path('tests/v312_v311_reference.json').write_text(json.dumps(capture311(), indent=2))
    Path('tests/v312_legacy_actor_reference.json').write_text(json.dumps(actor_reference(), indent=2))


if __name__ == '__main__':
    main()
