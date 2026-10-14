"""Read-only theoretical and Monte Carlo checks using the existing shared-Gaussian weapon."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv


def monte_carlo(weapon,distance,azimuth,elevation,samples,seed):
    rng=np.random.default_rng(seed)
    hits=sum(weapon.attempt_hit(rng,distance,azimuth,elevation) for _ in range(samples))
    expected=weapon.expected_hit_probability(distance,azimuth,elevation)
    observed=hits/samples
    standard_error=math.sqrt(expected*(1-expected)/samples)
    return dict(samples=samples,seed=seed,hits=hits,observed_probability=observed,
        expected_probability=expected,error=observed-expected,standard_error=standard_error,
        within_six_standard_errors=abs(observed-expected)<=6*standard_error+1/samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=1000000)
    p.add_argument('--seed',type=int,default=335)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v335_validation/hit_probability_audit.json')
    args=p.parse_args()
    if args.samples<=0:raise ValueError('samples must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits; no CPU fallback')
    envs={v:MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{v}.yaml') for v in ('334','335')}
    perfect={v:{str(d):e.weapon.expected_hit_probability(d,0,0) for d in (0,250,500,750,1000)} for v,e in envs.items()}
    normalized=[]
    for rho in (0,.25,.5,.75,1):
        old=math.pi*math.exp(-rho*4000/(4000/math.log(6)))
        new=math.pi*math.exp(-rho*1000/envs['335'].weapon.effective_hit_distance)
        assert old==new
        normalized.append(dict(rho=rho,threshold_old_4km=old,threshold_new_1km=new,equal=True))
    result=dict(device=torch.cuda.get_device_name(0),samples_per_case=args.samples,
        perfect_alignment=perfect,normalized_threshold=normalized,
        note='Diagnostic only. Every sample calls unchanged BodyFrameGaussianWeapon.attempt_hit, with one shared epsilon. Six-SE is an audit flag, not an environment threshold. Default sample count and angle cases are explicitly chosen diagnostic settings.',cases=[])
    angles=((0,0),(15,15),(30,30),(30,-30))
    for v,e in envs.items():
        for d in (0,250,500,750,1000):
            for az,el in angles:
                # Common seed per case permits paired old/new comparisons; states are not modified.
                row=monte_carlo(e.weapon,d,math.radians(az),math.radians(el),args.samples,args.seed)
                result['cases'].append(dict(version='3.'+v[-2:],distance_m=d,azimuth_deg=az,elevation_deg=el,**row))
            print(f'v{e.environment_version}: Monte Carlo {d} m complete',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('all_six_SE_flags',all(r['within_six_standard_errors'] for r in result['cases']),flush=True)


if __name__=='__main__':main()
