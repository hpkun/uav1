"""Shared-Gaussian analytic/Monte Carlo and normalized-distance validation."""
from pathlib import Path
from statistics import NormalDist
import argparse
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from tools.audit_v339_attack_geometry import write_csv

ANGLES=((0,0),(15,15),(30,30),(15,-15),(30,-30))


def analytic(weapon,distance,ata,ha):
    threshold=math.pi*math.exp(-distance/weapon.effective_hit_distance)
    lo=max((-threshold-ata)/weapon.azimuth_noise_scale,(-threshold-ha)/weapon.elevation_noise_scale)
    hi=min((threshold-ata)/weapon.azimuth_noise_scale,(threshold-ha)/weapon.elevation_noise_scale)
    return max(0.,NormalDist().cdf(hi)-NormalDist().cdf(lo)) if hi>=lo else 0.


def weapon(version):
    return MultiUAVCombatEnv(load_config(ROOT/f'configs/combat_environment_v{version}.yaml')).weapon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=200000)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v341_hit_calibration_preflight')
    args=p.parse_args()
    if args.samples<1000:p.error('at least 1000 samples required')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    ws={v:weapon(v) for v in ('339','340','341')}
    rows=[];max_error=0.;max_z=0.
    for di,d in enumerate((500,1000,1500,2000)):
        for ai,(ata_deg,ha_deg) in enumerate(ANGLES):
            ata,ha=np.deg2rad([ata_deg,ha_deg]);seed=341000+di*10+ai
            for v in ('340','341'):
                w=ws[v];formal=w.expected_hit_probability(d,ata,ha);independent=analytic(w,d,ata,ha)
                assert abs(formal-independent)<1e-15
                rng=np.random.default_rng(seed)
                hits=sum(w.attempt_hit(rng,d,ata,ha) for _ in range(args.samples))
                frequency=hits/args.samples;se=math.sqrt(formal*(1-formal)/args.samples)
                error=abs(frequency-formal);limit=6*se+2/args.samples
                assert error<=limit,(v,d,ata_deg,ha_deg,frequency,formal)
                z=error/se if se>1e-12 else 0.
                max_z=max(max_z,z);max_error=max(max_error,error)
                rows.append(dict(version='3.'+v[1:],distance_m=d,range_fraction=d/2000,ata_deg=ata_deg,ha_deg=ha_deg,
                    threshold_rad=math.pi*math.exp(-d/w.effective_hit_distance),probability=formal,
                    independent_probability=independent,samples=args.samples,seed=seed,mc_hits=hits,mc_frequency=frequency,
                    mc_abs_error=error,mc_standard_error=se,mc_z=z,acceptance_abs_error=limit,
                    method='actual scalar attempt_hit; one shared epsilon per shot; six-SE+2/n bound'))
    write_csv(out/'probability_audit.csv',rows)
    normalized=[]
    for ratio in (.25,.5,.75,1.,1.5,2.):
        for ata_deg,ha_deg in ANGLES:
            ata,ha=np.deg2rad([ata_deg,ha_deg]);d=ratio*1000
            p40=ws['340'].expected_hit_probability(d,ata,ha)
            p41=ws['341'].expected_hit_probability(2*d,ata,ha)
            p39=ws['339'].expected_hit_probability(d,ata,ha)
            assert abs(p40-p41)<1e-15 and p39==p40
            t39=math.pi*math.exp(-d/ws['339'].effective_hit_distance)
            t41=math.pi*math.exp(-2*d/ws['341'].effective_hit_distance)
            assert t39==t41
            normalized.append(dict(historical_range_fraction=ratio,ata_deg=ata_deg,ha_deg=ha_deg,
                v339_distance_m=d,v340_distance_m=d,v341_distance_m=2*d,
                v339_probability=p39,v340_probability=p40,v341_probability=p41,
                v339_threshold=t39,v341_threshold=t41,
                note='theory beyond legal ranges remains a mathematical diagnostic'))
    write_csv(out/'normalized_probability.csv',normalized)
    # Actual methods must consume exactly one scalar normal, even when intervals do not overlap.
    class Spy:
        def __init__(self,seed):self.rng=np.random.default_rng(seed);self.calls=0
        def normal(self,mean,std):
            assert (mean,std)==(0.,1.);self.calls+=1;return self.rng.normal(mean,std)
    x,y=Spy(3141),Spy(3141);differences=0
    for i in range(10000):
        a,b=map(math.radians,ANGLES[i%len(ANGLES)])
        differences+=ws['340'].attempt_hit(x,1500,a,b)!=ws['341'].attempt_hit(y,1500,a,b)
        assert x.calls==y.calls==i+1 and x.rng.bit_generator.state==y.rng.bit_generator.state
    result=dict(cases=len(rows),normalized_cases=len(normalized),samples_per_case=args.samples,
        actual_attempt_hit_samples=len(rows)*args.samples,mc_max_abs_error=max_error,mc_max_z=max_z,
        rng_calls_each=x.calls,rng_state_equal_each_step=True,different_hit_results=differences,
        effective_hit_distance=ws['341'].effective_hit_distance,
        calibration_origin='historical project v3.35 convention, not a paper-specified numerical value')
    (out/'probability_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
