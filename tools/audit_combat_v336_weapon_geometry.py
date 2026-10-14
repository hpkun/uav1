"""Read-only same-snapshot body versus ATA/HA windows and short policy rollouts."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.fixed_policy import NearestTargetPursuitPolicy
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v317_weapon import BodyFrameGaussianWeapon
from env.v336_weapon import PaperATAHAWeapon
from env.v36_geometry import angle_within


def compare_snapshot(attackers, targets, ammo, body, paper, emit=None, **context):
    """No mutation or random draws. Both weapons receive identical states/ammo."""
    old, _ = body.eligibility(attackers, targets, ammo)
    new, _ = paper.eligibility(attackers, targets, ammo)
    counts = Counter(dict(live_pair_steps=0, range_qualified=0, range_ata_qualified=0,
        range_ata_ha_qualified=0, v335_eligible=0, v336_eligible=0,
        agreement=0, both_eligible=0, both_ineligible=0, v335_only=0, v336_only=0))
    for i,a in enumerate(attackers):
        for j,b in enumerate(targets):
            if not a.alive or not b.alive:
                continue
            g = engagement_geometry(a,b)
            in_range = paper.range_min <= g.distance <= paper.range_max
            ata_ok = in_range and angle_within(abs(g.ata),paper.ata_max)
            full = ata_ok and angle_within(abs(g.ha),paper.ha_max)
            x,y = bool(old[i,j]),bool(new[i,j])
            for key,value in dict(live_pair_steps=1,range_qualified=in_range,
                range_ata_qualified=ata_ok,range_ata_ha_qualified=full,
                v335_eligible=x,v336_eligible=y,agreement=x==y,
                both_eligible=x and y,both_ineligible=not x and not y,
                v335_only=x and not y,v336_only=y and not x).items():
                counts[key] += int(value)
            if x != y and emit is not None:
                emit(dict(**context,attacker=i,target=j,ammo=int(ammo[i]),distance_m=g.distance,
                    ata_deg=math.degrees(g.ata),ha_deg=math.degrees(g.ha),
                    body_azimuth_deg=math.degrees(g.boresight_azimuth_error),
                    body_elevation_deg=math.degrees(g.boresight_elevation_error),
                    attacker_theta_deg=math.degrees(a.theta),v335_eligible=x,v336_eligible=y,
                    v335_expected_hit_probability=body.expected_hit_probability(g.distance,g.boresight_azimuth_error,g.boresight_elevation_error),
                    v336_expected_hit_probability=paper.expected_hit_probability(g.distance,g.ata,g.ha)))
    return counts


class GeometryObserver:
    """Wrap only the pre-fire hook, forward it once, and restore it on exit."""
    def __init__(self, environment, body, paper, emit=None, **context):
        self.environment,self.body,self.paper = environment,body,paper
        self.emit,self.context = emit,context
        self.counts = {s:Counter() for s in ('red','blue')}
        self.first_fire = {s:None for s in ('red','blue')}

    def __enter__(self):
        e = self.environment
        self.present = '_entry_attempts' in e.__dict__
        self.previous = e.__dict__.get('_entry_attempts')
        self.original = e._entry_attempts
        def entry(attackers,targets,states,side):
            row = compare_snapshot(attackers,targets,getattr(e,side+'_ammo'),self.body,self.paper,
                self.emit,**self.context,side=side,step=e.steps)
            self.counts[side].update(row)
            result = self.original(attackers,targets,states,side)
            if result and self.first_fire[side] is None:
                self.first_fire[side] = dict(step=e.steps,time_s=e.steps*e.dt,
                    distances_m=[engagement_geometry(attackers[i],targets[j]).distance for i,j,_ in result])
            return result
        e._entry_attempts = entry
        return self

    def __exit__(self,*args):
        if self.present:self.environment._entry_attempts = self.previous
        else:del self.environment._entry_attempts


def policy_actions(environment, policy, mode):
    """Probe only Red actions; Blue always retains its automatic fixed policy."""
    actions = policy.team_actions(environment.red,environment.blue)
    if mode == 'pitch_probe':
        # Deterministic diagnostic input, never an environment/policy change.
        desired = np.deg2rad(20)*np.sin((environment.steps+np.arange(4)*20)*2*math.pi/100)
        maximum = environment.config['action']['command']['pitch_delta_max']
        actions[:,1] = np.clip((desired-np.asarray([a.theta for a in environment.red]))/maximum,-1,1)
    return actions


def rollout(version,mode,seeds,body,paper,emit):
    e = MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml')
    policy = NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    rows = []
    totals = {s:Counter() for s in ('red','blue')}
    for seed in seeds:
        e.reset(seed)
        assert [e.fixed_policy.nearest_target_index(b,e.red) for b in e.blue] == [0]*4
        with GeometryObserver(e,body,paper,emit,source=f'v{version}_{mode}',seed=seed) as observer:
            while True:
                _,_,terminated,truncated,info = e.step(policy_actions(e,policy,mode))
                if terminated or truncated:break
        row = dict(seed=seed,episode_length=e.steps,red_win=bool(info['red_win']),
            blue_win=bool(info['blue_win']),draw=bool(info['draw']),timeout=bool(truncated),
            mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),sides={})
        for side in ('red','blue'):
            values = dict(observer.counts[side])
            values.update({k:int(e.combat_counts[side][k]) for k in ('fire_attempts','weapon_hits','attack_kills','ammo_used')})
            row['sides'][side] = dict(**values,first_fire=observer.first_fire[side])
            totals[side].update(values)
        rows.append(row)
        if (seed+1)%5 == 0:print(f'{version} {mode}: {seed+1}/{len(seeds)}',flush=True)
    return dict(episodes=len(rows),totals={s:dict(v) for s,v in totals.items()},
        outcomes={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction')},
        mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),rows=rows)


def grid(body,paper,emit):
    totals = Counter()
    for d in (500,1000):
        for ata_deg in range(-40,41,5):
            for ha_deg in range(-40,41,5):
                for theta_deg in (-40,-20,0,20,40):
                    ata,ha,theta = np.deg2rad([ata_deg,ha_deg,theta_deg])
                    a = AircraftState(0,0,-3000,225,theta,0)
                    b = AircraftState(d*np.cos(ha)*np.cos(ata),d*np.cos(ha)*np.sin(ata),-3000-d*np.sin(ha),225,0,math.pi)
                    # Rounding can put the nominal 1 km point one ULP above it.
                    # Keep the intended grid point strictly in range; production
                    # eligibility retains its unchanged strict range comparison.
                    g = engagement_geometry(a,b)
                    if g.distance>d:
                        scale = np.nextafter(float(d),0)/g.distance
                        b.x*=scale;b.y*=scale;b.z=a.z+(b.z-a.z)*scale
                    totals.update(compare_snapshot([a],[b],[6],body,paper,emit,
                        source='controlled_geometry_grid',seed=-1,side='single',step=0))
    return dict(totals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v336_validation/geometry_audit')
    args = p.parse_args()
    if args.episodes<=0:raise ValueError('episodes must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for audits; no CPU fallback')
    out = args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Use a fresh audit output directory')
    body = BodyFrameGaussianWeapon(**load_config(ROOT/'configs/combat_environment_v335.yaml')['weapon'])
    paper = PaperATAHAWeapon(**load_config(ROOT/'configs/combat_environment_v336.yaml')['weapon'])
    fields = ['source','seed','side','step','attacker','target','ammo','distance_m','ata_deg','ha_deg',
        'body_azimuth_deg','body_elevation_deg','attacker_theta_deg','v335_eligible','v336_eligible',
        'v335_expected_hit_probability','v336_expected_hit_probability']
    with (out/'disagreement_pairs.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        result = dict(device=torch.cuda.get_device_name(0),seeds=list(range(args.episodes)),
            comparison='Both eligibility calls use the exact same pre-fire states and ammo; no RNG draws.',
            denominator='Live opposing pair-steps; range/angle counters ignore ammo/cooldown; eligible includes ammo, excludes cooldown.',
            scope='Fixed-pursuit and deterministic Red pitch-probe diagnostics. Blue policy remains automatic. Separate version rollouts can diverge; A/B counts are evaluated within each same snapshot.',
            controlled_grid=grid(body,paper,writer.writerow),rollouts={})
        for mode in ('fixed','pitch_probe'):
            for version in ('335','336'):
                result['rollouts'][f'v{version}_{mode}'] = rollout(version,mode,range(args.episodes),body,paper,writer.writerow)
                (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{n:v[n] for n in ('outcomes','mean_episode_length','totals')} for k,v in result['rollouts'].items()},indent=2))


if __name__ == '__main__':main()
