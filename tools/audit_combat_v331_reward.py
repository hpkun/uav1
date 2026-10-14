"""CUDA-guarded, network-free R4 scale audit on identical fixed-policy paths."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v323_reward import tactical_rewards_v323

COMPONENTS = ('r1', 'r2', 'r3', 'tactical_offense', 'tactical_defense')


def pair_states(distance=2000., azimuth=0., elevation=0., aspect=0., threat=False):
    az, el, aa = np.deg2rad([azimuth, elevation, aspect])
    own = AircraftState(0., 0., -3000., 225., 0., 0.)
    target = AircraftState(distance*np.cos(el)*np.cos(az),
                           distance*np.cos(el)*np.sin(az),
                           -3000.-distance*np.sin(el), 225., theta=0., psi=az+aa)
    red, blue = ([target], [own]) if threat else ([own], [target])
    return red, blue


def static_table(old, new):
    cases = [dict(case=f'{side}_{level}', azimuth=angle, threat=side=='threat')
             for side in ('offense', 'threat')
             for level, angle in ((30,20.),(15,10.),(5,2.))]
    cases += [dict(case='AA_gate',aspect=31.), dict(case='azimuth_gate',azimuth=31.),
              dict(case='elevation_gate',elevation=31.), dict(case='distance_zero',distance=3000.),
              dict(case='R3_active',distance=4000.), dict(case='full_w4',distance=800.),
              dict(case='AA_gate_reverse',aspect=31.,threat=True),
              dict(case='azimuth_gate_reverse',azimuth=31.,threat=True),
              dict(case='elevation_gate_reverse',elevation=31.,threat=True)]
    rows = []
    for case in cases:
        kwargs={k:v for k,v in case.items() if k!='case'}
        red,blue=pair_states(**kwargs)
        a=tactical_rewards_v323(red,blue,old['reward'],old['weapon'])
        b=tactical_rewards_v323(red,blue,new['reward'],new['weapon'])
        forward=engagement_geometry(red[0],blue[0])
        reverse=engagement_geometry(blue[0],red[0])
        row=dict(case=case['case'],distance=forward.distance,
                 body_azimuth=np.rad2deg(forward.boresight_azimuth_error),
                 body_elevation=np.rad2deg(forward.boresight_elevation_error),AA=np.rad2deg(forward.aa),
                 reverse_body_azimuth=np.rad2deg(reverse.boresight_azimuth_error),
                 reverse_body_elevation=np.rad2deg(reverse.boresight_elevation_error),reverse_AA=np.rad2deg(reverse.aa))
        for name in ('guide','tactical_offense','tactical_defense'):
            x,y=float(a[name][0]),float(b[name][0]);scale=1 if name=='guide' else 2
            assert y==scale*x
            row[name]=dict(v330=x,v331=y,ratio=y/x if x else None)
        np.testing.assert_allclose(b['tactical'],a['guide']+2*(a['tactical_offense']+a['tactical_defense']),rtol=0,atol=1e-16)
        rows.append(row)
    return rows


def statistics(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),
                max_absolute=float(np.abs(values).max()))


def summarize(rows,version):
    names=(*COMPONENTS,'r4','dense','total')
    result={}
    for scope in ('episode_team_sum','episode_agent'):
        result[scope]={}
        for name in names:
            if scope=='episode_team_sum':
                values=[np.sum(row[version][name]) for row in rows]
            else:
                values=[value for row in rows for value in row[version][name]]
            result[scope][name]=statistics(values)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v331_validation/reward_audit.json')
    args=p.parse_args()
    if args.episodes<200:raise ValueError('at least 200 seeds required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audit')
    old=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v330.yaml')
    new=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v331.yaml')
    table=static_table(old.config,new.config)
    policy=NearestTargetPursuitPolicy(old.config['blue_policy'],old.config['action'])
    rows=[];steps=0
    for seed in range(args.episodes):
        a,_=old.reset(seed);b,_=new.reset(seed);np.testing.assert_array_equal(a,b)
        while True:
            actions=policy.team_actions(old.red,old.blue)
            a,ra,ta,xa,ia=old.step(actions);b,rb,tb,xb,ib=new.step(actions);steps+=1
            np.testing.assert_array_equal(a,b)
            assert old.red==new.red and old.blue==new.blue and (ta,xa)==(tb,xb)
            assert old.rng.bit_generator.state==new.rng.bit_generator.state
            for field in ('red_ammo','blue_ammo','red_fire_cooldown_steps','blue_fire_cooldown_steps','blue_step_target_indices'):
                np.testing.assert_array_equal(getattr(old,field),getattr(new,field))
            for side in ('red','blue'):
                assert old.combat_counts[side]==new.combat_counts[side]
            for field in ('red_win','blue_win','draw','red_success','termination_reason'):
                assert ia[field]==ib[field]
            for c in ('r1','r2','r3'):
                np.testing.assert_array_equal(ia[c+'_rewards'],ib[c+'_rewards'])
            for c in ('tactical_offense','tactical_defense','r4'):
                np.testing.assert_array_equal(2*ia[c+'_rewards'],ib[c+'_rewards'])
            np.testing.assert_allclose(rb,ia['r1_rewards']+ia['r2_rewards']+ia['r3_rewards']+2*ia['r4_rewards'],rtol=0,atol=1e-14)
            if ta or xa:break
        row=dict(seed=seed,episode_length=old.steps,
                 **{k:ia[k] for k in ('red_win','blue_win','draw','termination_reason','red_attack_kills','blue_attack_kills')},
                 timeout=xa,mutual_destruction=not old.red_alive_mask.any() and not old.blue_alive_mask.any(),
                 red_noncombat_losses=ia['red_boundary_exits']+ia['red_ground_losses'],
                 blue_noncombat_losses=ia['blue_boundary_exits']+ia['blue_ground_losses'])
        for version,e in (('v330',old),('v331',new)):
            d={c:e.episode_reward_components[c].tolist() for c in COMPONENTS}
            d['r4']=(np.array(d['tactical_offense'])+d['tactical_defense']).tolist()
            d['dense']=(np.array(d['r3'])+d['r4']).tolist();d['total']=e.episode_extended_rewards.tolist()
            row[version]=d
        rows.append(row)
        if (seed+1)%10==0:
            print(f'[REWARD AUDIT] {seed+1}/{args.episodes}',flush=True)
    totals={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction',
             'red_attack_kills','blue_attack_kills','red_noncombat_losses','blue_noncombat_losses')}
    result=dict(episodes=args.episodes,checked_transitions=steps,static_geometry=table,totals=totals,
                r4_active_episodes=sum(any(value!=0 for term in ('tactical_offense','tactical_defense')
                                          for value in row['v330'][term]) for row in rows),
                same_trajectory_checks='Every transition: exact states/obs/RNG/ammo/cooldown/targets/events/outcome/R1-R3; exact per-step R4 x2.',
                statistics={v:summarize(rows,v) for v in ('v330','v331')},rows=rows,
                note='No network training; policy mathematics/seeds match v3.30 fairness audit. Ratios for zero terms are undefined.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(totals=totals,statistics=result['statistics']),indent=2),flush=True)


if __name__=='__main__':main()
