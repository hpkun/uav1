"""Paired complete episodes: frozen actions, exact dynamics/R1-R3 and R4 ledger.

Four fixed policies, at least 20 identical seeds per version for acceptance.
No weights/training; CUDA is explicitly required by repository audit policy.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.geometry import engagement_geometry
from env.v346_reward import tactical_pair_rewards, tactical_rewards
from tools.audit_v339_attack_geometry import red_actions, POLICIES, states


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def episode_pair(policy,seed):
    envs=[MultiUAVCombatEnv(load_config(ROOT/f'configs/combat_environment_v{v}.yaml')) for v in ('344','346')]
    a,b=envs
    x,y=a.reset(seed),b.reset(seed)
    np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(states(a),states(b))
    initial=hashlib.sha256(states(a).tobytes()).hexdigest()
    initial_r4_nonzero=int(np.count_nonzero(b.dense_combat_reward()[0]))
    sums=np.zeros((2,4,4));totals=np.zeros((2,4));nz=np.zeros(2,dtype=int)
    applied=np.zeros((2,3),dtype=int);candidate=applied.copy();aspects=np.zeros(3,dtype=int)
    samples=0;physics=hashlib.sha256();actions_hash=hashlib.sha256();max_step=0.;max_delta=0.
    phi=a.potential()[0];discounted=np.zeros((2,4));observations=[]
    original=b.dense_combat_reward
    def observe():
        nonlocal samples
        off,de=tactical_pair_rewards(b.red,b.blue,b.config['reward'])
        r=tactical_rewards(b.red,b.blue,b.config['reward'])
        samples+=sum(s.alive for s in b.red)
        for i,(pos,neg) in enumerate(((.1,-.15),(.02,-.025),(.01,-.015))):
            candidate[0,i]+=int(np.count_nonzero((off==pos).any(axis=1)))
            candidate[1,i]+=int(np.count_nonzero((de==neg).any(axis=1)))
            applied[0,i]+=int(np.count_nonzero(r['tactical_offense']==pos))
            applied[1,i]+=int(np.count_nonzero(r['tactical_defense']==neg))
        for own in b.red:
            for target in b.blue:
                if own.alive and target.alive:
                    aa=abs(engagement_geometry(own,target).aa)
                    aspects[0 if aa>=5*np.pi/6 else 2 if aa<=np.pi/6 else 1]+=1
        return original()
    b.dense_combat_reward=observe
    while True:
        action,_=red_actions(a,policy)
        np.testing.assert_array_equal(action,red_actions(b,policy)[0])
        actions_hash.update(action.tobytes())
        results=[e.step(action) for e in envs]
        x,y=results
        np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(x[0],y[0])
        assert x[2:4]==y[2:4]
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        physics.update(states(a).tobytes())
        for key in x[4]:
            if ('dwell_' in key or key.endswith(('attack_kills','losses'))
                    or key in ('executed_blue_actions','blue_pursuit_target_indices','termination_reason',
                               'phi_current','phi_next','phi_next_for_shaping')):
                np.testing.assert_equal(x[4][key],y[4][key])
        parts=[]
        for j,result in enumerate(results):
            info=result[4];r=result[1];p=np.stack([info[f'r{k}_rewards'] for k in range(1,5)])
            np.testing.assert_array_equal(r,p.sum(axis=0))
            max_step=max(max_step,float(np.max(np.abs(r-p.sum(axis=0)))))
            sums[j]+=p;totals[j]+=r;nz[j]+=np.count_nonzero(p[3]);parts.append(p)
            discounted[j]+=.99**(a.steps-1)*p[2]
        np.testing.assert_array_equal(parts[0][:3],parts[1][:3])
        delta=(x[1]-y[1])-(parts[0][3]-parts[1][3])
        max_delta=max(max_delta,float(np.max(np.abs(delta))))
        np.testing.assert_allclose(delta,0,atol=4e-15,rtol=0)
        if x[2] or x[3]:break
    rows=[]
    for j,(e,result) in enumerate(zip(envs,results)):
        info=result[4]
        np.testing.assert_allclose(totals[j],sums[j].sum(axis=0),atol=4e-12,rtol=0)
        np.testing.assert_allclose(discounted[j],-phi,atol=5e-15,rtol=0)
        for k in range(4):np.testing.assert_array_equal(sums[j,k],info[f'episode_r{k+1}_rewards'])
        row=dict(version=e.environment_version,policy=policy,seed=seed,steps=e.steps,
            red_win=info['red_win'],blue_win=info['blue_win'],draw=info['draw'],
            mutual_annihilation=info['termination_reason']=='draw_mutual_destruction',
            red_losses=info['red_losses'],blue_losses=info['blue_losses'],
            initial_sha256=initial,trajectory_sha256=physics.hexdigest(),actions_sha256=actions_hash.hexdigest(),
            live_red_agent_steps=samples,r4_nonzero_samples=int(nz[j]),
            initial_paper_r4_nonzero=initial_r4_nonzero,
            ledger_max_abs_error=float(np.max(np.abs(totals[j]-sums[j].sum(axis=0)))))
        for k in range(4):row[f'r{k+1}_sum']=float(sums[j,k].sum())
        for side in ('red','blue'):
            for key in ('dwell_started','dwell_completed','dwell_neutralizations'):
                row[f'{side}_{key}']=int(info[f'{side}_{key}'])
        rows.append(row)
    for branch,ix in (('offense',0),('defense',1)):
        for i,degree in enumerate((5,15,30)):
            rows[1][f'{branch}_{degree}_samples']=int(applied[ix,i])
            rows[1][f'{branch}_{degree}_candidate_samples']=int(candidate[ix,i])
    for label,n in zip(('head_on','side','tail'),aspects):
        for row in rows:row[f'{label}_pair_steps']=int(n)
    return rows,dict(policy=policy,seed=seed,complete_episode_exact=True,
        initial_state_motion_observation_events_dwell_rng_r1_r2_r3_equal=True,
        max_step_ledger_error=max_step,max_reward_delta_residual=max_delta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.seeds<20 or args.workers<1:p.error('acceptance requires >=20 seeds and positive workers')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True,exist_ok=True);rows=[];paired=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(episode_pair,policy,seed) for policy in POLICIES for seed in range(args.seeds)]
        for i,f in enumerate(as_completed(futures),1):
            group,proof=f.result();rows.extend(group);paired.append(proof)
            if i%5==0:print(f'{i}/{len(futures)} paired episodes verified',flush=True)
    summaries=[]
    for version in ('3.44','3.46'):
        for policy in POLICIES:
            group=[r for r in rows if r['version']==version and r['policy']==policy]
            s=dict(version=version,policy=policy,episodes=len(group))
            for key in ('red_win','blue_win','draw','mutual_annihilation','red_losses','blue_losses','steps',
                        'r1_sum','r2_sum','r3_sum','r4_sum',*[f'{side}_{key}' for side in ('red','blue')
                        for key in ('dwell_started','dwell_completed','dwell_neutralizations')]):
                s[key+'_mean']=float(np.mean([r[key] for r in group]))
            denom=sum(r['live_red_agent_steps'] for r in group)
            s['r4_nonzero_fraction']=sum(r['r4_nonzero_samples'] for r in group)/max(1,denom)
            s['initial_paper_r4_nonzero']=sum(r['initial_paper_r4_nonzero'] for r in group)
            for label in ('head_on','side','tail'):
                s[label+'_fraction']=sum(r[label+'_pair_steps'] for r in group)/max(1,sum(
                    r[k+'_pair_steps'] for r in group for k in ('head_on','side','tail')))
            if version=='3.46':
                for branch in ('offense','defense'):
                    for deg in (5,15,30):
                        for suffix in ('samples','candidate_samples'):
                            s[f'{branch}_{deg}_{suffix}_fraction']=sum(r[f'{branch}_{deg}_{suffix}'] for r in group)/max(1,denom)
                s['sparsity_risk_below_1pct']=s['r4_nonzero_fraction']<.01
            summaries.append(s)
    rows.sort(key=lambda r:(r['version'],r['policy'],r['seed']))
    for name,data in (('episodes',rows),('paired',paired),('comparison',summaries)):
        write_csv(args.output/f'{name}.csv',data)
    payload=dict(episodes=len(rows),paired_episodes=len(paired),summary=summaries,
        all_complete_pairs_exact=True,cuda_available=True,formal_training_started=False,
        aspect_denominator='all living Red-Blue pair steps before settlement; head abs(AA)>=150, tail <=30, side otherwise',
        tier_denominator='living Red agent steps before settlement; applied mutually exclusive; candidates may overlap',
        max_step_ledger_error=max(r['max_step_ledger_error'] for r in paired),
        max_reward_delta_residual=max(r['max_reward_delta_residual'] for r in paired),
        max_episode_ledger_error=max(r['ledger_max_abs_error'] for r in rows))
    (args.output/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
