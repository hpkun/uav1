"""Fixed-seed invariants and symmetric 1000-episode audit; no tuning/training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import json
import multiprocessing as mp
import time
import numpy as np
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.sensor import is_detected
from algorithm.common.evaluator import aggregate_combat_records,episode_return_metrics

CONFIG=None


def init_worker(config):
    global CONFIG
    CONFIG=config


def verify_transition(env,observation,reward,info):
    if not np.isfinite(observation).all() or not np.isfinite(reward).all():raise AssertionError('nonfinite transition')
    for side,opposing in (('red','blue'),('blue','red')):
        ammo=getattr(env,f'{side}_ammo')
        assert ((0<=ammo)&(ammo<=6)).all()
        assert info[f'{side}_ammo_used']==info[f'{side}_fire_attempts']==30-int(ammo.sum())
        assert info[f'{side}_attack_kills']<=info[f'{side}_weapon_hits']<=info[f'{side}_fire_attempts']
        assert info[f'{side}_losses']==info[f'{opposing}_attack_kills']+info[f'{side}_boundary_exits']+info[f'{side}_ground_losses']


def episode(task):
    seed,mode=task
    e=MultiUAVCombatEnv(CONFIG);obs,_=e.reset(seed)
    rng=np.random.default_rng(seed+100_000_000)
    returns=np.zeros(5)
    while True:
        actions=e.fixed_policy.team_actions(e.red,e.blue) if mode=='symmetric' else rng.uniform(-1,1,(5,3))
        obs,reward,terminated,truncated,info=e.step(actions)
        verify_transition(e,obs,reward,info)
        returns+=reward
        if terminated or truncated:
            total,mean=episode_return_metrics(returns)
            assert np.isclose(total,sum(info[f'episode_{n}_total'] for n in ('event','outcome','adv','safe')),atol=2e-4)
            # JSON-safe completed record includes all common evaluator fields.
            record={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in info.items()}
            record.update(seed=seed,episode_return=total,mean_agent_episode_return=mean)
            return record


def invariant_audits(config, samples=10000, hidden_samples=1000):
    rng=np.random.default_rng(300001)
    e=MultiUAVCombatEnv(config);e.reset(1)
    qualified=0
    for _ in range(samples):
        a=AircraftState(*map(float,[0,0,-3000,rng.uniform(150,300),rng.uniform(-np.pi/3,np.pi/3),rng.uniform(-np.pi,np.pi)]))
        b=AircraftState(*map(float,[rng.uniform(-3500,3500),rng.uniform(-3500,3500),rng.uniform(-6000,0),rng.uniform(150,300),rng.uniform(-np.pi/3,np.pi/3),rng.uniform(-np.pi,np.pi)]))
        if e.weapon.qualifies(a,b,6):
            qualified+=1
            assert is_detected(a,b,config['sensor']),'weapon target outside sensor'
    e.red=[AircraftState(0,0,-3000,225,0,0,False) for _ in range(5)];e.red[0].alive=True
    for _ in range(hidden_samples):
        e.blue=[AircraftState(rng.uniform(-5000,-1),rng.uniform(-2000,2000),rng.uniform(-6000,0),rng.uniform(150,300),rng.uniform(-1,1),rng.uniform(-np.pi,np.pi)) for _ in range(5)]
        assert all(not is_detected(e.red[0],b,config['sensor']) for b in e.blue)
        assert not e._observations()[0,36:].any()
        phi,diag=e.potential()
        assert phi[0]==diag['distance'][0]==diag['angle'][0]==0
    return {'random_geometry_samples':samples,'eligible_weapon_cases':qualified,
            'weapon_sensor_violations':0,'hidden_state_samples':hidden_samples,'hidden_information_violations':0}


def summarize(records):
    result=aggregate_combat_records(records)
    result['absolute_side_win_rate_gap']=abs(result['red_win_rate']-result['blue_win_rate'])
    result['red_win_episodes']=sum(r['red_win'] for r in records)
    wins=[r for r in records if r['red_win']]
    result['blue_deaths_in_red_win_episodes']={
        'attack':sum(r['red_attack_kills'] for r in wins),
        'horizontal_boundary':sum(r['blue_boundary_exits']-r['blue_ceiling_losses'] for r in wins),
        'ground':sum(r['blue_ground_losses'] for r in wins),
        'ceiling':sum(r['blue_ceiling_losses'] for r in wins)}
    result['red_win_without_red_attack_kills_count']=sum(r['red_attack_kills']==0 for r in wins)
    result['red_win_blue_majority_noncombat_death_count']=sum(
        r['blue_boundary_exits']+r['blue_ground_losses']>r['red_attack_kills'] for r in wins)
    result['exploit_risk_observed']=bool(result['red_win_blue_majority_noncombat_death_count'])
    result['total_counts']={f'{side}_{key}':sum(r[f'{side}_{key}'] for r in records)
        for side in ('red','blue') for key in ('attack_kills','fire_attempts','ammo_used','boundary_exits','ground_losses','ceiling_losses')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config',default=str(ROOT/'configs/combat_environment_v30.yaml'))
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--random-episodes',type=int,default=32)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--seed-base',type=int,default=30_000_000)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.episodes<1000 or args.random_episodes<1 or args.workers<1:raise ValueError('audit requires >=1000 symmetric episodes')
    # Environment dynamics execute on CPU as in the training workers; no
    # checkpoint or learned actor is loaded. Require the repository audit runtime.
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    config=load_config(args.env_config)
    if config['environment_version']!='3.0':raise ValueError('audit requires v3.0')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    invariant=invariant_audits(config)
    print('[INVARIANTS]',json.dumps(invariant),flush=True)
    start=time.monotonic();records=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker,initargs=(config,)) as pool:
        tasks=[(args.seed_base+i,'symmetric') for i in range(args.episodes)]
        for record in pool.imap_unordered(episode,tasks,chunksize=1):
            records.append(record)
            if len(records)%25==0:
                print(f'[AUDIT] symmetric={len(records)}/{args.episodes} elapsed={time.monotonic()-start:.1f}s',flush=True)
                (output/'progress.json').write_text(json.dumps({'completed':len(records),'target':args.episodes,'summary':summarize(records)},indent=2))
        random_records=list(pool.imap_unordered(episode,[(args.seed_base+1_000_000+i,'random') for i in range(args.random_episodes)]))
    records.sort(key=lambda r:r['seed']);random_records.sort(key=lambda r:r['seed'])
    result={'environment_version':'3.0','seed_base':args.seed_base,'invariants':invariant,
            'symmetric':summarize(records),'random_red':summarize(random_records),
            'elapsed_seconds':time.monotonic()-start,'formal_training_started':False,'parameters_tuned':False}
    (output/'symmetric_episodes.json').write_text(json.dumps(records,indent=2))
    (output/'random_episodes.json').write_text(json.dumps(random_records,indent=2))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
