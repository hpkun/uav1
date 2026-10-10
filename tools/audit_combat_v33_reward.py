"""Short paired reward/physical isolation audit, never RL training or tuning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from tools._capture_v33_reference import canonical,states
from tools.audit_v30_initial_policy_baselines import jsonable
from algorithm.common.evaluator import aggregate_combat_records

REWARD_FIELDS={'local_rewards','r1_rewards','r2_rewards','event_rewards','outcome_rewards',
    'individual_event_rewards','team_casualty_rewards'}|{f'episode_{name}_{suffix}'
    for name in ('r1','r2','event','outcome','individual_event','team_casualty') for suffix in ('rewards','total')}

def paired_rollout(seed,mode):
    old=MultiUAVCombatEnv(load_config(ROOT/'configs/combat_environment_v32.yaml'))
    new=MultiUAVCombatEnv(load_config(ROOT/'configs/combat_environment_v33.yaml'))
    left,_=old.reset(seed);right,_=new.reset(seed);rng=np.random.default_rng(seed+100000000)
    returns=np.zeros((2,5));steps=0
    while True:
        np.testing.assert_array_equal(states(old),states(new));np.testing.assert_array_equal(left,right)
        before=new.red_alive_mask.copy()
        actions=np.zeros((5,3)) if mode=='ZERO' else old.fixed_policy.team_actions(old.red,old.blue) if mode=='MIRROR' else rng.uniform(-1,1,(5,3))
        blue=old.fixed_policy.team_actions(old.blue,old.red)
        np.testing.assert_array_equal(blue,new.fixed_policy.team_actions(new.blue,new.red))
        left,lr,lt,ltr,li=old.step(actions,blue);right,nr,nt,ntr,ni=new.step(actions,blue);steps+=1
        np.testing.assert_array_equal(states(old),states(new));np.testing.assert_array_equal(left,right)
        assert (lt,ltr)==(nt,ntr) and old.rng.bit_generator.state==new.rng.bit_generator.state
        # Compare every non-reward info field, not just selected combat counters.
        old_info={k:v for k,v in li.items() if k not in REWARD_FIELDS|{'environment_version'}}
        new_info={k:v for k,v in ni.items() if k not in REWARD_FIELDS|{'environment_version'}}
        assert canonical(old_info)==canonical(new_info)
        losses=int(np.count_nonzero(before-new.red_alive_mask))
        np.testing.assert_array_equal(ni['team_casualty_rewards'],-2*losses*before.astype(float))
        np.testing.assert_array_equal(ni['individual_event_rewards'],li['event_rewards'])
        np.testing.assert_array_equal(ni['event_rewards'],ni['individual_event_rewards']+ni['team_casualty_rewards'])
        np.testing.assert_array_equal(nr,sum((ni[f'{k}_rewards'] for k in ('event','outcome','adv','safe')),np.zeros(5)).astype(np.float32))
        assert np.isfinite(nr).all()
        returns[0]+=lr;returns[1]+=nr
        if nt or ntr:
            np.testing.assert_array_equal(ni['phi_next_for_shaping'],np.zeros(5))
            np.testing.assert_array_equal(ni['adv_rewards'],-ni['phi_current'])
            for label,row,total in (('v32',li,returns[0]),('v33',ni,returns[1])):
                np.testing.assert_allclose(total.sum(),sum(row[f'episode_{k}_total'] for k in ('event','outcome','adv','safe')),rtol=0,atol=2e-4)
            return jsonable(dict(seed=seed,mode=mode,steps=steps,physics_info_rng_mismatches=0,
                v32=dict(li,episode_return=returns[0].sum(),mean_agent_episode_return=returns[0].mean()),
                v33=dict(ni,episode_return=returns[1].sum(),mean_agent_episode_return=returns[1].mean())))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args()
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('use a fresh output directory')
    rows=[]
    for mode in ('ZERO','MIRROR','RANDOM'):
        for seed in range(38000000,38000016):
            rows.append(paired_rollout(seed,mode))
        print(f'[REWARD AUDIT] {mode}:16 paired episodes, no mismatches',flush=True)
    summary=dict(paired_episodes=48,physical_episodes=96,seed_base=38000000,seeds_per_mode=16,
        state_observation_combat_ammo_outcome_info_rng_mismatches=0,
        groups={mode:{v:aggregate_combat_records([r[v] for r in rows if r['mode']==mode]) for v in ('v32','v33')}
            for mode in ('ZERO','MIRROR','RANDOM')},parameters_tuned=False,formal_training_started=False)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
