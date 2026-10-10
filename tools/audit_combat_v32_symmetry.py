"""Five fixed diagnostic seed blocks; identical policies, no tuning/training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse,gzip,json,multiprocessing as mp,time
import numpy as np
from env.config import load_config
from tools import audit_combat_v31_initialization as audit
from tools.audit_v30_initial_policy_baselines import first_order,wilson

BLOCK_BASES=(33000000,34000000,35000000,36000000,37000000)

def distribution(values):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all()
    return dict(count=int(values.size),mean=float(values.mean()),std=float(values.std()),
                min=float(values.min()),max=float(values.max()))

def symmetry_summary(rows):
    result=audit.group_summary(rows);n=len(rows)
    red,blue=result['red_win_rate'],result['blue_win_rate'];gap=red-blue
    stderr=np.sqrt(max(0.,red+blue-gap*gap)/n)
    result.update(signed_gap=gap,absolute_gap=abs(gap),
        signed_gap_95_ci=[max(-1.,gap-1.959963984540054*stderr),min(1.,gap+1.959963984540054*stderr)],
        signed_gap_ci_method='multinomial episode-level normal interval for Red indicator minus Blue indicator',
        blue_win_95_ci=wilson(sum(r['blue_win'] for r in rows),n),
        draw_95_ci=wilson(sum(r['draw'] for r in rows),n))
    for side in ('red','blue'):
        hits=sum(r[f'{side}_weapon_hits'] for r in rows);attempts=sum(r[f'{side}_fire_attempts'] for r in rows)
        result[f'{side}_empirical_hit_probability']=hits/attempts if attempts else None
        result[f'{side}_hit_probability_95_ci']=wilson(hits,attempts)
        result[f'{side}_hit_count']=hits;result[f'{side}_attempt_count']=attempts
    result['first_event_order']={event:{order:sum(first_order(r,event)==order for r in rows)/n
        for order in ('red_first','blue_first','tie','neither')} for event in ('detection','fire_window','attempt','hit','kill')}
    adv=np.array([r['adv_rewards'] for r in rows])
    result['terminal_adv_per_agent']=distribution(adv)
    result['terminal_adv_team_sum']=distribution(adv.sum(1))
    assert all(np.array_equal(r['phi_next_for_shaping'],np.zeros(5)) for r in rows)
    assert all(np.allclose(r['adv_rewards'],-np.asarray(r['phi_current']),rtol=0,atol=1e-14) for r in rows)
    # Frozen reward scale=1 and |Phi| <= distance_weight+angle_weight=1.
    assert np.max(np.abs(adv))<=1+1e-12
    result['terminal_shaping_violations']=0
    result['nonfinite_terminal_adv']=0
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.workers<1:raise ValueError('workers must be positive')
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise RuntimeError('Use a fresh audit directory')
    config=load_config(ROOT/'configs/combat_environment_v32.yaml')
    tasks=[('3.2','MIRROR',seed+i) for i in range(2000) for seed in BLOCK_BASES]
    rows=[];start=time.monotonic()
    with gzip.open(output/'episodes.jsonl.gz','wt') as raw,mp.get_context('spawn').Pool(a.workers,
            initializer=audit.initialize,initargs=({'3.2':config},)) as pool:
        for n,row in enumerate(pool.imap_unordered(audit.episode,tasks,chunksize=1),1):
            raw.write(json.dumps(row)+'\n');rows.append(row)
            if n%100==0:
                print(f'[SYMMETRY] {n}/10000 elapsed={time.monotonic()-start:.1f}s',flush=True)
                (output/'progress.json').write_text(json.dumps(dict(completed=n,total=10000,elapsed_seconds=time.monotonic()-start)))
    rows.sort(key=lambda r:r['seed'])
    blocks={str(base):symmetry_summary([r for r in rows if base<=r['seed']<base+2000]) for base in BLOCK_BASES}
    overall=symmetry_summary(rows);gaps=[r['signed_gap'] for r in blocks.values()]
    summary=dict(environment_version='3.2',blocks=blocks,overall=overall,
        block_gap_statistics=dict(values=gaps,mean=float(np.mean(gaps)),std_sample=float(np.std(gaps,ddof=1)),min=min(gaps),max=max(gaps)),
        terminal_sanity_first_64=symmetry_summary(rows[:64]),
        diagnostic_seed_blocks=[dict(start=b,end=b+1999,episodes=2000) for b in BLOCK_BASES],
        physics_changed=False,policy='existing SensorLimitedPursuitPolicy on both sides',
        hit_ci_method='nominal Bernoulli Wilson interval over observed attempts; gap CI uses episodes',
        parameters_tuned=False,formal_training_started=False,side_swap_performed=False,
        elapsed_seconds=time.monotonic()-start)
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print('[COMPLETE]',flush=True)

if __name__=='__main__':main()
