"""Replay saved trace geometry, verify pair re-entry, and draw real trajectories."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState

def aircraft(row,side,index):
    s=row[side]
    return AircraftState(*s['positions'][index],s['speed'][index],s['pitch'][index],s['heading'][index],s['alive'][index])

def review(folder,output):
    env=MultiUAVCombatEnv(load_config(ROOT/'configs/combat_environment_v24.yaml'))
    reports=[]; plot_candidates=[]
    paths=sorted(folder.glob('seed_*.jsonl'))
    if len(paths)<10: raise RuntimeError('At least ten traces required')
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        by_step={row['step']:row for row in rows}
        seen={}; repeats=[]; projections=[]; attempts=[]
        for row in rows:
            for side in ('red','blue'):
                assert np.asarray(row[side]['positions']).shape==(8,3)
                assert np.isfinite(row[side]['positions']).all() and np.isfinite(row[side]['speed']).all()
            used=set()
            for event in row['attempts']:
                key=event['side'],event['attacker_index'],event['target_index']
                attacker_key=key[:2]
                assert attacker_key not in used,'Multiple attempts per attacker per step'
                used.add(attacker_key)
                assert 0<=event['distance']<=4000+1e-9
                assert event['off_boresight']<=np.pi/6+1e-10
                assert event['target_aspect']<=np.pi/4+1e-10
                assert event['attacker_speed']>=event['target_speed']-1e-10
                displacement=np.asarray(event['attacker_position'])-np.asarray(event['target_position'])
                heading=event['target_heading']
                projection=float(displacement[:2]@np.array([np.cos(heading),np.sin(heading)]))
                assert projection<=1e-8,'Attack did not originate behind target in horizontal aspect geometry'
                projections.append(projection);attempts.append(event)
                if key in seen:
                    other='blue' if key[0]=='red' else 'red'
                    invalid=[]
                    for step in range(seen[key]+1,event['step']):
                        previous=by_step[step]
                        if not env._in_fire_window(aircraft(previous,key[0],key[1]),aircraft(previous,other,key[2])):
                            invalid.append(step)
                    assert invalid,'Repeated pair attempt without leaving full qualification window'
                    repeats.append({'side':key[0],'attacker_index':key[1],'target_index':key[2],
                        'previous_attempt_step':seen[key],'invalid_step':invalid[0],'reentry_attempt_step':event['step']})
                seen[key]=event['step']
        reports.append({'seed':int(path.stem.split('_')[-1]),'trace':str(path),'steps':rows[-1]['step'],
            'attempt_count':len(attempts),'kill_count':sum(len(row['kills']) for row in rows),
            'rear_projection_max':max(projections) if projections else None,
            'first_attempt':attempts[0] if attempts else None,'verified_reentries':repeats,
            'final_red_survivors':sum(rows[-1]['red']['alive']),
            'final_blue_survivors':sum(rows[-1]['blue']['alive'])})
        if attempts:plot_candidates.append((path,rows))
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for path,rows in plot_candidates[:3]:
        fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(111,projection='3d')
        for side,color in [('red','tab:red'),('blue','tab:blue')]:
            for index in range(8):
                points=[]
                for row in rows:
                    points.append(row[side]['positions'][index])
                    if not row[side]['alive'][index]:break
                xyz=np.asarray(points)
                ax.plot(xyz[:,0],xyz[:,1],-xyz[:,2],color=color,alpha=.55,linewidth=1)
        event=next(event for row in rows for event in row['attempts'])
        xyz=np.asarray([event['attacker_position'],event['target_position']])
        ax.plot(xyz[:,0],xyz[:,1],-xyz[:,2],color='black',linewidth=3,label=f"First rear-qualified attempt, step {event['step']}")
        ax.scatter(xyz[:,0],xyz[:,1],-xyz[:,2],color=['darkorange','black'],s=40)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='Altitude (m)',title=f'v2.4 mirrored nearest-target, {path.stem}')
        ax.legend();fig.tight_layout();fig.savefig(output/f'{path.stem}_3d.png',dpi=150);plt.close(fig)
    result={'traces_checked':len(reports),'all_geometry_valid':True,
        'all_attempts_from_rear':True,'all_repeated_pairs_have_invalid_intervals':True,
        'verified_reentry_count':sum(len(row['verified_reentries']) for row in reports),'episodes':reports,
        'plots':[str(path) for path in sorted(output.glob('*_3d.png'))]}
    (output/'trace_review.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='episodes'}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trace-dir',type=Path,default=ROOT/'outputs/v24_combat_audit/v24/traces')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v24_combat_audit/trace_review');args=parser.parse_args()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for repository audits')
    review(args.trace_dir,args.output_dir)

if __name__=='__main__':main()
