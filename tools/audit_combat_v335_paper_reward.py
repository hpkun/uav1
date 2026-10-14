"""Read-only same-trajectory current v3.35 versus literal Eq.(25) reward audit."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from tools.v335_paper_reward_reference import comparison,event_comparison,aligned,AGGREGATION_NOTE

BINS=('0-1km','1-2km','2-3km','3-4km','>=4km')
COMPONENTS=('r3','offense','defense','r4','dense')


def bin_masks(distance):
    return [(distance>=low)&(distance<high) for low,high in ((0,1000),(1000,2000),(2000,3000),(3000,4000),(4000,np.inf))]


class Stats:
    def __init__(self):
        self.n=0;self.sums=Counter();self.counts=Counter()
    def update(self,variants,mask,w3=None,w4=None):
        n=int(np.count_nonzero(mask))
        if not n:return
        self.n+=n
        for name,v in variants.items():
            for component in COMPONENTS:self.sums[name+'_'+component]+=float(v[component][mask].sum())
        if w3 is not None:
            self.sums['w3']+=float(w3[mask].sum());self.sums['w4']+=float(w4[mask].sum())
        for label,component,sign in (('r3','r3',1),('offense','offense',1),('defense','defense',-1)):
            c=variants['D'][component][mask];p=variants['A'][component][mask]
            conditions=dict(current_active_paper_zero=(c*sign>0)&(p==0),current_zero_paper_active=(c==0)&(p*sign>0),
                current_greater=c>p,current_less=c<p,both_active=(c*sign>0)&(p*sign>0),both_zero=(c==0)&(p==0),
                current_active=c*sign>0,paper_active=p*sign>0)
            for k,v in conditions.items():self.counts[label+'_'+k]+=int(v.sum())
        delta=variants['A']['offense'][mask]-variants['D']['offense'][mask]
        improved=delta>0
        self.counts['paper_offense_greater_count']+=int(improved.sum())
        self.sums['paper_offense_greater_difference']+=float(delta[improved].sum())
    def result(self):
        means={k:v/self.n for k,v in self.sums.items()} if self.n else {}
        counts=dict(self.counts)
        for label in ('r3','offense','defense'):
            for name in ('current_active_paper_zero','current_zero_paper_active','current_greater','current_less','both_active','both_zero','current_active','paper_active'):
                counts.setdefault(label+'_'+name,0)
        count=counts.setdefault('paper_offense_greater_count',0)
        return dict(sample_count=self.n,sums=dict(self.sums),means=means,counts=counts,
            rates={k:v/self.n if self.n else None for k,v in counts.items()},
            paper_offense_greater_mean_difference=self.sums['paper_offense_greater_difference']/count if count else None,
            paper_offense_greater_cumulative_difference=self.sums['paper_offense_greater_difference'])


class AuditStats:
    def __init__(self):
        self.pair=Stats();self.agent=Stats()
        self.pair_bins={k:Stats() for k in BINS};self.agent_bins={k:Stats() for k in BINS}
        self.head={k:Stats() for k in BINS};self.rear={str(level):{k:Stats() for k in BINS} for level in (30,15,5)}
        self.weights={str(level):Counter() for level in (30,15,5)}
        self.weight_bins={str(level):{k:Counter() for k in BINS} for level in (30,15,5)}
    @staticmethod
    def weight_update(result,nominal,effective,w4,mask):
        n=int(mask.sum())
        if not n:return
        a=nominal[mask];b=effective[mask];positive=b>0
        result.update(sample_count=n,nominal_sum=float(a.sum()),effective_sum=float(b.sum()),lost_due_w4=float((a-b).sum()),
            zero_effective_count=int((~positive).sum()),positive_effective_count=int(positive.sum()),
            finite_nominal_effective_ratio_sum=float((a[positive]/b[positive]).sum()),w4_sum=float(w4[mask].sum()))
    def update(self,g,variants,agents,w3,w4):
        live=g['live'];d=g['distance'];masks=bin_masks(d)
        self.pair.update(variants,live,w3,w4)
        agent_live=live.any(axis=1);nearest=np.where(live,d,np.inf).min(axis=1)
        self.agent.update(agents,agent_live)
        for label,mask in zip(BINS,masks):
            self.pair_bins[label].update(variants,live&mask,w3,w4)
        for label,mask in zip(BINS,bin_masks(nearest)):
            self.agent_bins[label].update(agents,agent_live&mask)
        head=live&(np.abs(g['ata_r'])<=np.pi/6)&(np.abs(g['aa_r'])>=np.deg2rad(150))
        rear=live&(np.abs(g['aa_r'])<=np.pi/6)
        for label,mask in zip(BINS,masks):
            self.head[label].update(variants,head&mask,w3,w4)
            for level in (30,15,5):self.rear[str(level)][label].update(variants,rear&aligned(g,'r',level,'paper')&mask,w3,w4)
        # Isolate multiplication by w4 inside the current tactical gate. Range exclusion is separate in A/B/C/D.
        eligible=live&(d<=3000)&(variants['B']['offense']>0)
        for level in (30,15,5):
            mask=eligible&(variants['D']['tier']==level)
            self.weight_update(self.weights[str(level)],variants['D']['nominal_offense'],variants['D']['offense'],w4,mask)
            for label,bucket in zip(BINS,masks):self.weight_update(self.weight_bins[str(level)][label],variants['D']['nominal_offense'],variants['D']['offense'],w4,mask&bucket)
    @staticmethod
    def weight_result(counter):
        n=counter.get('sample_count',0);positive=counter.get('positive_effective_count',0)
        return dict(counter,sample_count=n,nominal_mean=counter['nominal_sum']/n if n else None,
            effective_mean=counter['effective_sum']/n if n else None,lost_mean=counter['lost_due_w4']/n if n else None,
            finite_nominal_effective_ratio_mean=counter['finite_nominal_effective_ratio_sum']/positive if positive else None,
            zero_effective_count=int(counter['zero_effective_count']))
    def result(self):
        return dict(pair=self.pair.result(),agent=self.agent.result(),
            pair_bins={k:v.result() for k,v in self.pair_bins.items()},agent_bins={k:v.result() for k,v in self.agent_bins.items()},
            head_on={k:v.result() for k,v in self.head.items()},
            rear_aspect={l:{k:v.result() for k,v in buckets.items()} for l,buckets in self.rear.items()},
            w4_suppression={k:self.weight_result(v) for k,v in self.weights.items()},
            w4_suppression_bins={l:{k:self.weight_result(v) for k,v in buckets.items()} for l,buckets in self.weight_bins.items()})


class Observer:
    def __init__(self,e,stats,pair_writer,agent_writer,event_stream,episode,seed):
        self.e=e;self.stats=stats;self.pair_writer=pair_writer;self.agent_writer=agent_writer;self.event_stream=event_stream
        self.episode=episode;self.seed=seed;self.pending=None;self.event_data=None
        self.current={k:np.zeros(4) for k in ('r1','r2','r3','r4','total')};self.paper={k:np.zeros(4) for k in self.current}
        self.old_dense=e.dense_combat_reward;self.old_events=e._event_reward_components
        self.instance_dense=e.__dict__.get('dense_combat_reward');self.instance_events=e.__dict__.get('_event_reward_components')
    def dense(self):
        # Capture exact pre-fire states, call authoritative method once, return untouched.
        self.snapshot_red=[r.copy() for r in self.e.red];self.snapshot_blue=[b.copy() for b in self.e.blue]
        self.snapshot_alive_red=[r.alive for r in self.snapshot_red];self.snapshot_alive_blue=[b.alive for b in self.snapshot_blue]
        self.actual_dense=self.old_dense()
        return self.actual_dense
    def events(self,red_exits,red_ground,red_kills,blue_kills):
        self.event_data=dict(red_exits=list(red_exits),red_ground=list(red_ground),
            red_kills={k:list(v) for k,v in red_kills.items()},blue_kills={k:list(v) for k,v in blue_kills.items()},
            horizontal_exits=[i for i in red_exits if np.hypot(self.e.red[i].x,self.e.red[i].y)>5000],
            ceiling_exits=[i for i in red_exits if self.e.red[i].altitude>6000])
        return self.old_events(red_exits,red_ground,red_kills,blue_kills)
    def __enter__(self):
        self.e.dense_combat_reward=self.dense;self.e._event_reward_components=self.events;return self
    def __exit__(self,*args):
        for name,value in (('dense_combat_reward',self.instance_dense),('_event_reward_components',self.instance_events)):
            if value is None:self.e.__dict__.pop(name,None)
            else:setattr(self.e,name,value)
    def consume(self,rewards,info):
        # All comparison reward calculations happen after env.step, on copied snapshots.
        g,v,ag,w3,w4=comparison(self.snapshot_red,self.snapshot_blue,self.e.config['reward'],self.e.config['weapon'])
        for key,source in (('r3','guide'),('offense','tactical_offense'),('defense','tactical_defense'),('dense','tactical')):
            np.testing.assert_array_equal(ag['D'][key],self.actual_dense[1][source])
        self.stats.update(g,v,ag,w3,w4)
        n=4;fields=('distance','ata_r','ha_r','aa_r','ata_b','ha_b','aa_b','body_az_r','body_el_r','body_az_b','body_el_b')
        for i in range(n):
            for j in range(n):
                row=[self.episode,self.seed,self.e.steps,i,j,int(self.snapshot_alive_red[i]),int(self.snapshot_alive_blue[j])]
                row += [float(g[k][i,j]) for k in fields]+[float(w3[i,j]),float(w4[i,j])]
                row += [float(v[name][k][i,j]) for name,k in (('D','r3'),('D','offense'),('D','defense'),('A','r3'),('A','offense'),('A','defense'))]
                row += [float(v[name][k][i,j]) for name in ('A','B','C','D') for k in COMPONENTS]
                row += [float(v['D']['nominal_offense'][i,j]),int(v['D']['tier'][i,j])]
                self.pair_writer.writerow(row)
        nearest=np.where(g['live'],g['distance'],np.inf).min(axis=1)
        ev=self.event_data
        paper_r1,paper_r2=event_comparison(ev['red_exits'],ev['red_ground'],ev['red_kills'],ev['blue_kills'],4)
        self.event_stream.write(json.dumps(dict(episode=self.episode,seed=self.seed,step=self.e.steps,**self.event_data))+'\n')
        current=dict(r1=np.asarray(info['r1_rewards']),r2=np.asarray(info['r2_rewards']),r3=np.asarray(info['r3_rewards']),r4=np.asarray(info['r4_rewards']),total=np.asarray(rewards))
        paper=dict(r1=paper_r1,r2=paper_r2,r3=ag['A']['r3'],r4=ag['A']['r4'],total=paper_r1+paper_r2+ag['A']['r3']+ag['A']['r4'])
        np.testing.assert_array_equal(current['r3'],ag['D']['r3']);np.testing.assert_array_equal(current['r4'],ag['D']['r4'])
        np.testing.assert_array_equal(current['r1'],paper_r1);np.testing.assert_array_equal(current['r2'],paper_r2)
        np.testing.assert_array_equal(current['total'],current['r1']+current['r2']+current['r3']+current['r4'])
        for k in current:self.current[k]+=current[k];self.paper[k]+=paper[k]
        for i in range(n):
            self.agent_writer.writerow([self.episode,self.seed,self.e.steps,i,int(self.snapshot_alive_red[i]),float(nearest[i]) if np.isfinite(nearest[i]) else '',
                *[float(current[k][i]) for k in ('r1','r2','r3','r4','total')],*[float(paper[k][i]) for k in ('r1','r2','r3','r4','total')],
                *[float(ag[name][k][i]) for name in ('A','B','C','D') for k in COMPONENTS]])


def description_stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),std=float(a.std()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))


def decomposition(stats):
    result={}
    for scope in ('pair','agent'):
        d=stats[scope];n=d['sample_count'];values=d['sums']
        combos={name:{k:dict(total=values.get(name+'_'+k,0.),mean=values.get(name+'_'+k,0.)/n if n else None) for k in COMPONENTS} for name in ('A','B','C','D')}
        effects={}
        for label,weights in (('geometry_hard_B_minus_A',{'B':1,'A':-1}),('geometry_continuous_D_minus_C',{'D':1,'C':-1}),
            ('distance_paper_C_minus_A',{'C':1,'A':-1}),('distance_body_D_minus_B',{'D':1,'B':-1}),('interaction_D_minus_C_minus_B_plus_A',{'D':1,'C':-1,'B':-1,'A':1})):
            effects[label]={k:sum(w*values.get(name+'_'+k,0.) for name,w in weights.items()) for k in COMPONENTS}
        result[scope]=dict(sample_count=n,variants=combos,effect_totals=effects)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200);p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--algorithm-config',type=Path)
    p.add_argument('--policy-mode',choices=('deterministic','stochastic'),default='deterministic')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v335_paper_reward_audit')
    args=p.parse_args()
    if args.episodes<=0:raise ValueError('episodes must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('choose a fresh output directory; refusing to overwrite audit')
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v335.yaml')
    fixed=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action']);trainer=None;provenance={}
    if args.checkpoint:
        from tools.record_combat_episode import load_recording_policy
        ac=args.algorithm_config or args.checkpoint.parent/'algorithm_config.yaml'
        config=yaml.safe_load(ac.read_text(encoding='utf-8'))
        trainer,state,provenance=load_recording_policy(args.checkpoint,e.config,config,'cuda')
        if str(state['algorithm']).lower()!='mappo':raise ValueError('this audit checkpoint mode requires MAPPO')
        provenance['training_smoke']=bool(state['extra'].get('training_smoke'))
    stats=AuditStats();episodes=[]
    pair_header=['episode','seed','step','red_index','blue_index','red_alive','blue_alive','distance','ATA_r','HA_r','AA_r','ATA_b','HA_b','AA_b','body_az_r','body_el_r','body_az_b','body_el_b','w3','w4']
    pair_header += ['current_R3_pair','current_R4_offense_pair','current_R4_defense_pair','paper_R3_pair','paper_R41_pair','paper_R42_pair']+[name+'_'+k for name in ('A','B','C','D') for k in COMPONENTS]+['current_nominal_offense','current_exclusive_tier']
    agent_header=['episode','seed','step','red_index','red_alive','nearest_live_blue_distance']+[side+'_'+k for side in ('current','paper_wrapper') for k in ('r1','r2','r3','r4','total')]+[name+'_'+k for name in ('A','B','C','D') for k in COMPONENTS]
    with gzip.open(out/'pair_steps.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1) as pf, gzip.open(out/'agent_steps.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1) as af, (out/'events.jsonl').open('w',encoding='utf-8') as ef:
        pw=csv.writer(pf);aw=csv.writer(af);pw.writerow(pair_header);aw.writerow(agent_header)
        for episode,seed in enumerate(range(args.seed_start,args.seed_start+args.episodes)):
            obs,_=e.reset(seed)
            if trainer is not None:torch.manual_seed(seed^335);torch.cuda.manual_seed_all(seed^335)
            with Observer(e,stats,pw,aw,ef,episode,seed) as observer:
                while True:
                    actions=fixed.team_actions(e.red,e.blue) if trainer is None else trainer.act(obs[None],e.red_alive_mask[None],deterministic=args.policy_mode=='deterministic')[0]
                    obs,rewards,terminated,truncated,info=e.step(actions);observer.consume(rewards,info)
                    if terminated or truncated:break
            episodes.append(dict(episode=episode,seed=seed,length=e.steps,red_win=info['red_win'],blue_win=info['blue_win'],draw=info['draw'],timeout=truncated,
                red_kills=info['red_attack_kills'],blue_kills=info['blue_attack_kills'],
                current={k:float(v.sum()) for k,v in observer.current.items()},paper_wrapper={k:float(v.sum()) for k,v in observer.paper.items()},
                current_per_agent={k:v.tolist() for k,v in observer.current.items()},paper_per_agent={k:v.tolist() for k,v in observer.paper.items()},
                total_difference=float(observer.current['total'].sum()-observer.paper['total'].sum()),
                red_ground_losses=info['red_ground_losses'],red_boundary_exits=info['red_boundary_exits'],red_ceiling_losses=int(e.combat_counts['red']['ceiling_losses'])))
            if (episode+1)%10==0:print(f'audit {episode+1}/{args.episodes}',flush=True)
    result=stats.result()
    meta=dict(environment_version='3.35',trajectory_source='fixed-policy' if trainer is None else 'MAPPO checkpoint',episodes=args.episodes,seed_start=args.seed_start,
        policy_mode=args.policy_mode,checkpoint=provenance,device=torch.cuda.get_device_name(0),aggregation_note=AGGREGATION_NOTE,
        sampling='post-advance/post-noncombat, pre-fire: the exact actual dense reward stage; event mapping taken from the same transition',
        angle_units='radians',distance_units='meters',
        nominal_column_semantics='Alignment-only body coefficient before AA/distance gates; w4 suppression statistics require full current aspect/alignment eligibility.',
        bins='[0,1000),[1000,2000),[2000,3000),[3000,4000),[4000,infinity); reward gates retain their own inclusive boundary',
        agent_bins='nearest live Blue distance, one sample per live Red with a surviving Blue; aggregated reward may be triggered by a different Blue',
        current_reconstruction_exact=True,event_wrapper_matches_current=True,
        outcomes={k:sum(r[k] for r in episodes) for k in ('red_win','blue_win','draw','timeout','red_kills','blue_kills')},
        episode_statistics={side:{k:description_stats([r[side][k] for r in episodes]) for k in ('r1','r2','r3','r4','total')} for side in ('current','paper_wrapper')},
        total_difference_statistics=description_stats([r['total_difference'] for r in episodes]),
        r1_r2_project_assumptions='shared simultaneous kill credit; ground loss as R1 own loss; horizontal/ceiling exit as R2; priority/order and event attribution are PAPER UNSPECIFIED')
    for filename,value in (('summary.json',dict(metadata=meta,**result)),('episode_summary.json',dict(aggregation_note=AGGREGATION_NOTE,episodes=episodes)),
        ('distance_bins.json',dict(pair=result['pair_bins'],agent=result['agent_bins'])),('pair_trigger_summary.json',dict(pair=result['pair'],agent=result['agent'])),('decomposition.json',decomposition(result))):
        (out/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    write_report(out,meta,result,decomposition(result))
    print(json.dumps(meta['episode_statistics'],indent=2),flush=True)


def write_report(out,meta,s,decomp):
    lines=['# v3.35 current Reward versus paper Eq.(25) audit','',AGGREGATION_NOTE,'',f"Source: {meta['trajectory_source']}; episodes={meta['episodes']}; seed_start={meta['seed_start']}; checkpoint={meta['checkpoint']}",'', 'Pair statistics count only live pairs at the actual pre-fire dense reward stage. Agent statistics count live Red with live opponents; agent distance bins use nearest live Blue and do not attribute the aggregate to that pair.','', '## Pair-level trigger comparison','', '| component | N | current active | paper active | current-only | paper-only | both active | both zero |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ('r3','offense','defense'):
        c=s['pair']['counts'];lines.append(f"| {label} | {s['pair']['sample_count']} | {c[label+'_current_active']} | {c[label+'_paper_active']} | {c[label+'_current_active_paper_zero']} | {c[label+'_current_zero_paper_active']} | {c[label+'_both_active']} | {c[label+'_both_zero']} |")
    lines += ['', '## Pair distance bins','', '| bin | N | current R3 mean | paper R3 mean | current R3 positive rate | paper R3 positive rate | mean w3 | mean w4 | current offense | paper offense | current defense | paper defense |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def value(v):return 'N/A' if v is None else f'{v:.9g}'
    for label,b in s['pair_bins'].items():
        m=b['means'];rates=b['rates'];keys=('D_r3','A_r3')
        vals=[m.get(k) for k in keys]+[rates['r3_current_active'],rates['r3_paper_active']]+[m.get(k) for k in ('w3','w4','D_offense','A_offense','D_defense','A_defense')]
        lines.append('| '+label+f" | {b['sample_count']} | "+' | '.join(value(v) for v in vals)+' |')
    lines += ['', '## Offense suppression','', 'For each scope, paper offense > current: count / rate / conditional mean difference / cumulative difference.']
    for scope in ('pair','agent'):
        b=s[scope];lines.append(f"- {scope}: {b['counts']['paper_offense_greater_count']} / {b['rates']['paper_offense_greater_count']} / {b['paper_offense_greater_mean_difference']} / {b['paper_offense_greater_cumulative_difference']}")
    lines += ['', 'Weight-only suppression selects current body-aligned, AA<=30 pairs inside d<=3000, with mutually exclusive strongest tiers. Zero samples mean N/A, not evidence of no possible attenuation. Ratios exclude zero effective reward and report that count separately.','', '| strongest tier | N | nominal sum | after-w4 sum | lost due w4 | finite nominal/effective mean ratio | zero effective |','|---|---:|---:|---:|---:|---:|---:|']
    for level,b in s['w4_suppression'].items():lines.append(f"| {level} | {b['sample_count']} | {b.get('nominal_sum',0)} | {b.get('effective_sum',0)} | {b.get('lost_due_w4',0)} | {b['finite_nominal_effective_ratio_mean']} | {b['zero_effective_count']} |")
    lines += ['', '## Head-on and rear-aspect','', 'Head-on: abs(ATA_r)<=30 and abs(AA_r)>=150; no HA gate added. Rear groups use AA<=30 and nested ATA/HA<=30/15/5.','', '| head-on bin | N | current R3 sum | paper R3 sum | current R4 sum | paper R4 sum |','|---|---:|---:|---:|---:|---:|']
    for label,b in s['head_on'].items():lines.append(f"| {label} | {b['sample_count']} | {b['sums'].get('D_r3',0)} | {b['sums'].get('A_r3',0)} | {b['sums'].get('D_r4',0)} | {b['sums'].get('A_r4',0)} |")
    lines += ['', '| rear tier (nested) | bin | N | current offense sum | paper offense sum |','|---|---|---:|---:|---:|']
    for level,buckets in s['rear_aspect'].items():
        for label,b in buckets.items():lines.append(f"| {level} | {label} | {b['sample_count']} | {b['sums'].get('D_offense',0)} | {b['sums'].get('A_offense',0)} |")
    lines += ['', '## 2x2 decomposition','', 'A=ATA/HA + hard 4km; B=body angles + hard 4km; C=ATA/HA + current continuous distance system; D=body angles + current continuous distance system. D is validated exactly against the actual environment at every step.','', '| scope | variant | R3 sum | offense sum | defense sum | dense sum |','|---|---|---:|---:|---:|---:|']
    for scope,result in decomp.items():
        for variant,b in result['variants'].items():lines.append(f"| {scope} | {variant} | {b['r3']['total']} | {b['offense']['total']} | {b['defense']['total']} | {b['dense']['total']} |")
    lines += ['', 'Geometry effects are B-A and D-C. Distance-system effects are C-A and D-B; these include both current 3km tactical cutoff and continuous weights. Paper helper uses exact abs<= thresholds; body/current variants preserve the existing 1e-12 rad numerical tolerance. No adapted paper training reward is introduced.','', '## Episode cumulative comparison (team sum over Red agents)','',AGGREGATION_NOTE,'', '| side | component | mean | std | median | min | max |','|---|---|---:|---:|---:|---:|---:|']
    for side,components in meta['episode_statistics'].items():
        for k,b in components.items():lines.append(f"| {side} | {k} | {b['mean']} | {b['std']} | {b['median']} | {b['min']} | {b['max']} |")
    lines += ['', 'current_total-paper_total: '+json.dumps(meta['total_difference_statistics']), '', '## PAPER-SPECIFIED / PROJECT-ASSUMED','', '- PAPER-SPECIFIED: literal R1 +/-10, R2 -10, R3 .001 at >=4000 with ATA/HA<=30, R4 at <=4000 with AA<=30, offense .01/.02/.10 and reverse defense -.015/-.025/-.15 with strongest alignment overriding lower levels. Exact 4000 boundary permits both R3/R4.','- Geometry source: engagement_geometry().ata/.ha/.aa, including reverse Blue-to-Red geometry. Body fields are used only for B/D and recorded current comparison.','- PROJECT ASSUMPTION / PAPER UNSPECIFIED: max/max/min aggregation; shared simultaneous kill credit; ground loss as R1; horizontal/ceiling exit as R2; event order and multi-attacker resolution. R1/R2 match numerically under this inherited event wrapper, not an independently specified paper event implementation.','- No environment, Reward, Weapon, algorithm, config or training code is modified. Observer calls authoritative methods once and returns their results unchanged, consumes no RNG, and restores instance methods after recording.','', 'Files: pair_steps.csv.gz, agent_steps.csv.gz, events.jsonl, summary.json, episode_summary.json, distance_bins.json, pair_trigger_summary.json, decomposition.json. No reward modification recommendations.']
    lines += ['', '## Pair-step / agent-step detailed comparisons', '', '| scope | component | N | current active rate | paper active rate | current-only | paper-only | current>paper | current<paper | both active | both zero |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('pair','agent'):
        data=s[scope];c=data['counts'];rates=data['rates']
        for component in ('r3','offense','defense'):
            row=[scope,component,str(data['sample_count']),value(rates[component+'_current_active']),value(rates[component+'_paper_active'])]
            row += [str(c[component+'_'+key]) for key in ('current_active_paper_zero','current_zero_paper_active','current_greater','current_less','both_active','both_zero')]
            lines.append('| '+' | '.join(row)+' |')
    lines += ['', '## Distance-bin active rates (R4 offense>0, defense<0)', '', '| scope | bin | N | current offense mean | paper offense mean | current defense mean | paper defense mean | current offense rate | paper offense rate | current defense rate | paper defense rate |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('pair','agent'):
        for label,b in s[scope+'_bins'].items():
            row=[scope,label,str(b['sample_count'])]+[value(b['means'].get(key)) for key in ('D_offense','A_offense','D_defense','A_defense')]+[value(b['rates'][key]) for key in ('offense_current_active','offense_paper_active','defense_current_active','defense_paper_active')]
            lines.append('| '+' | '.join(row)+' |')
    lines += ['', '## Agent-step R3 distance bins (nearest-live-distance convention)', '', '| bin | N | current mean | paper wrapper mean | current positive rate | paper positive rate |', '|---|---:|---:|---:|---:|---:|']
    for label,b in s['agent_bins'].items():lines.append('| '+label+' | '+str(b['sample_count'])+' | '+' | '.join(value(v) for v in (b['means'].get('D_r3'),b['means'].get('A_r3'),b['rates']['r3_current_active'],b['rates']['r3_paper_active']))+' |')
    lines += ['', '## Offense paper>current by distance bin', '', '| scope | bin | count | rate over live samples | conditional mean gap | cumulative positive gap |', '|---|---|---:|---:|---:|---:|']
    for scope in ('pair','agent'):
        for label,b in s[scope+'_bins'].items():lines.append('| '+scope+' | '+label+' | '+str(b['counts']['paper_offense_greater_count'])+' | '+' | '.join(value(v) for v in (b['rates']['paper_offense_greater_count'],b['paper_offense_greater_mean_difference'],b['paper_offense_greater_cumulative_difference']))+' |')
    lines += ['', '## Geometry/distance effects and interaction totals', '', '| scope | effect | R3 | offense | defense | dense |', '|---|---|---:|---:|---:|---:|']
    for scope,d in decomp.items():
        for effect,metrics in d['effect_totals'].items():lines.append('| '+scope+' | '+effect+' | '+' | '.join(value(metrics[k]) for k in ('r3','offense','defense','dense'))+' |')
    lines += ['', '## Canonical helper facts (not trajectory samples)', '', 'For an aligned rear-aspect pair, the current continuous factor is w4=0.75 at 1500 m and w4=0.25 at 2500 m. These retain 75%/25% of the nominal .01/.02/.10 coefficients; nominal/effective ratios are 4/3 and 4. At 3500 m the current 3000 m gate excludes offense while the paper 4000 m gate permits full tiers. This last difference is gate exclusion, not solely multiplication attenuation. Canonical unit tests cover all three levels. Empirical trajectory tables remain authoritative and do not substitute these hypothetical states for missing samples.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
