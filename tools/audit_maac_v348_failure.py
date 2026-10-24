"""Read-only CUDA diagnosis. No optimizer steps or changes to training semantics."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.v346_reward import tactical_pair_rewards
from env.v339_policy import DistanceTierNearestTargetBluePolicy
from tools.record_combat_episode import load_recording_policy
from algorithm.maac.trainer import score_function_loss


def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),
                quantiles=np.quantile(x,[.01,.1,.5,.9,.99]).tolist()) if len(x) else dict(n=0)


def episode(trainer,ec,seed,mode,out,collect):
    e=MultiUAVCombatEnv(ec);obs,_=e.reset(seed)
    # A separate controller prevents diagnostic rule probes from touching Blue's
    # otherwise control-neutral pursuit/target-switch reporting counters.
    rule_controller=DistanceTierNearestTargetBluePolicy(ec['blue_policy'],ec['action'])
    torch.manual_seed(40000000+seed)
    run=np.zeros((4,2),int);last_sign=np.zeros((4,2));longest=np.zeros(2,int)
    rows=[];blue_rows=[];returns=np.zeros(4);parts=np.zeros(4);eligible_samples=range_samples=aligned_samples=correct=error_pairs=0
    min_alpha=180.;first_qual=None;saturations=np.zeros(3);live_samples=0;r4sign=np.zeros(3);aligned_r4_positive=0
    while True:
        before=obs.copy();alive=e.red_alive_mask.copy();states=[s.copy() for s in e.red]
        blue_states=[s.copy() for s in e.blue]
        with torch.no_grad():
            dist=trainer.actor.distribution(trainer.tensor(obs));mu=dist.mean.cpu().numpy();std=dist.scale.cpu().numpy()
            deterministic=np.tanh(mu)
        action=rule_controller.team_actions(e.red,e.blue) if mode=='rule' else trainer.act(obs,e.red_alive_mask,deterministic=mode=='det')
        # Snapshot both teams pre-motion; evaluate alignment after Red's actual move
        # against the SAME target position to separate turning from target motion.
        geometries=[];targets=[]
        for s in e.red:
            js=[j for j,b in enumerate(e.blue) if s.alive and b.alive]
            j=min(js,key=lambda j:(engagement_geometry(s,e.blue[j]).distance,j)) if js else None
            targets.append(None if j is None else e.blue[j].copy())
            geometries.append(None if j is None else engagement_geometry(s,e.blue[j]))
        # Diagnostic wrapper observes post-motion/pre-event state only. No RNG draws.
        actual_dense=e.dense_combat_reward
        sampled={};attempts={};settlement={}
        original_entry=e._entry_attempts;original_resolve=e._resolve_combat
        def observe_entry(attackers,opponents,fire_states,side):
            result=original_entry(attackers,opponents,fire_states,side)
            attempts[side]=list(result);return result
        def observe_resolve(red,blue):
            result=original_resolve(red,blue);settlement.update(red=result[0],blue=result[1]);return result
        e._entry_attempts=observe_entry;e._resolve_combat=observe_resolve
        def observe_dense():
            val=actual_dense()
            cone=e.weapon.eligibility(e.red,e.blue,e.red_ammo)[0]
            alignment=e.weapon.eligibility(e.red,e.blue,np.full(4,6))[0].any(axis=1)
            sampled.update(off=val[1]['tactical_offense'].copy(),cone=cone.any(axis=1),alignment=alignment,
                ranges=np.array([any(a.alive and b.alive and engagement_geometry(a,b).distance<=2000 for b in e.blue) for a in e.red]))
            return val
        e.dense_combat_reward=observe_dense
        obs,r,term,trunc,info=e.step(action)
        e.dense_combat_reward=actual_dense
        e._entry_attempts=original_entry;e._resolve_combat=original_resolve
        returns+=r;parts+=np.array([info[f'r{i}_rewards'].sum() for i in range(1,5)])
        sign=np.sign(action[:,:2]);run=np.where((sign==last_sign)&(sign!=0)&(alive[:,None]>0),run+1,np.where((sign!=0)&(alive[:,None]>0),1,0));last_sign=sign
        longest=np.maximum(longest,run.max(axis=0))
        saturations+=(np.abs(action[alive>0])>=.99).sum(axis=0);live_samples+=int(alive.sum())
        rv=info['r4_rewards'][alive>0];r4sign+=np.array([(rv>0).sum(),(rv<0).sum(),(rv==0).sum()])
        eligible_samples+=int(sampled['cone'].sum());range_samples+=int(sampled['ranges'].sum())
        aligned_samples+=int((sampled['alignment']&sampled['ranges']).sum())
        aligned_r4_positive+=int((sampled['cone']&(info['r4_rewards']>0)).sum())
        if sampled['cone'].any() and first_qual is None:first_qual=e.steps*e.dt
        if collect and (e.steps%10==0 or term or trunc or info['red_step_fire_attempts'] or info['blue_step_fire_attempts']):
            collect.append(dict(observation=before,actions=action.copy(),rewards=r.copy(),next_observation=obs.copy(),alive=alive.copy(),next_alive=e.red_alive_mask.copy(),done=term or trunc,
                fired=bool(info['red_step_fire_attempts']+info['blue_step_fire_attempts']),killed=bool(info['red_step_attack_kills']+info['blue_step_attack_kills']),rule_actions=rule_controller.team_actions(states,blue_states)))
        for i,s in enumerate(states):
            if not alive[i]:continue
            g=geometries[i];after=e.red[i]
            ga=engagement_geometry(after,targets[i]) if targets[i] is not None else None
            dpsi=(after.psi-s.psi+np.pi)%(2*np.pi)-np.pi
            if g is not None and ga is not None and abs(dpsi)+abs(after.theta-s.theta)>1e-9:
                correct+=int(ga.off_boresight<g.off_boresight);error_pairs+=1
            if g and g.distance<=2000:min_alpha=min(min_alpha,float(np.rad2deg(g.off_boresight)))
            row=dict(seed=seed,mode=mode,step=e.steps,time_s=e.steps*e.dt,agent=i+1,alive_after=bool(after.alive),
                x=after.x,y=after.y,altitude=after.altitude,speed=after.v,psi=after.psi,theta=after.theta,
                actual_heading_delta=dpsi,actual_pitch_delta=after.theta-s.theta,actual_speed_delta=after.v-s.v,
                normal_load=None,roll_phi=None,control_semantics='v332_direct_kinematics_nz_phi_not_applicable',
                speed_clipped=bool(after.v in (e.spec.v_min,e.spec.v_max)),pitch_clipped=bool(after.theta in (e.spec.theta_min,e.spec.theta_max)),
                horizontal_margin=5000-np.hypot(after.x,after.y),ground_margin=after.altitude,ceiling_margin=6000-after.altitude,
                nearest_distance=None if g is None else g.distance,body_azimuth_error=None if g is None else g.boresight_azimuth_error,
                body_elevation_error=None if g is None else g.boresight_elevation_error,alpha_before=None if g is None else g.off_boresight,
                alpha_after_same_target=None if ga is None else ga.off_boresight,eligible=bool(sampled['cone'][i]),
                geometry_aligned=bool(sampled['alignment'][i]),ammo_after=int(e.red_ammo[i]),cooldown_after=int(e.red_fire_cooldown_steps[i]),
                agent_step_attempts=sum(j==i for j,_,_ in attempts['red']),agent_step_hits=sum(j==i and hit for j,_,hit in attempts['red']),
                agent_step_fractional_kills=sum(1/len(js) for js in settlement['red'].values() if i in js),
                actor_distribution_is_reference_only=mode=='rule',
                in_range=bool(sampled['ranges'][i]),r1=info['r1_rewards'][i],r2=info['r2_rewards'][i],r3=info['r3_rewards'][i],r4=info['r4_rewards'][i],
                cone_but_no_paper_offense=bool(sampled['cone'][i] and sampled['off'][i]==0),
                total_fire_attempts=info['red_fire_attempts'],total_hits=info['red_weapon_hits'],total_kills=info['red_attack_kills'])
            for j,axis in enumerate(('heading','pitch','speed')):
                row.update({axis+'_action':float(action[i,j]),axis+'_mu':float(mu[i,j]),axis+'_std':float(std[i,j]),axis+'_log_std':float(np.log(std[i,j])),axis+'_deterministic_action':float(deterministic[i,j])})
            rows.append(row)
        for i,s in enumerate(e.blue):
            blue_rows.append(dict(seed=seed,step=e.steps,time_s=e.steps*e.dt,agent=i+1,alive=bool(s.alive),x=s.x,y=s.y,altitude=s.altitude,speed=s.v,psi=s.psi,theta=s.theta,
                heading_action=float(info['executed_blue_actions'][i,0]),pitch_action=float(info['executed_blue_actions'][i,1]),speed_action=float(info['executed_blue_actions'][i,2]),
                ammo_after=int(e.blue_ammo[i]),cooldown_after=int(e.blue_fire_cooldown_steps[i]),pursuit_target=int(info['blue_pursuit_target_indices'][i]),
                agent_step_attempts=sum(j==i for j,_,_ in attempts['blue']),agent_step_hits=sum(j==i and hit for j,_,hit in attempts['blue'])))
        if term or trunc:break
    if rows:
        with out.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with out.with_name(out.stem+'_blue.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(blue_rows[0]));w.writeheader();w.writerows(blue_rows)
    losses=[]
    for i in range(4):
        own=[x for x in rows if x['agent']==i+1]
        if own and not own[-1]['alive_after']:
            recent=own[-10:];losses.append(dict(agent=i+1,time_s=own[-1]['time_s'],last_1s_actions={axis:float(np.mean([r[axis+'_action'] for r in recent])) for axis in ('heading','pitch','speed')},
                last_position=[own[-1][k] for k in ('x','y','altitude')]))
    return dict(seed=seed,mode=mode,steps=e.steps,return_sum=float(returns.sum()),red_win=info['red_win'],blue_win=info['blue_win'],draw=info['draw'],
        attempts=info['red_fire_attempts'],hits=info['red_weapon_hits'],kills=info['red_attack_kills'],red_losses=info['red_losses'],blue_losses=info['blue_losses'],
        boundary=info['red_boundary_exits'],ceiling=info['red_ceiling_losses'],ground=info['red_ground_losses'],
        live_samples=live_samples,saturation=(saturations/max(1,live_samples)).tolist(),longest_same_sign_seconds=(longest*e.dt).tolist(),
        error_reduction_ratio=correct/max(1,error_pairs),range_agent_steps=range_samples,cone_agent_steps=eligible_samples,
        alignment_rate_in_range=aligned_samples/max(1,range_samples),min_alpha_in_range_deg=None if min_alpha==180 else min_alpha,
        first_qualification_s=first_qual,r4_positive_negative_zero=(r4sign/max(1,live_samples)).tolist(),reward_totals=parts.tolist(),
        cone_with_positive_r4=aligned_r4_positive,max_abs_reward_ledger_error=float(abs(parts.sum()-returns.sum())),loss_precursors=losses)


def critic_diagnostics(t,data,out):
    arrays={k:np.array([r[k] for r in data]) for k in ('observation','actions','rewards','next_observation','alive','next_alive','done','fired','killed','rule_actions')}
    np.savez_compressed(out/'critic_samples.npz',**arrays)
    values={k:[] for k in ('q','target_q','td_residual','advantage','actor_grad','critic_grad','q_rule_minus_actor','q_bad_minus_actor')}
    events={k:[] for k in ('terminal','fire','kill','ordinary')}
    torch.manual_seed(901)
    # All values are off-policy Soft Q diagnostics. No raw MC return comparison.
    for start in range(0,len(data),64):
        b={k:t.tensor(v[start:start+64]) for k,v in arrays.items()}
        o,a,m=b['observation'],b['actions'],b['alive'];mask=m.bool()
        with torch.no_grad():
            q=t.critic(o,a,m).squeeze(-1);qt=t.target_critic(o,a,m).squeeze(-1)
            y=t.td_targets(b['rewards'],b['next_observation'],b['done'],b['next_alive'])
            act,_,raw=t.actor.sample(o);joint=act*m[...,None];qa=t.critic(o,joint,m).squeeze(-1)
            baseline=torch.stack([t.counterfactual_baseline(o,joint,m,i) for i in range(4)],1)
            advantage=qa-baseline
            rule=t.critic(o,b['rule_actions']*m[...,None],m).squeeze(-1)
            bad=b['rule_actions'].clone();bad[...,:2]*=-1
            wrong=t.critic(o,bad*m[...,None],m).squeeze(-1)
        for k,v in [('q',q),('target_q',qt),('td_residual',q-y),('advantage',advantage),('q_rule_minus_actor',rule-qa),('q_bad_minus_actor',wrong-qa)]:values[k].extend(v[mask].cpu().numpy().tolist())
        for k,key in [('terminal','done'),('fire','fired'),('kill','killed')]:
            selected=mask&b[key][:,None].bool();events[k].extend((q-y)[selected].cpu().numpy().tolist())
        selected=mask&~(b['done'].bool()|b['fired'].bool()|b['killed'].bool())[:,None]
        events['ordinary'].extend((q-y)[selected].cpu().numpy().tolist())
        lp=t.actor.log_prob(o,raw);loss=score_function_loss(lp,qa,baseline,t.alpha,m)
        gs=torch.autograd.grad(loss,t.actor.parameters());values['actor_grad'].append(float(torch.sqrt(sum(g.square().sum() for g in gs))))
        qc=t.critic(o,a,m).squeeze(-1);closs=((qc-y).square()*m).sum()/len(qc)
        params=list(t.critic.parameters());gs=torch.autograd.grad(closs,params);shared={id(p) for p in t.critic.shared_parameters()}
        values['critic_grad'].append(float(torch.sqrt(sum((g/(4 if id(p) in shared else 1)).square().sum() for p,g in zip(params,gs)))))
    result={k:stats(v) for k,v in values.items()};result['td_by_event']={k:stats(v) for k,v in events.items()}
    result['positive_advantage_fraction']=float(np.mean(np.array(values['advantage'])>0))
    result['actor_clip_fraction']=float(np.mean(np.array(values['actor_grad'])>.5));result['critic_clip_fraction']=float(np.mean(np.array(values['critic_grad'])>50))
    result['rule_q_above_wrong_fraction']=float(np.mean(np.array(values['q_rule_minus_actor'])>np.array(values['q_bad_minus_actor'])))
    result['scope']='saved-policy off-policy transitions, single target-action draw, 16 baseline samples; no optimizer steps; gradient norms are diagnostic mini-batches, not reconstructed training gradients'
    dump(out/'critic_diagnostics.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    if args.output.exists():raise FileExistsError(args.output)
    out=args.output;out.mkdir(parents=True);torch.set_num_threads(1)
    source=ROOT/'outputs/maac_v348_seed1_3m';ec=yaml.safe_load((source/'env_config.yaml').read_text());ac=yaml.safe_load((source/'algorithm_config.yaml').read_text())
    dump(out/'audit_protocol.json',dict(seeds=list(range(10000000,10000020))+[20000071],policy_rng='40000000+episode_seed',cuda=torch.cuda.get_device_name(),normal_load_roll='not applicable: v332 direct increment kinematics',no_optimizer_steps=True))
    trainers={};hashes={}
    for label,n in [('best','best_eval.pt'),('final','final.pt')]:
        t,_,_=load_recording_policy(source/n,ec,ac,'cuda');trainers[label]=t;hashes[n]=hashlib.sha256((source/n).read_bytes()).hexdigest()
    results={};collected={}
    for label,mode in [('best','det'),('final','det'),('best','stochastic'),('final','stochastic'),('best','rule')]:
        group=label+'_'+mode;(out/group).mkdir();res=[];data=[]
        # Collect list needs a truthy sentinel because episode checks collection enabled.
        if mode=='det':data=[None]
        for seed in list(range(10000000,10000020))+[20000071]:
            r=episode(trainers[label],ec,seed,mode,out/group/f'{seed}.csv',data);res.append(r)
            print(group,seed,'shots',r['attempts'],'win',r['red_win'],flush=True)
        results[group]=res;dump(out/(group+'_summary.json'),res)
        if mode=='det':collected[label]=data[1:]
    for label in ('best','final'):critic_diagnostics(trainers[label],collected[label],out/(label+'_det'))
    # Same observation bank separates parameter changes from trajectory shifts.
    bank=np.array([x['observation'] for x in collected['best'][::max(1,len(collected['best'])//256)]])
    np.save(out/'fixed_observation_bank.npy',bank)
    snapshots=[]
    paths=[source/'best_eval.pt']+sorted(source.glob('checkpoint_*.pt'))+[source/'final.pt']
    for path in paths:
        t,state,_=load_recording_policy(path,ec,ac,'cuda')
        with torch.no_grad():
            d=t.actor.distribution(t.tensor(bank));mu=d.mean.cpu().numpy();sd=d.scale.cpu().numpy()
        snapshots.append(dict(checkpoint=path.name,steps=state['sampled_steps'],mean=[stats(mu[...,i]) for i in range(3)],std=[stats(sd[...,i]) for i in range(3)],saturation=[float((np.abs(np.tanh(mu[...,i]))>=.99).mean()) for i in range(3)]))
        del t
    dump(out/'fixed_bank_checkpoint_snapshots.json',snapshots)
    reps=[a['seed'] for a,b in zip(results['best_det'],results['final_det']) if a['attempts']>0 and b['attempts']==0]
    dump(out/'representative_seeds.json',dict(best_fires_final_not=reps))
    for n,d in hashes.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==d
    dump(out/'checkpoint_hashes.json',hashes);dump(out/'all_episode_summaries.json',results)
    print('Read-only diagnosis complete',flush=True)


if __name__=='__main__':main()
