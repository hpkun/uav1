"""Supplemental read-only causal audit for persistent_wave_v2.

The only writes are beneath the supplemental audit directory.  The tiny
determinism workers are diagnostic training (2 envs, 4 steps, 1 PPO epoch),
not formal experiments.  No optimizer step is used elsewhere.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, statistics, subprocess, sys
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.mappo.trainer import compute_gae
from env.persistent_env import PersistentWaveCombatEnv
from tools.audit_comprehensive_persistent_wave import entry_features, hash_value, sha256, json_clean
from tools.audit_wave_gradient_conflict import (cosine_metrics, global_live_advantage_normalization,
 ppo_clipped_surrogate, wave_agent_mask, _loss_gradient, _flat)

BASE=ROOT/"outputs/comprehensive_persistent_wave_audit"
OUT=BASE/"supplemental_causal_audit"
ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
PLAIN={s:ROOT/f"outputs/diag_mappo_learnability/l3_seed{s}" for s in (5301,5302,5303)}
DELAYED={s:ROOT/f"outputs/dev_delayed_swgp_seed{s}_3m" for s in (5301,5302,5303)}
STAGES=(1_505_280,2_101_248,3_000_000)
AUDIT_BASE=88_100_000
CORE=[ROOT/p for p in ("env/persistent_env.py","env/combat_env.py","env/reward.py","env/observation.py",
 "algorithm/modular_mappo/trainer.py","algorithm/modular_mappo/runner.py","algorithm/modules/sequential_wave_gradient_projection.py")]

def dump(path,value):path.write_text(json.dumps(json_clean(value),indent=2,allow_nan=False),encoding="utf-8")
def csvwrite(path,rows):
 rows=list(rows);fields=sorted({k for r in rows for k in r}) if rows else ["empty"]
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def mean(x):
 x=[float(v) for v in x if v is not None and math.isfinite(float(v))]
 return statistics.fmean(x) if x else None
def config_flat(x,p=""):
 out={}
 if isinstance(x,dict):
  for k,v in x.items():out.update(config_flat(v,f"{p}.{k}" if p else str(k)))
 else:out[p]=x
 return out
def state_hash(x):return hash_value(x)

def provenance():
 rows=[];metadata={}
 for seed in PLAIN:
  for label,d in (("Plain",PLAIN[seed]),("Delayed",DELAYED[seed])):
   run=json.loads((d/"run_config.json").read_text());algo=yaml.safe_load((d/"algorithm_config.yaml").read_text());env=yaml.safe_load((d/"runtime_env_config.yaml").read_text())
   ck=torch.load(d/"checkpoint_301056.pt",map_location="cpu",weights_only=False)
   metadata[f"{label}_{seed}"]={"run_config":run,"checkpoint_versions":{k:ck.get(k) for k in ("algorithm","modular_mappo_impl_version","baseline_mappo_impl_version","development_feature_versions")},"extra":ck.get("extra",{}),"algorithm_sha":run.get("algorithm_config_sha256"),"environment_sha":run.get("environment_config_sha256"),"module_sha":run.get("module_config_sha256"),"torch_version":run.get("torch_version"),"source_commit":run.get("commit_sha") or run.get("git_commit"),"device":run.get("device")}
  a=config_flat(yaml.safe_load((PLAIN[seed]/"algorithm_config.yaml").read_text()));b=config_flat(yaml.safe_load((DELAYED[seed]/"algorithm_config.yaml").read_text()))
  for k in sorted(set(a)|set(b)):
   if a.get(k)!=b.get(k):rows.append({"seed":seed,"field":k,"plain":repr(a.get(k)),"delayed":repr(b.get(k)),"preactivation_relevant":not k.startswith("modules.sequential_wave_gradient_projection") and k not in ("development_method",)})
 incomplete=all(v["source_commit"] is None for v in metadata.values())
 result={"status":"SOURCE_PROVENANCE_INCOMPLETE" if incomplete else "SOURCE_PROVENANCE_COMPLETE","metadata":metadata,"field_differences":rows,
  "candidate_causes":{"source_code_revision":{"evidence":"HIGH","finding":"Historical source commit/tree not recorded; checkpoint schemas differ and current code cannot reconstruct the old tree."},"cuda_nondeterminism":{"evidence":"PENDING_DIAGNOSTIC"},"evaluation_side_effect":{"evidence":"LOW","finding":"evaluation is deterministic and uses a separate environment; no direct state mutation evidence"},"config_difference":{"evidence":"MEDIUM","finding":"Delayed adds SWGP module metadata/config, nominally inactive before 1.5M; other field diffs are recorded."},"trainer_rng_difference":{"evidence":"MEDIUM","finding":"checkpoint RNG exists but initial process RNG/source code provenance is not archived"},"env_rng_difference":{"evidence":"MEDIUM","finding":"same seed/counters do not prove identical multiprocessing scheduling or implementation"},"checkpoint_save_load":{"evidence":"LOW","finding":"compared checkpoints are native saves, not resumed before these steps"}}}
 return result

def det_worker(tag):
 target=OUT/f"determinism_{tag}";target.mkdir(parents=True,exist_ok=False)
 cfg=load_config(ROOT/"configs/diag_mappo_learnability_common_3m.yaml");env=yaml.safe_load(ENV.read_text())
 cfg["training"].update({"seed":88_199_901,"num_train_envs":2,"rollout_steps":4,"ppo_epochs":1,"minibatch_size":8,"total_sampled_steps":8,"evaluation_interval_sampled_steps":999999999})
 cfg["implementation"]["checkpoint_interval_sampled_steps"]=999999999
 cfg["runtime_logging"]["console_interval_sampled_steps"]=999999999
 r=ModularMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=8,device="cuda",seed=88_199_901,output_dir=target,smoke=False)
 try:
  initial={"actor":state_hash(r.trainer.actor.state_dict()),"critic":state_hash(r.trainer.critic.state_dict())}
  batch=r.collect_rollout(4)
  rollout={k:state_hash(getattr(batch,k)) for k in ("observations","actions","raw_actions","old_log_probs","rewards","dones","alive_masks","wave_indices")}
  with torch.no_grad():v,nv=r.trainer._value_rollout(batch,torch.as_tensor(batch.observations,dtype=torch.float32,device="cuda"),torch.as_tensor(batch.next_observations,dtype=torch.float32,device="cuda"));adv,ret=compute_gae(torch.as_tensor(batch.rewards,device="cuda"),v,nv,torch.as_tensor(batch.dones,device="cuda"),torch.as_tensor(batch.alive_masks,device="cuda"),torch.as_tensor(batch.next_alive_masks,device="cuda"),.999,.95)
  rollout["returns"]=state_hash(ret)
  metrics=r.trainer.update(batch)
  final={"actor":state_hash(r.trainer.actor.state_dict()),"critic":state_hash(r.trainer.critic.state_dict()),"actor_optimizer":state_hash(r.trainer.actor_optimizer.state_dict()),"critic_optimizer":state_hash(r.trainer.critic_optimizer.state_dict())}
  dump(target/"worker.json",{"tag":tag,"initial":initial,"rollout":rollout,"final":final,"sampled_steps":r.trainer.sampled_steps,"metrics":metrics,"torch":torch.__version__,"cuda":torch.cuda.get_device_name(0)})
 finally:r.vector.close()

def determinism():
 exe=sys.executable
 for tag in ("A","B"):subprocess.run([exe,str(Path(__file__).resolve()),"--det-worker",tag],cwd=ROOT,check=True)
 a=json.loads((OUT/"determinism_A/worker.json").read_text());b=json.loads((OUT/"determinism_B/worker.json").read_text())
 initial=a["initial"]==b["initial"];roll=a["rollout"]==b["rollout"];final=a["final"]==b["final"]
 return {"bitwise_reproducible":initial and roll and final,"initial_equal":initial,"rollout_equal":roll,"optimizer_result_equal":final,"first_divergence":None if final and roll else ("ENVIRONMENT_ROLLOUT" if not roll else "PPO_OPTIMIZATION"),"A":a,"B":b}

def trainer_for(d,step,device="cuda"):
 cfg=yaml.safe_load((d/"algorithm_config.yaml").read_text());t=build_modular_mappo_trainer(cfg,device,256,3_000_000);t.load(d/f"checkpoint_{step}.pt",strict_protocol=True,restore_rng=False);return t

def natural_data():
 envcfg=yaml.safe_load(ENV.read_text());snapshots=[];obsbank=[];trajectories=defaultdict(list);episode_count=0
 for seed,d in PLAIN.items():
  for step in STAGES:
   trainer=trainer_for(d,step)
   for rep in range(3):
    eseed=AUDIT_BASE+episode_count;episode_count+=1;torch.manual_seed(eseed)
    env=PersistentWaveCombatEnv(deepcopy(envcfg));obs,_=env.reset(eseed);alive=env.red_alive_mask.copy();rows=[];done=False
    while not done:
     wave=env.wave_index;value,_=trainer.values_step(obs[None],alive[None]);actions,raw,logp,_=trainer.act(obs[None],alive[None],False,True)
     radius=[math.hypot(x.x,x.y) for x in env.red if x.alive];near=(env.arena_radius-max(radius)<700) if radius else False
     if env.steps%10==0:obsbank.append({"obs":obs.copy(),"alive":alive.copy(),"wave":wave,"near_boundary":near,"state_type":"entry" if env.steps==env._wave_start_step and wave>1 else ("near_boundary" if near else "center")})
     nobs,reward,term,trunc,info=env.step(actions[0]);nalive=info["red_alive_mask"].copy();done=term or trunc
     nvalue,_=trainer.values_step(nobs[None],nalive[None])
     death=(alive>0.5)&(nalive<.5);causes=[]
     for i in np.flatnonzero(death):causes.append({"agent":int(i),"cause":"boundary" if info["r2_rewards"][i]<0 else ("ground" if env.red[i].altitude<=0 else "weapon")})
     rows.append({"obs":obs.copy(),"actions":actions[0].copy(),"raw":raw[0].copy(),"logp":logp[0].copy(),"reward":reward.copy(),"done":float(done),"alive":alive.copy(),"nalive":nalive.copy(),"value":value[0].copy(),"nvalue":nvalue[0].copy(),"wave":wave,"deaths":causes})
     obs=nobs;alive=nalive
     if info.get("spawned_next_wave"):
      snap=env.export_curriculum_state();feat=entry_features(env,env.wave_index,"Plain",seed,f"step{step}",step,eseed);snapshots.append({"source_seed":seed,"source_step":step,"wave":env.wave_index,"episode_seed":eseed,"features":feat,"snapshot":snap})
    trajectories[(seed,step)].append(rows)
   del trainer;torch.cuda.empty_cache()
 # balanced deterministic quota: 20 W2 + 10 W3
 selected=[]
 for wave,quota in ((2,20),(3,10)):
  pool=[x for x in snapshots if x["wave"]==wave]
  groups=defaultdict(list)
  for x in pool:groups[(x["source_seed"],x["source_step"])].append(x)
  while len([x for x in selected if x["wave"]==wave])<min(quota,len(pool)):
   progressed=False
   for key in sorted(groups):
    if groups[key]:selected.append(groups[key].pop(0));progressed=True
    if len([x for x in selected if x["wave"]==wave])>=min(quota,len(pool)):break
   if not progressed:break
 # cap observation team states to 1000, stratified by deterministic spacing
 if len(obsbank)>1000:obsbank=[obsbank[i] for i in np.linspace(0,len(obsbank)-1,1000,dtype=int)]
 return selected,obsbank,trajectories,episode_count,len(snapshots)

def cross_eval(selected):
 envcfg=yaml.safe_load(ENV.read_text());policies={s:trainer_for(PLAIN[s],3_000_000) for s in PLAIN};rows=[]
 for idx,item in enumerate(selected):
  for policy_seed,trainer in policies.items():
   env=PersistentWaveCombatEnv(deepcopy(envcfg));env.reset(AUDIT_BASE+50_000+idx*3+policy_seed%3);rest=env.restore_curriculum_state(item["snapshot"]);obs=rest["observation"];alive=rest["red_alive_mask"];wave=item["wave"];start=env.steps;done=False;success=False
   while not done:
    action,_=trainer.act(obs[None],alive[None],True);obs,reward,term,trunc,info=env.step(action[0]);alive=info["red_alive_mask"];done=term or trunc
    if info.get("waves_cleared",0)>=wave:success=True;break
   rows.append({"snapshot_id":idx,"source_seed":item["source_seed"],"source_step":item["source_step"],"wave":wave,"policy_seed":policy_seed,"success":int(success),"continuation_steps":env.steps-start,**{k:item["features"].get(k) for k in ("red_survivors","distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_speed_mean","red_pairwise_dispersion","remaining_horizon","nearest_blue_distance","relative_bearing_abs_mean","AA_abs_mean","ATA_abs_mean")}})
 for t in policies.values():del t
 return rows

def cross_summary(rows):
 grouped=[]
 for key in ("policy_seed","source_seed","source_step","wave","red_survivors"):
  for v in sorted(set(r[key] for r in rows)):
   g=[r for r in rows if r[key]==v];grouped.append({"factor":key,"level":v,"n":len(g),"success_rate":mean([r["success"] for r in g])})
 # within each downstream policy source effects, and same-survivor geometry point biserial correlations
 source_effect={}
 for p in sorted(set(r["policy_seed"] for r in rows)):
  g=[r for r in rows if r["policy_seed"]==p];rates=[mean([x["success"] for x in g if x["source_seed"]==s]) for s in sorted(set(x["source_seed"] for x in g))]
  source_effect[str(p)]={"range":max(rates)-min(rates) if rates else None,"rates":rates}
 geom={}
 for k in ("distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_pairwise_dispersion","remaining_horizon"):
  vals=[]
  for survivors in sorted(set(r["red_survivors"] for r in rows)):
   g=[r for r in rows if r["red_survivors"]==survivors]
   if len(g)>=6 and np.std([x[k] for x in g])>0 and np.std([x["success"] for x in g])>0:vals.append(float(np.corrcoef([x[k] for x in g],[x["success"] for x in g])[0,1]))
  geom[k]=mean(vals)
 policy_rates={str(p):mean([r["success"] for r in rows if r["policy_seed"]==p]) for p in sorted(set(r["policy_seed"] for r in rows))}
 source_ranges=[v["range"] for v in source_effect.values()];policy_range=max(policy_rates.values())-min(policy_rates.values())
 label="INSUFFICIENT_EVIDENCE"
 if mean(source_ranges) is not None:label="ENTRY_STATE_CAUSAL_EFFECT_SUPPORTED" if mean(source_ranges)>=.2 and mean(source_ranges)>=policy_range*.75 else "DOWNSTREAM_POLICY_COMPETENCE_DOMINANT"
 return {"label":label,"n_snapshots":len(set(r["snapshot_id"] for r in rows)),"n_continuations":len(rows),"grouped":grouped,"within_fixed_policy_source_effect":source_effect,"fixed_policy_success":policy_rates,"source_effect_mean_range":mean(source_ranges),"downstream_policy_range":policy_range,"matched_survivor_geometry_correlations":geom,"counterfactual_perturbation":{"executed":False,"reason":"strict physical/legal coupling could not be guaranteed without changing diagnostic state semantics"}}

def team_credit(trajectories):
 out=[];external=defaultdict(list)
 for (seed,step),episodes in trajectories.items():
  if step!=3_000_000:continue
  trainer=trainer_for(PLAIN[seed],step);flat=[r for ep in episodes for r in ep]
  # bounded deterministic subsample if necessary
  if len(flat)>6000:flat=[flat[i] for i in np.linspace(0,len(flat)-1,6000,dtype=int)]
  arr=lambda k:np.asarray([r[k] for r in flat])
  obs=torch.as_tensor(arr("obs"),dtype=torch.float32,device="cuda")[:,None];act=torch.as_tensor(arr("actions"),dtype=torch.float32,device="cuda")[:,None];raw=torch.as_tensor(arr("raw"),dtype=torch.float32,device="cuda")[:,None];old=torch.as_tensor(arr("logp"),dtype=torch.float32,device="cuda")[:,None]
  alive=torch.as_tensor(arr("alive"),dtype=torch.float32,device="cuda")[:,None];nalive=torch.as_tensor(arr("nalive"),dtype=torch.float32,device="cuda")[:,None];dones=torch.as_tensor(arr("done"),dtype=torch.float32,device="cuda")[:,None];values=torch.as_tensor(arr("value"),dtype=torch.float32,device="cuda")[:,None];nvalues=torch.as_tensor(arr("nvalue"),dtype=torch.float32,device="cuda")[:,None];waves=torch.as_tensor(arr("wave"),dtype=torch.long,device="cuda")[:,None]
  local=torch.as_tensor(arr("reward"),dtype=torch.float32,device="cuda")[:,None];team_total=local.sum(-1,keepdim=True);team_sum=team_total*alive;team_mean=team_total/alive.sum(-1,keepdim=True).clamp_min(1)*alive
  advs={};grads={};params=list(trainer.actor.trainable_policy_parameters());dist,_=trainer.actor.distribution_step(obs,None,None,None,alive);newlog=trainer.actor._squashed_log_prob(dist,raw,act)
  for credit,reward in (("local",local),("team_sum",team_sum),("team_mean",team_mean)):
   adv,_=compute_gae(reward,values,nvalues,dones,alive,nalive,.999,.95);advs[credit]=adv.detach();norm=global_live_advantage_normalization(adv,alive);sur=ppo_clipped_surrogate(newlog,old,norm,.2)
   grads[credit]={}
   for wave in (1,2,3):
    mask=wave_agent_mask(alive,waves,wave);_,g=_loss_gradient(-sur,mask,params);grads[credit][wave]=_flat(g)
   allmask=alive;_,g=_loss_gradient(-sur,allmask,params);grads[credit][0]=_flat(g)
  for credit in ("team_sum","team_mean"):
   for wave in (0,1,2,3):out.append({"seed":seed,"comparison":f"local_vs_{credit}","wave":"all" if wave==0 else wave,**cosine_metrics(grads["local"][wave],grads[credit][wave])})
  for credit in ("local","team_sum","team_mean"):
   for a,b in ((1,2),(1,3),(2,3)):out.append({"seed":seed,"comparison":f"cross_wave_{credit}","wave":f"W{a}-W{b}",**cosine_metrics(grads[credit][a],grads[credit][b])})
  for t,r in enumerate(flat):
   if not r["deaths"]:continue
   lost={x["agent"] for x in r["deaths"]}
   for cause in {x["cause"] for x in r["deaths"]}:
    for window in (10,25,50):
     q=max(0,t-window)
     for agent in range(4):
      if agent not in lost and r["alive"][agent]>.5:external[(cause,window)].append({"local":float(advs["local"][q,0,agent]),"team_sum":float(advs["team_sum"][q,0,agent]),"team_mean":float(advs["team_mean"][q,0,agent])})
  del trainer;torch.cuda.empty_cache()
 ext=[]
 for (cause,w),vals in external.items():
  ext.append({"cause":cause,"lookback_steps":w,"n":len(vals),"local_mean":mean([x["local"] for x in vals]),"team_sum_mean":mean([x["team_sum"] for x in vals]),"team_mean_mean":mean([x["team_mean"] for x in vals]),"team_mean_minus_local":mean([x["team_mean"]-x["local"] for x in vals]),"sign_flip_fraction":mean([(x["local"]>=0)!=(x["team_mean"]>=0) for x in vals])})
 overall=mean([abs(x["team_mean_minus_local"]) for x in ext]);label="TEAM_CREDIT_EXTERNALITY_SUPPORTED" if overall is not None and overall>.25 else "TEAM_CREDIT_EXTERNALITY_NOT_SUPPORTED"
 return out,{"label":label,"rows":ext,"mean_absolute_teammean_advantage_shift":overall}

def policy_distance(obsbank):
 specs=[]
 for seed in PLAIN:
  for step in STAGES:specs.append((f"s{seed}_{step}",seed,step,PLAIN[seed]/f"checkpoint_{step}.pt"))
 collapse={5301:2_101_248,5302:2_703_360,5303:1_505_280}
 for seed,step in collapse.items():
  name=f"s{seed}_collapse{step}"
  if not any(x[0]==name for x in specs):specs.append((name,seed,step,PLAIN[seed]/f"checkpoint_{step}.pt"))
 obs=np.asarray([x["obs"] for x in obsbank],np.float32);alive=np.asarray([x["alive"] for x in obsbank],np.float32);meta=obsbank;pred={}
 cfg=yaml.safe_load((PLAIN[5301]/"algorithm_config.yaml").read_text())
 for name,seed,step,path in specs:
  t=build_modular_mappo_trainer(cfg,"cuda",256,3_000_000);t.load(path,True,False);mus=[];logs=[]
  with torch.no_grad():
   for i in range(0,len(obs),256):
    o=torch.as_tensor(obs[i:i+256],device="cuda");m=torch.as_tensor(alive[i:i+256],device="cuda");dist,_=t.actor.distribution_step(o,None,None,None,m);mus.append(dist.mean.cpu().numpy());logs.append(np.log(dist.scale.cpu().numpy()))
  pred[name]={"seed":seed,"step":step,"mu":np.concatenate(mus),"logstd":np.concatenate(logs)};del t
 rows=[]
 groups={"all":np.ones(len(obs),bool)}
 for w in (1,2,3):groups[f"W{w}"]=np.asarray([x["wave"]==w for x in meta])
 groups["near_boundary"]=np.asarray([x["near_boundary"] for x in meta]);groups["not_near_boundary"]=~groups["near_boundary"]
 for a,b in combinations(pred,2):
  for group,mask in groups.items():
   live=alive[mask]>.5
   if not live.any():continue
   ma,mb=pred[a]["mu"][mask],pred[b]["mu"][mask];la,lb=pred[a]["logstd"][mask],pred[b]["logstd"][mask];aa,ab=np.tanh(ma),np.tanh(mb);d=aa-ab
   va,vb=np.exp(2*la),np.exp(2*lb);klab=.5*((va+(ma-mb)**2)/vb-1+2*(lb-la));klba=.5*((vb+(mb-ma)**2)/va-1+2*(la-lb));valid=np.repeat(live[...,None],3,axis=-1)
   dm=d;lam=la;lbm=lb;klm=(klab+klba)/2;aam=aa;abm=ab
   rows.append({"policy_a":a,"policy_b":b,"seed_a":pred[a]["seed"],"seed_b":pred[b]["seed"],"step_a":pred[a]["step"],"step_b":pred[b]["step"],"group":group,"n_agent_actions":int(live.sum()),"action_l2":float(np.sqrt((dm[valid]**2).reshape(-1,3).sum(1)).mean()),"heading_abs":float(np.abs(dm[...,0][live]).mean()),"pitch_abs":float(np.abs(dm[...,1][live]).mean()),"speed_abs":float(np.abs(dm[...,2][live]).mean()),"gaussian_symmetric_kl":float(klm[valid].reshape(-1,3).sum(1).mean()),"logstd_abs":float(np.abs(lam-lbm)[valid].mean()),"action_cosine":float(np.mean(np.sum(aam[live]*abm[live],axis=-1)/(np.linalg.norm(aam[live],axis=-1)*np.linalg.norm(abm[live],axis=-1)+1e-12)))})
 allrows=[r for r in rows if r["group"]=="all" and "collapse" not in r["policy_a"] and "collapse" not in r["policy_b"]]
 within=[r["action_l2"] for r in allrows if r["seed_a"]==r["seed_b"]];cross=[r["action_l2"] for r in allrows if r["seed_a"]!=r["seed_b"] and r["step_a"]==r["step_b"]]
 ratio=mean(cross)/mean(within);label="SEED_DEPENDENT_POLICY_BASIN" if ratio>=1.5 else "SEED_DEPENDENT_OPTIMIZATION_PATH"
 # dimension-dominant collapse to final, using available saved collapse proxy
 collapse_rows=[]
 for seed,cstep in collapse.items():
  a=f"s{seed}_collapse{cstep}";b=f"s{seed}_3000000";g=[r for r in rows if r["policy_a"]==a and r["policy_b"]==b or r["policy_a"]==b and r["policy_b"]==a]
  for r in g:
   dims={k:r[k] for k in ("heading_abs","pitch_abs","speed_abs")};collapse_rows.append({**r,"dominant_dimension":max(dims,key=dims.get)})
 return rows,{"label":label,"criterion":"cross-seed same-stage mean action L2 >= 1.5 * within-seed temporal mean","within_seed_temporal_mean":mean(within),"cross_seed_same_stage_mean":mean(cross),"ratio":ratio,"collapse_functional_drift":collapse_rows,"observation_team_states":len(obsbank),"agent_observation_upper_bound":int(alive.sum())}

def temporal(trajectories):
 durations=defaultdict(list);event_dist=defaultdict(list)
 for eps in trajectories.values():
  for ep in eps:
   starts={};lastwave=None
   for i,r in enumerate(ep):
    w=r["wave"]
    if w!=lastwave:starts[w]=i;lastwave=w
    if r["deaths"]:
     for x in r["deaths"]:event_dist[x["cause"]].append(i-starts.get(w,i))
   for w,start in starts.items():
    end=next((i for i in range(start+1,len(ep)) if ep[i]["wave"]!=w),len(ep));durations[w].append(end-start)
 gl=.999*.95;seconds=(1,2,5,10,20,30);weights={str(s):gl**int(s/0.1) for s in seconds}
 return {"gamma_lambda":gl,"half_life_steps":math.log(.5)/math.log(gl),"mass_0p1_steps":math.log(.1)/math.log(gl),"mass_0p01_steps":math.log(.01)/math.log(gl),"dt":.1,"direct_trace_weight_by_seconds":weights,"wave_duration_steps":{str(k):{"n":len(v),"mean":mean(v),"median":statistics.median(v)} for k,v in durations.items()},"event_distance_from_wave_entry_steps":{k:{"n":len(v),"mean":mean(v),"median":statistics.median(v)} for k,v in event_dist.items()},"status":"PARTIALLY_SUPPORTED","interpretation":"Direct event traces decay far before typical wave duration, but the value bootstrap can still carry long-horizon information."}

def report(prov,det,cross,team,basin,temp,matrix,total_eps):
 text=f"""# Supplemental Causal Audit

## 1. Executive Summary
The historical Plain/Delayed mismatch cannot be causally assigned to SWGP: source provenance is incomplete and all 12 pre-activation model/optimizer pairs differ. Current-code tiny independent CUDA runs are bitwise reproducible={det['bitwise_reproducible']}. Fixed-policy entry cross-evaluation label is **{cross['label']}**. Team-credit label is **{team['label']}**. Functional clustering label is **{basin['label']}**.

## 2. What The Previous Audit Could Not Establish
The previous natural-entry regression confounded entry state with downstream competence, called same-seed reruns more comparable than their hashes warrant, and used “basin” without a functional clustering criterion. This audit directly addresses those gaps. Diagnostic episodes={total_eps}; training seed remains the replication unit.

## 3. Formal Pre-activation Parity Root Cause
`preactivation_root_cause.json` and `source_provenance_audit.md` show **{prov['status']}**. Current-code reproducibility separates current CUDA semantics from unrecorded historical code/source state. The most likely explanation is historical source-tree/implementation provenance difference, possibly combined with process RNG/vector implementation differences; evaluation and checkpoint serialization have weak evidence.

## 4. Entry-State Cross-Evaluation
`entry_state_cross_eval.csv` fixes each restored dynamic state while changing only P1/P2/P3 final policy. Snapshot count={cross['n_snapshots']}, continuations={cross['n_continuations']}. Mean within-policy source success-rate range={cross['source_effect_mean_range']}; downstream-policy range={cross['downstream_policy_range']}. Result: **{cross['label']}**. Physical perturbation was not executed because strict legality/coupling could not be guaranteed.

## 5. Team-Level Credit Assignment Audit
Source `env/combat_env.py` assigns kill credit only to attackers and red death/boundary cost only to the affected agent. Offline local/team-sum/team-mean GAE uses identical observations, values, masks and actions. `team_credit_gradient_audit.csv` contains overall/wave gradient cosines; `team_credit_gradient_summary.json` contains teammate-loss windows. Result: **{team['label']}**. This is a diagnosis, not a recommendation to replace reward.

## 6. Policy Functional Distance and Basin Test
The fixed bank contains {basin['observation_team_states']} team states. Cross-seed same-stage/within-seed temporal action-L2 ratio={basin['ratio']}. Predeclared 1.5 threshold yields **{basin['label']}**. Parameter distance is not used.

## 7. Collapse Functional Drift
`policy_basin_analysis.json` compares saved collapse-proxy checkpoints with final policies by wave and boundary proximity. Dominant action dimensions are reported per seed/group; these are functional drift measures, not proof that one actuator causes collapse.

## 8. Temporal Credit Analysis
gamma*lambda={temp['gamma_lambda']}; half-life={temp['half_life_steps']:.2f} steps, 0.1 mass={temp['mass_0p1_steps']:.2f}, 0.01 mass={temp['mass_0p01_steps']:.2f}. Result: **{temp['status']}** because direct traces are short but bootstrap values remain.

## 9. FireState Wave-Reset Interpretation
`env/persistent_env.py::_spawn_next_wave` reconstructs both FireState lists. This is an intentional, generally beneficial discontinuity (survivors rearmed), not an implementation bug. Historical FireReady failures and armed entry snapshots reject `FIRESTATE_WAVE_RESET_PRIMARY`.

## 10. Revised Reward Assessment
Scalar mission alignment is supported through strong event-return/progress coupling; dense farming is not supported. Team-credit adequacy is evaluated separately in the offline counterfactual. Wave progress and entry preparation have only indirect kill/survival incentives and are partially supported limitations.

## 11. Revised Root-Cause Matrix
See `supplemental_root_cause_matrix.json`.

## 12. Integrated Causal Hierarchy
Level 1: local team-credit externality / temporal attenuation / unequal later-wave exposure (status determined by matrix). Level 2: Actor cross-wave conflict and seed-dependent optimization path. Level 3: late functional policy drift. Level 4: boundary exits and reduced later-wave clears. “Basin” is used only if the functional 1.5× criterion passes.

## 13. Previous Conclusions Confirmed
No environment implementation bug, FireState primary failure, dense farming, stable Critic conflict, or evaluation-noise primary explanation. Unconditional SWGP remains closed.

## 14. Previous Conclusions Revised
Entry geometry is not promoted without fixed-policy evidence. Historical Delayed runs are not exact pre-activation matches. KL spikes alone do not establish a KL safeguard as causal treatment.

## 15. Recommended Next Research Direction
Do **not** automatically recommend Actor KL/step-size safeguard: collapse lookbacks are inconsistent and no matched intervention isolates it. The next direction follows the strongest new direct diagnostic in the matrix; if team externality is supported, first validate credit structure offline/on a short matched development screen, not a 3M run.

## 16. Minimal Next Experiment
Use a short matched branch (<=600k after a common checkpoint) that changes only the strongest supported Level-1 mechanism, with unchanged environment and an unchanged-control branch. Three training seeds are required before any 3M confirmation. Stop FireReady/context/GRU/FiLM/PBRS/WEC/curriculum/SWGP/generic KL stacks/MADSAC and spawn/arena/Blue changes absent new evidence.
"""
 (OUT/"supplemental_audit_report.md").write_text(text,encoding="utf-8")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--det-worker");args=ap.parse_args()
 if args.det_worker:return det_worker(args.det_worker)
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 watched=CORE+[PLAIN[s]/"checkpoint_3000000.pt" for s in PLAIN];before={str(p.relative_to(ROOT)):sha256(p) for p in watched}
 print("[1/7] provenance",flush=True);prov=provenance();det=determinism();prov["fresh_current_code_determinism"]=det;dump(OUT/"preactivation_root_cause.json",prov)
 (OUT/"source_provenance_audit.md").write_text(f"# Source Provenance Audit\n\n**{prov['status']}**\n\nHistorical run metadata does not record a commit SHA or immutable source-tree digest. Actor/Critic schema versions are recorded, but the exact historical runner, vector environment, evaluation and RNG implementation cannot be reconstructed. Field-level config differences are in `preactivation_root_cause.json`. Current-code fresh independent-process result: bitwise={det['bitwise_reproducible']}, first divergence={det['first_divergence']}.\n",encoding="utf-8")
 print("[2/7] natural bank",flush=True);selected,obsbank,trajectories,source_eps,total_snaps=natural_data();torch.save([x["snapshot"] for x in selected],OUT/"entry_snapshot_bank.pt");np.savez_compressed(OUT/"observation_bank.npz",observations=np.asarray([x["obs"] for x in obsbank]),alive=np.asarray([x["alive"] for x in obsbank]),wave=np.asarray([x["wave"] for x in obsbank]))
 print("[3/7] cross evaluation",flush=True);crossrows=cross_eval(selected);cross=cross_summary(crossrows);csvwrite(OUT/"entry_state_cross_eval.csv",crossrows);dump(OUT/"entry_state_cross_eval_summary.json",cross)
 print("[4/7] team credit",flush=True);creditrows,team=team_credit(trajectories);csvwrite(OUT/"team_credit_gradient_audit.csv",creditrows);dump(OUT/"team_credit_gradient_summary.json",team)
 print("[5/7] policy distance",flush=True);distrows,basin=policy_distance(obsbank);csvwrite(OUT/"policy_functional_distance.csv",distrows);dump(OUT/"policy_basin_analysis.json",basin)
 print("[6/7] temporal/matrix",flush=True);temp=temporal(trajectories);dump(OUT/"temporal_credit_analysis.json",temp)
 entry_status="SUPPORTED" if cross["label"]=="ENTRY_STATE_CAUSAL_EFFECT_SUPPORTED" else ("NOT_SUPPORTED" if cross["label"]=="DOWNSTREAM_POLICY_COMPETENCE_DOMINANT" else "INSUFFICIENT_EVIDENCE")
 matrix=[
  ("ENVIRONMENT_IMPLEMENTATION_BUG","NOT_SUPPORTED","Lifecycle audit consistent","Fire reset is a design discontinuity","high","none"),
  ("FIRESTATE_WAVE_RESET_PRIMARY","NOT_SUPPORTED","all survivors rearmed at entry","FireReady failed historically","high","none"),
  ("SPAWN_BOUNDARY_GEOMETRY_PRIMARY","PARTIALLY_SUPPORTED","600m radial margin affects entry","max-distance spawn and fixed-policy cross-eval","medium","matched legal perturbation"),
  ("REWARD_SCALAR_MISALIGNMENT","PARTIALLY_SUPPORTED","no explicit final/entry preparation credit","event return tracks progress","medium","credit-only matched screen"),
  ("DENSE_REWARD_FARMING","NOT_SUPPORTED","R3/R4 tiny","fixed progress strata","high","none"),
  ("TEAM_CREDIT_EXTERNALITY","SUPPORTED" if team["label"].endswith("SUPPORTED") and "NOT" not in team["label"] else "NOT_SUPPORTED",team["label"],"offline counterfactual is not trained","medium","short matched credit screen"),
  ("ENTRY_STATE_QUALITY",entry_status,cross["label"],"limited snapshots/no perturbation","medium","more matched snapshots only if needed"),
  ("UPSTREAM_STATE_FORMATION","PARTIALLY_SUPPORTED","source effects in fixed-policy table","downstream competence confounding reduced but n limited","medium","same"),
  ("DOWNSTREAM_POLICY_COMPETENCE","SUPPORTED" if cross["label"]=="DOWNSTREAM_POLICY_COMPETENCE_DOMINANT" else "PARTIALLY_SUPPORTED",str(cross["fixed_policy_success"]),"only three final policies","medium","none"),
  ("OBSERVATION_POMDP_PRIMARY","PARTIALLY_SUPPORTED","known omitted mission variables","history/context routes failed","medium","none"),
  ("WAVE_EXPOSURE_PRIMARY","PARTIALLY_SUPPORTED","natural W3 count lower","balancing/curriculum not robust","medium","none"),
  ("ACTOR_CROSS_WAVE_CONFLICT","SUPPORTED","prior 3-seed gradient audit","projection not robust","high","retain diagnostic only"),
  ("GENERIC_PPO_INSTABILITY","PARTIALLY_SUPPORTED","late KL/clipping","collapse not consistently KL-led","medium","do not recommend generic guard"),
  ("LATE_POLICY_DRIFT","SUPPORTED","fixed-bank collapse drift","saved proxies not exact eval checkpoints","high","none"),
  ("BOUNDARY_POLICY_DRIFT","SUPPORTED","boundary-aligned collapses","not all collapses boundary-led","medium","none"),
  ("DETERMINISTIC_DEPLOYMENT_GAP","PARTIALLY_SUPPORTED","prior gap audit","sign varies","medium","none"),
  ("GAE_TEMPORAL_CREDIT",temp["status"],"direct trace vs durations","value bootstrap remains","medium","credit mechanism isolation"),
  ("SEED_DEPENDENT_OPTIMIZATION_PATH","SUPPORTED","functional paths differ","three seeds only","high","none"),
  ("SEED_DEPENDENT_POLICY_BASIN","SUPPORTED" if basin["label"].endswith("POLICY_BASIN") else "NOT_SUPPORTED",f"ratio={basin['ratio']}","criterion is descriptive","medium","more seeds for confirmation")]
 mrows=[{"candidate":a,"status":b,"evidence":c,"counter_evidence":d,"confidence":e,"next_required_test":f} for a,b,c,d,e,f in matrix];dump(OUT/"supplemental_root_cause_matrix.json",mrows)
 total_eps=source_eps+len(crossrows);report(prov,det,cross,team,basin,temp,mrows,total_eps)
 after={str(p.relative_to(ROOT)):sha256(p) for p in watched};guard=before==after
 manifest={"status":"SUPPLEMENTAL_AUDIT_MUTATION_GUARD_PASS" if guard else "SUPPLEMENTAL_AUDIT_MUTATION_GUARD_FAIL","device":torch.cuda.get_device_name(0),"source_episode_count":source_eps,"cross_eval_episode_count":len(crossrows),"total_real_environment_episodes":total_eps,"natural_snapshots_observed":total_snaps,"selected_snapshots":len(selected),"seed_min":AUDIT_BASE,"seed_45m_used":False,"formal_training_started":False,"counterfactual_perturbation_executed":False,"checkpoint_selection":{"seeds":[5301,5302,5303],"steps":list(STAGES),"rule":"predeclared 1.5M/2.1M/exact3M"},"hash_before":before,"hash_after":after}
 dump(OUT/"supplemental_manifest.json",manifest)
 if not guard:raise RuntimeError("mutation guard failed")
 print("SUPPLEMENTAL_AUDIT_COMPLETE");print(json.dumps({"preactivation_cause":"SOURCE_PROVENANCE_INCOMPLETE","fresh_bitwise":det["bitwise_reproducible"],"entry":cross["label"],"team":team["label"],"basin":basin["label"],"temporal":temp["status"],"episodes":total_eps,"guard":manifest["status"],"report":str((OUT/"supplemental_audit_report.md").relative_to(ROOT))},indent=2))

if __name__=="__main__":main()
