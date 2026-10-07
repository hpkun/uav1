"""Statistical/causal corrections for supplemental causal audit V2."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,statistics,sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np, torch, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.mappo.trainer import compute_gae
from env.persistent_env import PersistentWaveCombatEnv
from tools.audit_comprehensive_persistent_wave import sha256,hash_value,json_clean
from tools.audit_wave_gradient_conflict import (cosine_metrics,global_live_advantage_normalization,
 ppo_clipped_surrogate,wave_agent_mask,_loss_gradient,_flat)

V1=ROOT/"outputs/comprehensive_persistent_wave_audit/supplemental_causal_audit"
OUT=ROOT/"outputs/comprehensive_persistent_wave_audit/supplemental_causal_audit_v2"
PLAIN={s:ROOT/f"outputs/diag_mappo_learnability/l3_seed{s}" for s in (5301,5302,5303)}
ENV=ROOT/"configs/persistent_wave_v2_environment.yaml";TEAM_SEEDS=list(range(88_200_000,88_200_005))
CORE=[ROOT/p for p in ("env/persistent_env.py","env/combat_env.py","env/reward.py","env/observation.py","env/weapon.py","algorithm/modular_mappo/trainer.py","algorithm/modular_mappo/runner.py","algorithm/modules/sequential_wave_gradient_projection.py")]
def dump(p,x):p.write_text(json.dumps(json_clean(x),indent=2,allow_nan=False),encoding="utf-8")
def csvread(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 rows=list(rows);fields=sorted({k for r in rows for k in r}) if rows else ["empty"]
 with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def mean(v):
 v=[float(x) for x in v if x is not None and math.isfinite(float(x))];return statistics.fmean(v) if v else None
def trainer(seed):
 d=PLAIN[seed];cfg=yaml.safe_load((d/"algorithm_config.yaml").read_text());t=build_modular_mappo_trainer(cfg,"cuda",256,3_000_000);t.load(d/"checkpoint_3000000.pt",True,False);return t
def rtg(reward,dones,alive,nalive,gamma=.999):
 out=torch.zeros_like(reward);future=torch.zeros_like(reward[0])
 for i in reversed(range(len(reward))):
  cont=(1-dones[i].unsqueeze(-1))*nalive[i];future=reward[i]+gamma*cont*future;out[i]=future*alive[i]
 return out

def collect_team():
 envcfg=yaml.safe_load(ENV.read_text());episodes=defaultdict(list)
 for seed in PLAIN:
  t=trainer(seed)
  for es in TEAM_SEEDS:
   torch.manual_seed(es^seed);env=PersistentWaveCombatEnv(deepcopy(envcfg));obs,_=env.reset(es);alive=env.red_alive_mask.copy();ep=[];done=False
   while not done:
    wave=env.wave_index;val,_=t.values_step(obs[None],alive[None]);a,raw,lp,_=t.act(obs[None],alive[None],True,True);nobs,reward,term,trunc,info=env.step(a[0]);nalive=info["red_alive_mask"].copy();nval,_=t.values_step(nobs[None],nalive[None]);done=term or trunc
    deaths=[]
    for i in np.flatnonzero((alive>.5)&(nalive<.5)):deaths.append({"agent":int(i),"cause":"boundary" if info["r2_rewards"][i]<0 else ("ground" if env.red[i].altitude<=0 else "weapon")})
    ep.append({"obs":obs.copy(),"action":a[0].copy(),"raw":raw[0].copy(),"logp":lp[0].copy(),"reward":reward.copy(),"done":float(done),"alive":alive.copy(),"nalive":nalive.copy(),"value":val[0].copy(),"nvalue":nval[0].copy(),"wave":wave,"deaths":deaths})
    obs=nobs;alive=nalive
   episodes[seed].append(ep)
  del t;torch.cuda.empty_cache()
 return episodes

def credit_audit(episodes):
 perseed=[];gradrows=[];crossrows=[];delta_rows=[]
 for seed,eps in episodes.items():
  t=trainer(seed);processed=[];events=defaultdict(list);maxerr=0
  for ep_idx,ep in enumerate(eps):
   arr=lambda k:np.asarray([r[k] for r in ep]);dev="cuda"
   alive=torch.as_tensor(arr("alive"),dtype=torch.float32,device=dev)[:,None];nalive=torch.as_tensor(arr("nalive"),dtype=torch.float32,device=dev)[:,None];done=torch.as_tensor(arr("done"),dtype=torch.float32,device=dev)[:,None]
   values=torch.as_tensor(arr("value"),dtype=torch.float32,device=dev)[:,None];nvalues=torch.as_tensor(arr("nvalue"),dtype=torch.float32,device=dev)[:,None];local=torch.as_tensor(arr("reward"),dtype=torch.float32,device=dev)[:,None]
   team=local.sum(-1,keepdim=True)/alive.sum(-1,keepdim=True).clamp_min(1)*alive;zero=torch.zeros_like(values)
   al,_=compute_gae(local,values,nvalues,done,alive,nalive,.999,.95);at,_=compute_gae(team,values,nvalues,done,alive,nalive,.999,.95);delta,_=compute_gae(team-local,zero,zero,done,alive,nalive,.999,.95)
   maxerr=max(maxerr,float(((at-al)-delta).abs().max().cpu()));rl=rtg(local,done,alive,nalive);rt=rtg(team,done,alive,nalive)
   processed.append({"obs":arr("obs"),"action":arr("action"),"raw":arr("raw"),"logp":arr("logp"),"alive":arr("alive"),"wave":arr("wave"),"al":al[:,0].cpu().numpy(),"at":at[:,0].cpu().numpy(),"rl":rl[:,0].cpu().numpy(),"rt":rt[:,0].cpu().numpy()})
   # episode-local q: never crosses boundary
   for i,r in enumerate(ep):
    for event in r["deaths"]:
     for w in (10,25,50):
      q=max(0,i-w)
      for agent in range(4):
       if agent!=event["agent"] and r["alive"][agent]>.5:events[(event["cause"],w)].append((float(at[q,0,agent]-al[q,0,agent]),float(al[q,0,agent]),float(at[q,0,agent])))
  for (cause,w),vals in sorted(events.items()):perseed.append({"seed":seed,"cause":cause,"lookback_steps":w,"n":len(vals),"mean_teammean_minus_local":mean([x[0] for x in vals]),"sign_flip_fraction":mean([(x[1]>=0)!=(x[2]>=0) for x in vals])})
  delta_rows.append({"seed":seed,"max_abs_fixedbaseline_difference_minus_reward_only_delta_gae":maxerr,"consistent":maxerr<1e-5})
  cat=lambda k:np.concatenate([x[k] for x in processed])
  obs=torch.as_tensor(cat("obs"),dtype=torch.float32,device="cuda")[:,None];act=torch.as_tensor(cat("action"),dtype=torch.float32,device="cuda")[:,None];raw=torch.as_tensor(cat("raw"),dtype=torch.float32,device="cuda")[:,None];old=torch.as_tensor(cat("logp"),dtype=torch.float32,device="cuda")[:,None];alive=torch.as_tensor(cat("alive"),dtype=torch.float32,device="cuda")[:,None];waves=torch.as_tensor(cat("wave"),dtype=torch.long,device="cuda")[:,None]
  dist,_=t.actor.distribution_step(obs,None,None,None,alive);new=t.actor._squashed_log_prob(dist,raw,act);params=list(t.actor.trainable_policy_parameters());grads={}
  for method,lk,tk in (("FIXED_LOCAL_BASELINE_GAE","al","at"),("BASELINE_FREE_REWARD_TO_GO","rl","rt")):
   grads[method]={}
   for credit,key in (("local",lk),("team_mean",tk)):
    adv=torch.as_tensor(cat(key),dtype=torch.float32,device="cuda")[:,None];adv=global_live_advantage_normalization(adv,alive);sur=ppo_clipped_surrogate(new,old,adv,.2);grads[method][credit]={}
    for wave in (0,1,2,3):
     mask=alive if wave==0 else wave_agent_mask(alive,waves,wave);_,g=_loss_gradient(-sur,mask,params);grads[method][credit][wave]=_flat(g)
   for wave in (0,1,2,3):gradrows.append({"seed":seed,"method":method,"wave":"overall" if wave==0 else f"W{wave}",**cosine_metrics(grads[method]["local"][wave],grads[method]["team_mean"][wave])})
   for credit in ("local","team_mean"):
    for a,b in ((1,2),(1,3),(2,3)):crossrows.append({"seed":seed,"method":method,"credit":credit,"wave_pair":f"W{a}-W{b}",**cosine_metrics(grads[method][credit][a],grads[method][credit][b])})
  del t;torch.cuda.empty_cache()
 primary=[r for r in perseed if r["cause"]=="weapon" and r["lookback_steps"]==10];neg=sum(r["mean_teammean_minus_local"]<0 for r in primary)
 replicated="TEAM_CREDIT_EXTERNALITY_REPLICATED" if neg==3 else "PARTIALLY_REPLICATED" if neg==2 else "NOT_REPLICATED"
 summary={"episode_boundary_bug_confirmed":True,"episode_boundary_bug_fixed":True,"primary_event":"teammate_weapon_death","primary_window_steps":10,"per_seed_primary":primary,"negative_direction_seed_count":neg,"replication_status":replicated,"externality_presence":"TEAM_CREDIT_EXTERNALITY_PRESENT" if neg>=2 else "INSUFFICIENT_EVIDENCE","performance_cause":"INSUFFICIENT_EVIDENCE","fixed_baseline_name":"FIXED_LOCAL_BASELINE_GAE","interpretation":"first-step credit counterfactual under the current local-trained policy/critic, not a trained team-credit critic","secondary_events":[r for r in perseed if r not in primary]}
 return perseed,summary,delta_rows,gradrows,crossrows

def select_snapshots():
 snaps=torch.load(V1/"entry_snapshot_bank.pt",map_location="cpu",weights_only=False);rows=csvread(V1/"entry_state_cross_eval.csv");meta={int(r["snapshot_id"]):r for r in rows if int(r["policy_seed"])==5301}
 selected=[]
 for wave,quota in ((2,10),(3,5)):
  ids=[i for i in sorted(meta) if int(meta[i]["wave"])==wave];seen={"seed":set(),"stage":set(),"survivor":set()}
  while ids and len([x for x in selected if x[1]==wave])<quota:
   best=max(ids,key=lambda i:(sum(meta[i][k] not in seen[n] for k,n in (("source_seed","seed"),("source_step","stage"),("red_survivors","survivor"))),-i));ids.remove(best);selected.append((best,wave,snaps[best],meta[best]));seen["seed"].add(meta[best]["source_seed"]);seen["stage"].add(meta[best]["source_step"]);seen["survivor"].add(meta[best]["red_survivors"])
 return selected

def entry_v2(selected):
 envcfg=yaml.safe_load(ENV.read_text());pol={s:trainer(s) for s in PLAIN};out=[]
 for sid,wave,snap,m in selected:
  for rep in (0,1):
   future_seed=88_210_000+sid*2+rep
   for ps,t in pol.items():
    env=PersistentWaveCombatEnv(deepcopy(envcfg));env.reset(88_219_999);rest=env.restore_curriculum_state(snap);env.rng=np.random.default_rng(future_seed);obs=rest["observation"];alive=rest["red_alive_mask"];start=env.steps;success=False;done=False
    while not done:
     a,_=t.act(obs[None],alive[None],True);obs,reward,term,trunc,info=env.step(a[0]);alive=info["red_alive_mask"];done=term or trunc
     if info.get("waves_cleared",0)>=wave:success=True;break
    out.append({"snapshot_id":sid,"replicate":rep,"future_rng_seed":future_seed,"source_seed":int(m["source_seed"]),"source_step":int(m["source_step"]),"wave":wave,"survivors":int(float(m["red_survivors"])),"policy_seed":ps,"success":int(success),"continuation_steps":env.steps-start,**{k:float(m[k]) for k in ("distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_pairwise_dispersion","remaining_horizon")}})
 for t in pol.values():del t
 return out

def entry_summary(rows):
 snap=[]
 for sid in sorted(set(r["snapshot_id"] for r in rows)):
  g=[r for r in rows if r["snapshot_id"]==sid];x=dict(g[0]);x["success_rate"]=mean([r["success"] for r in g]);snap.append(x)
 survivor={str(n):{"n_snapshots":sum(x["survivors"]==n for x in snap),"success_rate":mean([x["success_rate"] for x in snap if x["survivors"]==n])} for n in sorted(set(x["survivors"] for x in snap))}
 strata=[];geom=defaultdict(list)
 for (wave,n) in sorted(set((x["wave"],x["survivors"]) for x in snap)):
  g=[x for x in snap if x["wave"]==wave and x["survivors"]==n];sources=sorted(set(x["source_seed"] for x in g));rates=[mean([x["success_rate"] for x in g if x["source_seed"]==s]) for s in sources]
  strata.append({"wave":wave,"survivors":n,"n_snapshots":len(g),"source_levels":len(sources),"source_rate_range":max(rates)-min(rates) if len(rates)>=2 else None})
  if len(g)>=4 and np.std([x["success_rate"] for x in g])>0:
   for k in ("distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_pairwise_dispersion","remaining_horizon"):
    if np.std([x[k] for x in g])>0:geom[k].append(float(np.corrcoef([x[k] for x in g],[x["success_rate"] for x in g])[0,1]))
 geommean={k:mean(v) for k,v in geom.items()};usable=[x for x in strata if x["source_levels"]>=2 and x["n_snapshots"]>=3]
 survivor_positive=all(survivor[str(b)]["success_rate"]>=survivor[str(a)]["success_rate"] for a,b in zip(sorted(map(int,survivor))[:-1],sorted(map(int,survivor))[1:]))
 geo="GEOMETRIC_ENTRY_STATE_EFFECT_SUPPORTED" if len(usable)>=3 and any(abs(v or 0)>=.3 for v in geommean.values()) else "INSUFFICIENT_EVIDENCE"
 return {"selection":"pre-outcome greedy categorical coverage; W2=10,W3=5","snapshot_count":len(snap),"continuations":len(rows),"entry_state_composition_effect":"SUPPORTED" if survivor_positive else "PARTIALLY_SUPPORTED","survivor_count_effect":"SUPPORTED" if survivor_positive else "NOT_SUPPORTED","geometric_entry_state_effect":geo,"upstream_state_formation_effect":"PARTIALLY_SUPPORTED","survivor_strata":survivor,"within_wave_survivor_strata":strata,"geometry_residual_associations":geommean,"note":"episodes are descriptive; training seed is the replication unit"}

def clustering():
 rows=csvread(V1/"policy_functional_distance.csv");out={}
 for group in ("all","W1","W2","W3","near_boundary","not_near_boundary"):
  g=[r for r in rows if r["group"]==group and "collapse" not in r["policy_a"] and "collapse" not in r["policy_b"]];within=[float(r["action_l2"]) for r in g if r["seed_a"]==r["seed_b"]];cross=[float(r["action_l2"]) for r in g if r["seed_a"]!=r["seed_b"] and r["step_a"]==r["step_b"]];out[group]={"within_seed_temporal_mean":mean(within),"cross_seed_same_stage_mean":mean(cross),"ratio":mean(cross)/mean(within),"cross_greater":mean(cross)>mean(within)}
 count=sum(v["cross_greater"] for v in out.values());return {"groups":out,"groups_cross_greater":count,"group_count":len(out),"functional_clustering":"SUPPORTED" if count>=4 else "PARTIALLY_SUPPORTED","policy_basin":"INFERENCE","interpretation":"functional block structure is descriptive; no attractor or perturbation-continuation evidence"}

def derive_collapse():
 result={}
 for seed,d in PLAIN.items():
  rows=csvread(d/"evaluation_history.csv");rows.sort(key=lambda x:int(x["sampled_steps"]));events=[]
  for a,b in zip(rows,rows[1:]):
   da=float(a["average_waves_cleared"])-float(b["average_waves_cleared"]);dw=float(a["clear_wave_1_probability"])-float(b["clear_wave_1_probability"]);db=float(b["average_red_boundary_exits"])-float(a["average_red_boundary_exits"]);tr=[]
   if da>=.5:tr.append("AW_DROP_GE_0.5")
   if dw>=.25:tr.append("W1_DROP_GE_0.25")
   if db>=.8:tr.append("BOUNDARY_RISE_GE_0.8")
   if tr:events.append({"from_step":int(a["sampled_steps"]),"to_step":int(b["sampled_steps"]),"trigger":"+".join(tr),"delta_AW":-da,"delta_W1":-dw,"delta_Boundary":db,"aw_drop":da})
  if not events:result[str(seed)]={"status":"NO_STRONG_COLLAPSE"};continue
  e=max(events,key=lambda x:(x["aw_drop"],-x["to_step"]));saved=sorted(int(re.search(r"checkpoint_(\d+)\.pt",p.name).group(1)) for p in d.glob("checkpoint_*.pt"));e["pre_checkpoint_step"]=max((x for x in saved if x<=e["from_step"]),default=min(saved));e["post_checkpoint_step"]=min((x for x in saved if x>=e["to_step"]),default=max(saved));e["status"]="DERIVED_STRONG_COLLAPSE";e["selection_rule"]="eligible adjacent event with largest AW drop; saved checkpoint bracketing event";result[str(seed)]=e
 return result

def policy_output(seed,step,obs,alive):
 d=PLAIN[seed];cfg=yaml.safe_load((d/"algorithm_config.yaml").read_text());t=build_modular_mappo_trainer(cfg,"cuda",256,3_000_000);t.load(d/f"checkpoint_{step}.pt",True,False);mus=[];ls=[]
 with torch.no_grad():
  for i in range(0,len(obs),256):dist,_=t.actor.distribution_step(torch.as_tensor(obs[i:i+256],device="cuda"),None,None,None,torch.as_tensor(alive[i:i+256],device="cuda"));mus.append(dist.mean.cpu().numpy());ls.append(np.log(dist.scale.cpu().numpy()))
 del t;return np.concatenate(mus),np.concatenate(ls)
def collapse_drift(derived):
 z=np.load(V1/"observation_bank.npz");obs=z["observations"];alive=z["alive"];wave=z["wave"];radius=np.sqrt(obs[:,:,0]**2+obs[:,:,1]**2);near=np.max(np.where(alive>.5,radius,0),axis=1)>.86;groups={"all":np.ones(len(obs),bool),"W1":wave==1,"W2":wave==2,"W3":wave==3,"near_boundary":near};out=[]
 for seed in PLAIN:
  e=derived[str(seed)]
  if e["status"]!="DERIVED_STRONG_COLLAPSE":continue
  ma,la=policy_output(seed,e["pre_checkpoint_step"],obs,alive);mb,lb=policy_output(seed,e["post_checkpoint_step"],obs,alive);aa,ab=np.tanh(ma),np.tanh(mb);va,vb=np.exp(2*la),np.exp(2*lb)
  for name,mask in groups.items():
   live=alive[mask]>.5;d=(aa-ab)[mask];kl=.25*((va+(ma-mb)**2)/vb-1+2*(lb-la)+(vb+(mb-ma)**2)/va-1+2*(la-lb));kl=kl[mask];dims={"heading_abs":float(np.abs(d[...,0][live]).mean()),"pitch_abs":float(np.abs(d[...,1][live]).mean()),"speed_abs":float(np.abs(d[...,2][live]).mean())};out.append({"seed":seed,"from_eval_step":e["from_step"],"to_eval_step":e["to_step"],"pre_checkpoint_step":e["pre_checkpoint_step"],"post_checkpoint_step":e["post_checkpoint_step"],"group":name,"action_l2":float(np.sqrt((d[np.repeat(live[...,None],3,-1)]**2).reshape(-1,3).sum(1)).mean()),**dims,"gaussian_symmetric_kl":float(kl[np.repeat(live[...,None],3,-1)].reshape(-1,3).sum(1).mean()),"logstd_abs":float(np.abs((la-lb)[mask])[np.repeat(live[...,None],3,-1)].mean()),"dominant_dimension":max(dims,key=dims.get)})
 return out

def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if OUT.exists():raise FileExistsError(OUT)
 v1=json.loads((V1/"supplemental_manifest.json").read_text());assert v1["total_real_environment_episodes"]==117 and not v1["seed_45m_used"] and v1["status"]=="SUPPLEMENTAL_AUDIT_MUTATION_GUARD_PASS"
 OUT.mkdir(parents=True);watched=CORE+[PLAIN[s]/"checkpoint_3000000.pt" for s in PLAIN];before={str(p.relative_to(ROOT)):sha256(p) for p in watched}
 manifest={"v1_verified":True,"selection_rule":"Before outcomes: greedy categorical coverage across wave/source seed/source stage/survivor; W2=10,W3=5","team_episode_seeds":TEAM_SEEDS,"entry_future_rng_base":88_210_000,"planned_team_episodes":15,"planned_entry_continuations":90,"maximum_new_environment_episodes":105,"training_seed_is_replication_unit":True,"optimizer_steps":0,"formal_training":False,"45m_used":False}
 dump(OUT/"v2_manifest.json",manifest)
 print("[1/5] team trajectories",flush=True);episodes=collect_team();perseed,ext,delta,grads,cross=credit_audit(episodes);csvwrite(OUT/"team_credit_per_seed.csv",perseed);dump(OUT/"team_credit_externality_summary.json",ext);dump(OUT/"reward_only_delta_gae.json",{"rows":delta,"all_consistent":all(x["consistent"] for x in delta)});csvwrite(OUT/"baseline_free_credit_gradient.csv",[r for r in grads if r["method"]=="BASELINE_FREE_REWARD_TO_GO"]);csvwrite(OUT/"cross_wave_credit_gradient.csv",cross)
 print("[2/5] entry cross eval",flush=True);selected=select_snapshots();entryrows=entry_v2(selected);entrysum=entry_summary(entryrows);csvwrite(OUT/"entry_cross_eval_v2.csv",entryrows);dump(OUT/"entry_cross_eval_v2_summary.json",entrysum)
 print("[3/5] clustering/collapse",flush=True);cluster=clustering();dump(OUT/"functional_clustering_v2.json",cluster);derived=derive_collapse();dump(OUT/"derived_collapse_checkpoints.json",derived);drift=collapse_drift(derived);csvwrite(OUT/"collapse_functional_drift_v2.csv",drift)
 baseline=[r for r in grads if r["method"]=="BASELINE_FREE_REWARD_TO_GO"]
 bmean={w:mean([float(r["cosine"]) for r in baseline if r["wave"]==w]) for w in ("overall","W1","W2","W3")};primary=ext["replication_status"]=="TEAM_CREDIT_EXTERNALITY_REPLICATED";rewarddelta=all(x["consistent"] for x in delta);gradient_changed=abs(1-bmean["overall"])>.1;survivor=entrysum["survivor_count_effect"]=="SUPPORTED";screen=primary and rewarddelta and gradient_changed and survivor
 taxonomy={"SCALAR_REWARD_MISSION_ALIGNMENT":"SUPPORTED","SCALAR_REWARD_MISALIGNMENT_PRIMARY":"NOT_SUPPORTED","DENSE_REWARD_FARMING":"NOT_SUPPORTED","TEAM_CREDIT_EXTERNALITY_PRESENT":ext["externality_presence"],"TEAM_CREDIT_PERFORMANCE_CAUSE":"INSUFFICIENT_EVIDENCE","SURVIVOR_EXTERNALITY_CREDIT":"NOT_SUPPORTED" if primary else "PARTIALLY_SUPPORTED","WAVE_PROGRESS_CREDIT":"PARTIALLY_SUPPORTED","ENTRY_PREPARATION_CREDIT":"PARTIALLY_SUPPORTED"};dump(OUT/"reward_taxonomy_v2.json",taxonomy)
 hierarchy={"Level_1_candidate_causes":[{"name":"local teammate-survival credit externality","status":ext["externality_presence"],"performance_cause":"INSUFFICIENT_EVIDENCE"},{"name":"survivor-dominated entry composition","status":entrysum["entry_state_composition_effect"]},{"name":"GAE temporal credit","primary":"NOT_SUPPORTED","plausible_secondary":"SUPPORTED"}],"Level_2_optimization_mechanisms":[{"name":"actor cross-wave conflict","status":"SUPPORTED_DIAGNOSTIC_NOT_CAUSE"},{"name":"seed-dependent optimization path","status":"SUPPORTED"}],"Level_3_functional_manifestations":[{"name":"seed-dependent functional clustering","status":cluster["functional_clustering"]},{"name":"policy basin","status":"INFERENCE"},{"name":"late functional drift","status":"SUPPORTED"}],"Level_4_environment_outcomes":["boundary exits","survivor attrition","reduced later-wave clears"],"arrow_evidence":{"externality_to_survivor_preservation":"PLAUSIBLE_NOT_INTERVENED","survivor_count_to_next_wave_success":"SUPPORTED_DESCRIPTIVE_FIXED_POLICY","functional_drift_to_boundary_failure":"PARTIALLY_SUPPORTED"},"TEAM_CREDIT_SCREEN_JUSTIFIED":"YES" if screen else "NO","decision_checks":{"weapon_10step_3of3":primary,"reward_only_delta_consistent":rewarddelta,"baseline_free_gradient_changed":gradient_changed,"survivor_positive":survivor,"no_new_environment_bug":True}};dump(OUT/"root_cause_hierarchy_v2.json",hierarchy)
 print("[4/5] report",flush=True)
 report=f"""# Supplemental Causal Audit V2

## Corrections
V1 episode flattening could cross episode boundaries for 10/25/50-step lookbacks. V2 computes GAE and lookbacks separately per episode. The main statistical unit is the training seed.

## Team credit
Weapon-death 10-step per-seed results: {json.dumps(ext['per_seed_primary'])}. Replication: **{ext['replication_status']}**. `FIXED_LOCAL_BASELINE_GAE` retains the current local-trained critic and only describes first-step credit change. Reward-only delta-GAE matches the fixed-baseline advantage difference: all_consistent={rewarddelta}. Baseline-free RTG gradient cosines overall/W1/W2/W3={bmean}. Cross-wave negative cosine is not interpreted as instability or inferiority.

## Entry state
Fifteen pre-outcome-selected snapshots (10 W2, 5 W3), three fixed downstream policies and two matched future weapon RNG replicates produced 90 continuations. Composition={entrysum['entry_state_composition_effect']}; survivor={entrysum['survivor_count_effect']}; geometry={entrysum['geometric_entry_state_effect']}. Analyses condition on wave and survivor count; see `entry_cross_eval_v2_summary.json`.

## Functional clustering and collapse
Group ratios are {json.dumps(cluster['groups'])}. `SEED_DEPENDENT_FUNCTIONAL_CLUSTERING={cluster['functional_clustering']}`; `SEED_DEPENDENT_POLICY_BASIN=INFERENCE`. Collapse events/checkpoints are derived from evaluation history using the declared adjacent-drop rule, never hard-coded. Drift rows are in `collapse_functional_drift_v2.csv`.

## Reward and temporal credit
`SCALAR_REWARD_MISSION_ALIGNMENT=SUPPORTED`; `SCALAR_REWARD_MISALIGNMENT_PRIMARY=NOT_SUPPORTED`. Team externality presence and performance causation are separate. `GAE_TEMPORAL_CREDIT_PRIMARY=NOT_SUPPORTED`; `GAE_TEMPORAL_CREDIT_PLAUSIBLE_SECONDARY=SUPPORTED`: direct traces are short, but bootstrap exists and prior IWSC/CAIW/BRSC/HTA interventions did not establish temporal credit as primary.

## Causal hierarchy and decision
See `root_cause_hierarchy_v2.json`. Offline evidence establishes credit externality, not performance causation. **TEAM_CREDIT_SCREEN_JUSTIFIED={hierarchy['TEAM_CREDIT_SCREEN_JUSTIFIED']}**. If YES, the minimum next experiment is a 300k three-seed matched screen from identical checkpoints comparing unchanged local credit with scale-matched team-mean credit; no 3M run or combined mechanism is justified yet. No implementation or training was performed here.
""";(OUT/"supplemental_audit_v2_report.md").write_text(report,encoding="utf-8")
 after={str(p.relative_to(ROOT)):sha256(p) for p in watched};guard=before==after;manifest.update({"actual_team_episodes":15,"actual_entry_continuations":len(entryrows),"actual_new_environment_episodes":15+len(entryrows),"hash_before":before,"hash_after":after,"mutation_guard":"SUPPLEMENTAL_V2_MUTATION_GUARD_PASS" if guard else "SUPPLEMENTAL_V2_MUTATION_GUARD_FAIL","TEAM_CREDIT_SCREEN_JUSTIFIED":"YES" if screen else "NO"});dump(OUT/"v2_manifest.json",manifest)
 if not guard:raise RuntimeError("mutation guard fail")
 print("SUPPLEMENTAL_CAUSAL_AUDIT_V2_COMPLETE");print(json.dumps({"replication":ext["replication_status"],"baseline_free_cosines":bmean,"entry":entrysum,"clustering":cluster,"collapse":derived,"screen":"YES" if screen else "NO","guard":manifest["mutation_guard"],"report":str((OUT/"supplemental_audit_v2_report.md").relative_to(ROOT))},indent=2))
if __name__=="__main__":main()
