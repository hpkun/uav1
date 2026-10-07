"""CUDA trajectory collector for the clean MAPPO single-wave stability audit."""
from __future__ import annotations
import argparse, copy, json, math, sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.factory import make_combat_environment
from env.geometry import engagement_geometry
from tools.single_wave_stability_common import (CROSS_HORIZON_BASE,DIAGNOSTIC_BASE,action_saturation,agent_attempt_transitions,boundary_descriptor,checkpoint_specs,counterfactual_score,cross_horizon_config,deterministic_policy_data,heading_relative_to_outward,load_clean_mappo,precursor_indices,radial_velocity,refuse_existing_output,strict_checkpoint_identity,symmetric_cone_components,validate_seed_bank,weapon_hit_probability,write_csv)
LAGS=(50,20,10,5,1);DIMS=("heading","pitch","speed")

def nearest_geometry(env,agent):
 own=env.red[agent];targets=[x for x in env.blue if x.alive]
 if not own.alive or not targets:return {"nearest_blue_distance":None,"nearest_blue_off_boresight":None,"nearest_blue_ATA":None,"nearest_blue_AA":None,"closing_velocity":None,"fire_window":False}
 target=min(targets,key=lambda x:engagement_geometry(own,x).distance);g=engagement_geometry(own,target)
 rel=np.asarray([target.x-own.x,target.y-own.y,target.z-own.z]);relv=target.velocity_vector()-own.velocity_vector()
 return {"nearest_blue_distance":g.distance,"nearest_blue_off_boresight":g.off_boresight,"nearest_blue_ATA":g.ata,"nearest_blue_AA":g.aa,"closing_velocity":-float(np.dot(rel,relv)/max(g.distance,1e-12)),"fire_window":bool(env.weapon.in_fire_window(g))}

def snapshot(env,obs,action,raw,log_std,last_attempt_steps,last_any_attempt):
 out=[]
 for i,own in enumerate(env.red):
  vel=own.velocity_vector();r=math.hypot(own.x,own.y)
  out.append({"agent":i,"x":own.x,"y":own.y,"radius":r,"boundary_margin":env.arena_radius-r,"radial_velocity":radial_velocity(own.x,own.y,vel[0],vel[1]),"altitude":own.altitude,"speed":own.v,"heading":own.psi,"pitch":own.theta,"heading_relative_to_outward":heading_relative_to_outward(own.x,own.y,own.psi),**nearest_geometry(env,i),"weapon_range_max":float(env.weapon.range_max),"fire_ready":bool(env.red_fire_states[i].armed),**{f"action_{d}":float(action[i,j]) for j,d in enumerate(DIMS)},**{f"raw_mean_{d}":float(raw[i,j]) for j,d in enumerate(DIMS)},**{f"log_std_{d}":float(log_std[i,j]) for j,d in enumerate(DIMS)},"steps_since_own_red_attempt":1_000_000 if last_attempt_steps[i] is None else env.steps-last_attempt_steps[i],"steps_since_any_red_attempt":1_000_000 if last_any_attempt is None else env.steps-last_any_attempt,"observation":obs[i].astype(np.float32).copy()})
 return out

def evaluate_episode(trainer,env_config,episode_seed,capture=True):
 env=make_combat_environment(env_config);obs,_=env.reset(int(episode_seed));returns=np.zeros(4);history=deque(maxlen=max(LAGS)+1);precursors=[];actions_all=[];state_records=[];first_boundary=None;last_attempt_steps=[None]*4;last_any_attempt=None;last_kill=None
 while True:
  alive=env.red_alive_mask.astype(np.float32);action,raw,log_std=deterministic_policy_data(trainer,obs,alive)
  if capture:
   actions_all.append(action[alive>.5].copy());state_records.extend({"step":env.steps,"agent":int(i),"observation":obs[i].copy(),"action":action[i].copy()} for i in np.flatnonzero(alive>.5))
  before=np.asarray([x.alive for x in env.red]);armed_before=np.asarray([x.armed for x in env.red_fire_states]);history.append({"step":env.steps,"agents":snapshot(env,obs,action,raw,log_std,last_attempt_steps,last_any_attempt)})
  obs,reward,terminated,truncated,info=env.step(action);returns+=reward
  armed_after=np.asarray([x.armed for x in env.red_fire_states]);attempted=agent_attempt_transitions(armed_before,armed_after,before)
  for i in np.flatnonzero(attempted):last_attempt_steps[int(i)]=env.steps
  if bool(attempted.any()):last_any_attempt=env.steps
  if info.get("red_step_attack_kills",0):last_kill=env.steps
  after=np.asarray([x.alive for x in env.red])
  for agent in np.flatnonzero(before&~after):
   if math.hypot(env.red[agent].x,env.red[agent].y)<=env.arena_radius:continue
   first_boundary=env.steps if first_boundary is None else first_boundary;event=[]
   for lag,source_step in precursor_indices(env.steps,[x["step"] for x in history],LAGS).items():
    source=next(x for x in history if x["step"]==source_step);snap=source["agents"][agent]
    event.append({"episode_seed":episode_seed,"agent":int(agent),"boundary_step":env.steps,"source_step":source_step,"lag_steps":lag,**{k:v for k,v in snap.items() if k!="observation"},"_observation":snap["observation"]})
   label=boundary_descriptor(event,env.steps,last_kill)
   for row in event:row.update({"boundary_descriptor":label,"descriptor_warning":"trajectory descriptor/candidate; not an inference of policy intent"})
   precursors.extend(event)
  if terminated or truncated:break
 outcome={"episode_seed":int(episode_seed),"red_success":bool(info["red_success"]),"termination_reason":info["termination_reason"],"episode_length":int(info["episode_length"]),"episode_return":float(returns.sum()),"mean_agent_return":float(returns.mean()),**{k:int(info[k]) for k in ("red_losses","blue_losses","red_attack_kills","blue_attack_kills","red_boundary_exits","blue_boundary_exits","red_ground_losses","blue_ground_losses")},**{f"R{i}":float(info[f"episode_r{i}_total"]) for i in range(1,5)},"first_fire_window_step":info.get("red_first_fire_window_step"),"first_attempt_step":info.get("red_first_attempt_step"),"first_hit_step":info.get("red_first_hit_step"),"first_kill_step":info.get("red_first_kill_step"),"first_boundary_exit_step":first_boundary,**{k:info.get(k) for k in ("blue_ground_guard_decision_steps","blue_ground_guard_override_steps","blue_ground_guard_activations","blue_ground_guard_activation_ratio","blue_ground_guard_max_duration_steps")}}
 boundary_keys={(int(r["source_step"]),int(r["agent"])) for r in precursors}
 non_boundary_records=[r for r in state_records if (r["step"],r["agent"]) not in boundary_keys]
 return outcome,precursors if capture else [],actions_all if capture else [],[r["action"] for r in non_boundary_records] if capture else [],[r["observation"] for r in non_boundary_records] if capture else []

def reservoir(bank,items,limit,rng,seen):
 for item in items:
  seen[0]+=1
  if len(bank)<limit:bank.append(np.asarray(item,np.float32))
  else:
   j=int(rng.integers(0,seen[0]))
   if j<limit:bank[j]=np.asarray(item,np.float32)

def policy_drift(specs,banks):
 rows=[];pairs=defaultdict(dict)
 for spec in specs:pairs[spec["training_seed"]][spec["checkpoint_role"]]=spec
 for seed,pair in pairs.items():
  best,_=load_clean_mappo(pair["best"]);final,_=load_clean_mappo(pair["final"])
  for source in ("best","final"):
   for cls in ("non_boundary","boundary_precursor"):
    data=banks[(seed,source,cls)]
    if not data:continue
    obs=torch.as_tensor(np.asarray(data),dtype=torch.float32,device="cuda")
    with torch.no_grad():bd,fd=best.actor.distribution(obs),final.actor.distribution(obs);bm,fm=bd.mean,fd.mean;ba,fa=torch.tanh(bm),torch.tanh(fm);bl,fl=torch.log(bd.scale),torch.log(fd.scale)
    row={"training_seed":seed,"state_source":source,"state_class":cls,"n_states":len(data)}
    for name,(left,right) in {"raw_mean":(bm,fm),"deterministic_action":(ba,fa),"log_std":(bl,fl)}.items():
     diff=right-left;row[f"{name}_l2"]=float(torch.linalg.vector_norm(diff,dim=-1).mean())
     for i,dim in enumerate(DIMS):row[f"{name}_{dim}_mean_abs_delta"]=float(diff[:,i].abs().mean())
    rows.append(row)
  del best,final;torch.cuda.empty_cache()
 return rows

def weapon_table(cfg):
 w=cfg["weapon"];rows=[]
 for distance in (500,1000,2000,3000,3900):
  for deg in (0,10,20,30):
   angle=math.radians(deg);sym=symmetric_cone_components(angle)
   for orientation,(az,el) in {"azimuth_edge":(angle,0.),"elevation_edge":(0.,angle),"symmetric_3d_cone":sym}.items():rows.append({"distance":distance,"off_boresight_degrees":deg,"orientation":orientation,"azimuth_error":az,"elevation_error":el,"hit_probability":weapon_hit_probability(distance,az,el,w["effective_hit_distance"],w["attack_noise_scale"],w["height_noise_scale"])})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--episodes",type=int,default=50);p.add_argument("--seed-base",type=int,default=DIAGNOSTIC_BASE);p.add_argument("--cross-horizon-seed-base",type=int,default=CROSS_HORIZON_BASE);p.add_argument("--output-dir",default="outputs/single_wave_checkpoint_trajectory_audit");p.add_argument("--state-bank-per-source",type=int,default=512);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory")
 seeds=validate_seed_bank(range(a.seed_base,a.seed_base+a.episodes));hseeds=validate_seed_bank(range(a.cross_horizon_seed_base,a.cross_horizon_seed_base+a.episodes))
 if set(seeds)&set(hseeds):raise RuntimeError("H3000/H1000 seed banks overlap")
 output=ROOT/a.output_dir
 # Validate every identity before creating any formal output directory.
 prevalidated=[strict_checkpoint_identity(spec) for spec in checkpoint_specs()]
 refuse_existing_output(output)
 output.mkdir(parents=True);specs=checkpoint_specs();outcomes=[];precursors=[];cross=[];saturation=[];counter=[];banks=defaultdict(list);seen=defaultdict(lambda:[0]);rng=np.random.default_rng(89_299_991);identities=[]
 for spec in specs:
  trainer,identity=load_clean_mappo(spec);identities.append({k:v for k,v in identity.items() if not k.endswith("config")});ident={k:identity[k] for k in ("training_seed","checkpoint_role","checkpoint_step","checkpoint_sha256")};action_chunks=[];nonboundary_chunks=[]
  for epseed in seeds:
   outcome,event_rows,acts,nonboundary_acts,states=evaluate_episode(trainer,identity["environment_config"],epseed);outcomes.append({**ident,**outcome});action_chunks.extend(acts);nonboundary_chunks.extend(nonboundary_acts);bobs=[]
   for row in event_rows:bobs.append(row.pop("_observation"));precursors.append({**ident,**row})
   reservoir(banks[(spec["training_seed"],spec["checkpoint_role"],"non_boundary")],states,a.state_bank_per_source,rng,seen[(spec["training_seed"],spec["checkpoint_role"],"non_boundary")]);reservoir(banks[(spec["training_seed"],spec["checkpoint_role"],"boundary_precursor")],bobs,a.state_bank_per_source,rng,seen[(spec["training_seed"],spec["checkpoint_role"],"boundary_precursor")])
   for bp in (-10,-20,-30):
    for tp in (0,-10,-20):
     for wb in (0,10,20):
      for lp in (0,-10,-20):
       score=counterfactual_score(outcome,bp,tp,wb,lp)
       counter.append({**ident,"episode_seed":epseed,"boundary_penalty":bp,"timeout_penalty":tp,"win_bonus":wb,"mission_loss_penalty":lp,"fixed_trajectory_score":score,"warning":"fixed-trajectory rescoring only; does not predict retrained policy behavior"})
  allacts=np.concatenate(action_chunks) if action_chunks else np.empty((0,3));nonacts=np.asarray(nonboundary_chunks);ownpre=[r for r in precursors if r["training_seed"]==spec["training_seed"] and r["checkpoint_role"]==spec["checkpoint_role"]];preacts=np.asarray([[r[f"action_{d}"] for d in DIMS] for r in ownpre])
  for scope,vals in (("all_alive_states",allacts),("non_boundary_states",nonacts),("boundary_precursors",preacts)):
   for threshold in (.9,.99):saturation.append({**ident,"scope":scope,"threshold":threshold,"n_agent_states":len(vals),**action_saturation(vals,threshold)})
  h1000=cross_horizon_config(identity["environment_config"])
  for epseed in hseeds:
   for horizon,cfg in ((3000,identity["environment_config"]),(1000,h1000)):
    outcome,_,_,_,_=evaluate_episode(trainer,cfg,epseed,False);cross.append({**ident,**outcome,"evaluation_label":"CROSS_HORIZON_POLICY_EVALUATION","training_max_steps":3000,"evaluation_max_steps":horizon,"paired_environment_seed":epseed})
  del trainer;torch.cuda.empty_cache()
 summary=[]
 for spec in specs:
  sub=[r for r in outcomes if r["training_seed"]==spec["training_seed"] and r["checkpoint_role"]==spec["checkpoint_role"]];events={(r["episode_seed"],r["agent"],r["boundary_step"]):r["boundary_descriptor"] for r in precursors if r["training_seed"]==spec["training_seed"] and r["checkpoint_role"]==spec["checkpoint_role"]};cnt=Counter(events.values());total=sum(cnt.values());nbe=sum(r["red_boundary_exits"]>0 for r in sub)
  labels=(("escape_like","ESCAPE_LIKE_TRAJECTORY_DESCRIPTOR"),("tactical_overshoot","TACTICAL_OVERSHOOT_CANDIDATE"),("recent_kill_overshoot","RECENT_KILL_OVERSHOOT_CANDIDATE"),("unclassified","UNCLASSIFIED_BOUNDARY_EXIT"))
  summary.append({"training_seed":spec["training_seed"],"checkpoint_role":spec["checkpoint_role"],"episodes":len(sub),"episodes_with_boundary_exit":nbe,"boundary_episode_rate":nbe/max(len(sub),1),"total_boundary_exits":sum(r["red_boundary_exits"] for r in sub),**{f"{name}_count":cnt[label] for name,label in labels},**{f"{name}_fraction":cnt[label]/max(total,1) for name,label in labels}})
 drift=policy_drift(specs,banks);tmp,identity=load_clean_mappo(specs[0]);envcfg=identity["environment_config"];del tmp;torch.cuda.empty_cache()
 for name,data in (("episode_outcomes.csv",outcomes),("boundary_precursors.csv",precursors),("boundary_behavior_summary.csv",summary),("action_saturation_summary.csv",saturation),("policy_drift_summary.csv",drift),("offline_counterfactual_reward_scores.csv",counter),("weapon_probability_audit.csv",weapon_table(envcfg)),("cross_horizon_evaluation.csv",cross)):write_csv(output/name,data)
 manifest={"status":"TRAJECTORY_COLLECTION_COMPLETE","deployment":"deterministic_tanh_mean","episodes_per_checkpoint":a.episodes,"diagnostic_seed_base":seeds[0],"diagnostic_seed_end":seeds[-1],"cross_horizon_seed_base":hseeds[0],"cross_horizon_seed_end":hseeds[-1],"formal_48m_seeds_accessed":False,"checkpoint_identities":identities,"boundary_descriptor_rules":{"RECENT_KILL_OVERSHOOT_CANDIDATE":"team Red kill within 100 steps and current <=20-step precursors no longer satisfy active pursuit; recent-kill descriptor, not post-mission flight","TACTICAL_OVERSHOOT_CANDIDATE":"within <=20-step precursors: fire window, or distance<=weapon range + target in front hemisphere + positive closing velocity","ESCAPE_LIKE_TRAJECTORY_DESCRIPTOR":"at least two <=20-step points: outward velocity, heading within 90deg outward, no fire window, no active pursuit geometry, own-agent attempt older than 20 steps","UNCLASSIFIED_BOUNDARY_EXIT":"none of the above","warning":"not an inference of policy intent"},"cross_horizon_semantics":"3000-trained policy, only max_steps changed to 1000; direct deployment effect, not training-dynamics effect"}
 (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps({"status":manifest["status"],"output":str(output),"episodes":len(outcomes),"cross_horizon_episodes":len(cross)},indent=2))
if __name__=="__main__":main()
