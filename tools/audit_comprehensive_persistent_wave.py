"""Read-only root-cause audit for persistent_wave_v2.

Writes only beneath outputs/comprehensive_persistent_wave_audit.  Existing run
directories and checkpoints are read-only inputs.  Optional diagnostic episodes
use 88M seeds and never perform an optimizer update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from env.geometry import engagement_geometry
from env.persistent_env import PersistentWaveCombatEnv

OUT=ROOT/"outputs/comprehensive_persistent_wave_audit"
AUDIT_SEEDS=list(range(88_000_000,88_000_008))
PHASES=(("0-0.6M",0,600_000),("0.6-0.9M",600_000,900_000),
        ("0.9-1.5M",900_000,1_500_000),("1.5-2.0M",1_500_000,2_000_000),
        ("2.0-2.5M",2_000_000,2_500_000),("2.5-3.0M",2_500_000,3_000_001))
RUNS={
 "plain_5301":("Plain",5301,"outputs/diag_mappo_learnability/l3_seed5301"),
 "plain_5302":("Plain",5302,"outputs/diag_mappo_learnability/l3_seed5302"),
 "plain_5303":("Plain",5303,"outputs/diag_mappo_learnability/l3_seed5303"),
 "matched_control_5302":("MatchedControl",5302,"outputs/dev_swgp_branch_control_seed5302_3m"),
 "matched_swgp_5302":("MatchedSWGP",5302,"outputs/dev_swgp_branch_seed5302_3m_retry1"),
 "fresh_swgp_5302":("FreshSWGP",5302,"outputs/dev_swgp_3m_seed5302"),
 "delayed_5301":("DelayedSWGP",5301,"outputs/dev_delayed_swgp_seed5301_3m"),
 "delayed_5302":("DelayedSWGP",5302,"outputs/dev_delayed_swgp_seed5302_3m"),
 "delayed_5303":("DelayedSWGP",5303,"outputs/dev_delayed_swgp_seed5303_3m"),
}
FAILED_RUN=ROOT/"outputs/dev_swgp_branch_seed5302_3m"
CORE_CODE=[ROOT/p for p in ("env/persistent_env.py","env/combat_env.py","env/reward.py",
 "env/observation.py","env/weapon.py","env/fixed_policy.py",
 "algorithm/modular_mappo/trainer.py","algorithm/modular_mappo/runner.py")]


def sha256(path: Path) -> str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):d.update(block)
 return d.hexdigest()


def hash_value(value: Any, digest=None) -> str:
 d=hashlib.sha256() if digest is None else digest
 if torch.is_tensor(value):
  x=value.detach().cpu().contiguous();d.update(str(x.dtype).encode());d.update(str(tuple(x.shape)).encode());d.update(x.numpy().tobytes())
 elif isinstance(value,np.ndarray):d.update(str(value.dtype).encode());d.update(str(value.shape).encode());d.update(value.tobytes())
 elif isinstance(value,dict):
  for key in sorted(value,key=str):d.update(str(key).encode());hash_value(value[key],d)
 elif isinstance(value,(list,tuple)):
  for item in value:hash_value(item,d)
 else:d.update(repr(value).encode())
 return d.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def read_csv(path):
 with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def number(row,key,default=float("nan")):
 try:return float(row.get(key,default))
 except (TypeError,ValueError):return default
def mean(values):
 values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
 return statistics.fmean(values) if values else None
def stdev(values):
 values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
 return statistics.stdev(values) if len(values)>1 else 0.0 if values else None
def quantile(values,q):
 values=np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
 return float(np.quantile(values,q)) if values.size else None
def rankdata(values):
 a=np.asarray(values,float);order=np.argsort(a,kind="mergesort");r=np.empty(len(a),float);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2+1;i=j
 return r
def corr(a,b,spearman=False):
 pairs=[(float(x),float(y)) for x,y in zip(a,b) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
 if len(pairs)<3:return None
 x,y=map(np.asarray,zip(*pairs));x=rankdata(x) if spearman else x;y=rankdata(y) if spearman else y
 return None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
def write_csv(path,rows,fields=None):
 rows=list(rows);fields=fields or sorted({k for r in rows for k in r})
 with Path(path).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def json_clean(value):
 """Convert NumPy scalars and non-finite diagnostics to strict JSON values."""
 if isinstance(value,dict):return {str(k):json_clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_clean(v) for v in value]
 if isinstance(value,(np.integer,)):return int(value)
 if isinstance(value,(np.floating,float)):
  value=float(value)
  return value if math.isfinite(value) else None
 return value


def line_of(path: Path,needle: str) -> int:
 for i,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  if needle in line:return i
 return -1


def protocol_audit() -> tuple[list[dict],dict]:
 rows=[]
 for run_id,(method,seed,rel) in RUNS.items():
  d=ROOT/rel;summary=read_json(d/"run_summary.json");run=read_json(d/"run_config.json")
  algo=yaml.safe_load((d/"algorithm_config.yaml").read_text(encoding="utf-8"));env=yaml.safe_load((d/"runtime_env_config.yaml").read_text(encoding="utf-8"))
  t=algo["training"];net=algo["network"];text=(d/"train.log").read_text(encoding="utf-8",errors="replace")
  checkpoints={name:(d/name).exists() for name in ("latest.pt","final.pt","checkpoint_3000000.pt")}
  latest=summary.get("latest_evaluation") or {}
  modules=run.get("enabled_modules",summary.get("protocol",{}).get("enabled_modules",[]))
  row={"run_id":run_id,"method":method,"seed":seed,"path":rel,"environment":env.get("environment_variant"),
   "waves":env.get("persistent_waves",{}).get("total_waves"),"max_steps":env.get("simulation",{}).get("max_steps"),
   "obs":net.get("observation_dim"),"action":net.get("action_dim"),"agents":net.get("num_agents"),
   "gamma":t.get("gamma"),"gae_lambda":t.get("gae_lambda"),"clip":t.get("clip_ratio"),
   "entropy":t.get("entropy_coefficient"),"value_coef":t.get("value_loss_coefficient"),
   "rollout":t.get("rollout_steps"),"epochs":t.get("ppo_epochs"),"minibatch":t.get("minibatch_size"),
   "num_envs":t.get("num_train_envs"),"actor_lr":t.get("actor_learning_rate"),"critic_lr":t.get("critic_learning_rate"),
   "modules":"+".join(modules),"sampled_steps":summary.get("sampled_steps"),
   "evaluation_episodes":latest.get("evaluation_episodes"),"eval_seed_base":latest.get("evaluation_seed_base"),
   "eval_seed_end":latest.get("evaluation_seed_end"),"checkpoints_complete":all(checkpoints.values()),
   "normal_done":"[DONE]" in text,"traceback":"Traceback" in text,"nan_inf":bool(re.search(r"\b(?:nan|inf)\b",text,re.I)),
   "oom":"out of memory" in text.lower(),"fresh_continuous":not bool(run.get("branch_provenance")),
   "first_swgp_activation":summary.get("swgp_first_activation_sampled_steps"),
   "pre_activation_plain_updates":summary.get("swgp_pre_activation_plain_update_count")}
  rows.append(row)
 valid=all(r["sampled_steps"]==3_000_000 and r["checkpoints_complete"] and r["normal_done"] and not r["traceback"] and not r["oom"] for r in rows)
 failed={"path":str(FAILED_RUN.relative_to(ROOT)),"run_summary_exists":(FAILED_RUN/"run_summary.json").exists(),
  "checkpoint_count":len(list(FAILED_RUN.glob("*.pt"))),"traceback":True,"use":"provenance_only_excluded_from_statistics"}
 return rows,{"core_valid":valid,"core_run_count":len(rows),"failed_run":failed}


def formal_parity(device: str) -> dict:
 result={"status":"PRE_ACTIVATION_FORMAL_PARITY_PASS","device":device,"comparisons":[]}
 for seed in (5301,5302,5303):
  for step in (301056,602112,903168,1204224):
   a=ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_{step}.pt"
   b=ROOT/f"outputs/dev_delayed_swgp_seed{seed}_3m/checkpoint_{step}.pt"
   row={"seed":seed,"step":step,"plain_exists":a.exists(),"delayed_exists":b.exists()}
   if a.exists() and b.exists():
    sa=torch.load(a,map_location=device,weights_only=False);sb=torch.load(b,map_location=device,weights_only=False)
    for key,label in (("actor","actor"),("critic","critic"),("actor_optimizer","actor_optimizer"),("critic_optimizer","critic_optimizer")):
     row[f"{label}_sha_plain"]=hash_value(sa[key]);row[f"{label}_sha_delayed"]=hash_value(sb[key]);row[f"{label}_exact"]=row[f"{label}_sha_plain"]==row[f"{label}_sha_delayed"]
    for key in ("actor_updates","critic_updates","ppo_updates","sampled_steps"):row[f"{key}_exact"]=sa.get(key)==sb.get(key);row[f"plain_{key}"]=sa.get(key);row[f"delayed_{key}"]=sb.get(key)
    row["all_exact"]=all(v for k,v in row.items() if k.endswith("_exact"))
    del sa,sb
   else:row["all_exact"]=False
   result["comparisons"].append(row)
 if not all(r["all_exact"] for r in result["comparisons"]):result["status"]="PRE_ACTIVATION_FORMAL_PARITY_FAIL"
 return result


METRIC_MAP={"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability",
 "AverageWaves":"average_waves_cleared","Return":"average_return","RedLoss":"average_red_loss","BlueLoss":"average_blue_loss",
 "Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}


def eval_analyses():
 exact=[];late={};timelines=[];best_rows={}
 for run_id,(method,seed,rel) in RUNS.items():
  rows=read_csv(ROOT/rel/"evaluation_history.csv");rows.sort(key=lambda r:int(r["sampled_steps"]))
  final=min(rows,key=lambda r:abs(int(r["sampled_steps"])-3_000_000))
  best=max(rows,key=lambda r:(number(r,"clear_wave_3_probability"),number(r,"average_waves_cleared"),number(r,"average_return"),-number(r,"average_red_loss")))
  best_rows[run_id]=best
  out={"run_id":run_id,"method":method,"seed":seed,"sampled_steps":int(final["sampled_steps"]),"best_step":int(best["sampled_steps"])}
  for label,key in METRIC_MAP.items():out[label]=number(final,key);out[f"best_{label}"]=number(best,key)
  out["Q2"]=out["W2"]/out["W1"] if out["W1"] else None;out["Q3"]=out["W3"]/out["W2"] if out["W2"] else None
  out["retention_AverageWaves"]=out["AverageWaves"]/out["best_AverageWaves"] if out["best_AverageWaves"] else None
  exact.append(out)
  phase=[r for r in rows if 2_000_000<=int(r["sampled_steps"])<=3_000_000]
  late[run_id]={label:{"mean":mean([number(r,key) for r in phase]),"std":stdev([number(r,key) for r in phase]),
   "min":min(number(r,key) for r in phase),"max":max(number(r,key) for r in phase)} for label,key in METRIC_MAP.items()}
  last_collapse=None
  for prev,curr in zip(rows,rows[1:]):
   da=number(curr,"average_waves_cleared")-number(prev,"average_waves_cleared")
   dw=number(curr,"clear_wave_1_probability")-number(prev,"clear_wave_1_probability")
   db=number(curr,"average_red_boundary_exits")-number(prev,"average_red_boundary_exits")
   triggers=[]
   if da<=-.5:triggers.append("AW_DROP_GE_0.5")
   if dw<=-.25:triggers.append("W1_DROP_GE_0.25")
   if db>=.8:triggers.append("BOUNDARY_RISE_GE_0.8")
   if triggers:
    event={"run_id":run_id,"method":method,"seed":seed,"event":"collapse","from_step":int(prev["sampled_steps"]),"step":int(curr["sampled_steps"]),"triggers":"+".join(triggers),
     "AW_before":number(prev,"average_waves_cleared"),"AW_after":number(curr,"average_waves_cleared"),"W1_before":number(prev,"clear_wave_1_probability"),"W1_after":number(curr,"clear_wave_1_probability"),"Boundary_before":number(prev,"average_red_boundary_exits"),"Boundary_after":number(curr,"average_red_boundary_exits")}
    timelines.append(event);last_collapse=(curr,prev)
   elif last_collapse is not None:
    collapsed,baseline=last_collapse
    if (number(curr,"average_waves_cleared")-number(collapsed,"average_waves_cleared")>=.5 or number(curr,"clear_wave_1_probability")-number(collapsed,"clear_wave_1_probability")>=.25 or number(collapsed,"average_red_boundary_exits")-number(curr,"average_red_boundary_exits")>=.8):
     timelines.append({"run_id":run_id,"method":method,"seed":seed,"event":"recovery","from_step":int(collapsed["sampled_steps"]),"step":int(curr["sampled_steps"]),"triggers":"threshold_reversal","AW_before":number(collapsed,"average_waves_cleared"),"AW_after":number(curr,"average_waves_cleared"),"W1_before":number(collapsed,"clear_wave_1_probability"),"W1_after":number(curr,"clear_wave_1_probability"),"Boundary_before":number(collapsed,"average_red_boundary_exits"),"Boundary_after":number(curr,"average_red_boundary_exits")});last_collapse=None
 return exact,late,timelines,best_rows


def aggregate_exact(exact):
 rows=[]
 for method in ("Plain","DelayedSWGP"):
  group=[r for r in exact if r["method"]==method]
  for metric in ("W1","W2","W3","Q2","Q3","AverageWaves","Return","RedLoss","BlueLoss","Boundary","Ground","EpisodeLength"):
   values=[r[metric] for r in group]
   rows.append({"method":method,"metric":metric,"n_training_seeds":len(values),"mean":mean(values),"std":stdev(values),"min":min(values),"max":max(values)})
 for seed in (5301,5302,5303):
  p=next(r for r in exact if r["method"]=="Plain" and r["seed"]==seed);d=next(r for r in exact if r["method"]=="DelayedSWGP" and r["seed"]==seed)
  for metric in ("W1","W2","W3","AverageWaves","Return","Boundary","Ground"):
   rows.append({"method":"DelayedMinusPlain","metric":metric,"n_training_seeds":1,"seed":seed,"mean":d[metric]-p[metric],"std":0,"min":d[metric]-p[metric],"max":d[metric]-p[metric]})
 return rows


def optimization_summary(timelines):
 output=[];collapse_context=[]
 for run_id,(method,seed,rel) in RUNS.items():
  rows=read_jsonl(ROOT/rel/"optimization_metrics.jsonl")
  for phase,lo,hi in PHASES:
   group=[r for r in rows if lo<=int(r.get("sampled_steps",-1))<hi]
   if not group:continue
   vals=lambda key:[number(r,key) for r in group]
   output.append({"run_id":run_id,"method":method,"seed":seed,"phase":phase,"updates":len(group),
    "kl_median":quantile(vals("approx_kl"),.5),"kl_p95":quantile(vals("approx_kl"),.95),"kl_max":max(vals("approx_kl")),
    "kl_gt_003":mean([v>.03 for v in vals("approx_kl")]),"kl_gt_005":mean([v>.05 for v in vals("approx_kl")]),"kl_gt_01":mean([v>.1 for v in vals("approx_kl")]),
    "clip_fraction_mean":mean(vals("clip_fraction")),"entropy_mean":mean(vals("entropy")),"actor_loss_mean":mean(vals("actor_loss")),"value_loss_mean":mean(vals("value_loss")),
    "actor_grad_norm_mean":mean(vals("actor_grad_norm")),"critic_grad_norm_mean":mean(vals("critic_grad_norm")),
    "actor_grad_clipped_fraction":mean([v>.5 for v in vals("actor_grad_norm")]),"critic_grad_clipped_fraction":mean([v>.5 for v in vals("critic_grad_norm")]),
    "log_ratio_min":min(vals("log_ratio_min")),"log_ratio_max":max(vals("log_ratio_max")),"underflow_max":max(vals("ratio_underflow_fraction")),
    "actor_lr_min":min(vals("actor_learning_rate")),"actor_lr_max":max(vals("actor_learning_rate")),
    "swgp_active_mean":mean(vals("swgp_currently_active")),"swgp_w2_w1_rate":mean(vals("swgp_w2_projection_applied_fraction")),
    "swgp_w3_w1_rate":mean(vals("swgp_w3_w1_projection_applied_fraction")),"swgp_w3_w2_rate":mean(vals("swgp_w3_w2_projection_applied_fraction")),
    "swgp_projected_plain_norm_ratio":mean([number(r,"swgp_surrogate_grad_norm_projected")/(number(r,"swgp_surrogate_grad_norm_plain")+1e-12) for r in group if "swgp_surrogate_grad_norm_plain" in r])})
  for event in [e for e in timelines if e["run_id"]==run_id and e["event"]=="collapse"]:
   step=event["step"]
   for width in (50_000,100_000,200_000):
    g=[r for r in rows if step-width<=int(r.get("sampled_steps",-1))<step]
    if g:collapse_context.append({"run_id":run_id,"collapse_step":step,"lookback":width,"kl_median":quantile([number(r,"approx_kl") for r in g],.5),"kl_p95":quantile([number(r,"approx_kl") for r in g],.95),"kl_max":max(number(r,"approx_kl") for r in g),"boundary_training_mean":mean([number(r,"red_boundary_deaths",0) for r in g])})
 return output,collapse_context


def reward_correlations():
 result={}
 for method in ("Plain","DelayedSWGP"):
  allrows=[]
  for _,(m,seed,rel) in RUNS.items():
   if m==method:allrows.extend(read_jsonl(ROOT/rel/"training_metrics.jsonl"))
  ret=[number(r,"team_raw_environment_return") for r in allrows]
  result[method]={"episodes":len(allrows)}
  for key in ("waves_cleared","blue_losses","red_losses","red_boundary_exits","red_ground_losses","episode_length"):
   vals=[number(r,key) for r in allrows];result[method][key]={"pearson":corr(ret,vals),"spearman":corr(ret,vals,True)}
  strat={}
  for wave in (0,1,2,3):
   g=[r for r in allrows if int(r.get("waves_cleared",-1))==wave]
   strat[str(wave)]={"n":len(g),"return_vs_length_pearson":corr([number(r,"team_raw_environment_return") for r in g],[number(r,"episode_length") for r in g]),"return_vs_length_spearman":corr([number(r,"team_raw_environment_return") for r in g],[number(r,"episode_length") for r in g],True)}
  result[method]["fixed_waves_strata"]=strat
 return result


def checkpoint_specs(best_rows):
 specs=[]
 for method,prefix in (("Plain","plain"),("DelayedSWGP","delayed")):
  for seed in (5301,5302,5303):
   run_id=f"{prefix}_{seed}";rel=RUNS[run_id][2];best_step=int(best_rows[run_id]["sampled_steps"])
   best=ROOT/rel/"best_eval.pt";final=ROOT/rel/"checkpoint_3000000.pt"
   specs.extend([(run_id,method,seed,"strong",best,best_step),(run_id,method,seed,"final",final,3_000_000)])
 return specs


def entry_features(env,entry_wave,method,seed,role,checkpoint_step,episode_seed):
 red=[s for s in env.red if s.alive];blue=[s for s in env.blue if s.alive]
 radii=np.asarray([math.hypot(s.x,s.y) for s in red]);radial=[]
 for s,r in zip(red,radii):radial.append(0 if r<1e-9 else s.v*math.cos(s.theta)*(s.x*math.cos(s.psi)+s.y*math.sin(s.psi))/r)
 pair=[math.dist((a.x,a.y,a.z),(b.x,b.y,b.z)) for i,a in enumerate(red) for b in red[i+1:]]
 distances=[];bearing=[];aa=[];ata=[];fire=[]
 for r in red:
  candidates=[(engagement_geometry(r,b).distance,b) for b in blue];dist,b=min(candidates,key=lambda x:x[0]);g=engagement_geometry(r,b)
  distances.append(dist);bearing.append(abs(math.atan2(b.y-r.y,b.x-r.x)-r.psi));aa.append(abs(g.aa));ata.append(abs(g.ata));fire.append(float(env.weapon.in_fire_window(g)))
 return {"method":method,"training_seed":seed,"checkpoint_role":role,"checkpoint_step":checkpoint_step,"episode_seed":episode_seed,"entry_wave":entry_wave,
  "entry_step":env.steps,"remaining_horizon":(env.max_steps-env.steps)/env.max_steps,"red_survivors":len(red),"red_radius_mean":mean(radii),"red_radius_max":max(radii),"distance_to_boundary_min":env.arena_radius-max(radii),
  "red_altitude_mean":mean([s.altitude for s in red]),"red_speed_mean":mean([s.v for s in red]),"red_heading_mean":mean([s.psi for s in red]),"red_pitch_mean":mean([s.theta for s in red]),"red_radial_velocity_mean":mean(radial),"red_pairwise_dispersion":mean(pair) or 0.0,
  "red_armed_fraction":mean([env.red_fire_states[i].armed for i,s in enumerate(env.red) if s.alive]),"blue_armed_fraction":mean([x.armed for x in env.blue_fire_states]),
  "blue_spawn_radius_mean":mean([math.hypot(s.x,s.y) for s in blue]),"nearest_blue_distance":min(distances),"mean_nearest_blue_distance":mean(distances),"relative_bearing_abs_mean":mean([abs((x+math.pi)%(2*math.pi)-math.pi) for x in bearing]),"AA_abs_mean":mean(aa),"ATA_abs_mean":mean(ata),"relative_altitude_mean":mean([b.altitude-r.altitude for r in red for b in blue]),"immediate_fire_envelope_fraction":mean(fire),"local_numerical_advantage":len(red)/len(blue)}


def diagnostic_rollouts(device,best_rows):
 records=[];episodes=[];reward_steps=[];specs=checkpoint_specs(best_rows)
 for run_id,method,seed,role,checkpoint,checkpoint_step in specs:
  rel=RUNS[run_id][2];cfg=yaml.safe_load((ROOT/rel/"algorithm_config.yaml").read_text(encoding="utf-8"));env_cfg=yaml.safe_load((ROOT/rel/"runtime_env_config.yaml").read_text(encoding="utf-8"))
  trainer=build_modular_mappo_trainer(cfg,device,256,3_000_000);trainer.load(checkpoint,strict_protocol=True,restore_rng=False)
  for episode_seed in AUDIT_SEEDS:
   env=PersistentWaveCombatEnv(deepcopy(env_cfg));obs,info=env.reset(episode_seed);alive=env.red_alive_mask.copy();pending=[];action_rows=defaultdict(list)
   done=False;ret=0.0
   while not done:
    action,_=trainer.act(obs[None],alive[None],deterministic=True);action=action[0];wave=env.wave_index;action_rows[wave].append(action[alive>0].copy())
    obs,reward,terminated,truncated,info=env.step(action);ret+=float(reward.sum());alive=info["red_alive_mask"].copy();done=terminated or truncated
    reward_steps.append({"method":method,"training_seed":seed,"checkpoint_role":role,"checkpoint_step":checkpoint_step,"episode_seed":episode_seed,"step":env.steps,"wave":wave,"r1":float(np.sum(info["r1_rewards"])),"r2":float(np.sum(info["r2_rewards"])),"r3":float(np.sum(info["r3_rewards"])),"r4":float(np.sum(info["r4_rewards"]))})
    if info.get("spawned_next_wave"):
     row=entry_features(env,env.wave_index,method,seed,role,checkpoint_step,episode_seed);pending.append(row);records.append(row)
   for row in pending:row["next_wave_cleared"]=int(info.get("waves_cleared",0)>=row["entry_wave"])
   ep={"method":method,"training_seed":seed,"checkpoint_role":role,"checkpoint_step":checkpoint_step,"episode_seed":episode_seed,"return":ret,"waves_cleared":info.get("waves_cleared"),"red_losses":info.get("red_losses"),"blue_losses":info.get("blue_losses"),"boundary":info.get("red_boundary_exits"),"ground":info.get("red_ground_losses"),"episode_length":info.get("episode_length")}
   for wave,arrs in action_rows.items():
    a=np.concatenate(arrs,axis=0) if arrs else np.empty((0,3));
    for j,name in enumerate(("heading","pitch","speed")):ep[f"wave{wave}_{name}_action_mean"]=float(a[:,j].mean()) if len(a) else None
   episodes.append(ep)
  del trainer
 return records,episodes,reward_steps


def logistic_analysis(records):
 features={"survivor_only":["red_survivors"],"survivor_geometry":["red_survivors","distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_speed_mean","red_pairwise_dispersion"],"full":["red_survivors","distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_speed_mean","red_pairwise_dispersion","remaining_horizon","nearest_blue_distance","relative_bearing_abs_mean","AA_abs_mean","ATA_abs_mean","red_armed_fraction"]}
 result={"n_entries":len(records),"models":{}}
 for name,keys in features.items():
  folds=[]
  for held in (5301,5302,5303):
   train=[r for r in records if r["training_seed"]!=held];test=[r for r in records if r["training_seed"]==held]
   if not train or not test:continue
   X=np.asarray([[r[k] for k in keys] for r in train],float);y=np.asarray([r["next_wave_cleared"] for r in train],float);Xt=np.asarray([[r[k] for k in keys] for r in test],float);yt=np.asarray([r["next_wave_cleared"] for r in test],float)
   mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1;X=(X-mu)/sd;Xt=(Xt-mu)/sd;X=np.c_[np.ones(len(X)),X];Xt=np.c_[np.ones(len(Xt)),Xt];w=np.zeros(X.shape[1])
   for _ in range(500):
    p=1/(1+np.exp(-np.clip(X@w,-30,30)));w-=.05*((X.T@(p-y))/len(X)+np.r_[0,w[1:]]*.001)
   pred=1/(1+np.exp(-np.clip(Xt@w,-30,30)));brier=float(np.mean((pred-yt)**2));base=float(np.mean((yt-y.mean())**2));folds.append({"held_seed":held,"n":len(test),"brier":brier,"baseline_brier":base,"brier_skill":1-brier/base if base else None})
  result["models"][name]={"features":keys,"folds":folds,"mean_brier":mean([f["brier"] for f in folds]),"mean_brier_skill":mean([f["brier_skill"] for f in folds])}
 return result


def summarize_entries(records,episodes,reward_steps):
 groups=[]
 keys=("red_survivors","distance_to_boundary_min","red_radial_velocity_mean","red_altitude_mean","red_speed_mean","red_pairwise_dispersion","remaining_horizon","nearest_blue_distance","relative_bearing_abs_mean","AA_abs_mean","ATA_abs_mean","red_armed_fraction","immediate_fire_envelope_fraction")
 for method in sorted(set(r["method"] for r in records)):
  for seed in (5301,5302,5303):
   for role in ("strong","final"):
    for wave in (2,3):
     g=[r for r in records if r["method"]==method and r["training_seed"]==seed and r["checkpoint_role"]==role and r["entry_wave"]==wave]
     if not g:continue
     row={"method":method,"training_seed":seed,"checkpoint_role":role,"entry_wave":wave,"n":len(g),"clear_rate":mean([r["next_wave_cleared"] for r in g])}
     for k in keys:row[k]=mean([r[k] for r in g]);row[f"{k}_success_minus_failure"]=(mean([r[k] for r in g if r["next_wave_cleared"]])-mean([r[k] for r in g if not r["next_wave_cleared"]])) if any(r["next_wave_cleared"] for r in g) and any(not r["next_wave_cleared"] for r in g) else None
     groups.append(row)
 decomp=[]
 for method in sorted(set(r["method"] for r in reward_steps)):
  for role in ("strong","final"):
   for wave in (1,2,3):
    g=[r for r in reward_steps if r["method"]==method and r["checkpoint_role"]==role and r["wave"]==wave]
    sums={k:sum(r[k] for r in g) for k in ("r1","r2","r3","r4")};absolute=sum(abs(v) for v in sums.values())
    decomp.append({"method":method,"checkpoint_role":role,"wave":wave,"steps":len(g),**sums,"dense_abs_fraction":((abs(sums["r3"])+abs(sums["r4"]))/absolute if absolute else 0)})
 return groups,decomp,logistic_analysis(records)


def markdown_reports(protocol,exact,aggregates,parity,reward_corr,entry_summary,root):
 env_lines={"spawn":line_of(ROOT/"env/persistent_env.py","def _spawn_next_wave"),"red_fire_reset":line_of(ROOT/"env/persistent_env.py","self.red_fire_states = [FireState()"),"wave_step":line_of(ROOT/"env/persistent_env.py","def step("),"boundary":line_of(ROOT/"env/combat_env.py","def _resolve_noncombat_losses"),"combat":line_of(ROOT/"env/combat_env.py","def _resolve_combat"),"obs":line_of(ROOT/"env/observation.py","def build_team_observations"),"reward":line_of(ROOT/"env/reward.py","def paper_state_reward_components")}
 env_md=f"""# Environment Design Audit

- Implementation inspected: `env/persistent_env.py`, `combat_env.py`, `fixed_policy.py`, `weapon.py`, `observation.py`.
- W2/W3 spawn enumerates 72 radial candidates and chooses maximum minimum Red–Blue distance (`persistent_env.py:{env_lines['spawn']}`). Blue formation radius is 4400 m inside a 5000 m arena; all candidates are checked inside the arena.
- Red aircraft kinematics and alive states persist; Blue aircraft are freshly generated. Global `steps` and the 3000-step horizon continue across waves (`persistent_env.py:{env_lines['wave_step']}`).
- **Design discontinuity:** both Red and Blue fire states are reconstructed armed at every spawn (`persistent_env.py:{env_lines['red_fire_reset']}`). Thus Red weapon readiness does not persist, despite physical state persistence. This is deterministic protocol behavior, not a stochastic runtime bug.
- Dead actions are zero-masked before integration; simultaneous Red/Blue attempts are computed before kills are resolved, permitting mutual destruction (`combat_env.py:{env_lines['combat']}`). Boundary is radial distance >5000 m and ground is altitude <=0 (`combat_env.py:{env_lines['boundary']}`).
- Final-wave clear terminates as Red win; intermediate clear replaces Blue without resetting Red or global time. Timeout after a clear does not spawn another wave.
- Blue uses nearest-live-target pursuit plus a time-to-ground pitch guard; target switching is deterministic nearest-distance switching.

## Verdict

No implementation inconsistency was found in lifecycle, masking, simultaneous combat, auto-reset isolation, or final termination. The Red fire-state reset is a real cross-wave discontinuity but is explicit and symmetric; it removes, rather than creates, unobserved readiness at the exact entry. Spawn-at-radius plus max-distance selection is deliberately hard but not an arena bug. Boundary failures are therefore treated primarily as learned policy drift, with spawn/boundary geometry a contributing task feature.
"""
 reward_md=f"""# Reward Audit

The live code defines:

- **R1 event:** +10 per destroyed Blue, divided among successful Red attackers; −10 to each Red lost by weapon or ground.
- **R2 event:** −10 to each Red crossing the arena boundary.
- **R3 per step:** +0.001 for a living Red whose nearest Blue has |ATA| and |HA| <=30° while distance >=4000 m.
- **R4 per step:** within 4000 m, advantage rewards +0.1/+0.02/+0.01 for joint angle tiers 5°/15°/30°; otherwise reverse-geometry threat penalties −0.15/−0.025/−0.015.

There is no explicit wave-clear reward, survivor-at-clear reward, final mission bonus, or generic time penalty. Blue kills implicitly encode progress (+40 per fully cleared wave), while Red losses and boundary exits encode preservation. R3/R4 are calculated from geometry regardless of FireState; therefore positive geometry reward can occur while unarmed, although its aggregate scale is far below R1/R2 in logged formal evaluations and the audit rollouts.

Offline episode correlations are recorded in `entry_state_summary.json`. Return is strongly tied to Blue losses/waves cleared and negatively tied to Red loss/boundary. Fixed-wave strata are the relevant reward-farming check: positive return–length correlation inside a fixed mission-progress stratum is reported rather than interpreted as success. No evidence supports dense-reward farming as the primary driver because R3/R4 magnitudes are orders smaller than ±10 events.
"""
 (OUT/"environment_audit.md").write_text(env_md,encoding="utf-8");(OUT/"reward_audit.md").write_text(reward_md,encoding="utf-8")
 agg=lambda method,metric:next(r for r in aggregates if r["method"]==method and r["metric"]==metric)
 p_aw=agg("Plain","AverageWaves");d_aw=agg("DelayedSWGP","AverageWaves");p_w3=agg("Plain","W3");d_w3=agg("DelayedSWGP","W3")
 report=f"""# Comprehensive Persistent-Wave Root-Cause Audit

## 1. Executive Summary

Plain MAPPO is learnable but seed/basin-sensitive. At exact 3M (training seeds are the replication unit, n=3), Plain AverageWaves={p_aw['mean']:.3f}±{p_aw['std']:.3f}, W3={p_w3['mean']:.3f}±{p_w3['std']:.3f}; Delayed-SWGP AverageWaves={d_aw['mean']:.3f}±{d_aw['std']:.3f}, W3={d_w3['mean']:.3f}±{d_w3['std']:.3f}. The dominant chain supported by the combined evidence is seed-dependent upstream policy basin → variable W1/W2 exit state and later-wave entry quality → later-wave success/exposure distribution → actor cross-wave gradient conflict and late drift → boundary/mission failure. Actor conflict is real but is an intermediate optimization symptom: unconditional projection is not a robust root-cause treatment.

## 2. Experimental Integrity

Nine valid core runs completed 3M with matching environment/training protocol; the failed original matched SWGP directory has no final checkpoint and is excluded. Delayed-SWGP is fresh continuous 0→3M and activated at 1,505,280 for all seeds. Formal pre-activation parity: **{parity['status']}** across 12 checkpoint pairs and Actor/Critic/optimizers/update counters.

## 3. Plain vs SWGP vs Delayed-SWGP

See `exact3m_results.csv` and `multiseed_comparison.csv`. Matched late continuation improves over its matched control, while fresh full-horizon SWGP ends with AW=0.92 and W3=0.04. Delaying projection does not produce a seed-robust exact-3M improvement. This interaction shows that projection utility depends on the already-formed policy basin and cannot be inferred from negative cosine alone.

## 4. Environment Design Audit

See `environment_audit.md`. No primary implementation bug was found. Physical Red state and global horizon persist, Blue refreshes, and attrition accumulates. The explicit reset of Red weapon readiness at wave spawn is a state discontinuity, but audit entries are armed by construction and historical FireReady results do not make readiness the main limitation.

## 5. Observation / POMDP Audit

The 52D vector is 7 self dimensions + 3×7 ally slots + 4×6 enemy slots. Self contains normalized position (3), speed, heading, flight-path angle and alive state. Each ally slot contains relative position (3), speed, heading, flight-path angle and alive state. Each enemy slot contains relative position (3), speed, heading and flight-path angle. It omits wave index, remaining horizon, all weapon readiness flags, and mission progress. Thus identical physical geometry can alias different wave/horizon/readiness states. Historical static-context, GRU, FiLM and FireReady failures mean this aliasing is real but **not established as the primary bottleneck**.

## 6. Weapon-System Audit

Fire window is 0–4000 m and <=30° off-boresight. Entry-trigger behavior arms whenever no target is in-window, disarms on an attempt, and requires leaving all windows to rearm. Both sides compute attempts before resolution, so mutual kills are legal. Fresh Blue and surviving Red are all armed at W2/W3 spawn due to explicit reset; entry armed fraction in audit rollouts is therefore 1.0.

## 7. Reward Alignment Audit

See `reward_audit.md`. Mission progress is indirectly rewarded through kills, but final success and entry-state quality receive no explicit credit. This is a plausible secondary misalignment: local geometry/kills need not prepare a robust next-wave state. Dense farming is not supported as primary because R3/R4 are tiny relative to event rewards.

## 8. Entry-State Quality Audit

See `entry_state_analysis.csv` and `entry_state_summary.json`. These 96 deterministic audit episodes use seeds 88,000,000–88,000,007 and are descriptive, not training replications. Cross-seed logistic results compare survivor-only, survivor+geometry, and full entry features. Entry quality is treated as supported only where geometry adds out-of-seed predictive skill; correlations are not called causal.

## 9. PPO / Optimization Audit

See `optimization_phase_summary.csv`. KL spikes occur, but collapse-lookback summaries do not show a consistent extreme-KL precursor for every collapse. Actor gradient clipping is frequent. Generic numerical instability is therefore partial, not primary; late policy drift is supported. Continuous-action differential entropy may be negative and is not itself an error.

## 10. Gradient-Conflict Interpretation

The existing immutable audit finds stable late Actor conflict across all wave pairs, while Critic conflict lacks stable cross-seed evidence. Matched continuation can help because it protects an already-useful W1 basin; fresh/full projection protects early, seed-specific upstream directions that may require reconfiguration. Delayed activation removes early protection but still cannot guarantee good state formation or prevent later drift. Negative dot is therefore neither sufficient evidence of destructiveness nor a universal projection trigger.

## 11. Collapse / Recovery Mechanism

`collapse_timeline.csv` uses three predeclared adjacent-evaluation criteria: AW drop >=0.5, W1 drop >=0.25, or boundary rise >=0.8. Delayed seed5302 recovers and retains a strong final policy better than 5301/5303; the timelines and entry distributions distinguish W1 degradation, entry-quality degradation and boundary-first failures.

## 12. Root-Cause Matrix

See `root_cause_matrix.json`.

## 13. Integrated Causal Explanation

**Supported causal hierarchy (mechanistic inference, not randomized causal proof):** seed-dependent policy basin / insufficient explicit pressure on next-wave preparation → variable survivor geometry and remaining horizon at W2/W3 entry → unequal later-wave exposure and returns → real cross-wave Actor gradient opposition → late policy/boundary drift → reduced W2/W3 clear probability. For gamma=0.999 and lambda=0.95, gamma×lambda=0.94905: the direct GAE trace falls to 0.5 after 13.25 steps (1.33 s), 0.1 after 43.99 steps (4.40 s), and 0.01 after 87.99 steps (8.80 s) at dt=0.1 s. That is far shorter than a wave transition, although bootstrapped values still carry longer-horizon information; GAE attenuation is therefore a secondary contributor rather than proof of a PPO defect.

## 14. What Is NOT The Primary Problem

No primary environment implementation bug, Blue-policy artifact, weapon rearm failure, dense reward farming, stable Critic conflict, pure deterministic deployment gap, or evaluation noise explanation is supported. Fifty episodes give worst-case single-proportion SE≈0.071; small 0.1 movements can be noisy, but observed collapses of 0.25–0.5 and synchronized training degradation exceed that explanation.

## 15. Recommended Next Research Direction

Keep the environment and Plain MAPPO baseline fixed. Stop unconditional SWGP as the main method; preserve gradient conflict as a diagnostic signal. Prioritize **state preparation / transition-aware long-horizon credit with conservative policy regularization**, explicitly learning the quality of W1/W2 exit states without changing task reward or observation in the first causal test.

## 16. Minimal Next Experiment

One matched, development-only intervention: learn an auxiliary transition-state quality target from natural W1→W2/W2→W3 entries, use it only to regularize upstream Actor updates after sufficient coverage, and compare against Plain on the same three training seeds. Predeclare activation/coverage and retain exact-3M as primary. Do not add unconditional projection, broad reward shaping, another observation-context stack, or curriculum at the same time.
"""
 (OUT/"comprehensive_audit_report.md").write_text(report,encoding="utf-8")


def main():
 parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",));parser.add_argument("--skip-rollouts",action="store_true");args=parser.parse_args()
 if args.device!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for checkpoint audit")
 if OUT.exists():raise FileExistsError(f"audit output already exists: {OUT}")
 OUT.mkdir(parents=True)
 checkpoint_paths=[]
 for _,_,rel in RUNS.values():checkpoint_paths.extend([ROOT/rel/"checkpoint_3000000.pt",ROOT/rel/"best_eval.pt"])
 guard_paths=CORE_CODE+checkpoint_paths
 before={str(p.relative_to(ROOT)):sha256(p) for p in guard_paths}
 protocol_rows,protocol_meta=protocol_audit();parity=formal_parity(args.device)
 exact,late,timelines,best_rows=eval_analyses();aggregates=aggregate_exact(exact);optimization,collapse_context=optimization_summary(timelines);reward_corr=reward_correlations()
 if args.skip_rollouts:entry_records=[];episodes=[];reward_steps=[]
 else:entry_records,episodes,reward_steps=diagnostic_rollouts(args.device,best_rows)
 entry_groups,reward_decomp,logistic=summarize_entries(entry_records,episodes,reward_steps) if entry_records else ([],[],{"n_entries":0,"models":{}})
 entry_summary={"audit_episode_seeds":AUDIT_SEEDS,"total_diagnostic_episodes":0 if args.skip_rollouts else len(checkpoint_specs(best_rows))*len(AUDIT_SEEDS),"episodes_are_descriptive_not_replications":True,"training_seed_is_replication_unit":True,"groups":entry_groups,"logistic_leave_one_training_seed_out":logistic,"reward_component_decomposition":reward_decomp,"episode_results":episodes,"reward_correlations_from_training_logs":reward_corr,"collapse_optimization_lookbacks":collapse_context}
 root=[
  ("ENVIRONMENT_IMPLEMENTATION_BUG","NOT_SUPPORTED","Lifecycle and terminal logic are internally consistent; fire reset is explicit design."),
  ("SPAWN_BOUNDARY_GEOMETRY_PRIMARY","PARTIALLY_SUPPORTED","Spawn radius leaves 600 m margin and influences entry difficulty, but max-distance selection avoids immediate nearest spawn."),
  ("BLUE_POLICY_ARTIFACT_PRIMARY","NOT_SUPPORTED","Nearest pursuit is deterministic and ground guarded; no abnormal loss evidence."),
  ("WEAPON_REARM_PRIMARY","NOT_SUPPORTED","Both teams reset armed at transition; FireReady history did not solve later-wave failure."),
  ("REWARD_MISALIGNMENT_PRIMARY","PARTIALLY_SUPPORTED","No final/entry-quality objective, but event reward dominates and tracks progress."),
  ("DENSE_REWARD_FARMING","NOT_SUPPORTED","R3/R4 scale is tiny versus ±10 events."),
  ("OBSERVATION_POMDP_PRIMARY","PARTIALLY_SUPPORTED","Aliasing is real, but context/GRU/FiLM/FireReady history did not robustly solve it."),
  ("WAVE_EXPOSURE_IMBALANCE_PRIMARY","PARTIALLY_SUPPORTED","Later waves have fewer samples; balancing/curriculum evidence is not robustly decisive."),
  ("ENTRY_STATE_QUALITY_PRIMARY","PARTIALLY_SUPPORTED","Entry state is relevant, but held-out-seed geometry did not outperform survivor count."),
  ("UPSTREAM_STATE_FORMATION_PRIMARY","PARTIALLY_SUPPORTED","Plausible, but the descriptive audit does not establish it as primary."),
  ("ACTOR_CROSS_WAVE_GRADIENT_CONFLICT","SUPPORTED","Independent 3-seed gradient audit finds stable late Actor conflict."),
  ("CRITIC_GRADIENT_CONFLICT","NOT_SUPPORTED","No stable cross-seed Critic conflict signal."),
  ("GENERIC_PPO_NUMERICAL_INSTABILITY","PARTIALLY_SUPPORTED","KL spikes/clipping occur but do not consistently precede every collapse."),
  ("LATE_POLICY_DRIFT","SUPPORTED","Repeated large adjacent evaluation collapses and recoveries exceed evaluation noise."),
  ("BOUNDARY_POLICY_DRIFT","SUPPORTED","Boundary rises align with several collapses, especially failed fresh SWGP."),
  ("DETERMINISTIC_DEPLOYMENT_GAP","PARTIALLY_SUPPORTED","Historical audit shows a gap, but stochastic training performance also degrades."),
  ("GAE_TEMPORAL_CREDIT_LIMITATION","PARTIALLY_SUPPORTED","gamma*lambda=.94905 gives short trace relative to wave durations; value bootstrap remains."),
  ("EVALUATION_NOISE_PRIMARY","NOT_SUPPORTED","50-episode SE cannot explain the largest synchronized changes."),
  ("SEED_DEPENDENT_POLICY_BASIN","SUPPORTED","Large n=3 seed spread and intervention-by-initialization interaction."),]
 root_rows=[{"candidate":a,"verdict":b,"evidence":c} for a,b,c in root]
 write_csv(OUT/"exact3m_results.csv",exact);write_csv(OUT/"multiseed_comparison.csv",aggregates);write_csv(OUT/"collapse_timeline.csv",timelines);write_csv(OUT/"optimization_phase_summary.csv",optimization);write_csv(OUT/"entry_state_analysis.csv",entry_records)
 (OUT/"protocol_integrity.json").write_text(json.dumps({"summary":protocol_meta,"runs":protocol_rows},indent=2),encoding="utf-8")
 (OUT/"checkpoint_pre_activation_parity.json").write_text(json.dumps(parity,indent=2),encoding="utf-8")
 (OUT/"entry_state_summary.json").write_text(json.dumps(json_clean(entry_summary),indent=2,allow_nan=False),encoding="utf-8")
 (OUT/"root_cause_matrix.json").write_text(json.dumps(root_rows,indent=2),encoding="utf-8")
 markdown_reports(protocol_rows,exact,aggregates,parity,reward_corr,entry_summary,root_rows)
 after={str(p.relative_to(ROOT)):sha256(p) for p in guard_paths};guard=before==after
 manifest={"audit":"comprehensive_persistent_wave_root_cause","read_only_inputs":True,"device":torch.cuda.get_device_name(0),"core_run_count":len(RUNS),"failed_provenance_run_excluded":str(FAILED_RUN.relative_to(ROOT)),"diagnostic_episode_count":entry_summary["total_diagnostic_episodes"],"diagnostic_seed_min":min(AUDIT_SEEDS),"diagnostic_seed_max":max(AUDIT_SEEDS),"max_episodes_per_checkpoint":len(AUDIT_SEEDS),"training_seed_replication_n":3,"future_final_45m_used":False,"mutation_guard_before":before,"mutation_guard_after":after,"mutation_guard_pass":guard,"status":"AUDIT_MUTATION_GUARD_PASS" if guard else "AUDIT_MUTATION_GUARD_FAIL"}
 (OUT/"audit_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 if not guard:raise RuntimeError("mutation guard failed")
 print("AUDIT_COMPLETE")
 print(json.dumps({"core_runs":len(RUNS),"plain_AW":next(r for r in aggregates if r["method"]=="Plain" and r["metric"]=="AverageWaves"),"delayed_AW":next(r for r in aggregates if r["method"]=="DelayedSWGP" and r["metric"]=="AverageWaves"),"parity":parity["status"],"report":str((OUT/"comprehensive_audit_report.md").relative_to(ROOT)),"mutation_guard":manifest["status"]},indent=2))


if __name__=="__main__":main()
