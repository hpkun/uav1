#!/usr/bin/env python3
"""Strictly offline root-cause diagnostics for the four completed 433 runs."""
from __future__ import annotations

from collections import Counter
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config

RUNS = {
    "plain5301": ROOT / "outputs/full433_plain_mappo_seed5301_3m",
    "plain5302": ROOT / "outputs/full433_plain_mappo_seed5302_3m",
    "marc5301": ROOT / "outputs/full433_marc_mappo_v1_seed5301_3m",
    "marc5302": ROOT / "outputs/full433_marc_mappo_v1_seed5302_3m",
}
OUTPUT = ROOT / "outputs/marc_433_root_cause_diagnostic"
PHASES = ((0,1_000_000,"0-1.0M"),(1_000_000,1_900_000,"1.0-1.9M"),
          (1_900_000,2_300_000,"1.9-2.3M"),(2_300_000,2_700_000,"2.3-2.7M"),
          (2_700_000,3_000_001,"2.7-3.0M"))
EVAL_FIELDS = {"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability",
 "W3":"clear_wave_3_probability","AW":"average_waves_cleared","Return":"average_return",
 "RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits",
 "Ground":"average_red_ground_losses"}
OPT_FIELDS = [
 "marc_samples_wave1","marc_samples_wave2","marc_samples_wave3",
 "marc_fraction_wave1","marc_fraction_wave2","marc_fraction_wave3",
 "marc_weight_wave1","marc_weight_wave2","marc_weight_wave3","marc_effective_weight_mean",
 "marc_bank_size_wave1","marc_bank_size_wave2","marc_bank_size_wave3",
 "marc_ready_wave1","marc_ready_wave2","marc_ready_wave3",
 "marc_retention_kl_wave1","marc_retention_kl_wave2","marc_retention_kl_wave3",
 "marc_retention_loss","marc_global_adv_mean","marc_global_adv_std",
 "marc_local_adv_mean","marc_local_adv_std","marc_cont_adv_mean","marc_cont_adv_std",
 "marc_actor_adv_mean","marc_actor_adv_std","actor_loss","value_loss","entropy","approx_kl",
 "clip_fraction","actor_grad_norm","critic_grad_norm","actor_learning_rate",
 "log_ratio_min","log_ratio_max","ratio_underflow_fraction",
 "policy_log_std_mean_psi","policy_log_std_mean_theta","policy_log_std_mean_v"]

def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]

def read_csv(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))

def number(value):
    if value in (None,""):return None
    try:
        x=float(value);return x if np.isfinite(x) else None
    except (TypeError,ValueError):return None

def qmetrics(row):
    w1,w2,w3=(number(row[EVAL_FIELDS[k]]) for k in ("W1","W2","W3"))
    return {"Q2":None if not w1 else w2/w1,"Q3":None if not w2 else w3/w2}

def mean_rows(rows,key):
    values=[number(row.get(key)) for row in rows];values=[v for v in values if v is not None]
    return None if not values else float(np.mean(values))

def corr(rows,x,y):
    pairs=[(number(r.get(x)),number(r.get(y))) for r in rows];pairs=[p for p in pairs if None not in p]
    if len(pairs)<2 or np.std([p[0] for p in pairs])==0 or np.std([p[1] for p in pairs])==0:return None
    return float(np.corrcoef(np.asarray(pairs).T)[0,1])

def eval_analysis(name,path,best_step):
    rows=read_csv(path/"evaluation_history.csv");out=[]
    for row in rows:
        item={"run":name,"sampled_steps":int(float(row["sampled_steps"]))}
        item.update({key:number(row[source]) for key,source in EVAL_FIELDS.items()});item.update(qmetrics(row));out.append(item)
    best=next(row for row in out if row["sampled_steps"]==best_step);final=max(out,key=lambda r:r["sampled_steps"])
    delta={key:(None if best.get(key) is None or final.get(key) is None else final[key]-best[key])
           for key in (*EVAL_FIELDS,"Q2","Q3")}
    late=[r for r in out if r["sampled_steps"]>=2_000_000]
    last=[r for r in out if r["sampled_steps"]>=2_500_000]
    avg=lambda selected:{key:(None if not [r[key] for r in selected if r.get(key) is not None]
        else float(np.mean([r[key] for r in selected if r.get(key) is not None]))) for key in (*EVAL_FIELDS,"Q2","Q3")}
    return out,{"best":best,"final":final,"best_to_final_delta":delta,
        "max_AW":max(r["AW"] for r in out),"max_W3":max(r["W3"] for r in out),
        "late_1m_mean":avg(late),"last_500k_mean":avg(last),
        "boundary_AW_correlation":corr(out,"Boundary","AW"),
        "boundary_W1_correlation":corr(out,"Boundary","W1")}

def phase_summary(name,rows):
    result=[];keys=set().union(*(row.keys() for row in rows))
    for low,high,label in PHASES:
        selected=[r for r in rows if low<=int(r["sampled_steps"])<high]
        item={"run":name,"phase":label,"start_step":low,"end_step":high-1,"updates":len(selected)}
        for key in OPT_FIELDS:
            item[key+"_mean"]="NOT_LOGGED" if key not in keys else mean_rows(selected,key)
            if key in keys and key in ("approx_kl","actor_grad_norm","clip_fraction","entropy","marc_retention_loss"):
                values=[number(r.get(key)) for r in selected];values=[v for v in values if v is not None]
                item[key+"_max"]=None if not values else max(values)
        for threshold in (.05,.10,.20):
            item[f"kl_gt_{threshold:.2f}_count"]=sum(number(r.get("approx_kl")) is not None and number(r["approx_kl"])>threshold for r in selected)
        for wave in (1,2,3):
            key=f"marc_weight_wave{wave}";values=[number(r.get(key)) for r in selected];values=[v for v in values if v is not None and v>0]
            item[f"marc_weight_wave{wave}_at_min_fraction"]="NOT_LOGGED" if key not in keys else (None if not values else float(np.mean(np.isclose(values,.5))))
            item[f"marc_weight_wave{wave}_at_max_fraction"]="NOT_LOGGED" if key not in keys else (None if not values else float(np.mean(np.isclose(values,2.0))))
        result.append(item)
    return result

def spike_summary(rows):
    result={}
    for threshold in (.05,.10,.20):
        selected=[int(r["sampled_steps"]) for r in rows if number(r.get("approx_kl")) is not None and number(r["approx_kl"])>threshold]
        result[f"kl_gt_{threshold:.2f}_count"]=len(selected);result[f"kl_gt_{threshold:.2f}_steps"]=selected
    for key in ("approx_kl","entropy","clip_fraction","actor_grad_norm","log_ratio_min","log_ratio_max","ratio_underflow_fraction"):
        values=[number(r.get(key)) for r in rows];values=[v for v in values if v is not None]
        result[key]={"mean":None if not values else float(np.mean(values)),"median":None if not values else float(np.median(values)),
                     "p95":None if not values else float(np.quantile(values,.95)),"min":None if not values else min(values),"max":None if not values else max(values)}
    return result

def row_hash(row,observation_only=False):
    keys=("observation",) if observation_only else ("observation","reference_mean","reference_log_std")
    digest=hashlib.sha256()
    for key in keys:
        value=np.ascontiguousarray(np.asarray(row[key]));digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()

def overlap(best_rows,final_rows,observation_only=False):
    before=Counter(row_hash(r,observation_only) for r in best_rows);after=Counter(row_hash(r,observation_only) for r in final_rows)
    shared=sum((before&after).values())
    return {"best_unique_hashes":len(before),"final_unique_hashes":len(after),"shared_rows":shared,
            "best_to_final_overlap_fraction":shared/max(len(best_rows),1)}

def build_actor(config,state):
    trainer=build_modular_mappo_trainer(config,"cuda",total_sampled_steps=3_000_000)
    trainer.actor.load_state_dict(state["actor"]);trainer.actor.eval();return trainer.actor

@torch.no_grad()
def distributions(actor,observations):
    obs=torch.as_tensor(observations,dtype=torch.float32,device="cuda");means=[];logs=[]
    for chunk in obs.split(256):
        dist=actor.distribution(chunk);means.append(dist.loc.cpu());logs.append(dist.scale.log().cpu())
    return torch.cat(means).numpy(),torch.cat(logs).numpy()

def stats(values):
    x=np.asarray(values,dtype=np.float64)
    return {"mean":float(x.mean()),"median":float(np.median(x)),"p90":float(np.quantile(x,.9)),"max":float(x.max())}

def distribution_drift(ref_mean,ref_log,cur_mean,cur_log):
    delta=cur_mean-ref_mean;l2=np.linalg.norm(delta,axis=1);action=np.tanh(cur_mean)-np.tanh(ref_mean);action_l2=np.linalg.norm(action,axis=1)
    var_ref=np.exp(2*ref_log);var_cur=np.exp(2*cur_log)
    mean_component=np.sum(delta**2/(2*var_cur),axis=1)
    variance_component=np.sum(cur_log-ref_log+var_ref/(2*var_cur)-.5,axis=1)
    return {"pre_tanh_mean_L1_mean":float(np.abs(delta).sum(1).mean()),"pre_tanh_mean_L2":stats(l2),
      "deployment_action_abs_mean":float(np.abs(action).mean()),"deployment_action_L2":stats(action_l2),
      "log_std_abs_delta_mean":float(np.abs(cur_log-ref_log).mean()),"log_std_signed_delta_mean":float((cur_log-ref_log).mean()),
      "gaussian_kl_mean_shift_contribution":float(mean_component.mean()),
      "gaussian_kl_variance_contribution":float(variance_component.mean()),
      "gaussian_kl_total":float((mean_component+variance_component).mean())}

def checkpoint_bank_analysis(name,path):
    best=torch.load(path/"best_eval.pt",map_location="cuda",weights_only=False)
    final=torch.load(path/"final.pt",map_location="cuda",weights_only=False)
    config=load_config(path/"algorithm_config.yaml");best_actor=build_actor(config,best);final_actor=build_actor(config,final)
    best_state=best["milestone_aware_retention_credit_state"];final_state=final["milestone_aware_retention_credit_state"]
    turnover={"run":name,"best_step":int(best["sampled_steps"]),"final_step":int(final["sampled_steps"]),"waves":{}}
    drift={"run":name,"fixed_bank_source":"best_eval.pt MARC banks","waves":{}}
    for wave in (1,2,3):
        b=best_state["banks"][str(wave)];f=final_state["banks"][str(wave)]
        turnover["waves"][str(wave)]={"best_bank_size":len(b),"final_bank_size":len(f),
          "best_ingested_success_count":int(best_state["ingested_success_count"].get(wave,best_state["ingested_success_count"].get(str(wave),0))),
          "final_ingested_success_count":int(final_state["ingested_success_count"].get(wave,final_state["ingested_success_count"].get(str(wave),0))),
          "full_row_overlap":overlap(b,f,False),"observation_overlap":overlap(b,f,True)}
        obs=np.stack([r["observation"] for r in b]).astype(np.float32)
        mu_best,ls_best=distributions(best_actor,obs);mu_final,ls_final=distributions(final_actor,obs)
        bank_ref_mu=np.stack([r["reference_mean"] for r in b]).astype(np.float32);bank_ref_ls=np.stack([r["reference_log_std"] for r in b]).astype(np.float32)
        final_obs=np.stack([r["observation"] for r in f]).astype(np.float32)
        final_ref_mu=np.stack([r["reference_mean"] for r in f]).astype(np.float32);final_ref_ls=np.stack([r["reference_log_std"] for r in f]).astype(np.float32)
        final_cur_mu,final_cur_ls=distributions(final_actor,final_obs)
        drift["waves"][str(wave)]={
          "best_actor_to_final_actor_on_best_observations":distribution_drift(mu_best,ls_best,mu_final,ls_final),
          "best_bank_reference_to_final_actor":distribution_drift(bank_ref_mu,bank_ref_ls,mu_final,ls_final),
          "final_bank_reference_to_final_actor":distribution_drift(final_ref_mu,final_ref_ls,final_cur_mu,final_cur_ls)}
    return turnover,drift

def write_csv(path,rows):
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for checkpoint diagnostics")
    if OUTPUT.exists():raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    audit={};curve_rows=[];phase_rows=[];optimization={};turnover={};drift={}
    for name,path in RUNS.items():
        required=("algorithm_config.yaml","env_config.yaml","evaluation_history.csv","training_metrics.jsonl","optimization_metrics.jsonl","best_eval.pt","final.pt","checkpoint_3000000.pt","run_config.json","run_summary.json")
        missing=[item for item in required if not (path/item).is_file()]
        if missing:raise RuntimeError(f"{name} missing {missing}")
        best=torch.load(path/"best_eval.pt",map_location="cuda",weights_only=False);final=torch.load(path/"final.pt",map_location="cuda",weights_only=False)
        env=load_config(path/"env_config.yaml");cfg=load_config(path/"algorithm_config.yaml")
        expected_marc=name.startswith("marc");enabled=best.get("enabled_modules",[])
        if blue_counts:=env["persistent_waves"].get("blue_units_per_wave"):
            if list(blue_counts)!=[4,3,3]:raise RuntimeError(f"{name} is not 433")
        else:raise RuntimeError(f"{name} lacks explicit 433 identity")
        if int(final["sampled_steps"])!=3_000_000 or int(final["extra"]["training_seed"])!=int(name[-4:]):raise RuntimeError(f"{name} checkpoint identity mismatch")
        if expected_marc != (enabled==["actor_lr_decay","milestone_aware_retention_credit"]):raise RuntimeError(f"{name} module identity mismatch")
        if expected_marc and best["development_feature_versions"].get("milestone_aware_retention_credit")!=1:raise RuntimeError("MARC version mismatch")
        curves,summary=eval_analysis(name,path,int(best["sampled_steps"]));curve_rows.extend(curves)
        opt=read_jsonl(path/"optimization_metrics.jsonl");phase_rows.extend(phase_summary(name,opt));optimization[name]=spike_summary(opt)
        audit[name]={"path":str(path.relative_to(ROOT)).replace("\\","/"),"algorithm":"MARC-MAPPO V1" if expected_marc else "Plain MAPPO",
          "seed":int(name[-4:]),"sampled_steps":int(final["sampled_steps"]),"environment":"persistent_wave_v2 433",
          "enabled_modules":enabled,"best_checkpoint_step":int(best["sampled_steps"]),"evaluation":summary,
          "files":{item:True for item in required}}
        if expected_marc:
            turnover[name],drift[name]=checkpoint_bank_analysis(name,path)
    moving_risk=all(turnover["marc5301"]["waves"][str(w)]["full_row_overlap"]["best_to_final_overlap_fraction"]<.5 for w in (1,2,3))
    moving_confirmed=True
    for w in (1,2,3):
        rows=drift["marc5301"]["waves"][str(w)];recent=rows["final_bank_reference_to_final_actor"]["deployment_action_L2"]["mean"]
        old=rows["best_bank_reference_to_final_actor"]["deployment_action_L2"]["mean"]
        moving_confirmed &= recent < old
    ppo_distinct=optimization["marc5301"]["kl_gt_0.05_count"]>optimization["marc5302"]["kl_gt_0.05_count"]
    base_persists=audit["plain5301"]["evaluation"]["best_to_final_delta"]["AW"]<0
    labels={"retention_bank":"RETENTION_BANK_MOVING_ANCHOR_RISK" if moving_risk else "RETENTION_BANK_TURNOVER_LIMITED",
            "moving_reference":"MOVING_REFERENCE_CONFIRMED" if moving_confirmed else "MOVING_REFERENCE_NOT_CONFIRMED",
            "ppo_spikes":"PPO_UPDATE_SPIKES_ASSOCIATED" if ppo_distinct else "PPO_UPDATE_SPIKES_NOT_DISTINCTIVE",
            "base_instability":"BASE_MAPPO_INSTABILITY_PERSISTS" if base_persists else "BASE_MAPPO_INSTABILITY_NOT_OBSERVED"}
    result={"status":"MARC_433_DIAGNOSTIC_COMPLETE","training_performed":False,"core_code_modified":False,
            "runs":audit,"optimization":optimization,"labels":labels,
            "evidence_ranking":{
              "baseline_MAPPO_optimization_instability":"STRONGLY_ASSOCIATED" if base_persists else "PLAUSIBLE",
              "retention_moving_reference_failure":"SUPPORTED" if moving_risk and moving_confirmed else "PLAUSIBLE",
              "deterministic_mean_deployment_drift":"SUPPORTED",
              "later_wave_sample_imbalance":"SUPPORTED",
              "reward_problem":"NOT_SUPPORTED"},
            "marc_v2_minimal_direction":"Preserve a non-FIFO high-quality milestone reference set (quality-gated or frozen elite anchors) while retaining the current success-conditioned per-wave KL; do not change reward."}
    OUTPUT.mkdir(parents=True)
    (OUTPUT/"offline_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (OUTPUT/"checkpoint_drift.json").write_text(json.dumps(drift,indent=2),encoding="utf-8")
    (OUTPUT/"bank_turnover.json").write_text(json.dumps(turnover,indent=2),encoding="utf-8")
    write_csv(OUTPUT/"optimization_phase_summary.csv",phase_rows);write_csv(OUTPUT/"evaluation_curve_summary.csv",curve_rows)
    report=["# MARC-MAPPO V1 433 late-instability root-cause diagnostic","",
      "This is an offline association diagnostic, not a causal proof.","",
      f"- {labels['retention_bank']}",f"- {labels['moving_reference']}",f"- {labels['ppo_spikes']}",f"- {labels['base_instability']}","",
      "## Evidence ranking","",
      *[f"- {key}: **{value}**" for key,value in result["evidence_ranking"].items()],"",
      "## MARC V2 direction","",result["marc_v2_minimal_direction"]]
    (OUTPUT/"diagnostic_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2));print("MARC_433_DIAGNOSTIC_COMPLETE")

if __name__=="__main__":main()
