#!/usr/bin/env python3
"""Read-only PPO clipping-geometry audit for matched Plain/WSAI/WSMH runs."""
from __future__ import annotations
import csv,hashlib,json,math,statistics
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/clipping_geometry_audit"
SEEDS=(5301,5302,5303);TARGET=1_805_280;SOURCE_STEP=1_505_280;MAX_NORM=.5
METHODS={"Plain":"dev_pwtr_plain_seed{seed}_300k","WSAI":"dev_wsai_seed{seed}_300k","WSMH":"dev_wsmh_seed{seed}_300k"}
EXPECTED_METHOD={"Plain":"pwtr_plain_matched_control","WSAI":"wsai_mappo","WSMH":"wsmh_mappo"}
WINDOWS={"EARLY":(1_505_280,1_603_584),"MIDDLE":(1_603_584,1_701_888),"LATE":(1_701_888,1_805_280)}
COLLAPSE_WINDOWS={"PRE_COLLAPSE":(1_603_584,1_701_888),"POST_BEST":(1_701_888,1_800_192),"FINAL_TAIL":(1_800_192,1_805_280)}
EVAL_STEPS=(1_603_584,1_701_888,1_800_192,1_805_280)
TRUST_FIELDS=("approx_kl","clip_fraction","entropy","ratio_underflow_fraction","max_abs_log_ratio","log_ratio_min","log_ratio_max")

def run_dir(method,seed):return ROOT/"outputs"/METHODS[method].format(seed=seed)
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def load_json(path):return json.loads(path.read_text(encoding="utf-8"))
def load_jsonl(path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def load_csv(path):
 with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def write_csv(path,rows):
 rows=list(rows);fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def safe_div(numerator,denominator,epsilon=1e-30):return float(numerator)/max(float(denominator),epsilon)
def clip_scale_proxy(norm,max_norm=MAX_NORM):return min(1.,safe_div(max_norm,norm))
def wsai_post_pre_proxy(post,pre):return safe_div(post,pre)
def window_for_step(step,windows=WINDOWS):
 for name,(lower,upper) in windows.items():
  if lower<int(step)<=upper:return name
 return None
def exact_intersection(step_sets):return set.intersection(*(set(values) for values in step_sets))
def paired_geometry(high,low):
 ratio=safe_div(high,low);high_scale=clip_scale_proxy(high);low_scale=clip_scale_proxy(low)
 return {"preclip_ratio":ratio,"log_preclip_ratio":math.log(ratio),"clip_scale_proxy_delta":high_scale-low_scale,
         "clip_severity_ratio":safe_div(low_scale,high_scale),"scale_ratio":safe_div(high_scale,low_scale)}
def quantile(values,q):
 values=sorted(float(v) for v in values);position=(len(values)-1)*q;lower=int(math.floor(position));upper=int(math.ceil(position))
 return values[lower] if lower==upper else values[lower]*(upper-position)+values[upper]*(position-lower)
def describe(values,quantiles=(.1,.5,.9,.95)):
 values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
 if not values:return "N/A"
 out={"n":len(values),"mean":statistics.mean(values),"median":statistics.median(values),"min":min(values),"max":max(values)}
 for q in quantiles:out[f"p{int(q*100)}"]=quantile(values,q)
 return out
def fraction(rows,predicate):return sum(bool(predicate(row)) for row in rows)/len(rows) if rows else None
def plain_status(per_seed):
 flags={str(seed):(per_seed[str(seed)]["clip_pressure_fraction"]>=.80 and per_seed[str(seed)]["clip_scale_proxy"]["median"]<=.25) for seed in SEEDS}
 count=sum(flags.values());return ("STRONG" if count>=2 else "WEAK" if count==0 else "MIXED"),flags
def excess_status(paired):
 flags={str(seed):(paired[str(seed)]["median_preclip_ratio"]>=1.25 and paired[str(seed)]["median_scale_ratio"]<=.80) for seed in SEEDS}
 count=sum(flags.values());return ("STRONG" if count>=2 else "WEAK" if count==0 else "MIXED"),flags
def collapse_alignment(pre,later,final_tail_count):
 scale_drop=later["exact_scale_median"]<=.8*pre["exact_scale_median"]
 norm_rise=later["preclip_median"]>=1.25*pre["preclip_median"]
 auxiliary=(later["w1_later_median"]>=1.25*pre["w1_later_median"] or later["backbone_median"]>=1.25*pre["backbone_median"] or later["kl_gt_005"]>=pre["kl_gt_005"]+.10)
 criteria=scale_drop and norm_rise and auxiliary
 label="MIXED" if final_tail_count<=2 else ("YES" if criteria else "NO")
 return {"label":label,"criteria_met":criteria,"exact_scale_drop_20pct":scale_drop,"global_preclip_rise_25pct":norm_rise,"auxiliary_deterioration":auxiliary,"final_tail_row_count":final_tail_count}
def metric_or_na(rows,key):return describe([row.get(key) for row in rows if row.get(key) is not None])

def enrich(method,row):
 value=dict(row);pre=float(row["actor_grad_norm"] if method=="Plain" else row[f"{method.lower()}_global_actor_grad_norm_preclip"])
 value.update({"_preclip":pre,"_pressure":float(pre>MAX_NORM),"_scale_proxy":clip_scale_proxy(pre),"_over_threshold":pre/MAX_NORM})
 if method=="WSMH":
  exact=float(row["wsmh_global_clip_scale"]);value["_exact_scale"]=exact;value["_proxy_exact_gap"]=value["_scale_proxy"]-exact
  names=("backbone","logstd","mean1","mean2","mean3");keys=("wsmh_shared_backbone_grad_norm_preclip","wsmh_shared_logstd_grad_norm_preclip","wsmh_wave1_mean_grad_norm_preclip","wsmh_wave2_mean_grad_norm_preclip","wsmh_wave3_mean_grad_norm_preclip")
  components={name:float(row[key]) for name,key in zip(names,keys)};total=sum(components.values());heads=math.sqrt(sum(components[name]**2 for name in ("mean1","mean2","mean3")))
  value.update({f"_{name}":number for name,number in components.items()});value["_shared_proxy"]=math.hypot(components["backbone"],components["logstd"]);value["_heads_proxy"]=heads
  value["_w1_later_ratio"]=safe_div(components["mean1"],math.hypot(components["mean2"],components["mean3"]));value["_backbone_heads_ratio"]=safe_div(components["backbone"],heads)
  for name,number in components.items():value[f"_{name}_fraction_proxy"]=safe_div(number,total)
 elif method=="WSAI":value["_post_pre_proxy"]=wsai_post_pre_proxy(float(row["wsai_global_actor_grad_norm_postclip"]),pre)
 return value

def method_summary(rows,method):
 result={"row_count":len(rows),"preclip_norm":describe([r["_preclip"] for r in rows]),"clip_pressure_fraction":statistics.mean(r["_pressure"] for r in rows),"clip_scale_proxy":describe([r["_scale_proxy"] for r in rows]),"preclip_over_threshold":describe([r["_over_threshold"] for r in rows])}
 if method=="WSMH":result.update({"exact_mean_minibatch_clip_scale":describe([r["_exact_scale"] for r in rows]),"proxy_exact_gap":{"summary":describe([r["_proxy_exact_gap"] for r in rows]),"max_abs":max(abs(r["_proxy_exact_gap"]) for r in rows)}})
 if method=="WSAI":result["post_pre_ratio_proxy"]=describe([r["_post_pre_proxy"] for r in rows])
 return result
def trust_summary(rows):
 result={key:metric_or_na(rows,key) for key in TRUST_FIELDS}
 if result["approx_kl"]!="N/A":
  result.update({"fraction_kl_gt_003":fraction(rows,lambda r:float(r["approx_kl"])>.03),"fraction_kl_gt_005":fraction(rows,lambda r:float(r["approx_kl"])>.05),"fraction_kl_gt_010":fraction(rows,lambda r:float(r["approx_kl"])>.10)})
 return result
def component_summary(rows):
 keys=("_backbone","_logstd","_mean1","_mean2","_mean3","_shared_proxy","_heads_proxy","_w1_later_ratio","_backbone_heads_ratio","_backbone_fraction_proxy","_logstd_fraction_proxy","_mean1_fraction_proxy","_mean2_fraction_proxy","_mean3_fraction_proxy")
 return {key.removeprefix("_"):describe([row[key] for row in rows]) for key in keys}
def collapse_window_summary(rows):
 return {"row_count":len(rows),"preclip":describe([r["_preclip"] for r in rows]),"preclip_median":statistics.median(r["_preclip"] for r in rows),"exact_scale":describe([r["_exact_scale"] for r in rows]),"exact_scale_median":statistics.median(r["_exact_scale"] for r in rows),"common_scale_proxy":describe([r["_scale_proxy"] for r in rows]),"backbone":describe([r["_backbone"] for r in rows]),"backbone_median":statistics.median(r["_backbone"] for r in rows),"logstd":describe([r["_logstd"] for r in rows]),"mean1":describe([r["_mean1"] for r in rows]),"mean2":describe([r["_mean2"] for r in rows]),"mean3":describe([r["_mean3"] for r in rows]),"w1_to_later":describe([r["_w1_later_ratio"] for r in rows]),"w1_later_median":statistics.median(r["_w1_later_ratio"] for r in rows),"backbone_to_heads":describe([r["_backbone_heads_ratio"] for r in rows]),"kl_gt_005":fraction(rows,lambda r:float(r.get("approx_kl",0))>.05),"trust_region":trust_summary(rows)}

def validate_protocol():
 report={};trainer_hashes=set()
 for method in METHODS:
  for seed in SEEDS:
   path=run_dir(method,seed);key=f"{method}_seed{seed}"
   for required in ("run_config.json","run_summary.json","branch_from.json","algorithm_config.yaml","evaluation_history.csv","optimization_metrics.jsonl","latest.pt","final.pt"):
    if not (path/required).is_file():raise RuntimeError(f"{key} missing {required}")
   run=load_json(path/"run_config.json");branch=load_json(path/"branch_from.json");config=yaml.safe_load((path/"algorithm_config.yaml").read_text(encoding="utf-8"));training=config["training"]
   source=ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt";source_sha=sha256(source)
   evaluations=load_csv(path/"evaluation_history.csv");endpoint=[row for row in evaluations if int(row["sampled_steps"])==TARGET]
   checks={"training_seed":int(run["seed"])==seed,"development_method":run["development_method"]==EXPECTED_METHOD[method],"parent_source":int(branch["parent_sampled_steps"])==SOURCE_STEP and int(branch["source_training_seed"])==seed and branch["parent_checkpoint_sha256"]==source_sha,"target":int(run["total_sampled_steps"])==TARGET,"evaluation":len(endpoint)==1 and int(endpoint[0]["evaluation_episodes"])==50 and int(endpoint[0]["evaluation_seed_base"])==44_000_000 and int(endpoint[0]["evaluation_seed_end"])==44_000_049,"uses_45m":all(int(r["evaluation_seed_base"])<45_000_000 for r in evaluations),"environment":run["environment_variant"]=="persistent_wave_v2" and int(run["effective_training_total_waves"])==3 and int(run["effective_training_max_steps"])==3000,"num_envs":int(run["num_envs"])==24,"ppo":int(training["rollout_steps"])==256 and int(training["ppo_epochs"])==10 and int(training["minibatch_size"])==512 and float(training["max_grad_norm"])==.5,"actor_lr":all(float(row["actor_learning_rate"])==1e-4 for row in load_jsonl(path/"optimization_metrics.jsonl"))}
   if not all(checks.values()):raise RuntimeError(f"protocol failure {key}: {checks}")
   manifest={item["path"]:item["sha256"] for item in run["runtime_source_manifest_files"]};trainer_hashes.add(manifest["algorithm/modular_mappo/trainer.py"])
   report[key]={"status":"PASS","checks":checks,"modular_mappo_impl_version":run["modular_mappo_impl_version"],"runtime_trainer_sha256":manifest["algorithm/modular_mappo/trainer.py"],"parent_checkpoint_sha256":source_sha}
 report["summary"]={"status":"PASS","runs":9,"runtime_trainer_hash_count":len(trainer_hashes),"all_modular_impl_version_2":all(v.get("modular_mappo_impl_version")==2 for k,v in report.items() if k!="summary")}
 return report

def metric_semantics():
 source=(ROOT/"algorithm/modular_mappo/trainer.py").read_text(encoding="utf-8");evidence={
  "plain_clip_return_assigned":("ag=nn.utils.clip_grad_norm_(self.actor.trainable_policy_parameters(),self.max_grad_norm)" in source),
  "plain_row_records_actor_grad_norm":('"actor_grad_norm":float(ag)' in source),
  "aggregate_nonweighted_values_use_row_mean":('result[key]=float(np.sum(values*counts)/total) if key in weighted else float(values.mean())' in source),
  "wsmh_preclip_computed_before_clip":("global_pre=self._gradient_norm(all_parameters);clip_scale=min(1.,self.max_grad_norm/global_pre)" in source),
  "wsmh_single_global_clip":("ag=nn.utils.clip_grad_norm_(all_parameters,self.max_grad_norm)" in source),
  "wsai_pre_and_post_recorded":('"wsai_global_actor_grad_norm_preclip":global_pre' in source and '"wsai_global_actor_grad_norm_postclip":global_post' in source)}
 # Historical manifests retain hashes but not historical source bodies. Additive
 # implementation version 2 metadata supports compatibility, not strict proof.
 return {"current_source_evidence":evidence,"plain_actor_grad_semantics_verified":False,"historical_compatibility_assumption":True,"reason":"All Plain runs declare modular_mappo_impl_version=2 and actor_grad_norm is numerically compatible with the current pre-clip contract, but their historical trainer SHA differs and no archived source body was found. Strict historical source-level proof is unavailable.","plain_actor_grad_norm_interpretation":"conditional on the version-2 compatibility assumption: row mean of per-minibatch pre-clipping total Actor norm","wsmh_global_actor_grad_norm_preclip":"row mean of per-minibatch exact pre-clip global norm","wsmh_global_clip_scale":"row mean of exact per-minibatch min(1,0.5/global_preclip)","wsai_scale_quantities":"row-level proxies only"}

def existing_conflict_evidence(protocol):
 manifest_path=ROOT/"outputs/wave_gradient_conflict_audit/audit_manifest.json";summary_path=ROOT/"outputs/wave_gradient_conflict_audit/wave_gradient_conflict_summary.json"
 if not manifest_path.is_file() or not summary_path.is_file():return {"status":"DIRECT_CROSS_WAVE_GRADIENT_COSINE = UNAVAILABLE"}
 manifest=load_json(manifest_path);selected=[row for row in manifest.get("selected_checkpoints",[]) if int(row.get("checkpoint_step",-1))==SOURCE_STEP]
 matched=len(selected)==3 and {int(row["training_seed"]) for row in selected}==set(SEEDS)
 parent_match=matched and all(row["checkpoint_sha256"]==protocol[f"Plain_seed{row['training_seed']}"]["parent_checkpoint_sha256"] for row in selected)
 return {"status":"AVAILABLE_AT_SHARED_SOURCE_BOUNDARY_ONLY" if parent_match else "DIRECT_CROSS_WAVE_GRADIENT_COSINE = UNAVAILABLE","provenance_match":parent_match,"training_seeds":list(SEEDS),"sampled_step":SOURCE_STEP,"environment_config_sha256":manifest.get("environment_config_sha256"),"scope":"full Actor cross-wave cosine at the shared parent only; no mean-head/backbone decomposition and no coverage of the 300k continuation","usable_for_mechanism":"auxiliary, not direct identification of WSMH clipping"}

def main():
 if OUT.exists():raise FileExistsError(f"refusing to overwrite {OUT}")
 protocol=validate_protocol();semantics=metric_semantics();data={};schema={};grid={};window_rows=[];component_csv=[];paired_csv=[];trust_csv=[];alignment=[]
 for method in METHODS:
  data[method]={};schema[method]={}
  for seed in SEEDS:
   raw=load_jsonl(run_dir(method,seed)/"optimization_metrics.jsonl");rows=[enrich(method,row) for row in raw];data[method][seed]=rows
   schema[method][str(seed)]={"field_count":len(raw[0]),"fields":sorted(raw[0]),"required_metrics_available":{key:all(key in row for row in raw) for key in TRUST_FIELDS}}
   steps=[int(r["sampled_steps"]) for r in rows];grid[f"{method}_seed{seed}"]={"row_count":len(rows),"first_sampled_steps":steps[0],"last_sampled_steps":steps[-1],"step_increment_set":sorted(set(b-a for a,b in zip(steps,steps[1:])))}
 for seed in SEEDS:
  sets=[{int(r["sampled_steps"]) for r in data[m][seed]} for m in METHODS];intersection=exact_intersection(sets);same=all(values==sets[0] for values in sets)
  grid[f"seed{seed}_paired"]={"paired_step_grid":same,"intersection_count":len(intersection),**{f"missing_{m}":sorted(set.union(*sets)-sets[i]) for i,m in enumerate(METHODS)}}
  for high in ("WSMH","WSAI"):
   h={int(r["sampled_steps"]):r for r in data[high][seed]};p={int(r["sampled_steps"]):r for r in data["Plain"][seed]}
   for step in sorted(intersection):paired_csv.append({"seed":seed,"method":high,"sampled_steps":step,**paired_geometry(h[step]["_preclip"],p[step]["_preclip"])})
 summaries={m:{str(s):method_summary(data[m][s],m) for s in SEEDS} for m in METHODS}
 paired={}
 for method in ("WSMH","WSAI"):
  paired[method]={}
  for seed in SEEDS:
   rows=[r for r in paired_csv if r["method"]==method and r["seed"]==seed];paired[method][str(seed)]={"n_exact_paired_steps":len(rows),"median_preclip_ratio":statistics.median(r["preclip_ratio"] for r in rows),"median_scale_ratio":statistics.median(r["scale_ratio"] for r in rows),"preclip_ratio":describe([r["preclip_ratio"] for r in rows]),"scale_ratio":describe([r["scale_ratio"] for r in rows]),"clip_severity_ratio":describe([r["clip_severity_ratio"] for r in rows])}
 windowed={};components={};trust={}
 for method in METHODS:
  windowed[method]={};trust[method]={}
  for seed in SEEDS:
   windowed[method][str(seed)]={};trust[method][str(seed)]=trust_summary(data[method][seed]);trust_csv.append({"method":method,"seed":seed,**{f"{k}_{stat}":v.get(stat) if isinstance(v,dict) else v for k,v in trust[method][str(seed)].items() for stat in (("median","p95","max") if isinstance(v,dict) else ("value",))}})
   for name,bounds in WINDOWS.items():
    rows=[r for r in data[method][seed] if bounds[0]<int(r["sampled_steps"])<=bounds[1]];summary=method_summary(rows,method);windowed[method][str(seed)][name]=summary
    window_rows.append({"method":method,"seed":seed,"window":name,"row_count":len(rows),"preclip_mean":summary["preclip_norm"]["mean"],"preclip_median":summary["preclip_norm"]["median"],"preclip_p95":summary["preclip_norm"]["p95"],"clip_pressure_fraction":summary["clip_pressure_fraction"],"scale_proxy_mean":summary["clip_scale_proxy"]["mean"],"scale_proxy_median":summary["clip_scale_proxy"]["median"],"exact_scale_mean":summary.get("exact_mean_minibatch_clip_scale",{}).get("mean") if isinstance(summary.get("exact_mean_minibatch_clip_scale"),dict) else None})
    if method=="WSMH":
     comp=component_summary(rows);components.setdefault(str(seed),{})[name]=comp;component_csv.append({"seed":seed,"window":name,"row_count":len(rows),**{f"{k}_median":v["median"] for k,v in comp.items()}})
 collapse={};seed=5302
 for name,bounds in COLLAPSE_WINDOWS.items():collapse[name]=collapse_window_summary([r for r in data["WSMH"][seed] if bounds[0]<int(r["sampled_steps"])<=bounds[1]])
 later_rows=[r for name in ("POST_BEST","FINAL_TAIL") for r in data["WSMH"][seed] if COLLAPSE_WINDOWS[name][0]<int(r["sampled_steps"])<=COLLAPSE_WINDOWS[name][1]];later=collapse_window_summary(later_rows);alignment_label=collapse_alignment(collapse["PRE_COLLAPSE"],later,collapse["FINAL_TAIL"]["row_count"]);collapse["POST_BEST_PLUS_FINAL_TAIL"]=later;collapse["alignment"]=alignment_label
 eval_data={}
 for method in METHODS:
  eval_data[method]={}
  for seed in SEEDS:
   evaluations={int(r["sampled_steps"]):r for r in load_csv(run_dir(method,seed)/"evaluation_history.csv")};optimizations={int(r["sampled_steps"]):r for r in data[method][seed]};eval_data[method][str(seed)]={}
   for step in EVAL_STEPS:
    if step not in evaluations:raise RuntimeError(f"missing exact evaluation {method} seed{seed} step{step}")
    row=evaluations[step];w1=float(row["clear_wave_1_probability"]);w2=float(row["clear_wave_2_probability"]);w3=float(row["clear_wave_3_probability"])
    item={"AW":float(row["average_waves_cleared"]),"W1":w1,"W2":w2,"W3":w3,"Q2":None if w1==0 else w2/w1,"Q3":None if w2==0 else w3/w2,"Return":float(row["average_return"]),"Boundary":float(row["average_red_boundary_exits"]),"Ground":float(row["average_red_ground_losses"]),"optimization_exact_match":step in optimizations}
    eval_data[method][str(seed)][str(step)]=item;alignment.append({"method":method,"seed":seed,"sampled_steps":step,**item,"preclip_norm":optimizations.get(step,{}).get("_preclip"),"clip_scale_proxy":optimizations.get(step,{}).get("_scale_proxy"),"wsmh_exact_clip_scale":optimizations.get(step,{}).get("_exact_scale")})
 expected_5302={1_603_584:(.88,None),1_701_888:(1.28,0.),1_800_192:(1.10,.38),1_805_280:(.56,1.88)};verified={str(step):{"AW":eval_data["WSMH"]["5302"][str(step)]["AW"],"Boundary":eval_data["WSMH"]["5302"][str(step)]["Boundary"]} for step in EVAL_STEPS};collapse["evaluation_reverification"]=verified
 plain_label,plain_flags=plain_status(summaries["Plain"]);wsmh_label,wsmh_flags=excess_status(paired["WSMH"]);wsai_label,wsai_flags=excess_status(paired["WSAI"])
 decision=("CASE_C_BOTH" if plain_label=="STRONG" and wsmh_label=="STRONG" else "CASE_A_GLOBAL_PPO_CLIPPING" if plain_label=="STRONG" else "CASE_B_ARCHITECTURE_INDUCED_CLIPPING" if wsmh_label=="STRONG" else "CASE_D_CLIPPING_NOT_PRIMARY_SIGNAL")
 next_class={"CASE_A_GLOBAL_PPO_CLIPPING":"test baseline/global PPO update geometry before more wave-specific architecture","CASE_B_ARCHITECTURE_INDUCED_CLIPPING":"a clipping-decoupling causal control is justified","CASE_C_BOTH":"separate baseline clipping severity from architecture-induced excess clipping with a minimal factorial control","CASE_D_CLIPPING_NOT_PRIMARY_SIGNAL":"do not spend the next experiment on clipping without stronger evidence"}[decision]
 conflict=existing_conflict_evidence(protocol)
 limitations=["training seed is the replication unit (n=3); optimization rows are temporally autocorrelated diagnostics, not independent replicates","Plain historical actor_grad_norm source semantics are compatibility-supported but not strictly source-body verified","row-level common clipping scales are proxies; only WSMH stores the mean exact minibatch coefficient","component norm fractions are descriptive proxies, not a gradient-energy decomposition","seed5302 FINAL_TAIL contains only %d row(s), so temporal alignment is not inferential"%collapse["FINAL_TAIL"]["row_count"],"gradient-cosine evidence covers the shared 1,505,280 source boundary only, not the complete continuations"]
 report={"protocol_validation":protocol,"metric_semantics":semantics,"step_grid_validation":grid,"plain_per_seed":summaries["Plain"],"wsmh_per_seed":summaries["WSMH"],"wsai_per_seed":summaries["WSAI"],"paired_wsmh_vs_plain":paired["WSMH"],"paired_wsai_vs_plain":paired["WSAI"],"windowed_summary":windowed,"wsmh_component_geometry":components,"seed5302_collapse_audit":collapse,"ppo_trust_region_diagnostics":trust,"evaluation_alignment":eval_data,"existing_direct_gradient_conflict_evidence":conflict,"structural_geometry_explanation":{"plain":"g_mean=g1+g2+g3; squared norm includes 2*gi^T*gj cross terms","wsmh":"block norm squared=||g1||^2+||g2||^2+||g3||^2","inference":"negative head-gradient cosine could mechanically increase the separated block norm, but this is mechanistically plausible and not directly identified by the available full-Actor source-boundary evidence"},"PLAIN_GLOBAL_CLIP_STATUS":plain_label,"plain_seed_labels":plain_flags,"WSMH_EXCESS_CLIPPING_VS_PLAIN":wsmh_label,"wsmh_seed_labels":wsmh_flags,"WSAI_EXCESS_CLIPPING_VS_PLAIN":wsai_label,"wsai_seed_labels":wsai_flags,"SEED5302_COLLAPSE_ALIGNMENT":alignment_label["label"],"NEXT_OPTIMIZATION_DIAGNOSIS":decision,"evidence_based_next_experiment_class":next_class,"replication_unit":"training_seed","n":3,"limitations":limitations,"training_run":False,"evaluation_run":False,"cuda_used":False,"uses_45m":False}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8");(OUT/"protocol_validation.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8");(OUT/"optimization_schema.json").write_text(json.dumps(schema,indent=2),encoding="utf-8");write_csv(OUT/"paired_step_clipping.csv",paired_csv);write_csv(OUT/"window_clipping_summary.csv",window_rows);write_csv(OUT/"wsmh_component_geometry.csv",component_csv);write_csv(OUT/"ppo_trust_region_summary.csv",trust_csv);write_csv(OUT/"clipping_eval_alignment.csv",alignment)
 decision_text=f"PLAIN_GLOBAL_CLIP_STATUS={plain_label}\nWSMH_EXCESS_CLIPPING_VS_PLAIN={wsmh_label}\nWSAI_EXCESS_CLIPPING_VS_PLAIN={wsai_label}\nSEED5302_COLLAPSE_ALIGNMENT={alignment_label['label']}\nNEXT_OPTIMIZATION_DIAGNOSIS={decision}\nNEXT_EXPERIMENT_CLASS={next_class}\n"
 (OUT/"decision_support.txt").write_text(decision_text,encoding="utf-8");print(json.dumps({"status":"CLIPPING_GEOMETRY_AUDIT_COMPLETE","plain":plain_label,"wsmh_excess":wsmh_label,"wsai_excess":wsai_label,"collapse_alignment":alignment_label["label"],"decision":decision,"output":str(OUT)},indent=2))
if __name__=="__main__":main()
