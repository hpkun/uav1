"""Three-training-seed confirmation analysis for fixed M5 and M6+M8(.03).

The module contains no training entry point.  It provides source/config
preflight, fresh deterministic evaluation, seed-level aggregation, conditional
timeout diagnostics, and report/plot generation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/"outputs"/"multiseed_confirmation"
CACHE=OUT/"evaluation_cache"
PW_ENV=ROOT/"configs"/"persistent_wave_v2_environment.yaml"
DIRECT_ENV=ROOT/"configs"/"combat_environment.yaml"
TRAINING_SEEDS=(2023,2024,2025)
M5_SEED_BASE=36_000_000
M8_PW_SEED_BASE=36_100_000
M8_DIRECT_SEED_BASE=36_200_000
EPISODES=100
FORMAL_HOLDOUT=set(range(20_000_000,20_000_200))
DEVELOPMENT_SEEDS=(set(range(35_000_000,35_000_050)) |
                   set(range(35_100_000,35_100_030)) |
                   set(range(35_200_000,35_200_030)))

CORE={"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability",
      "W3":"clear_wave_3_probability","average_waves":"average_waves_cleared",
      "return":"average_return","red_loss":"average_red_loss","blue_loss":"average_blue_loss",
      "K_L":"kill_loss_ratio","boundary":"average_red_boundary_exits",
      "ground":"average_red_ground_losses","timeout":"timeout_rate",
      "episode_length":"average_episode_length"}


def validate_confirmation_seeds(seeds) -> list[int]:
    values=[int(seed) for seed in seeds]
    if not values or len(values)!=len(set(values)): raise ValueError("confirmation seeds must be non-empty and unique")
    if set(values)&FORMAL_HOLDOUT: raise ValueError("20M formal holdout seeds are forbidden")
    if set(values)&DEVELOPMENT_SEEDS: raise ValueError("35M development screening seeds are forbidden")
    return values


def _json(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def _yaml(path:Path)->dict[str,Any]: return yaml.safe_load(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    """Convert scalar values produced by NumPy/pandas to JSON primitives."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def classify_m8_confirmation(
    better_than_control: list[bool], adaptation_supported: list[bool]
) -> str:
    """Apply the preregistered strict three-seed M8 success rules."""
    preservation_consistent = all(better_than_control)
    adaptation_count = sum(adaptation_supported)
    if preservation_consistent and adaptation_count >= 2:
        return "MULTISEED_SUPPORTED"
    if preservation_consistent:
        return "PRESERVATION_ONLY"
    if adaptation_count == 1 and adaptation_supported[0]:
        return "NOT_REPLICATED"
    return "MIXED"


def file_sha256(path:str|Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def checkpoint_source_metadata(path:str|Path)->dict[str,Any]:
    import torch
    source=Path(path).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    state=torch.load(source,map_location="cpu",weights_only=False);extra=state.get("extra",{})
    return {"path":str(source),"sha256":file_sha256(source),"sampled_steps":int(state.get("sampled_steps",0)),
        "training_seed":int(extra.get("training_seed",-1)),"environment_variant":extra.get("environment_variant","direct_v2_3"),
        "gamma":float(extra.get("training_gamma",np.nan)),"algorithm":state.get("algorithm")}


def find_direct_source(outputs:Path,training_seed:int)->Path:
    found=[]
    for directory in outputs.iterdir():
        run_path,summary_path,best=directory/"run_config.json",directory/"run_summary.json",directory/"best_eval.pt"
        if not run_path.is_file() or not summary_path.is_file() or not best.is_file():continue
        try: run,summary=_json(run_path),_json(summary_path)
        except (OSError,json.JSONDecodeError):continue
        if (run.get("algorithm")=="MAPPO" and run.get("environment_variant")=="direct_v2_3"
            and int(run.get("seed",-1))==int(training_seed) and np.isclose(float(run.get("training_gamma",np.nan)),.999)
            and int(summary.get("sampled_steps",0))==3_000_000): found.append(best)
    if len(found)!=1:raise RuntimeError(f"Direct source discovery for seed {training_seed} expected one match, found {found}")
    metadata=checkpoint_source_metadata(found[0])
    validate_source_for_seed(metadata,training_seed)
    return found[0].resolve()


def validate_source_for_seed(metadata:dict[str,Any],training_seed:int)->None:
    if metadata["algorithm"]!="MAPPO":raise RuntimeError("Direct source algorithm mismatch")
    if int(metadata["training_seed"])!=int(training_seed):raise RuntimeError("source training seed does not match adaptation training seed")
    if metadata["environment_variant"]!="direct_v2_3":raise RuntimeError("source environment variant mismatch")
    if not np.isclose(float(metadata["gamma"]),.999):raise RuntimeError("source gamma mismatch")


def validate_same_source(warm:str|Path,reference:str|Path,training_seed:int)->dict[str,Any]:
    left,right=checkpoint_source_metadata(warm),checkpoint_source_metadata(reference)
    fields=("sha256","sampled_steps","training_seed","environment_variant","gamma")
    mismatch=[field for field in fields if left[field]!=right[field]]
    if mismatch:raise RuntimeError(f"warm/reference source mismatch: {mismatch}")
    validate_source_for_seed(left,training_seed)
    return left


def resolved_configs_matched(seed:int)->bool:
    from algorithm.train_modular_mappo import load_config
    alloff=load_config(ROOT/"configs/pw_alloff_matched_1p5m.yaml")
    m5=load_config(ROOT/"configs/pw_m5_wave_balance.yaml")
    alloff["training"]["seed"]=seed;m5["training"]["seed"]=seed
    left,right=deepcopy(alloff),deepcopy(m5)
    left["modules"].pop("wave_balancing");right["modules"].pop("wave_balancing")
    if left!=right:raise RuntimeError(f"All-Off/M5 configs are not matched for seed {seed}")
    return True


def preflight(direct_2024:str|Path,direct_2025:str|Path)->dict[str,Any]:
    from algorithm.train_modular_mappo import load_config
    sources={seed:validate_same_source(path,path,seed) for seed,path in ((2024,direct_2024),(2025,direct_2025))}
    for seed in (2024,2025):resolved_configs_matched(seed)
    control=load_config(ROOT/"configs/pw_m6_screen_control_300k.yaml")
    anchor=load_config(ROOT/"configs/pw_m6_m8_anchor_c003_300k.yaml")
    if float(anchor["modules"]["policy_anchor"]["coefficient"])!=.03:raise RuntimeError("confirmation anchor coefficient must equal .03")
    for config in (control,anchor):
        if int(config["training"]["total_sampled_steps"])!=300_000:raise RuntimeError("M8 confirmation budget mismatch")
    return {"sources":sources,"m5_matched":True,"anchor_coefficient":.03}


def discover_confirmation_runs(outputs:Path=ROOT/"outputs")->dict[str,Any]:
    result={"direct":{},"alloff":{},"m5":{},"control":{},"anchor":{}}
    for seed in TRAINING_SEEDS: result["direct"][seed]=find_direct_source(outputs,seed).parent
    for directory in outputs.iterdir():
        rp,sp=directory/"run_config.json",directory/"run_summary.json"
        if not rp.is_file() or not sp.is_file():continue
        try:run,summary=_json(rp),_json(sp)
        except (OSError,json.JSONDecodeError):continue
        seed=int(run.get("seed",-1)); modules=set(run.get("enabled_modules",[]))
        if seed not in TRAINING_SEEDS or run.get("algorithm")!="modular_mappo" or run.get("environment_variant")!="persistent_wave_v2":continue
        steps=int(summary.get("sampled_steps",0)); cfg=_yaml(directory/"algorithm_config.yaml")
        if not np.isclose(float(cfg["training"]["gamma"]),.999) or int(run.get("num_envs",-1))!=24:continue
        key=None
        if steps==1_500_000 and not modules:key="alloff"
        elif steps==1_500_000 and modules=={"wave_balancing"}:key="m5"
        elif steps==300_000 and modules=={"warm_start"}:key="control"
        elif (steps==300_000 and modules=={"warm_start","policy_anchor"}
              and np.isclose(float(cfg["modules"]["policy_anchor"]["coefficient"]),.03)):key="anchor"
        if key:
            if seed in result[key]:raise RuntimeError(f"ambiguous {key} seed {seed}")
            result[key][seed]=directory
    missing={key:[s for s in TRAINING_SEEDS if s not in result[key]] for key in result}
    missing={key:value for key,value in missing.items() if value}
    if missing:raise RuntimeError(f"confirmation runs missing: {missing}")
    for seed in TRAINING_SEEDS:
        source=checkpoint_source_metadata(result["direct"][seed]/"best_eval.pt")
        validate_source_for_seed(source,seed)
        for key in ("control","anchor"):
            run=_json(result[key][seed]/"run_config.json");warm=run.get("warm_start_provenance",{})
            if warm.get("source_checkpoint_sha256")!=source["sha256"]:raise RuntimeError(f"{key} seed {seed} source hash mismatch")
            if int(warm.get("source_training_seed",-1))!=seed:raise RuntimeError(f"{key} seed {seed} source seed mismatch")
            if key=="anchor" and run.get("policy_anchor_provenance",{}).get("source_checkpoint_sha256")!=source["sha256"]:
                raise RuntimeError(f"anchor seed {seed} reference hash mismatch")
    return result


def task(name:str,checkpoint:Path,env:Path,seed_base:int,cross:bool=False)->dict[str,Any]:
    validate_confirmation_seeds(range(seed_base,seed_base+EPISODES))
    return {"name":name,"checkpoint":str(checkpoint.resolve()),"env":str(env.resolve()),
            "seed_base":seed_base,"episodes":EPISODES,"allow_cross_variant":cross}


def build_plan(runs:dict[str,Any])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    tasks=[];mapping=[]
    def add(group,seed,method,role,checkpoint,env,seed_base,cross=False):
        name=f"{group}_{seed}_{method}_{role}".replace(".","p")
        tasks.append(task(name,checkpoint,env,seed_base,cross));mapping.append({"group":group,"training_seed":seed,
            "method":method,"checkpoint_role":role,"checkpoint":str(checkpoint),"evaluation":name})
    for seed in TRAINING_SEEDS:
        for method,key in (("All-Off","alloff"),("M5","m5")):
            add("m5",seed,method,"best",runs[key][seed]/"best_eval.pt",PW_ENV,M5_SEED_BASE)
            add("m5",seed,method,"latest",runs[key][seed]/"latest.pt",PW_ENV,M5_SEED_BASE)
        source=runs["direct"][seed]/"best_eval.pt"
        add("m8_pw",seed,"Direct source","source",source,PW_ENV,M8_PW_SEED_BASE,True)
        add("m8_direct",seed,"Direct source","source",source,DIRECT_ENV,M8_DIRECT_SEED_BASE)
        for method,key in (("M6 control","control"),("M8 lambda=.03","anchor")):
            for role,name in (("latest","latest.pt"),("best","best_eval.pt")):
                add("m8_pw",seed,method,role,runs[key][seed]/name,PW_ENV,M8_PW_SEED_BASE)
                add("m8_direct",seed,method,role,runs[key][seed]/name,DIRECT_ENV,M8_DIRECT_SEED_BASE,True)
    return tasks,mapping


def _run_instrumented(spec:dict[str,Any])->dict[str,Any]:
    import tools.modular_1p5m_screening as common
    trainer,kind,env_config,metadata=common.load_policy(spec)
    records=[]
    from env.factory import make_combat_environment
    for seed in validate_confirmation_seeds(range(spec["seed_base"],spec["seed_base"]+spec["episodes"])):
        env=make_combat_environment(env_config);observation,_=env.reset(seed);alive=env.red_alive_mask.copy()
        returns=np.zeros(4,dtype=np.float64);wave=1;total=int(env_config.get("persistent_waves",{}).get("total_waves",1))
        actor_hidden=critic_hidden=None;episode_mask=np.zeros(1,dtype=np.float32)
        if kind=="modular":actor_hidden,critic_hidden=trainer.initial_hidden(1)
        while True:
            if kind=="baseline":actions=trainer.act(observation,alive,deterministic=True)
            else:
                from algorithm.modules.wave_survival_pbrs import mission_context_numpy
                context=mission_context_numpy(trainer,np.asarray([wave]),np.asarray([total]),
                    env.blue_alive_mask[None],np.asarray([env.steps]),env.max_steps)
                actions,actor_hidden=trainer.act(observation[None],alive[None],True,False,context,actor_hidden,episode_mask)
                _,critic_hidden=trainer.values_step(observation[None],alive[None],context,critic_hidden,episode_mask);actions=actions[0]
            observation,reward,terminated,truncated,info=env.step(actions);returns+=reward
            alive=np.asarray(info["red_alive_mask"],dtype=np.float32);episode_mask[:]=1
            wave=int(info.get("wave_index",1));total=int(info.get("total_waves",total))
            if terminated or truncated:break
        scalar=lambda value:value.item() if isinstance(value,np.generic) else value
        record={key:scalar(value) for key,value in info.items() if isinstance(value,(bool,str,int,float,np.generic)) or value is None}
        record.update({"seed":seed,"episode_return":float(returns.sum()),"mean_agent_episode_return":float(returns.mean()),
            "timeout":int(info.get("termination_reason")=="red_failure_timeout"),"waves_cleared":int(info.get("waves_cleared",0)),
            "total_waves":int(info.get("total_waves",1))})
        per_wave=[dict(row) for row in info.get("per_wave_metrics",[])]
        for index in (1,2,3):
            record[f"clear_wave_{index}"]=int(record["waves_cleared"]>=index)
            cleared=next((row for row in per_wave if int(row["wave_index"])==index and row.get("wave_cleared")),None)
            any_wave=next((row for row in per_wave if int(row["wave_index"])==index),None)
            record[f"time_to_clear_wave_{index}"]=int(cleared["end_step"]) if cleared else None
            record[f"time_spent_in_wave_{index}"]=int(any_wave["duration_steps"]) if any_wave else None
        record["reached_wave_2"]=int(record["waves_cleared"]>=1);record["reached_wave_3"]=int(record["waves_cleared"]>=2)
        record["episode_kill_loss_ratio"]=float(info.get("blue_losses",0))/max(float(info.get("red_losses",0)),1.)
        records.append(record)
    payload={"task":spec,"metadata":metadata,"summary":common.summarize_episodes(records),"episodes":records}
    (CACHE/f"{spec['name']}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return {"name":spec["name"],"episodes":len(records)}


def _worker(spec,cache):
    global CACHE;CACHE=Path(cache);output=CACHE/f"{spec['name']}.json"
    if output.is_file() and _json(output).get("task")==spec:return {"name":spec["name"],"cached":True}
    return _run_instrumented(spec)


def evaluate(tasks:list[dict[str,Any]],workers:int)->None:
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_worker,spec,str(CACHE)) for spec in tasks]
        for future in as_completed(futures):print(future.result(),flush=True)


def conditional_timeout_metrics(records:list[dict[str,Any]])->dict[str,Any]:
    def mean_present(key,subset=None):
        rows=records if subset is None else [row for row in records if subset(row)]
        values=[float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    result={"probability_reaching_W2":mean_present("reached_wave_2"),"probability_reaching_W3":mean_present("reached_wave_3"),
        "timeout_conditioned_reached_W2":mean_present("timeout",lambda row:bool(row["reached_wave_2"])),
        "timeout_conditioned_reached_W3":mean_present("timeout",lambda row:bool(row["reached_wave_3"])),
        "mean_time_to_clear_W1":mean_present("time_to_clear_wave_1"),"mean_time_to_clear_W2":mean_present("time_to_clear_wave_2"),
        "mean_time_spent_in_W3":mean_present("time_spent_in_wave_3")}
    for waves in range(4):result[f"episode_length_conditioned_waves_cleared_{waves}"]=mean_present("episode_length",lambda row,w=waves:int(row["waves_cleared"])==w)
    return result


def seed_level_summary(values:list[float])->dict[str,Any]:
    array=np.asarray(values,dtype=float);n=len(array);mean=float(array.mean());std=float(array.std(ddof=1)) if n>1 else None
    half=4.302652729*std/np.sqrt(n) if n==3 else None
    return {"n_training_seeds":n,"mean":mean,"std":std,"ci95_low":mean-half if half is not None else None,
            "ci95_high":mean+half if half is not None else None,"ci_note":"descriptive t-CI; n=3"}


def is_m8_primary_checkpoint(role:str)->bool:
    return str(role)=="latest"


def _payload(name):return _json(CACHE/f"{name}.json")
def _summary(mapping):
    payload=_payload(mapping["evaluation"]);s=payload["summary"]
    return {**mapping,"checkpoint_step":payload["metadata"]["sampled_steps"],"win":s["win_rate"],
            **{label:s[field] for label,field in CORE.items()}}


def _paired_delta(candidate,baseline,metrics):
    left=pd.DataFrame(candidate);right=pd.DataFrame(baseline)
    if left.seed.duplicated().any() or right.seed.duplicated().any():raise ValueError("duplicate evaluation seed")
    merged=left.merge(right,on="seed",suffixes=("_candidate","_baseline"),validate="one_to_one")
    if len(merged)!=EPISODES:raise ValueError("evaluation is not paired for all 100 seeds")
    return {label:float((merged[f"{field}_candidate"]-merged[f"{field}_baseline"]).mean()) for label,field in metrics.items()}


def build_outputs(runs,mapping)->dict[str,Any]:
    OUT.mkdir(parents=True,exist_ok=True)
    m5map=[row for row in mapping if row["group"]=="m5"]
    m5=pd.DataFrame([_summary(row) for row in m5map]);m5.to_csv(OUT/"m5_per_seed_summary.csv",index=False)
    timeout=[];delta=[]
    episode_metrics={"W3":"clear_wave_3","average_waves":"waves_cleared","return":"episode_return","red_loss":"red_losses",
        "K_L":"episode_kill_loss_ratio","boundary":"red_boundary_exits","ground":"red_ground_losses","timeout":"timeout"}
    for row in m5map:
        timeout.append({"training_seed":row["training_seed"],"method":row["method"],"checkpoint_role":row["checkpoint_role"],
                        **conditional_timeout_metrics(_payload(row["evaluation"])["episodes"])})
    for seed in TRAINING_SEEDS:
        for role in ("best","latest"):
            get=lambda method:next(row for row in m5map if row["training_seed"]==seed and row["method"]==method and row["checkpoint_role"]==role)
            d=_paired_delta(_payload(get("M5")["evaluation"])["episodes"],_payload(get("All-Off")["evaluation"])["episodes"],episode_metrics)
            for metric,value in d.items():delta.append({"row_type":"training_seed","checkpoint_role":role,"training_seed":seed,"metric":metric,"delta":value})
    for role in ("best","latest"):
        for metric in episode_metrics:
            values=[row["delta"] for row in delta if row["checkpoint_role"]==role and row["metric"]==metric]
            delta.append({"row_type":"aggregate","checkpoint_role":role,"training_seed":None,"metric":metric,**seed_level_summary(values)})
    pd.DataFrame(timeout).to_csv(OUT/"m5_timeout_diagnostics.csv",index=False)
    pd.DataFrame(delta).to_csv(OUT/"m5_seed_level_delta.csv",index=False)
    tables={}
    for group,filename in (("m8_pw","m8_per_seed_pw.csv"),("m8_direct","m8_per_seed_direct.csv")):
        frame=pd.DataFrame([_summary(row) for row in mapping if row["group"]==group]);frame.to_csv(OUT/filename,index=False);tables[group]=frame
    m8delta=[]
    metrics_pw=("W3","average_waves","return","red_loss","K_L","boundary","ground")
    metrics_direct=("win","return","red_loss","K_L","boundary","ground")
    for group,metrics in (("m8_pw",metrics_pw),("m8_direct",metrics_direct)):
        frame=tables[group]
        for seed in TRAINING_SEEDS:
            source=frame[(frame.training_seed==seed)&(frame.method=="Direct source")].iloc[0]
            for role in ("latest","best"):
                control=frame[(frame.training_seed==seed)&(frame.method=="M6 control")&(frame.checkpoint_role==role)].iloc[0]
                anchor=frame[(frame.training_seed==seed)&(frame.method=="M8 lambda=.03")&(frame.checkpoint_role==role)].iloc[0]
                for comparison,left,right in (("control-source",control,source),("anchor-source",anchor,source),("anchor-control",anchor,control)):
                    for metric in metrics:m8delta.append({"environment":group,"training_seed":seed,"checkpoint_role":role,
                        "primary":is_m8_primary_checkpoint(role),"comparison":comparison,"metric":metric,"delta":float(left[metric]-right[metric])})
    pd.DataFrame(m8delta).to_csv(OUT/"m8_seed_level_delta.csv",index=False)
    mechanism=[]
    for seed in TRAINING_SEEDS:
        directory=runs["anchor"][seed];opt=pd.read_json(directory/"optimization_metrics.jsonl",lines=True).sort_values("sampled_steps")
        pw=tables["m8_pw"][(tables["m8_pw"].training_seed==seed)&(tables["m8_pw"].method=="M8 lambda=.03")&(tables["m8_pw"].checkpoint_role=="latest")].iloc[0]
        direct=tables["m8_direct"][(tables["m8_direct"].training_seed==seed)&(tables["m8_direct"].method=="M8 lambda=.03")&(tables["m8_direct"].checkpoint_role=="latest")].iloc[0]
        source=checkpoint_source_metadata(runs["direct"][seed]/"best_eval.pt")
        mechanism.append({"training_seed":seed,"source_checkpoint":source["path"],"source_sha256":source["sha256"],
            "source_sampled_steps":source["sampled_steps"],"final_anchor_kl":float(opt.iloc[-1].anchor_kl),
            "anchor_kl_mean":float(opt.anchor_kl.mean()),"anchor_kl_p95":float(opt.anchor_kl.quantile(.95)),
            "Direct_win":direct.win,"PW_W3":pw.W3,"PW_waves":pw.average_waves,"PW_return":pw["return"]})
    pd.DataFrame(mechanism).to_csv(OUT/"m8_anchor_mechanism.csv",index=False)
    delta_frame=pd.DataFrame(delta);best_delta=delta_frame[(delta_frame.row_type=="training_seed")&(delta_frame.checkpoint_role=="best")]
    m5_means={metric:float(best_delta[best_delta.metric==metric].delta.mean()) for metric in episode_metrics}
    direction=lambda metric,value:(value<0 if metric=="red_loss" else value>0)
    consistent=sum(sum(direction(metric,value) for value in best_delta[best_delta.metric==metric].delta)>=2
                   for metric in ("average_waves","return","red_loss","K_L"))
    w3_not_worse=sum(best_delta[best_delta.metric=="W3"].delta>=0)>=2
    per_seed_m5=[]
    for seed in TRAINING_SEEDS:
        rows=best_delta[best_delta.training_seed==seed].set_index("metric").delta
        core_count=sum(direction(metric,float(rows[metric])) for metric in ("average_waves","return","red_loss","K_L"))
        per_seed_m5.append({"training_seed":seed,"core_favorable_count":core_count,"W3_not_worse":bool(rows["W3"]>=0),
                            "seed_support":bool(core_count>=3 and rows["W3"]>=0)})
    seed_support_count=sum(row["seed_support"] for row in per_seed_m5)
    if consistent>=3 and w3_not_worse and seed_support_count>=2:m5_rating="MULTISEED_SUPPORTED"
    elif seed_support_count==1 and per_seed_m5[0]["seed_support"]:m5_rating="NOT_REPLICATED"
    else:m5_rating="MIXED"
    per_seed_m8=[]
    pw_latest=tables["m8_pw"];direct_latest=tables["m8_direct"]
    for seed in TRAINING_SEEDS:
        ps=pw_latest[(pw_latest.training_seed==seed)&(pw_latest.method=="Direct source")].iloc[0]
        pa=pw_latest[(pw_latest.training_seed==seed)&(pw_latest.method=="M8 lambda=.03")&(pw_latest.checkpoint_role=="latest")].iloc[0]
        dc=direct_latest[(direct_latest.training_seed==seed)&(direct_latest.method=="M6 control")&(direct_latest.checkpoint_role=="latest")].iloc[0]
        ds=direct_latest[(direct_latest.training_seed==seed)&(direct_latest.method=="Direct source")].iloc[0]
        da=direct_latest[(direct_latest.training_seed==seed)&(direct_latest.method=="M8 lambda=.03")&(direct_latest.checkpoint_role=="latest")].iloc[0]
        preservation=bool(da.win-ds.win>=-.10);control_improvement=bool(da.win>dc.win)
        improvements=sum((pa.W3>ps.W3,pa.average_waves>ps.average_waves,pa["return"]>ps["return"],pa.red_loss<ps.red_loss,pa.K_L>ps.K_L))
        adaptation=bool(preservation and control_improvement and improvements>=3 and (pa.W3>ps.W3 or pa.average_waves>ps.average_waves))
        per_seed_m8.append({"training_seed":seed,"direct_preserved":preservation,"better_than_control":control_improvement,
                            "persistent_improvement_count":improvements,"adaptation_supported":adaptation})
    m8_rating=classify_m8_confirmation(
        [row["better_than_control"] for row in per_seed_m8],
        [row["adaptation_supported"] for row in per_seed_m8],
    )
    summary={"training_seeds":list(TRAINING_SEEDS),"statistical_unit":"training_seed","n_training_seeds":3,
        "evaluation_episodes_per_policy":100,"formal_holdout_used":False,"development_seeds_reused":False,
        "m8_primary_checkpoint":"latest/300k","anchor_coefficient":.03,"m5_rating":m5_rating,
        "m5_best_delta_means":m5_means,"m5_per_seed_decisions":per_seed_m5,
        "m8_rating":m8_rating,"m8_per_seed_decisions":per_seed_m8}
    (OUT/"multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    write_report(m5,pd.DataFrame(delta),tables,pd.DataFrame(mechanism));return summary


def write_report(m5,delta,tables,mechanism):
    lines=["# 3-Training-Seed Confirmation","","Statistical unit: training seed (n=3). Evaluation episodes are paired diagnostics, not independent training replicates.","",
        "## M5 per-seed results","",m5.to_markdown(index=False),"","## M5 seed-level deltas","",delta.to_markdown(index=False),"",
        "## M8 Persistent","",tables["m8_pw"].to_markdown(index=False),"","## M8 Direct","",tables["m8_direct"].to_markdown(index=False),"",
        "## Anchor mechanism","",mechanism.to_markdown(index=False),"","The 95% t intervals are descriptive and highly unstable at n=3."]
    (OUT/"multiseed_report.md").write_text("\n".join(lines),encoding="utf-8")


def plot_outputs():
    m5=pd.read_csv(OUT/"m5_seed_level_delta.csv");m5=m5[(m5.row_type=="training_seed")&(m5.checkpoint_role=="best")]
    for metric,name in (("W3","m5_seed_delta.png"),):
        frame=m5[m5.metric==metric];plt.figure(figsize=(5,4));plt.axhline(0,color="k",lw=1);plt.bar(frame.training_seed.astype(str),frame.delta);plt.ylabel(f"M5-AllOff delta {metric}");plt.tight_layout();plt.savefig(OUT/name,dpi=180);plt.close()
    summary=pd.read_csv(OUT/"m5_per_seed_summary.csv")
    fig,ax=plt.subplots(figsize=(6,4))
    for method in ("All-Off","M5"):
        frame=summary[summary.method==method];best=frame[frame.checkpoint_role=="best"].set_index("training_seed");latest=frame[frame.checkpoint_role=="latest"].set_index("training_seed")
        ax.plot(TRAINING_SEEDS,best.loc[list(TRAINING_SEEDS),"W3"],marker="o",label=f"{method} best");ax.plot(TRAINING_SEEDS,latest.loc[list(TRAINING_SEEDS),"W3"],marker="x",ls="--",label=f"{method} latest")
    ax.legend();ax.set(xlabel="training seed",ylabel="PW W3");fig.tight_layout();fig.savefig(OUT/"m5_best_latest_stability.png",dpi=180);plt.close(fig)
    timeout=pd.read_csv(OUT/"m5_timeout_diagnostics.csv");pivot=timeout[timeout.checkpoint_role=="best"].pivot(index="training_seed",columns="method",values="timeout_conditioned_reached_W3");pivot.plot.bar();plt.ylabel("timeout | reached W3");plt.tight_layout();plt.savefig(OUT/"m5_timeout_by_wave.png",dpi=180);plt.close()
    pw=pd.read_csv(OUT/"m8_per_seed_pw.csv");direct=pd.read_csv(OUT/"m8_per_seed_direct.csv")
    for frame,metric,name,ylabel in ((pw,"W3","m8_pw_w3_by_seed.png","PW W3"),(direct,"win","m8_direct_win_by_seed.png","Direct win")):
        part=frame[(frame.checkpoint_role.isin(["source","latest"]))].pivot(index="training_seed",columns="method",values=metric);part.plot.bar();plt.ylabel(ylabel);plt.tight_layout();plt.savefig(OUT/name,dpi=180);plt.close()
    mechanism=pd.read_csv(OUT/"m8_anchor_mechanism.csv");plt.figure(figsize=(5,4));plt.bar(mechanism.training_seed.astype(str),mechanism.final_anchor_kl);plt.ylabel("final anchor KL");plt.tight_layout();plt.savefig(OUT/"m8_anchor_kl_by_seed.png",dpi=180);plt.close()
    plt.figure(figsize=(5,4));plt.scatter(mechanism.Direct_win,mechanism.PW_W3)
    for _,row in mechanism.iterrows():plt.annotate(str(int(row.training_seed)),(row.Direct_win,row.PW_W3))
    plt.xlabel("Direct win");plt.ylabel("PW W3");plt.tight_layout();plt.savefig(OUT/"m8_preservation_adaptation.png",dpi=180);plt.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("evaluate","report","all"),default="all");parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--print-direct-source",type=int,choices=TRAINING_SEEDS);parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--direct-2024");parser.add_argument("--direct-2025");parser.add_argument("--plot-only",action="store_true");args=parser.parse_args()
    if args.print_direct_source is not None:print(find_direct_source(ROOT/"outputs",args.print_direct_source));return
    if args.preflight:
        if not args.direct_2024 or not args.direct_2025:raise ValueError("--preflight requires --direct-2024 and --direct-2025")
        print(json.dumps(preflight(args.direct_2024,args.direct_2025),indent=2));return
    if args.plot_only:plot_outputs();return
    runs=discover_confirmation_runs();tasks,mapping=build_plan(runs)
    if args.mode in ("evaluate","all"):evaluate(tasks,args.workers)
    if args.mode in ("report","all"):
        missing=[spec["name"] for spec in tasks if not (CACHE/f"{spec['name']}.json").is_file()]
        if missing:raise RuntimeError(f"missing evaluation caches: {missing}")
        build_outputs(runs,mapping);subprocess.run([sys.executable,str(Path(__file__).resolve()),"--plot-only"],check=True)


if __name__=="__main__":main()
