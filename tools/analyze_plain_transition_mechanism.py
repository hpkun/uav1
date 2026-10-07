"""Strictly offline analysis of the Plain transition-mechanism diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.plain_transition_diagnostic_common import OUTPUT_DIR,POLICY_SEEDS,stable_file_sha256


def scalar(value: str):
    if value=="" or value is None:return None
    if value in ("True","False"):return value=="True"
    try:
        number=float(value)
        return int(number) if number.is_integer() else number
    except ValueError:return value


def read_csv(path: Path) -> list[dict[str,Any]]:
    with path.open(newline="",encoding="utf-8") as stream:
        return [{k:scalar(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    if not rows:path.write_text("",encoding="utf-8");return
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def stats(values) -> dict[str,Any]:
    x=np.asarray([float(v) for v in values if v is not None],dtype=float)
    if not x.size:return {"N":0,"mean":None,"median":None,"q1":None,"q3":None}
    return {"N":int(x.size),"mean":float(x.mean()),"median":float(np.median(x)),"q1":float(np.quantile(x,.25)),"q3":float(np.quantile(x,.75))}


def rankdata(values: list[float]) -> np.ndarray:
    order=np.argsort(values);ranks=np.empty(len(values),float);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        ranks[order[i:j]]=(i+j-1)/2+1;i=j
    return ranks


def spearman(values,targets):
    pairs=[(float(x),float(y)) for x,y in zip(values,targets) if x is not None and y is not None]
    if len(pairs)<3:return None
    x,y=zip(*pairs);rx,ry=rankdata(list(x)),rankdata(list(y))
    if np.std(rx)==0 or np.std(ry)==0:return None
    return float(np.corrcoef(rx,ry)[0,1])


def paired_outcome_summary(rows, left_key, right_key):
    pairs=[(int(r[left_key]),int(r[right_key])) for r in rows if r.get(left_key) is not None and r.get(right_key) is not None]
    wins=sum(a>b for a,b in pairs);losses=sum(a<b for a,b in pairs);ties=len(pairs)-wins-losses
    return {"N":len(pairs),"wins":wins,"losses":losses,"ties":ties,
            "mean_paired_outcome_delta":float(np.mean([a-b for a,b in pairs])) if pairs else None}


def effect_label(deltas):
    values=[float(x) for x in deltas if x is not None]
    if not values:return "ENTRY_NOT_SUPPORTED"
    mean=float(np.mean(values));positive=sum(x>0 for x in values)
    if all(x>0 for x in values):return "ENTRY_SUPPORTED_STRONG"
    if mean>0 and positive>=4:return "ENTRY_SUPPORTED_DIRECTIONALLY"
    if mean>0:return "ENTRY_MIXED_POSITIVE"
    return "ENTRY_NOT_SUPPORTED"


def controller_label(deltas):
    values=[float(x) for x in deltas if x is not None]
    if len(values)>=2 and all(x>0 for x in values):return "CONTROLLER_SUPPORTED_DIRECTIONALLY"
    if values and float(np.mean(values))>0:return "CONTROLLER_MIXED_POSITIVE"
    return "CONTROLLER_NOT_SUPPORTED"


def overall_entry_label(wave2, wave3):
    supported={"ENTRY_SUPPORTED_STRONG","ENTRY_SUPPORTED_DIRECTIONALLY"}
    if wave2 in supported and wave3 in supported:return "ENTRY_OVERALL_SUPPORTED"
    if ((wave2 in supported and wave3=="ENTRY_MIXED_POSITIVE") or
        (wave3 in supported and wave2=="ENTRY_MIXED_POSITIVE")):return "ENTRY_OVERALL_MIXED"
    return "ENTRY_OVERALL_NOT_ESTABLISHED"


def overall_controller_label(wave2, wave3):
    supported="CONTROLLER_SUPPORTED_DIRECTIONALLY"
    if wave2==supported and wave3==supported:return "CONTROLLER_OVERALL_SUPPORTED"
    if ((wave2==supported and wave3=="CONTROLLER_MIXED_POSITIVE") or
        (wave3==supported and wave2=="CONTROLLER_MIXED_POSITIVE")):return "CONTROLLER_OVERALL_MIXED"
    return "CONTROLLER_OVERALL_NOT_ESTABLISHED"


def classify_overall_mechanism(entry_label, controller_label_value):
    entry_ok=entry_label=="ENTRY_OVERALL_SUPPORTED"
    controller_ok=controller_label_value=="CONTROLLER_OVERALL_SUPPORTED"
    if entry_ok and controller_ok:return "STATE_AND_CONTROLLER_BOTH"
    if entry_ok:return "STATE_QUALITY_DOMINANT"
    if controller_ok:return "CONTROLLER_QUALITY_DOMINANT"
    return "NEITHER_RESOLVED"


def paired_controller_effect(rows, next_wave):
    indexed={(int(r["evaluation_seed"]),int(r["source_policy_seed"]),int(r["continuation_policy_seed"])):int(r["next_wave_clear"])
             for r in rows if int(r["next_wave"])==next_wave}
    out=[]
    for other in (5301,5302):
        deltas=[]
        for case,source in sorted({(case,source) for case,source,_ in indexed}):
            a=indexed.get((case,source,5303));b=indexed.get((case,source,other))
            if a is not None and b is not None:deltas.append(a-b)
        out.append({"next_wave":next_wave,"contrast":f"controller5303-controller{other}","N":len(deltas),
                    "wins":sum(x>0 for x in deltas),"losses":sum(x<0 for x in deltas),"ties":sum(x==0 for x in deltas),
                    "mean_paired_outcome_delta":float(np.mean(deltas)) if deltas else None})
    return out


def paired_entry_effect(rows, next_wave):
    indexed={(int(r["evaluation_seed"]),int(r["source_policy_seed"]),int(r["continuation_policy_seed"])):int(r["next_wave_clear"])
             for r in rows if int(r["next_wave"])==next_wave}
    cases={case for case,_,_ in indexed
           if all(all((case,source,controller) in indexed for controller in POLICY_SEEDS)
                  for source in POLICY_SEEDS)}
    out=[]
    for controller in POLICY_SEEDS:
        for other in (5301,5302):
            deltas=[indexed[(case,5303,controller)]-indexed[(case,other,controller)] for case in sorted(cases)
                    if (case,5303,controller) in indexed and (case,other,controller) in indexed]
            out.append({"next_wave":next_wave,"continuation_policy":controller,"contrast":f"source5303-source{other}",
                        "N":len(deltas),"wins":sum(x>0 for x in deltas),"losses":sum(x<0 for x in deltas),
                        "ties":sum(x==0 for x in deltas),"mean_paired_outcome_delta":float(np.mean(deltas)) if deltas else None})
    return out


def case_matrix(direct):
    by={(int(r["evaluation_seed"]),int(r["policy_training_seed"])):r for r in direct};rows=[]
    for case in sorted({k[0] for k in by}):
        row={"evaluation_seed":case}
        for seed in POLICY_SEEDS:
            source=by[(case,seed)]
            for out,name in (("waves_cleared","waves"),("ground_losses","ground"),("boundary_losses","boundary"),("red_losses","red_loss"),("blue_losses","blue_loss"),("episode_return","return"),("episode_length","episode_length")):
                row[f"{name}_{seed}"]=source[out]
        rows.append(row)
    return rows


def case_counts(rows):
    result={"all_same_waves":0,"all_zero":0,"all_three":0,"5303_strictly_better_than_both":0,"5303_unique_three_wave":0,"5301_or_5302_three_but_5303_not":0}
    diffs={5301:{i:0 for i in range(-3,4)},5302:{i:0 for i in range(-3,4)}}
    disagreement={"5301_vs_5302":0,"5301_vs_5303":0,"5302_vs_5303":0}
    discriminating=[]
    for r in rows:
        a,b,c=(int(r[f"waves_{s}"]) for s in POLICY_SEEDS)
        result["all_same_waves"]+=int(a==b==c);result["all_zero"]+=int(a==b==c==0);result["all_three"]+=int(a==b==c==3)
        result["5303_strictly_better_than_both"]+=int(c>a and c>b)
        result["5303_unique_three_wave"]+=int(c==3 and a<3 and b<3)
        result["5301_or_5302_three_but_5303_not"]+=int(c<3 and (a==3 or b==3))
        disagreement["5301_vs_5302"]+=int(a!=b);disagreement["5301_vs_5303"]+=int(a!=c);disagreement["5302_vs_5303"]+=int(b!=c)
        diffs[5301][c-a]+=1;diffs[5302][c-b]+=1
        discriminating.append((max(a,b,c)-min(a,b,c),r["evaluation_seed"],a,b,c))
    result["disagreement"]=disagreement;result["5303_minus_others_distribution"]={str(k):v for s in (5301,5302) for k,v in diffs[s].items() if False}
    result["5303_minus_5301"]={str(k):v for k,v in diffs[5301].items()};result["5303_minus_5302"]={str(k):v for k,v in diffs[5302].items()}
    result["mean_paired_waves_difference"]={"5303-5301":float(np.mean([r["waves_5303"]-r["waves_5301"] for r in rows])),"5303-5302":float(np.mean([r["waves_5303"]-r["waves_5302"] for r in rows]))}
    result["most_discriminating_cases"]=[{"evaluation_seed":x[1],"waves_5301":x[2],"waves_5302":x[3],"waves_5303":x[4]} for x in sorted(discriminating,reverse=True)[:10]]
    result["5303_three_while_both_others_at_most_one"] = sum(int(r["waves_5303"]==3 and r["waves_5301"]<=1 and r["waves_5302"]<=1) for r in rows)
    return result


def survivor_summary(direct):
    rows=[]
    for seed in POLICY_SEEDS:
        rr=[r for r in direct if r["policy_training_seed"]==seed]
        for wave in (1,2,3):
            record=[r[f"wave_{wave}_red_survivors_end"] for r in rr if r[f"wave_{wave}_recorded"]]
            clear=[r[f"wave_{wave}_red_survivors_end"] for r in rr if r[f"wave_{wave}_cleared"]]
            rows.append({"policy_seed":seed,"wave":wave,"record_N":len(record),"conditional_on_record":float(np.mean(record)) if record else None,"clear_N":len(clear),"conditional_on_clear":float(np.mean(clear)) if clear else None})
    return rows


FEATURES=("red_survivor_count","red_altitude_min","red_altitude_mean","red_boundary_margin_min","red_boundary_margin_mean","red_formation_spread","red_pairwise_distance_mean","remaining_horizon","minimum_spawn_distance","minimum_red_blue_distance","red_current_time_to_ground_min")


def matched_transition(transitions):
    rows=[];summary=[]
    for nxt in (2,3):
        available={seed:{int(r["evaluation_seed"]) for r in transitions if r["source_policy_seed"]==seed and r["next_wave"]==nxt} for seed in POLICY_SEEDS}
        matched=set.intersection(*(available[s] for s in POLICY_SEEDS))
        for case in sorted(matched):
            for seed in POLICY_SEEDS:
                r=next(x for x in transitions if x["next_wave"]==nxt and x["source_policy_seed"]==seed and x["evaluation_seed"]==case)
                rows.append(r)
        for feature in FEATURES+("source_wave_ground_risk_ratio",):
            for other in (5301,5302):
                deltas=[]
                for case in matched:
                    a=next(x for x in rows if x["next_wave"]==nxt and x["evaluation_seed"]==case and x["source_policy_seed"]==5303).get(feature)
                    b=next(x for x in rows if x["next_wave"]==nxt and x["evaluation_seed"]==case and x["source_policy_seed"]==other).get(feature)
                    if a is not None and b is not None:deltas.append(float(a)-float(b))
                summary.append({"next_wave":nxt,"feature":feature,"contrast":f"5303-{other}","N":len(deltas),"mean_difference":float(np.mean(deltas)) if deltas else None,"median_difference":float(np.median(deltas)) if deltas else None,"positive_direction_fraction":float(np.mean(np.asarray(deltas)>0)) if deltas else None})
    return rows,summary


def ground_summary(risks):
    rows=[]
    for seed in POLICY_SEEDS:
        for wave in (0,1,2,3):
            rr=[r for r in risks if r["policy_training_seed"]==seed and r["wave"]==wave]
            living=sum(r["living_agent_decision_steps"] for r in rr);risk=sum(r["ground_risk_steps"] for r in rr)
            rows.append({"policy_seed":seed,"wave":"all" if wave==0 else wave,"cases":len(rr),"living_agent_decision_steps":living,"ground_risk_steps":risk,"ground_risk_ratio":risk/max(living,1),"ground_risk_activation_count":sum(r["ground_risk_activation_count"] for r in rr),"max_consecutive_ground_risk_steps":max((r["max_consecutive_ground_risk_steps"] for r in rr),default=0),"minimum_boundary_margin":min((r["minimum_boundary_margin"] for r in rr if r["minimum_boundary_margin"] is not None),default=None),"minimum_positive_time_to_boundary":min((r["minimum_positive_time_to_boundary"] for r in rr if r["minimum_positive_time_to_boundary"] is not None),default=None)})
    return rows


def death_summary(path: Path):
    events=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[]
    mapping={1:0,5:4,10:9,25:24,50:49}
    fields=("altitude","pitch","vertical_speed","action_pitch","commanded_pitch","time_to_ground")
    for seed in POLICY_SEEDS:
        ground=[e for e in events if e["policy_seed"]==seed and e["death_type"]=="ground"]
        for label,index in mapping.items():
            samples=[e["trace"][index] for e in ground if len(e["trace"])>index]
            row={"policy_seed":seed,"t_minus":label,"death_event_N":len(ground),"available_N":len(samples)}
            for field in fields:row[f"median_{field}"]=statistics.median([x[field] for x in samples if x.get(field) is not None]) if any(x.get(field) is not None for x in samples) else None
            row["guard_risk_fraction"]=float(np.mean([x["would_trigger_blue_ground_guard"] for x in samples])) if samples else None
            rows.append(row)
    return rows,events


def death_counts(events):
    rows=[]
    for seed in POLICY_SEEDS:
        for wave in (1,2,3):
            selected=[e for e in events if int(e["policy_seed"])==seed and int(e["wave"])==wave]
            rows.append({"policy_seed":seed,"wave":wave,
                         "ground_death_count":sum(e["death_type"]=="ground" for e in selected),
                         "boundary_death_count":sum(e["death_type"]=="boundary" for e in selected),
                         "combat_death_count":sum(e["death_type"]=="combat" for e in selected),
                         "simultaneous_boundary_ground_count":sum(bool(e.get("simultaneous_boundary_ground",False)) for e in selected)})
    return rows


def feature_associations(transitions):
    rows=[]
    for nxt in (2,3):
        for scope in ("pooled",*POLICY_SEEDS):
            rr=[r for r in transitions if r["next_wave"]==nxt and (scope=="pooled" or r["source_policy_seed"]==scope)]
            targets=[int(r["next_wave_clear"]) for r in rr]
            for feature in FEATURES+("source_wave_ground_risk_ratio","spawn_candidate_index","spawn_radial_angle"):
                success=[r.get(feature) for r in rr if r["next_wave_clear"]]
                failure=[r.get(feature) for r in rr if not r["next_wave_clear"]]
                ss,fs=stats(success),stats(failure)
                rows.append({"scope":scope,"next_wave":nxt,"feature":feature,"success_N":ss["N"],"success_mean":ss["mean"],"success_median":ss["median"],"success_q1":ss["q1"],"success_q3":ss["q3"],"failure_N":fs["N"],"failure_mean":fs["mean"],"failure_median":fs["median"],"failure_q1":fs["q1"],"failure_q3":fs["q3"],"spearman":spearman([r.get(feature) for r in rr],targets)})
    return rows


def survivor_probability(transitions):
    rows=[]
    for nxt in (2,3):
        for scope in ("pooled",*POLICY_SEEDS):
            rr=[r for r in transitions if r["next_wave"]==nxt and (scope=="pooled" or r["source_policy_seed"]==scope)]
            for n in (1,2,3,4):
                x=[r for r in rr if r["red_survivor_count"]==n]
                rows.append({"scope":scope,"next_wave":nxt,"survivors":n,"N":len(x),"clear_probability":float(np.mean([r["next_wave_clear"] for r in x])) if x else None})
    return rows


def continuation_matrix(rows,next_wave):
    out=[]
    for source in POLICY_SEEDS:
        for controller in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==source and r["continuation_policy_seed"]==controller]
            out.append({"state_source":source,"continuation_policy":controller,"N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None,"mean_survivors_end":float(np.mean([r["red_survivors_end"] for r in rr])) if rr else None,"mean_ground_losses":float(np.mean([r["ground_losses"] for r in rr])) if rr else None,"mean_boundary_losses":float(np.mean([r["boundary_losses"] for r in rr])) if rr else None,"mean_continuation_steps":float(np.mean([r["continuation_steps"] for r in rr])) if rr else None})
    return out


def continuation_effects(rows,next_wave):
    sources={s:{r["evaluation_seed"] for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==s} for s in POLICY_SEEDS}
    matched=set.intersection(*(sources[s] for s in POLICY_SEEDS));state=[]
    for controller in POLICY_SEEDS:
        for source in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["continuation_policy_seed"]==controller and r["source_policy_seed"]==source and r["evaluation_seed"] in matched]
            state.append({"next_wave":next_wave,"continuation_policy":controller,"state_source":source,"matched_case_N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None})
    control=[]
    for source in POLICY_SEEDS:
        for controller in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==source and r["continuation_policy_seed"]==controller]
            control.append({"next_wave":next_wave,"state_source":source,"continuation_policy":controller,"N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None})
    return state,control


def duration_summary(direct):
    rows=[]
    for seed in POLICY_SEEDS:
        rr=[r for r in direct if r["policy_training_seed"]==seed]
        for wave in (1,2,3):
            record=[r[f"wave_{wave}_duration"] for r in rr if r[f"wave_{wave}_recorded"]]
            clear=[r[f"wave_{wave}_duration"] for r in rr if r[f"wave_{wave}_cleared"]]
            rows.append({"policy_seed":seed,"wave":wave,"record_duration_mean":float(np.mean(record)) if record else None,"clear_duration_mean":float(np.mean(clear)) if clear else None,"record_N":len(record),"clear_N":len(clear)})
    return rows


def matched_ground_risk(direct, risks):
    clear={(int(r["evaluation_seed"]),int(r["policy_training_seed"]),w):bool(r[f"clear_wave_{w}"])
           for r in direct for w in (1,2)}
    risk={(int(r["evaluation_seed"]),int(r["policy_training_seed"]),int(r["wave"])):float(r["ground_risk_ratio"]) for r in risks}
    out=[]
    for wave in (1,2):
        cases=[case for case in sorted({k[0] for k in clear}) if all(clear.get((case,s,wave),False) for s in POLICY_SEEDS)]
        for other in (5301,5302):
            delta=[risk[(case,5303,wave)]-risk[(case,other,wave)] for case in cases]
            out.append({"wave":wave,"contrast":f"5303-{other}","N":len(delta),
                        "mean_difference":float(np.mean(delta)) if delta else None,
                        "median_difference":float(np.median(delta)) if delta else None,
                        "direction_fraction_5303_lower":float(np.mean([x<0 for x in delta])) if delta else None})
    return out


def ground_episode_exposure(direct, risks, events):
    deaths={(int(e["policy_seed"]),int(e["evaluation_seed"])) for e in events if e["death_type"]=="ground"}
    out=[]
    for seed in POLICY_SEEDS:
        all_rows=[r for r in risks if int(r["policy_training_seed"])==seed and int(r["wave"])==0]
        for label,flag in (("with_ground_death",True),("without_ground_death",False)):
            rows=[r for r in all_rows if (((seed,int(r["evaluation_seed"])) in deaths)==flag)]
            out.append({"policy_seed":seed,"episode_group":label,"N":len(rows),
                        "mean_ground_risk_ratio":float(np.mean([r["ground_risk_ratio"] for r in rows])) if rows else None})
    return out


def ground_classification(ground, matched, death, integrity):
    overall={int(r["policy_seed"]):float(r["ground_risk_ratio"]) for r in ground if r["wave"]=="all"}
    a=overall[5301]>overall[5303] and overall[5302]>overall[5303]
    contrasts=[r for r in matched if r["mean_difference"] is not None]
    b=sum(r["mean_difference"]<0 for r in contrasts)>=3
    precursor=[r for r in death if r["t_minus"] in (1,5) and r["guard_risk_fraction"] is not None]
    c=any(r["guard_risk_fraction"]>0 for r in precursor)
    aw={int(k):float(v["diagnostic"]["average_waves"]) for k,v in integrity["policies"].items()}
    d=overall[5303]==min(overall.values()) and aw[5303]==max(aw.values()) and any(aw[s]<aw[5303] for s in (5301,5302) if overall[s]>overall[5303])
    label=("GROUND_PRIMARY_MECHANISM_SUPPORTED" if all((a,b,c,d)) else
           "GROUND_HIGH_PRIORITY_PLAUSIBLE" if a and b else
           "GROUND_ASSOCIATED_NOT_PRIMARY" if a else "GROUND_INCONCLUSIVE")
    return {"label":label,"criteria":{"A_overall_exposure":a,"B_matched_exposure":b,"C_death_precursor":c,"D_task_association":d},
            "overall_ground_risk_ratio":overall,"matched_lower_contrasts":sum(r["mean_difference"]<0 for r in contrasts),"matched_contrast_count":len(contrasts)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--input-dir",default=str(OUTPUT_DIR));a=p.parse_args();base=Path(a.input_dir);analysis=base/"analysis";analysis.mkdir(exist_ok=True)
    direct=read_csv(base/"direct_case_results.csv");transitions=read_csv(base/"transition_states.csv");risks=read_csv(base/"ground_risk_cases.csv");continuations=read_csv(base/"continuation_results.csv")
    canonical_transitions=read_csv(base/"canonical_transition_states.csv");canonical_continuations=read_csv(base/"canonical_continuation_results.csv")
    matrix=case_matrix(direct);write_csv(analysis/"case_policy_matrix.csv",matrix)
    survivors=survivor_summary(direct);write_csv(analysis/"transition_summary.csv",survivors)
    matched,matched_summary=matched_transition(transitions);write_csv(analysis/"matched_transition_case_results.csv",matched);write_csv(analysis/"matched_transition_summary.csv",matched_summary)
    ground=ground_summary(risks);write_csv(analysis/"ground_risk_summary.csv",ground)
    death,events=death_summary(base/"death_pretrace.jsonl");write_csv(analysis/"ground_death_summary.csv",death)
    deaths_by_policy_wave=death_counts(events);write_csv(analysis/"death_counts_by_policy_wave.csv",deaths_by_policy_wave)
    associations=feature_associations(transitions);write_csv(analysis/"feature_success_association.csv",associations)
    surv_prob=survivor_probability(transitions);write_csv(analysis/"survivor_success_probability.csv",surv_prob)
    m2=continuation_matrix(continuations,2);m3=continuation_matrix(continuations,3);write_csv(analysis/"continuation_matrix_wave2.csv",m2);write_csv(analysis/"continuation_matrix_wave3.csv",m3)
    s2,c2=continuation_effects(continuations,2);s3,c3=continuation_effects(continuations,3);write_csv(analysis/"state_source_effect.csv",s2+s3);write_csv(analysis/"controller_effect.csv",c2+c3)
    duration=duration_summary(direct);write_csv(analysis/"wave_duration_summary.csv",duration)
    integrity=json.loads((base/"replay_integrity.json").read_text(encoding="utf-8"));metadata=json.loads((base/"run_metadata.json").read_text(encoding="utf-8"))
    simultaneous=sum(bool(e.get("simultaneous_boundary_ground",False)) for e in events)
    native_controller=paired_controller_effect(continuations,2)+paired_controller_effect(continuations,3)
    canonical_controller=paired_controller_effect(canonical_continuations,2)+paired_controller_effect(canonical_continuations,3)
    native_entry=paired_entry_effect(continuations,2)+paired_entry_effect(continuations,3)
    canonical_entry=paired_entry_effect(canonical_continuations,2)+paired_entry_effect(canonical_continuations,3)
    write_csv(analysis/"paired_controller_effect_native.csv",native_controller);write_csv(analysis/"paired_controller_effect_canonical.csv",canonical_controller)
    write_csv(analysis/"paired_entry_effect_native.csv",native_entry);write_csv(analysis/"paired_entry_effect_canonical.csv",canonical_entry)
    labels={}
    for mode,entry_rows,controller_rows in (("native",native_entry,native_controller),("canonical",canonical_entry,canonical_controller)):
        for wave in (2,3):
            labels[f"{mode}_entry_wave{wave}"]=effect_label([r["mean_paired_outcome_delta"] for r in entry_rows if r["next_wave"]==wave])
            labels[f"{mode}_controller_wave{wave}"]=controller_label([r["mean_paired_outcome_delta"] for r in controller_rows if r["next_wave"]==wave])
    labels["canonical_entry_overall"]=overall_entry_label(labels["canonical_entry_wave2"],labels["canonical_entry_wave3"])
    labels["canonical_controller_overall"]=overall_controller_label(labels["canonical_controller_wave2"],labels["canonical_controller_wave3"])
    labels["final"]=classify_overall_mechanism(labels["canonical_entry_overall"],labels["canonical_controller_overall"])
    supported={"ENTRY_SUPPORTED_STRONG","ENTRY_SUPPORTED_DIRECTIONALLY"}
    def same_direction(left,right):
        return (left in supported and right in supported) or left==right
    agreements=sum(same_direction(labels[f"native_entry_wave{w}"],labels[f"canonical_entry_wave{w}"]) for w in (2,3))
    consistency="CONSISTENT" if agreements==2 else "PARTIALLY_CONSISTENT" if agreements==1 else "INCONSISTENT"
    matched_ground=matched_ground_risk(direct,risks);write_csv(analysis/"matched_ground_risk_comparison.csv",matched_ground)
    episode_ground=ground_episode_exposure(direct,risks,events);write_csv(analysis/"ground_risk_by_episode_outcome.csv",episode_ground)
    ground_label=ground_classification(ground,matched_ground,death,integrity)
    entry_mag=float(np.mean([abs(r["mean_paired_outcome_delta"]) for r in canonical_entry if r["mean_paired_outcome_delta"] is not None]))
    controller_mag=float(np.mean([abs(r["mean_paired_outcome_delta"]) for r in canonical_controller if r["mean_paired_outcome_delta"] is not None]))
    final=labels["final"]
    recommended=("minimal inter-wave state-quality credit intervention" if final=="STATE_QUALITY_DOMINANT" or (final=="STATE_AND_CONTROLLER_BOTH" and entry_mag>=controller_mag) else
                 "minimal persistent safety/control stabilization intervention" if final=="CONTROLLER_QUALITY_DOMINANT" or final=="STATE_AND_CONTROLLER_BOTH" else
                 "no new training; mechanism diagnosis remains unresolved")
    result={"status":"ANALYSIS_COMPLETE","integrity":integrity,"run_metadata":metadata,"case_outcomes":case_counts(matrix),"clear_and_record_survivors":survivors,"matched_transition_summary":matched_summary,"ground_risk":ground,"matched_ground_risk":matched_ground,"ground_death_precursors":death,"ground_episode_exposure":episode_ground,"simultaneous_boundary_ground_count":simultaneous,"previous_death_precedence_bug_realized_impact":"ZERO_REALIZED_IMPACT" if simultaneous==0 else "AFFECTED_DIAGNOSTIC_DEATH_LABELS","survivor_success_probability":surv_prob,"wave_duration":duration,"native_entry_effect":{"wave2":[r for r in native_entry if r["next_wave"]==2],"wave3":[r for r in native_entry if r["next_wave"]==3]},"canonical_entry_effect":{"wave2":[r for r in canonical_entry if r["next_wave"]==2],"wave3":[r for r in canonical_entry if r["next_wave"]==3]},"native_controller_effect":{"wave2":[r for r in native_controller if r["next_wave"]==2],"wave3":[r for r in native_controller if r["next_wave"]==3]},"canonical_controller_effect":{"wave2":[r for r in canonical_controller if r["next_wave"]==2],"wave3":[r for r in canonical_controller if r["next_wave"]==3]},"mechanism_classification":labels,"ground_classification":ground_label,"native_canonical_direction_consistency":consistency,"effect_interpretation_note":"Entry effects are entry-condition effects, not PURE_RED_STATE_CAUSAL_EFFECT.","matched_future_rng_note":"matched initial future RNG stream, not event-wise coupled randomness","statistical_unit_note":"Training seed is the algorithm replication unit; 44M cases are matched mechanism diagnostics.","reward_credit_interpretation":"Mechanism labels are development diagnostics and do not by themselves establish a reward intervention.","recommended_next_experiment":recommended}
    result["ground_death_counts"]=deaths_by_policy_wave
    (analysis/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    report=["# Plain transition mechanism diagnostic closure audit","","## Protocol", "Plain MAPPO seeds 5301/5302/5303 at 3M final; deterministic 44M cases only.","","## Replay integrity",f"`{integrity['status']}`","","## Counts",f"Direct={metadata['direct_episode_count']}, native transitions={metadata['native_transition_count']}, canonical transitions={metadata['canonical_transition_count']}, native continuations={metadata['native_continuation_count']}, canonical continuations={metadata['canonical_continuation_count']}.","","## Death precedence impact",f"Count={simultaneous}; `{result['previous_death_precedence_bug_realized_impact']}`.","","## Case outcomes",json.dumps(result["case_outcomes"],indent=2),"","## Clear-conditioned survivors",json.dumps(survivors,indent=2),"","## Native entry effects",json.dumps(result["native_entry_effect"],indent=2),"","## Canonical entry effects",json.dumps(result["canonical_entry_effect"],indent=2),"","## Controller effects",json.dumps({"native":result["native_controller_effect"],"canonical":result["canonical_controller_effect"]},indent=2),"","## Native-vs-canonical consistency",consistency,"","## Ground risk and death precursor",json.dumps({"classification":ground_label,"matched":matched_ground,"precursors":death},indent=2),"","## Wave duration",json.dumps(duration,indent=2),"","## Final mechanism classification",json.dumps(labels,indent=2),"","## Reward/credit interpretation",result["reward_credit_interpretation"],"","## Next experiment",recommended,"","No training, no policy update, and no 45M use occurred. Matched future RNG is initial-stream matched, not event-wise coupled."]
    (analysis/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    artifacts={name:stable_file_sha256(base/name) for name in ("direct_case_results.csv","transition_states.csv","canonical_transition_states.csv","continuation_results.csv","canonical_continuation_results.csv","ground_risk_cases.csv","death_pretrace.jsonl")}
    artifacts["analysis/analysis.json"]=stable_file_sha256(analysis/"analysis.json")
    snapshot={"source_commit":None,"diagnostic_protocol_version":2,"checkpoint_seeds":list(POLICY_SEEDS),"evaluation_seed_range":[44000000,44000049],"counts":{"direct":metadata["direct_episode_count"],"native_transitions":metadata["native_transition_count"],"canonical_transitions":metadata["canonical_transition_count"],"native_continuations":metadata["native_continuation_count"],"canonical_continuations":metadata["canonical_continuation_count"]},"replay_integrity":integrity["status"],"conditional_metric_fix":True,"death_precedence_fix":True,"simultaneous_boundary_ground_count":simultaneous,"case_outcomes":result["case_outcomes"],"clear_conditioned_survivors":survivors,"ground_risk_summary":ground,"matched_ground_risk_summary":matched_ground,"native_entry_effect":result["native_entry_effect"],"canonical_entry_effect":result["canonical_entry_effect"],"native_controller_effect":result["native_controller_effect"],"canonical_controller_effect":result["canonical_controller_effect"],"mechanism_classification":labels,"ground_classification":ground_label,"reward_credit_interpretation":result["reward_credit_interpretation"],"recommended_next_experiment":recommended,"training":False,"policy_update":False,"future_final_45m_used":False,"input_artifact_sha256":artifacts}
    exp=ROOT/"experiments";(exp/"plain_transition_mechanism_diagnostic_result.json").write_text(json.dumps(snapshot,indent=2),encoding="utf-8")
    (exp/"plain_transition_mechanism_diagnostic_report.md").write_text((analysis/"report.md").read_text(encoding="utf-8"),encoding="utf-8")
    print(json.dumps({"status":result["status"],"replay":integrity["status"],"analysis_dir":str(analysis)},indent=2))


if __name__=="__main__":main()
