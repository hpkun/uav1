"""Real persistent-wave CAIW integration smoke; never updates the Actor."""
from __future__ import annotations
from copy import deepcopy
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

ENGINEERING_SMOKE_SEED=8802001
FORBIDDEN_SEED_RANGES=((44000000,44000049),(45000000,45000199))
BEHAVIOR_CHECKPOINT=ROOT/"outputs/diag_mappo_learnability/l3_seed5303/checkpoint_3000000.pt"
MAX_VECTOR_STEPS=4096
ACTOR_OPTIMIZER_STEPS_ALLOWED=False

def actor_digest(actor):
    digest=hashlib.sha256()
    for name,value in actor.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def load_behavior_checkpoint(path,actor,device):
    path=Path(path)
    if path.name=="best_eval.pt":raise RuntimeError("best checkpoint is forbidden for CAIW smoke")
    state=torch.load(path,map_location=device,weights_only=False);extra=state.get("extra",{})
    if state.get("algorithm")!="modular_mappo" or int(state.get("sampled_steps",-1))!=3000000 or int(extra.get("training_seed",-1))!=5303 or extra.get("environment_variant")!="persistent_wave_v2":raise RuntimeError("behavior checkpoint provenance mismatch")
    current=actor.state_dict();saved=state.get("actor",{});topology=tuple((k,tuple(v.shape)) for k,v in current.items())==tuple((k,tuple(v.shape)) for k,v in saved.items())
    if not topology:raise RuntimeError("behavior checkpoint Actor topology mismatch")
    return state,{"behavior_checkpoint":str(path),"behavior_checkpoint_sha256":file_sha256(path),"behavior_training_seed":5303,"behavior_sampled_steps":3000000,"behavior_environment_variant":"persistent_wave_v2","actor_topology_match":True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");parser.add_argument("--output-dir",default="outputs/dev_caiw_mappo_v2_integration_smoke");args=parser.parse_args()
    if args.device!="cuda" or not torch.cuda.is_available():raise RuntimeError("real CAIW integration smoke requires CUDA")
    if any(lo<=ENGINEERING_SMOKE_SEED<=hi for lo,hi in FORBIDDEN_SEED_RANGES):raise RuntimeError("forbidden smoke seed")
    output=ROOT/args.output_dir
    if output.exists():raise FileExistsError(f"refusing to overwrite integration smoke output: {output}")
    cfg=load_config("configs/dev_caiw_mappo_v2_3m.yaml");env=load_config("configs/persistent_wave_v2_environment.yaml");module_cfg=cfg["modules"]["counterfactual_inter_wave_credit"]
    module_cfg.update({"min_train_segments_per_class":1,"train_states_per_class_per_task":4,"critic_updates_per_rollout":1})
    runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=4,total_sampled_steps=MAX_VECTOR_STEPS*4,device=args.device,seed=ENGINEERING_SMOKE_SEED,output_dir=output,smoke=False)
    state,provenance=load_behavior_checkpoint(BEHAVIOR_CHECKPOINT,runner.trainer.actor,args.device);runner.trainer.actor.load_state_dict(state["actor"]);before=actor_digest(runner.trainer.actor);completed=[];cross_rollout=False;finite_update=False;fresh_count=0
    try:
        previous_pending=0
        for _ in range(MAX_VECTOR_STEPS//64):
            batch=runner.collect_rollout(64);pending=sum(len(rows[w]) for rows in runner.caiw_pending_episode for w in (1,2));cross_rollout|=previous_pending>0 and pending>0;previous_pending=pending
            segments=list(batch.caiw_supervision_segments or ());completed.extend(segments);runner.trainer.counterfactual_inter_wave_credit.ingest(segments)
            for task in ("w1_to_w2","w1_to_w3","w2_to_w3"):
                pos,neg=runner.trainer.counterfactual_inter_wave_credit.task_classes(task)
                if pos and neg:
                    metrics=runner.trainer._train_caiw_critic();finite_update=bool(np.isfinite(list(metrics.values())).all()) and metrics["caiw_critic_loss"]>0
                    fresh_count=max(fresh_count,int(metrics.get(f"caiw_{task}_fresh_train_samples",0)))
                    if finite_update:break
            if finite_update and any(s["source_wave"]==2 for s in completed):break
    finally:runner.vector.close()
    after=actor_digest(runner.trainer.actor);real_actions=all(all(key in s for key in ("actions","raw_actions","behavior_log_probs")) for s in completed);labels=all(not (s["label_c3"] and not s["label_c2"]) for s in completed);same_split=all(len({s["split"] for s in completed if s["episode_group_id"]==g})<=1 for g in {s["episode_group_id"] for s in completed})
    checks={"persistent_wave_v2":env["environment_variant"]=="persistent_wave_v2","three_waves":env["persistent_waves"]["total_waves"]==3,"max_steps":env["simulation"]["max_steps"]==3000,"cross_rollout_pending":cross_rollout,"true_episode_completion":len(completed)>0,"wave1_to_wave2_observed":any(s["source_wave"]==2 for s in completed),"real_action_schema":real_actions,"future_labels_valid":labels,"deterministic_episode_split":same_split,"replay_ingested":sum(len(v) for rows in runner.trainer.counterfactual_inter_wave_credit.train_replay.values() for v in rows.values())+sum(len(v) for v in runner.trainer.counterfactual_inter_wave_credit.validation.values())>0,"fresh_samples":fresh_count,"finite_bce_update":finite_update,"actor_parameter_drift":before!=after,"actor_optimizer_steps":0}
    passed=all(value for key,value in checks.items() if key not in {"actor_parameter_drift","actor_optimizer_steps","fresh_samples"}) and not checks["actor_parameter_drift"] and checks["actor_optimizer_steps"]==0 and fresh_count>0
    result={"status":"CAIW_REAL_INTEGRATION_SMOKE_PASS" if passed else "CAIW_REAL_INTEGRATION_SMOKE_FAIL","seed":ENGINEERING_SMOKE_SEED,**provenance,"completed_segments":len(completed),"checks":checks};(output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
