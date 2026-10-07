"""Real-environment BRSC collection smoke. It never trains the actor or evaluates a policy."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

def actor_digest(actor):
    digest=hashlib.sha256()
    for name,value in actor.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",));parser.add_argument("--max-chunks",type=int,default=160);parser.add_argument("--output-dir",default="outputs/dev_brsc_mappo_v1_integration_smoke");args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required but unavailable")
    output=ROOT/args.output_dir
    if output.exists():raise RuntimeError(f"refusing to overwrite integration smoke output: {output}")
    env=load_config("configs/persistent_wave_v2_environment.yaml");cfg=load_config("configs/dev_brsc_mappo_v1_3m.yaml")
    module_cfg=cfg["modules"]["boundary_redistributed_segment_credit"];module_cfg.update({"min_train_boundaries_per_class":1,"critic_samples_per_class_per_task":2,"critic_updates_per_rollout":1})
    runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=4,total_sampled_steps=1,device=args.device,seed=8803003,output_dir=output,smoke=False)
    checkpoint=ROOT/"outputs/diag_mappo_learnability/l3_seed5303/checkpoint_3000000.pt";state=torch.load(checkpoint,map_location=args.device,weights_only=False);runner.trainer.actor.load_state_dict(state["actor"],strict=True)
    before=actor_digest(runner.trainer.actor);recorded=0;exact_post=True;pending_seen=False;pending_crossed=False;completed=[];w1=w2=0;shape_batch=None
    original_step=runner.vector.step_batch
    def step(actions):
        result=original_step(actions);runner._brsc_smoke_last_step=result;return result
    runner.vector.step_batch=step
    original_record=runner._brsc_record_boundary
    def record(env_id,source_wave,entry_observation,entry_alive_mask,entry_remaining_horizon,spawned_next_wave,collection_sampled_steps):
        nonlocal recorded,exact_post,w1,w2
        if spawned_next_wave and int(source_wave) in (1,2):
            recorded+=1;w1+=int(source_wave==1);w2+=int(source_wave==2)
            exact_post=exact_post and np.array_equal(entry_observation,runner._brsc_smoke_last_step.transition_next_observations[env_id]) and np.array_equal(entry_alive_mask,runner._brsc_smoke_last_step.next_alive_masks[env_id])
        return original_record(env_id,source_wave,entry_observation,entry_alive_mask,entry_remaining_horizon,spawned_next_wave,collection_sampled_steps)
    runner._brsc_record_boundary=record
    try:
        for _ in range(args.max_chunks):
            had_pending=any(row[w] is not None for row in runner.brsc_pending_episode for w in (1,2));pending_seen|=had_pending
            batch=runner.collect_rollout(64);new=list(batch.brsc_supervision_boundaries or ());completed.extend(new)
            if int(np.asarray(batch.wave_transition_flags).sum())>=2:shape_batch=batch
            if had_pending and new:pending_crossed=True
            runner.trainer.boundary_redistributed_segment_credit.ingest(new)
            module=runner.trainer.boundary_redistributed_segment_credit
            trainable=any(min(map(len,module.task_classes(task)))>=module.min_train_boundaries_per_class for task in __import__("algorithm.modules",fromlist=["BRSC_TASKS"]).BRSC_TASKS)
            if trainable and recorded>0 and completed and shape_batch is not None:break
        metrics=runner.trainer._train_brsc_critic();finite_bce=bool(np.isfinite(metrics["brsc_critic_loss"]) and metrics["brsc_critic_loss"]>0)
        shape_ok=coverage_ok=False
        if shape_batch is not None:
            module=runner.trainer.boundary_redistributed_segment_credit
            for task in __import__("algorithm.modules",fromlist=["BRSC_TASKS"]).BRSC_TASKS:module.task_ready[task]=True
            device=runner.trainer.device;tt=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=device)
            advantage,active,coverage=runner.trainer._brsc_rollout_credit(shape_batch,tt(shape_batch.next_observations),tt(shape_batch.alive_masks),tt(shape_batch.next_alive_masks),tt(shape_batch.dones),tt(shape_batch.wave_indices,dtype=torch.long))
            shape_ok=advantage.shape==active.shape==shape_batch.wave_indices.shape
            coverage_values=(coverage.get("brsc_credited_transition_fraction"),coverage.get("brsc_credited_alive_agent_fraction"));coverage_ok=bool(np.isfinite(coverage_values).all() and all(0<=value<=1 for value in coverage_values))
    finally:runner.vector.close()
    after=actor_digest(runner.trainer.actor);splits={"train":sum(row["split"]=="train" for row in completed),"validation":sum(row["split"]=="validation" for row in completed)}
    checks={"environment":"persistent_wave_v2","waves":env["persistent_waves"]["total_waves"],"max_steps":env["simulation"]["max_steps"],"engineering_seed":8803003,"envs":4,"chunk_steps":64,
      "w1_to_w2_boundaries":w1,"w2_to_w3_boundaries":w2,"post_spawn_state_exact":exact_post and recorded>0,"pending_observed_at_chunk_boundary":pending_seen,"pending_completed_in_later_chunk":pending_crossed,
      "completed_boundaries":len(completed),"split_counts":splits,"finite_bce_update":finite_bce,"boundary_critic_loss":metrics["brsc_critic_loss"],
      "multi_boundary_rollout_shape":shape_ok,"coverage_metrics_finite_and_bounded":coverage_ok,
      "actor_drift":before!=after,"actor_optimizer_steps":runner.trainer.actor_update_count}
    passed=(w1>0 and checks["post_spawn_state_exact"] and pending_seen and pending_crossed and len(completed)>0 and finite_bce and shape_ok and coverage_ok and before==after and runner.trainer.actor_update_count==0)
    (output/"smoke_result.json").write_text(json.dumps({"status":"BRSC_REAL_INTEGRATION_SMOKE_PASS" if passed else "BRSC_REAL_INTEGRATION_SMOKE_FAIL",**checks},indent=2),encoding="utf-8")
    print(json.dumps({"status":"BRSC_REAL_INTEGRATION_SMOKE_PASS" if passed else "BRSC_REAL_INTEGRATION_SMOKE_FAIL",**checks},indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
