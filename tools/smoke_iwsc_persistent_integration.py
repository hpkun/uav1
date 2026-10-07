"""Real persistent-wave lifecycle smoke. No PPO/Actor optimization is performed."""
from __future__ import annotations
from copy import deepcopy
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

ENV_CONFIG_PATH="configs/persistent_wave_v2_environment.yaml"
ALGORITHM_CONFIG_PATH="configs/dev_iwsc_mappo_3m.yaml"
BEHAVIOR_CHECKPOINT_CANDIDATES=("outputs/diag_mappo_learnability/l3_seed5303/final.pt",
                                "outputs/diag_mappo_learnability/l3_seed5303/latest.pt")
ENGINEERING_SMOKE_SEED=8801000
NUM_ENVS=4
ROLLOUT_STEPS=64
MAX_VECTOR_STEPS=4096
ACTOR_OPTIMIZER_STEPS_ALLOWED=False
FORBIDDEN_SEED_RANGES=((44000000,44000049),(45000000,45000199))

def select_behavior_checkpoint():
    for relative in BEHAVIOR_CHECKPOINT_CANDIDATES:
        path=ROOT/relative
        if path.exists():
            state=torch.load(path,map_location="cpu",weights_only=False)
            extra=state.get("extra",{})
            if int(state.get("sampled_steps",-1))!=3000000 or int(extra.get("training_seed",-1))!=5303:
                raise RuntimeError("IWSC_REAL_ENV_SMOKE_REQUIRES_PLAIN_5303_3M_CHECKPOINT")
            return path,state
    raise RuntimeError("IWSC_REAL_ENV_SMOKE_REQUIRES_PLAIN_5303_3M_CHECKPOINT")

def validate_segment(segment):
    wave=int(segment["credit_wave"]);cleared=int(segment["episode_waves_cleared"])
    expected=((int(cleared>=2)+int(cleared>=3))/2 if wave==1 else int(cleared>=3))
    if float(segment["target"])!=float(expected):raise RuntimeError("real IW target mismatch")
    if len(segment["horizons"])>64:raise RuntimeError("real IW segment cap exceeded")
    indices=np.asarray(segment["sample_indices"]);original=int(segment["original_state_count"])
    if indices[0]!=0 or indices[-1]!=original-1:raise RuntimeError("real IW segment did not preserve endpoints")
    if ((wave==1 and cleared>=1) or (wave==2 and cleared>=2)) and not np.asarray(segment["boundary_flags"]).any():
        raise RuntimeError("real IW transition boundary was not preserved")
    return expected

def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for the real IWSC integration smoke")
    checkpoint,state=select_behavior_checkpoint()
    env=yaml.safe_load((ROOT/ENV_CONFIG_PATH).read_text(encoding="utf-8"));algorithm=load_config(ALGORITHM_CONFIG_PATH)
    if env.get("environment_variant")!="persistent_wave_v2" or env["persistent_waves"]["total_waves"]!=3 or env["simulation"]["max_steps"]!=3000:
        raise RuntimeError("real smoke environment contract mismatch")
    if any(low<=ENGINEERING_SMOKE_SEED<=high for low,high in FORBIDDEN_SEED_RANGES):raise RuntimeError("forbidden evaluation seed")
    with tempfile.TemporaryDirectory(prefix="iwsc_real_smoke_") as output:
      runner=ModularMAPPOTrainingRunner(env,algorithm,num_envs=NUM_ENVS,total_sampled_steps=MAX_VECTOR_STEPS*NUM_ENVS,
          device="cuda",seed=ENGINEERING_SMOKE_SEED,output_dir=output,smoke=False)
      try:
        if set(state["actor"])!=set(runner.trainer.actor.state_dict()):raise RuntimeError("Plain/IWSC Actor topology mismatch")
        runner.trainer.actor.load_state_dict(state["actor"],strict=True)
        actor_before={k:v.detach().clone() for k,v in runner.trainer.actor.state_dict().items()}
        previous_counts=None;previous_episode_indices=None;cross=False;w12=w23=0;all_segments=[];calls=0
        replay_before={str(w):len(runner.trainer.inter_wave_credit.replay[w]) for w in (1,2)}
        while runner.trainer.vector_steps<MAX_VECTOR_STEPS:
            batch=runner.collect_rollout(ROLLOUT_STEPS);calls+=1
            flags=np.asarray(batch.wave_transition_flags)>0.5;waves=np.asarray(batch.wave_indices)
            w12+=int((flags&(waves==1)).sum());w23+=int((flags&(waves==2)).sum())
            counts=np.asarray([[len(rows[w]) for w in (1,2)] for rows in runner.iw_pending_episode])
            episodes=runner.vector.episode_indices.copy()
            if previous_counts is not None:
                same=episodes==previous_episode_indices
                cross |= bool(np.any((counts>previous_counts)&same[:,None]&(previous_counts>0)))
            previous_counts=counts;previous_episode_indices=episodes
            segments=list(batch.iw_supervision_segments or [])
            for segment in segments:validate_segment(segment)
            all_segments.extend(segments);runner.trainer.inter_wave_credit.ingest(segments)
            groups={int(x["episode_group_id"]) for x in all_segments if int(x["credit_wave"])==1 and np.asarray(x["boundary_flags"]).any()}
            paired_groups={int(x["episode_group_id"]) for x in all_segments if int(x["credit_wave"])==2}
            if cross and w12>0 and bool(groups&paired_groups):break
        completed1=[x for x in all_segments if int(x["credit_wave"])==1];completed2=[x for x in all_segments if int(x["credit_wave"])==2]
        boundary_groups={int(x["episode_group_id"]) for x in completed1 if np.asarray(x["boundary_flags"]).any()}
        wave2_groups={int(x["episode_group_id"]) for x in completed2}
        lifecycle_complete=bool(boundary_groups&wave2_groups)
        if not (cross and w12>0 and lifecycle_complete):
            pending=[[len(rows[w]) for w in (1,2)] for rows in runner.iw_pending_episode]
            print(json.dumps({"result":"IWSC_REAL_ENV_SMOKE_INCOMPLETE","episodes_completed":runner.completed_episode_count,
              "wave1_to_wave2_count":w12,"wave2_to_wave3_count":w23,"pending_counts":pending,
              "vector_steps":runner.trainer.vector_steps},indent=2));raise SystemExit(2)
        critic=runner.trainer._train_iw_critic()
        replay_after={str(w):len(runner.trainer.inter_wave_credit.replay[w]) for w in (1,2)}
        drift=max(float((v-actor_before[k]).abs().max()) for k,v in runner.trainer.actor.state_dict().items())
        q=[]
        for segment in all_segments:
            obs=torch.as_tensor(segment["observations"],dtype=torch.float32,device=runner.trainer.device)
            alive=torch.as_tensor(segment["alive_masks"],dtype=torch.float32,device=runner.trainer.device)
            wave=torch.full((len(obs),),int(segment["credit_wave"]),dtype=torch.long,device=runner.trainer.device)
            horizon=torch.as_tensor(segment["horizons"],dtype=torch.float32,device=runner.trainer.device)
            with torch.no_grad():q.append(runner.trainer.iw_critic(obs,alive,wave,horizon).cpu().numpy())
        finite=all(np.isfinite(float(critic.get(key,0))) for key in ("iw_critic_loss","iw_mae_wave1","iw_mae_wave2")) and all(np.isfinite(x).all() and ((x>=0)&(x<=1)).all() for x in q)
        summary={"result":"IWSC_REAL_ENV_INTEGRATION_SMOKE_PASS","engineering_smoke_only":True,
          "environment_variant":env["environment_variant"],"max_steps":env["simulation"]["max_steps"],"total_waves":env["persistent_waves"]["total_waves"],
          "behavior_checkpoint":str(checkpoint),"behavior_checkpoint_seed":5303,"behavior_checkpoint_steps":3000000,
          "num_envs":NUM_ENVS,"rollout_steps":ROLLOUT_STEPS,"vector_steps_used":runner.trainer.vector_steps,"rollout_calls":calls,
          "cross_rollout_pending_observed":cross,"wave1_to_wave2_count":w12,"wave2_to_wave3_count":w23,
          "episodes_completed":runner.completed_episode_count,"completed_wave1_segments":len(completed1),"completed_wave2_segments":len(completed2),
          "wave1_boundary_segments":sum(bool(np.asarray(x["boundary_flags"]).any()) for x in completed1),
          "wave2_boundary_segments":sum(bool(np.asarray(x["boundary_flags"]).any()) for x in completed2),
          "observed_targets_wave1":sorted(set(float(x["target"]) for x in completed1)),"observed_targets_wave2":sorted(set(float(x["target"]) for x in completed2)),
          "replay_before":replay_before,"replay_after":replay_after,"iw_critic_update_count":critic["iw_critic_update_count"],
          "iw_critic_loss":critic["iw_critic_loss"],"iw_critic_mae_wave1":critic["iw_mae_wave1"],"iw_critic_mae_wave2":critic["iw_mae_wave2"],
          "iw_predictions_finite_and_bounded":finite,"training_actor_updated":False,"actor_parameter_drift_max":drift,"45m_used":False}
        if not finite or drift!=0:raise RuntimeError(summary)
        print(json.dumps(summary,indent=2));print("IWSC_REAL_ENV_INTEGRATION_SMOKE_PASS")
      finally:runner.vector.close()
if __name__=="__main__":main()
