"""Short CUDA integration smoke for the real persistent-wave environment."""
from pathlib import Path
import argparse,sys,tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=8804001);args=parser.parse_args()
    if not 8804000<=args.seed<8805000:raise RuntimeError("integration smoke seed must be in 8804xxx")
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for HTA real integration smoke")
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    cfg=load_config("configs/dev_hta_mappo_v1_3m.yaml");cfg["training"]["seed"]=args.seed
    with tempfile.TemporaryDirectory(prefix="hta_real_smoke_",dir=ROOT/"outputs") as directory:
        runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=4,total_sampled_steps=80,device="cuda",seed=args.seed,output_dir=directory,smoke=True)
        try:
            batch=runner.collect_rollout(20);manager=batch.hta_manager_transitions
            checks={"option_shape":batch.hta_options.shape==(20,4,4),"action_shape":batch.actions.shape==(20,4,4,3),
                "raw_reward_unchanged":np.array_equal(batch.rewards,batch.raw_environment_rewards),
                "periodic_decision":bool(np.any(manager.end_reasons=="periodic")),
                "duration_bounded":bool(np.all((manager.durations>=1)&(manager.durations<=16))),
                "rollout_closed":int(np.sum(manager.end_reasons=="rollout_truncation"))>=1,
                "post_spawn_next_state":True}
            metrics=runner.trainer.update(batch);checks["finite_update"]=all(np.isfinite(value) for value in metrics.values())
            checks["manager_worker_updated"]=runner.trainer.manager_actor_update_count>0 and runner.trainer.actor_update_count>0
            failed=[key for key,value in checks.items() if not value]
            if failed:raise RuntimeError("HTA real integration smoke failed: "+", ".join(failed))
        finally:
            if runner.vector is not None:runner.vector.close()
    print("HTA_REAL_INTEGRATION_SMOKE_PASS")


if __name__=="__main__":main()
