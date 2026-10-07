"""CUDA-only natural-trajectory tiny integration smoke for SWGP-MAPPO V1."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config
from env.persistent_env import PersistentWaveCombatEnv

SMOKE_SEED=88_130_001
OUTPUT=ROOT/"outputs/dev_swgp_mappo_tiny_smoke"


def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for SWGP smoke")
    if OUTPUT.exists():raise FileExistsError(f"smoke output already exists: {OUTPUT}")
    env=load_config("configs/persistent_wave_v2_environment.yaml");cfg=load_config("configs/dev_swgp_branch_3m.yaml")
    runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=16,device="cuda",seed=SMOKE_SEED,output_dir=OUTPUT,smoke=True)
    try:
        if runner.trainer.module_protocol()["enabled_modules"]!=["actor_lr_decay","sequential_wave_gradient_projection"]:raise RuntimeError("SWGP smoke module mismatch")
        rollout=runner.collect_rollout(4)
        if not np.all(rollout.wave_indices==1):raise RuntimeError("tiny natural smoke unexpectedly left W1")
        actor_before=runner.trainer.actor_update_count;critic_before=runner.trainer.critic_update_count
        metrics=runner.trainer.update(rollout)
        required=("actor_loss","value_loss","entropy","approx_kl","swgp_active","swgp_wave1_alive_fraction","swgp_surrogate_grad_norm_plain","swgp_surrogate_grad_norm_projected","swgp_entropy_grad_norm","swgp_actor_grad_norm_pre_clip","swgp_actor_grad_norm_post_clip","swgp_no_projection_fraction","swgp_single_wave_minibatch_fraction")
        if not all(math.isfinite(float(metrics[key])) for key in required):raise FloatingPointError("non-finite SWGP smoke metrics")
        expected=cfg["training"]["ppo_epochs"]
        if runner.trainer.actor_update_count-actor_before!=expected or runner.trainer.critic_update_count-critic_before!=expected:raise RuntimeError("SWGP smoke optimizer step count mismatch")
        if runner.trainer.sequential_wave_gradient_projection.total_swgp_minibatches!=expected:raise RuntimeError("SWGP minibatch counter mismatch")
        if metrics["swgp_single_wave_minibatch_fraction"]!=1 or metrics["swgp_no_projection_fraction"]!=1:raise RuntimeError("single-wave fallback was not Plain-equivalent")
        checkpoint=OUTPUT/"smoke_checkpoint.pt";runner.save_checkpoint(checkpoint)
        clone=build_modular_mappo_trainer(cfg,"cuda",64,16);clone.load(checkpoint)
        if clone.sequential_wave_gradient_projection.state_dict()!=runner.trainer.sequential_wave_gradient_projection.state_dict():raise RuntimeError("SWGP counter resume mismatch")
        evaluation_env=PersistentWaveCombatEnv(env);evaluation_env.reset(SMOKE_SEED+1)
        if evaluation_env.wave_index!=1:raise RuntimeError("evaluation reset did not start from W1")
        report={"status":"SWGP_CUDA_TINY_SMOKE_PASS","device":torch.cuda.get_device_name(0),"seed":SMOKE_SEED,
                "natural_rollout":True,"rollout_wave_indices":sorted(set(map(int,rollout.wave_indices.ravel()))),
                "actor_updates":runner.trainer.actor_update_count-actor_before,"critic_updates":runner.trainer.critic_update_count-critic_before,
                "swgp_counters":runner.trainer.sequential_wave_gradient_projection.state_dict(),
                "metrics":{key:float(metrics[key]) for key in required},"checkpoint_resume":True,"evaluation_first_wave":1,
                "formal_training":False,"formal_evaluation":False,"future_final_45m_used":False}
        (OUTPUT/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
    finally:runner.vector.close()


if __name__=="__main__":main()
