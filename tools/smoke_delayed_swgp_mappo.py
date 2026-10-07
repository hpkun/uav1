"""CUDA-only tiny parity/activation smoke for Delayed-SWGP."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from tools.audit_wave_gradient_conflict import state_sha256

SEED=88_150_001
OUTPUT=ROOT/"outputs/dev_delayed_swgp_tiny_smoke"


def batch(trainer,waves):
    rng=np.random.default_rng(731);t,e,a=4,2,4
    obs=rng.normal(size=(t,e,a,52)).astype("f");alive=np.ones((t,e,a),"f")
    raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad():
        tensor=lambda value:torch.as_tensor(value,device=trainer.device)
        old=trainer.actor._squashed_log_prob(trainer.actor.distribution(tensor(obs)),tensor(raw),tensor(actions)).cpu().numpy()
    rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,
        obs+.01,alive.copy(),np.asarray(waves,dtype=np.int64),np.full((t,e),3),ctx,ctx,
        episode_masks=np.ones((t,e),"f"))


def fingerprint(trainer):
    return {"actor":state_sha256(trainer.actor.state_dict()),"critic":state_sha256(trainer.critic.state_dict()),
        "actor_optimizer":state_sha256(trainer.actor_optimizer.state_dict()),
        "critic_optimizer":state_sha256(trainer.critic_optimizer.state_dict()),
        "counts":(trainer.actor_update_count,trainer.critic_update_count,trainer.ppo_update_count),
        "permutation_rng":deepcopy(trainer.rng.bit_generator.state)}


def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for Delayed-SWGP smoke")
    if OUTPUT.exists():raise FileExistsError(f"smoke output already exists: {OUTPUT}")
    plain_config=load_config("configs/diag_mappo_learnability_common_3m.yaml")
    delayed_config=load_config("configs/dev_delayed_swgp_3m.yaml")
    for config in (plain_config,delayed_config):
        config["training"]["ppo_epochs"]=1;config["training"]["minibatch_size"]=32
        config["training"]["seed"]=SEED
    plain=build_modular_mappo_trainer(plain_config,"cuda",32,3_000_000)
    delayed=build_modular_mappo_trainer(delayed_config,"cuda",32,3_000_000)
    plain.sampled_steps=delayed.sampled_steps=1_499_136
    rollout=batch(plain,[[1,2],[3,1],[2,3],[1,3]])
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    plain_metrics=plain.update(deepcopy(rollout));plain_torch=torch.get_rng_state();plain_cuda=torch.cuda.get_rng_state_all()
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    delayed_metrics=delayed.update(deepcopy(rollout));delayed_torch=torch.get_rng_state();delayed_cuda=torch.cuda.get_rng_state_all()
    pre_exact=fingerprint(plain)=={key:value for key,value in fingerprint(delayed).items()}
    rng_exact=torch.equal(plain_torch,delayed_torch) and all(torch.equal(a,b) for a,b in zip(plain_cuda,delayed_cuda))
    core=("actor_loss","value_loss","entropy","approx_kl","actor_learning_rate","critic_learning_rate")
    metric_exact=all(plain_metrics[key]==delayed_metrics[key] for key in core)

    active=build_modular_mappo_trainer(delayed_config,"cuda",32,3_000_000)
    active.sampled_steps=1_505_280
    active_metrics=active.update(batch(active,[[1,2],[3,1],[2,3],[1,3]]))
    if not all(math.isfinite(float(value)) for value in active_metrics.values()):
        raise FloatingPointError("non-finite Delayed-SWGP smoke metrics")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    checkpoint=OUTPUT/"smoke_checkpoint.pt";active.save(checkpoint)
    restored=build_modular_mappo_trainer(delayed_config,"cuda",32,3_000_000);restored.load(checkpoint)
    state_exact=restored.sequential_wave_gradient_projection.state_dict()==active.sequential_wave_gradient_projection.state_dict()
    checks={"pre_activation_plain_state_bitwise":pre_exact,"pre_activation_rng_bitwise":rng_exact,
        "pre_activation_core_metrics_exact":metric_exact,
        "pre_activation_projection_minibatches_zero":delayed.sequential_wave_gradient_projection.total_swgp_minibatches==0,
        "post_activation_path":active_metrics["swgp_currently_active"]==1,
        "actual_activation_step":active.sequential_wave_gradient_projection.first_activation_sampled_steps==1_505_280,
        "mixed_wave_batch":all(active_metrics[f"swgp_wave{k}_alive_fraction"]>0 for k in (1,2,3)),
        "post_activation_swgp_minibatches":active.sequential_wave_gradient_projection.total_swgp_minibatches>0,
        "checkpoint_roundtrip":state_exact}
    if not all(checks.values()):raise RuntimeError(f"Delayed-SWGP smoke failed: {checks}")
    report={"status":"DELAYED_SWGP_CUDA_TINY_SMOKE_PASS","device":torch.cuda.get_device_name(0),
        "seed":SEED,"checks":checks,"activation_state":active.sequential_wave_gradient_projection.state_dict(),
        "post_activation_metrics":{key:float(active_metrics[key]) for key in (
            "actor_loss","value_loss","entropy","approx_kl","swgp_wave1_alive_fraction",
            "swgp_wave2_alive_fraction","swgp_wave3_alive_fraction","swgp_w2_projection_applied_fraction",
            "swgp_w3_w1_projection_applied_fraction","swgp_w3_w2_projection_applied_fraction")},
        "formal_training":False,"formal_evaluation":False,"future_final_45m_used":False}
    (OUTPUT/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
