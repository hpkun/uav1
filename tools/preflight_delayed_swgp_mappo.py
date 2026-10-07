"""Static/CUDA construction preflight for fresh 0-to-3M Delayed-SWGP."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config


def validate_delayed_config(config: dict, env: dict, registry: dict) -> dict[str,bool]:
    if "development_branch" in config:
        raise RuntimeError("Delayed-SWGP fresh protocol forbids development_branch")
    training=config["training"];module=config["modules"]["sequential_wave_gradient_projection"]
    enabled=sorted(name for name,value in config["modules"].items()
                   if isinstance(value,dict) and value.get("enabled",False))
    future=registry["evaluation_ranges"]["45000000..45000199"]
    return {
        "method_identity":config.get("development_method")=="delayed_swgp_mappo",
        "fresh_no_development_branch":"development_branch" not in config,
        "environment_contract":env.get("environment_variant")=="persistent_wave_v2"
            and env["persistent_waves"]["total_waves"]==3
            and env["simulation"]["max_steps"]==3000,
        "observation_action_agents":(config["network"]["observation_dim"],
            config["network"]["action_dim"],config["network"]["num_agents"])==(52,3,4),
        "training_hyperparameters":(
            training["gamma"],training["gae_lambda"],training["clip_ratio"],
            training["entropy_coefficient"],training["value_loss_coefficient"],
            training["max_grad_norm"],training["rollout_steps"],training["ppo_epochs"],
            training["minibatch_size"],training["num_train_envs"],
            training["total_sampled_steps"],training["actor_learning_rate"],
            training["critic_learning_rate"]
        )==(.999,.95,.2,.01,.5,.5,256,10,512,24,3_000_000,3e-4,3e-4),
        "actor_lr_decay":config["modules"]["actor_lr_decay"]=={
            "enabled":True,"schedule":"delayed_linear","start_step":600000,
            "end_step":900000,"start_lr":.0003,"end_lr":.0001},
        "swgp_protocol":module=={"enabled":True,"mode":"ordered_upstream_pairwise",
            "epsilon":1e-12,"activation_start_step":1_500_000},
        "enabled_modules_exact":enabled==["actor_lr_decay","sequential_wave_gradient_projection"],
        "future_final_45m_unused":future.get("executed") is False,
    }


def run_preflight(device: str="cuda") -> dict:
    if device!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Delayed-SWGP preflight")
    env=load_config("configs/persistent_wave_v2_environment.yaml")
    config=load_config("configs/dev_delayed_swgp_3m.yaml")
    registry=json.loads((ROOT/"experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    checks=validate_delayed_config(config,env,registry)
    trainer=build_modular_mappo_trainer(config,device,32,3_000_000)
    checks.update({
        "trainer_constructs_on_cuda":trainer.device.type=="cuda",
        "inactive_before_threshold":not trainer.sequential_wave_gradient_projection.is_active(1_499_136),
        "active_at_threshold":trainer.sequential_wave_gradient_projection.is_active(1_500_000),
        "fresh_counters_zero":all(value in (0,None) for key,value in
            trainer.sequential_wave_gradient_projection.state_dict().items()
            if key.endswith("count") or key.endswith("minibatches") or key=="first_activation_sampled_steps"),
    })
    passed=all(checks.values())
    report={"status":"DELAYED_SWGP_PREFLIGHT_PASS" if passed else "DELAYED_SWGP_PREFLIGHT_FAIL",
        "device":torch.cuda.get_device_name(0),"checks":checks,
        "activation_start_step":1_500_000,"continuous_fresh_training":True,
        "branch_from_allowed":False,"formal_training_started":False,
        "formal_evaluation_run":False,"future_final_45m_used":False}
    print(json.dumps(report,indent=2));return report


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",))
    args=parser.parse_args();report=run_preflight(args.device)
    raise SystemExit(0 if report["status"].endswith("PASS") else 2)


if __name__=="__main__":main()
