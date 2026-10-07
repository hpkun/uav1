"""Strict protocol preflight and optional tiny CUDA smoke for actor mission context."""
from __future__ import annotations

import argparse, hashlib, json, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture,validate_modular_checkpoint
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner,validate_runtime_environment_contract
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
BASE=ROOT/"configs/diag_mappo_learnability_common_3m.yaml"
METHOD=ROOT/"configs/dev_actor_mission_context_3m.yaml"
MANIFEST=ROOT/"experiments/actor_mission_context_development_manifest.json"
REGISTRY=ROOT/"experiments/current_seed_provenance.json"
AUDIT=ROOT/"outputs/actor_mission_context_preflight"
SMOKE=ROOT/"outputs/dev_actor_mission_context_smoke_v1"
EXPECTED_SEEDS=(5301,5302,5303)


def load_yaml(path):return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _same_except_method(base,method):
    a,b=deepcopy(base),deepcopy(method)
    b.pop("development_method",None)
    a["modules"]["wave_context"]=deepcopy(b["modules"]["wave_context"])
    return a==b


def resolve_protocol(env=None,config=None,manifest=None,check_outputs=True):
    env=deepcopy(env if env is not None else load_yaml(ENV))
    config=deepcopy(config if config is not None else load_config(METHOD))
    base=load_config(BASE)
    manifest=deepcopy(manifest if manifest is not None else json.loads(MANIFEST.read_text(encoding="utf-8")))
    frozen=manifest["frozen_environment_contract"]
    if config.get("development_method")!="actor_mission_context":raise RuntimeError("development_method mismatch")
    wave=config["modules"]["wave_context"]
    if wave!={"enabled":True,"context_target":"actor_only","encoding":"mission_markov","max_waves":3}:
        raise RuntimeError("wave_context must be actor_only mission_markov")
    if not _same_except_method(base,config):raise RuntimeError("method config differs from baseline outside wave_context")
    expected_training={"actor_learning_rate":3e-4,"critic_learning_rate":3e-4,"gamma":.999,"gae_lambda":.95,
        "clip_ratio":.2,"value_loss_coefficient":.5,"entropy_coefficient":.01,"max_grad_norm":.5,
        "rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"num_train_envs":24,
        "total_sampled_steps":3_000_000,"evaluation_episodes":50,
        "evaluation_interval_sampled_steps":100_000,"device":"cuda"}
    bad={k:(config["training"].get(k),v) for k,v in expected_training.items() if config["training"].get(k)!=v}
    if bad:raise RuntimeError(f"training protocol mismatch: {bad}")
    enabled=[k for k,v in config["modules"].items() if v.get("enabled",False)]
    if enabled!=["wave_context","actor_lr_decay"]:raise RuntimeError(f"unexpected enabled modules: {enabled}")
    decay=config["modules"]["actor_lr_decay"]
    if decay!={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}:
        raise RuntimeError("actor LR decay protocol mismatch")
    if (env.get("environment_variant"),env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"]) != ("persistent_wave_v2",3,3000,4):
        raise RuntimeError("environment must be persistent_wave_v2 3-wave/3000-step/4-agent")
    effective=build_modular_mappo_trainer(config,"cpu",256,3_000_000).curriculum.runtime_config(env,0)
    contract=validate_runtime_environment_contract(env,effective,env,curriculum_enabled=False)
    if len({contract[k]["config_sha256"] for k in ("declared","effective_training","evaluation")})!=1:
        raise RuntimeError("declared/effective/evaluation environment hashes differ")
    actual_frozen={"environment_config_sha256":config_sha256(env),"action_config_sha256":config_sha256(env["action"]),
        "weapon_config_sha256":config_sha256(env["weapon"]),"reward_config_sha256":config_sha256(env["reward"]),
        "blue_policy_config_sha256":config_sha256(env["blue_policy"]),
        "blue_policy_source_sha256":hashlib.sha256((ROOT/"env/fixed_policy.py").read_bytes()).hexdigest()}
    for key,value in actual_frozen.items():
        if frozen.get(key)!=value:raise RuntimeError(f"frozen environment contract mismatch: {key}")
    method_trainer=build_modular_mappo_trainer(config,"cpu",256,3_000_000)
    baseline_trainer=build_modular_mappo_trainer(base,"cpu",256,3_000_000)
    ma,ba=checkpoint_architecture(method_trainer),checkpoint_architecture(baseline_trainer)
    if (method_trainer.wave_context.context_dim,ma["actor_context_dim"],ma["actor_input_dim"],ma["critic_context_dim"])!=(5,5,57,0):
        raise RuntimeError("actor-only architecture dimensions mismatch")
    if ma["critic_parameter_count"]!=ba["critic_parameter_count"] or method_trainer.critic.context_dim!=0:
        raise RuntimeError("critic architecture changed")
    if ma["actor_parameter_count"]-ba["actor_parameter_count"]!=5*256:
        raise RuntimeError("actor parameter delta is not exactly the 5D first-layer extension")
    if method_trainer.recurrent.enabled or method_trainer.entity_attention_enabled or method_trainer.curriculum.enabled:
        raise RuntimeError("forbidden actor architecture module enabled")
    if manifest["training_seeds"]!=list(EXPECTED_SEEDS):raise RuntimeError("training seed protocol mismatch")
    if manifest["development_evaluation"]!={"seed_start":44000000,"seed_end":44000049,"episodes":50,"deterministic":True,"common_scenarios":True,"is_holdout":False}:
        raise RuntimeError("development evaluation protocol mismatch")
    if manifest["future_final"]!={"seed_start":45000000,"seed_end":45000199,"executed":False}:
        raise RuntimeError("future-final block must remain unexecuted")
    registry=json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["evaluation_ranges"]["45000000..45000199"].get("executed") is not False:
        raise RuntimeError("seed registry marks future-final block executed")
    existing=[r["output_dir"] for r in manifest["runs"] if (ROOT/r["output_dir"]).exists()] if check_outputs else []
    if existing:raise RuntimeError(f"formal method output directories are not fresh: {existing}")
    return {"status":"READY_FOR_ACTOR_MISSION_CONTEXT_DEVELOPMENT","development_method":"actor_mission_context",
        "enabled_modules":enabled,"environment_contract":contract,"architecture":ma,
        "baseline_actor_parameter_count":ba["actor_parameter_count"],
        "baseline_critic_parameter_count":ba["critic_parameter_count"],"training":expected_training,
        "training_seeds":list(EXPECTED_SEEDS),"evaluation_seed_range":[44000000,44000049],
        "future_final":{"range":[45000000,45000199],"executed":False},
        "paired_initialization_note":manifest["paired_seed_note"],"formal_outputs_fresh":True}


def cuda_smoke(protocol):
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; CPU fallback forbidden")
    out=SMOKE/"seed88200001"
    if not out.exists():
        cfg=load_config(METHOD);cfg["training"].update({"seed":88_200_001,"num_train_envs":1,"rollout_steps":2,
            "ppo_epochs":1,"minibatch_size":8,"total_sampled_steps":8,"evaluation_episodes":1,
            "evaluation_interval_sampled_steps":10_000_000})
        cfg["implementation"]["evaluation_seed_base"]=88_201_000
        runner=ModularMAPPOTrainingRunner(load_yaml(ENV),cfg,1,8,"cuda",88_200_001,out,True)
        runner.run()
    state=torch.load(out/"latest.pt",map_location="cuda",weights_only=False);extra=state["extra"]
    validate_modular_checkpoint(state,load_yaml(ENV),extra["algorithm_config"])
    metrics=state.get("last_metrics",{})
    finite=all(torch.isfinite(v).all().item() for group in (state["actor"],state["critic"]) for v in group.values())
    required={"development_method":"actor_mission_context","actor_input_dim":57,"critic_context_dim":0,
              "mission_context_dim":5,"runtime_total_waves":3,"runtime_max_steps":3000}
    bad={k:(extra.get(k),v) for k,v in required.items() if extra.get(k)!=v}
    trainer=build_modular_mappo_trainer(extra["algorithm_config"],"cuda",64,8);trainer.load(out/"latest.pt")
    context=mission_context_numpy(trainer,np.asarray([1]),np.asarray([3]),np.ones((1,4),np.float32),np.asarray([0]),3000)
    opt=[json.loads(line) for line in (out/"optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    optimization_finite=bool(opt) and all(not isinstance(v,(int,float)) or np.isfinite(v) for row in opt for v in row.values())
    evaluation_rows=sum(1 for _ in (out/"evaluation_history.csv").read_text(encoding="utf-8").splitlines())-1
    if bad or not finite or not np.all(np.isfinite(context)) or not optimization_finite or evaluation_rows<1:
        raise RuntimeError(f"CUDA smoke provenance/finite failure: {bad}")
    return {"sampled_steps":int(state["sampled_steps"]),"seed":88_200_001,"evaluation_seed":88_201_000,
            "checkpoint_finite":finite,"context_finite":True,"optimization_metrics_finite":True,
            "evaluation_rows":evaluation_rows,"runtime_waves":extra["runtime_total_waves"],
            "runtime_max_steps":extra["runtime_max_steps"],"actor_input_dim":extra["actor_input_dim"],
            "critic_context_dim":extra["critic_context_dim"],"checkpoint":str(out/"latest.pt")}

def summary_line(result):
    env=result["environment_contract"]["effective_training"];arch=result["architecture"]
    return (f"[PREFLIGHT] READY | method={result['development_method']} | waves={env['total_waves']} "
            f"| max_steps={env['max_steps']} | actor={arch['actor_input_dim']} | critic_ctx={arch['critic_context_dim']} "
            f"| seeds={','.join(map(str,result['training_seeds']))} "
            f"| eval={result['evaluation_seed_range'][0]}..{result['evaluation_seed_range'][1]} | future_final=UNUSED")

def write_audit(result,output=AUDIT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (output/"preflight.md").write_text("# Actor mission-context preflight\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cuda-smoke",action="store_true")
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--summary",action="store_true");mode.add_argument("--verbose",action="store_true");args=parser.parse_args()
    result=resolve_protocol();result["cuda_smoke"]=cuda_smoke(result) if args.cuda_smoke else "NOT_REQUESTED"
    write_audit(result)
    print(json.dumps(result,indent=2) if args.verbose else summary_line(result))

if __name__=="__main__":main()
