"""Focused protocol preflight for the final Mission-Aware FiLM KL-guard diagnostic."""
from __future__ import annotations

import argparse,json,random,sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modules import ACTOR_KL_GUARD_VERSION
from algorithm.train_modular_mappo import load_config

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
ORIGINAL=ROOT/"configs/dev_mission_aware_film_3m.yaml"
GUARD=ROOT/"configs/dev_mission_aware_film_kl_guard_3m.yaml"
MANIFEST=ROOT/"experiments/mission_aware_film_kl_guard_manifest.json"
REGISTRY=ROOT/"experiments/current_seed_provenance.json"
FORMAL_ROOT=ROOT/"outputs/dev_mission_aware_film_kl_guard_3m"
AUDIT=ROOT/"outputs/mission_aware_film_kl_guard_preflight"
SMOKE=ROOT/"outputs/dev_mission_aware_film_kl_guard_cuda_smoke/seed88400001"
SEEDS=(5301,5302,5303)

def _json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))
def _yaml(path:Path)->dict[str,Any]:return yaml.safe_load(path.read_text(encoding="utf-8"))

def _equal(value_a:Any,value_b:Any)->bool:
    if isinstance(value_a,torch.Tensor):return isinstance(value_b,torch.Tensor) and torch.equal(value_a,value_b)
    if isinstance(value_a,dict):return isinstance(value_b,dict) and value_a.keys()==value_b.keys() and all(_equal(value_a[k],value_b[k]) for k in value_a)
    if isinstance(value_a,(list,tuple)):return type(value_a) is type(value_b) and len(value_a)==len(value_b) and all(_equal(a,b) for a,b in zip(value_a,value_b))
    return value_a==value_b

def _rollout(trainer)->ModularRolloutBatch:
    t,e,n=2,1,4;rng=np.random.default_rng(701)
    obs=rng.normal(size=(t,e,n,52)).astype(np.float32);alive=np.ones((t,e,n),np.float32)
    ctx=np.zeros((t,e,n,5),np.float32);ctx[...,0]=1.;ctx[...,3]=.1;ctx[...,4]=1.
    raw=np.zeros((t,e,n,3),np.float32);actions=np.tanh(raw).astype(np.float32)
    with torch.no_grad():
        o=torch.as_tensor(obs,device=trainer.device);c=torch.as_tensor(ctx,device=trainer.device)
        d,_=trainer.actor.distribution_step(o,c,None,None,torch.as_tensor(alive,device=trainer.device))
        old=trainer.actor._squashed_log_prob(d,torch.as_tensor(raw,device=trainer.device),torch.as_tensor(actions,device=trainer.device)).cpu().numpy()
    rewards=rng.normal(size=(t,e,n)).astype(np.float32);zeros=np.zeros((t,e),np.float32)
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),zeros,alive,obs.copy(),alive.copy(),
      np.ones((t,e),int),np.full((t,e),3,int),ctx,ctx.copy(),episode_masks=np.ones((t,e),np.float32))

def _simulated_update(config:dict[str,Any],epoch_kl:float|list[float],device:str="cpu",epochs:int=10)->dict[str,float]:
    cfg=deepcopy(config);cfg["training"]["seed"]=771;cfg["training"]["ppo_epochs"]=epochs;cfg["training"]["minibatch_size"]=512
    trainer=build_modular_mappo_trainer(cfg,device,32,1000)
    if isinstance(epoch_kl,list):
        values=iter(epoch_kl);trainer._full_rollout_kl=lambda *_:float(next(values))
    else:trainer._full_rollout_kl=lambda *_:float(epoch_kl)
    metrics=trainer.update(_rollout(trainer))
    if not all(np.isfinite(float(v)) for v in metrics.values()):raise RuntimeError("non-finite synthetic update")
    return metrics

def _no_trigger_exact_match(original:dict[str,Any],guard:dict[str,Any])->dict[str,bool]:
    a,b=deepcopy(original),deepcopy(guard);a["training"]["seed"]=b["training"]["seed"]=773
    ta=build_modular_mappo_trainer(a,"cpu",32,1000);tb=build_modular_mappo_trainer(b,"cpu",32,1000)
    rollout=_rollout(ta)
    if not (_equal(ta.actor.state_dict(),tb.actor.state_dict()) and _equal(ta.critic.state_dict(),tb.critic.state_dict())
            and _equal(ta.actor_optimizer.state_dict(),tb.actor_optimizer.state_dict())
            and _equal(ta.critic_optimizer.state_dict(),tb.critic_optimizer.state_dict())
            and _equal(ta.rng.bit_generator.state,tb.rng.bit_generator.state)):
        raise RuntimeError("no-trigger comparison did not start from identical trainer state")
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
    ta.update(rollout);end_python=random.getstate();end_numpy=np.random.get_state();end_torch=torch.get_rng_state()
    random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
    tb._full_rollout_kl=lambda *_:.01;tb.update(rollout)
    numpy_equal=all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(end_numpy,np.random.get_state()))
    result={"actor_bit_identical":_equal(ta.actor.state_dict(),tb.actor.state_dict()),
      "critic_bit_identical":_equal(ta.critic.state_dict(),tb.critic.state_dict()),
      "actor_optimizer_identical":_equal(ta.actor_optimizer.state_dict(),tb.actor_optimizer.state_dict()),
      "critic_optimizer_identical":_equal(ta.critic_optimizer.state_dict(),tb.critic_optimizer.state_dict()),
      "trainer_rng_identical":_equal(ta.rng.bit_generator.state,tb.rng.bit_generator.state),
      "torch_rng_identical":torch.equal(end_torch,torch.get_rng_state()),
      "python_rng_identical":end_python==random.getstate(),"numpy_rng_identical":numpy_equal,
      "actor_update_count_identical":ta.actor_update_count==tb.actor_update_count,
      "critic_update_count_identical":ta.critic_update_count==tb.critic_update_count,
      "sampled_steps_identical":ta.sampled_steps==tb.sampled_steps}
    if not all(result.values()):raise RuntimeError(f"no-trigger update is not exact matched: {result}")
    return result

def validate(check_outputs:bool=True)->dict[str,Any]:
    env=_yaml(ENV);original=load_config(ORIGINAL);guard=load_config(GUARD);manifest=_json(MANIFEST);registry=_json(REGISTRY)
    if guard.get("development_method")!="mission_aware_film_kl_guard":raise RuntimeError("development_method mismatch")
    expected_guard={"enabled":True,"hard_kl":.05,"actor_early_stop":True}
    if guard["modules"].get("actor_kl_guard")!=expected_guard:raise RuntimeError("actor_kl_guard config mismatch")
    changed={k for k in guard if guard[k]!=original.get(k)}
    if changed!={"development_method","modules"}:raise RuntimeError(f"unexpected top-level config changes: {changed}")
    modules_without_guard=deepcopy(guard["modules"]);modules_without_guard.pop("actor_kl_guard")
    if modules_without_guard!=original["modules"]:raise RuntimeError("guard config changes existing modules")
    enabled=sorted(k for k,v in guard["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
    if enabled!=["actor_kl_guard","actor_lr_decay","mission_film","wave_context"]:raise RuntimeError(f"enabled modules mismatch: {enabled}")
    training_keys=("actor_learning_rate","critic_learning_rate","gamma","gae_lambda","clip_ratio","entropy_coefficient","value_loss_coefficient","max_grad_norm","rollout_steps","ppo_epochs","minibatch_size","num_train_envs","total_sampled_steps","evaluation_interval_sampled_steps","evaluation_episodes")
    if any(guard["training"][k]!=original["training"][k] for k in training_keys):raise RuntimeError("base training protocol changed")
    if (env["environment_variant"],env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"])!=("persistent_wave_v2",3,3000,4):raise RuntimeError("environment contract mismatch")
    if (guard["network"]["observation_dim"],guard["network"]["action_dim"])!=(52,3):raise RuntimeError("network dimensions changed")
    decay=guard["modules"]["actor_lr_decay"]
    expected_decay={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}
    if decay!=expected_decay:raise RuntimeError("actor LR schedule changed")
    initialization={}
    for seed in SEEDS:
        a,b=deepcopy(original),deepcopy(guard);a["training"]["seed"]=b["training"]["seed"]=seed
        ta=build_modular_mappo_trainer(a,"cpu",256,3_000_000);tb=build_modular_mappo_trainer(b,"cpu",256,3_000_000)
        row={"actor_state_bit_identical":_equal(ta.actor.state_dict(),tb.actor.state_dict()),
             "critic_state_bit_identical":_equal(ta.critic.state_dict(),tb.critic.state_dict()),
             "actor_optimizer_initial_state_identical":_equal(ta.actor_optimizer.state_dict(),tb.actor_optimizer.state_dict()),
             "critic_optimizer_initial_state_identical":_equal(ta.critic_optimizer.state_dict(),tb.critic_optimizer.state_dict())}
        if not all(row.values()):raise RuntimeError(f"matched initialization failed for seed {seed}")
        initialization[str(seed)]=row
    lr_steps=(0,600000,750000,900000,3000000);lr_values={str(s):guard["modules"]["actor_lr_decay"] and build_modular_mappo_trainer(guard,"cpu",32,3_000_000).actor_lr_decay.learning_rate(s,.0003) for s in lr_steps}
    expected_lr={"0":.0003,"600000":.0003,"750000":.0002,"900000":.0001,"3000000":.0001}
    if any(abs(lr_values[k]-v)>1e-15 for k,v in expected_lr.items()):raise RuntimeError(f"actor LR schedule mismatch: {lr_values}")
    guard_module=build_modular_mappo_trainer(guard,"cpu",32,1000).actor_kl_guard
    no_parameters=not isinstance(guard_module,torch.nn.Module) and not any(isinstance(v,(torch.Tensor,torch.nn.Parameter)) for v in vars(guard_module).values())
    if not no_parameters:raise RuntimeError("actor_kl_guard unexpectedly owns parameters")
    architecture=checkpoint_architecture(build_modular_mappo_trainer(guard,"cpu",256,3_000_000))
    expected_arch={"actor_input_dim":52,"actor_context_dim":5,"critic_context_dim":0,"mission_film_enabled":True,"mission_film_alpha":.2}
    if any(architecture.get(k)!=v for k,v in expected_arch.items()):raise RuntimeError(f"FiLM architecture changed: {architecture}")
    exact=_no_trigger_exact_match(original,guard)
    low=_simulated_update(guard,.01);high=_simulated_update(guard,.06);third=_simulated_update(guard,[.01,.01,.06])
    if (low["actor_epochs_used"],low["critic_epochs_used"],low["kl_hard_stop_triggered"])!=(10.,10.,0.):raise RuntimeError("low-KL simulation failed")
    if (high["actor_epochs_used"],high["critic_epochs_used"],high["kl_hard_stop_triggered"])!=(1.,10.,1.):raise RuntimeError("high-KL simulation failed")
    if (third["actor_epochs_used"],third["critic_epochs_used"],third["kl_hard_stop_triggered"])!=(3.,10.,1.):raise RuntimeError("third-epoch KL simulation failed")
    future_registry=registry["evaluation_ranges"]["45000000..45000199"]
    if future_registry.get("executed") is not False or manifest["future_final"].get("executed") is not False:raise RuntimeError("45M future-final block is not untouched")
    existing=[str(FORMAL_ROOT/f"seed{s}") for s in SEEDS if (FORMAL_ROOT/f"seed{s}").exists()]
    if check_outputs and existing:raise RuntimeError(f"formal output directories are not fresh: {existing}")
    return {"status":"READY_FOR_MISSION_AWARE_FILM_KL_GUARD_3M","module_version":ACTOR_KL_GUARD_VERSION,
      "enabled_modules":enabled,"environment":{"variant":"persistent_wave_v2","waves":3,"max_steps":3000,"observation_dim":52,"action_dim":3},
      "training_seeds":list(SEEDS),"strict_serial":True,"architecture":architecture,"formal_outputs_fresh":not existing,"future_final":{"range":[45000000,45000199],"executed":False},
      "matched_initialization":initialization,"actor_lr_by_step":lr_values,"guard_has_no_parameters":no_parameters,
      "NO_TRIGGER_UPDATE_EXACT_MATCH":"PASS","no_trigger_update_exact_match":exact,
      "simulated_low_kl":{"epoch_kl":.01,"actor_epochs_used":low["actor_epochs_used"],"critic_epochs_used":low["critic_epochs_used"],"triggered":False},
      "simulated_high_kl":{"epoch_kl":.06,"actor_epochs_used":high["actor_epochs_used"],"critic_epochs_used":high["critic_epochs_used"],"triggered":True},
      "simulated_third_epoch_stop":{"epoch_kls":[.01,.01,.06],"actor_epochs_used":third["actor_epochs_used"],"critic_epochs_used":third["critic_epochs_used"],"triggered":True}}

def cuda_smoke()->dict[str,Any]:
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for smoke; CPU fallback forbidden")
    SMOKE.mkdir(parents=True,exist_ok=True)
    cfg=load_config(GUARD);cfg["training"]["seed"]=88_400_001
    trainer=build_modular_mappo_trainer(cfg,"cuda",32,8);trainer._full_rollout_kl=lambda *_:.06
    metrics=trainer.update(_rollout(trainer));trainer.sampled_steps=8
    path=SMOKE/"checkpoint_smoke.pt";trainer.save(path,{"smoke_seed":88_400_001,"environment_steps_executed":0,"evaluation_episodes":0})
    restored=build_modular_mappo_trainer(cfg,"cuda",32,8);restored.load(path)
    state=torch.load(path,map_location="cuda",weights_only=False)
    finite=all(torch.isfinite(v).all() for group in (state["actor"],state["critic"]) for v in group.values())
    if not finite or metrics["actor_epochs_used"]!=1 or metrics["critic_epochs_used"]!=10:raise RuntimeError("CUDA smoke failed")
    result={"seed":88_400_001,"device":"cuda","forward_update_finite":True,"guard_triggered":True,"actor_epochs_used":1,"critic_epochs_used":10,"checkpoint_roundtrip":True,"environment_steps_executed":0,"evaluation_episodes":0,"checkpoint":str(path)}
    (SMOKE/"smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result

def write_result(result:dict[str,Any])->None:
    AUDIT.mkdir(parents=True,exist_ok=True)
    (AUDIT/"preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (AUDIT/"preflight.md").write_text("# Mission-Aware FiLM Actor KL Guard preflight\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--cuda-smoke",action="store_true");parser.add_argument("--summary",action="store_true");args=parser.parse_args()
    result=validate();result["cuda_smoke"]=cuda_smoke() if args.cuda_smoke else "NOT_REQUESTED";write_result(result)
    if args.summary:print("[PREFLIGHT] READY | method=mission_aware_film_kl_guard | hard_kl=0.05 | actor_only=true | critic_epochs=10 | seeds=5301,5302,5303 | future_final=UNUSED")
    else:print(json.dumps(result,indent=2))

if __name__=="__main__":main()
