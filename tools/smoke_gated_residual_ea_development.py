"""Very short CUDA smoke for all four representation topologies; no validation seeds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.factory import make_combat_environment

METHODS={"mappo":"configs/dev_grea_mappo_400k.yaml","full_ea":"configs/dev_grea_full_ea_400k.yaml",
         "residual_ea":"configs/dev_grea_residual_ea_400k.yaml","gated_ea":"configs/dev_grea_gated_ea_400k.yaml"}
SMOKE_SEED_BASE=94100

def synthetic_rollout(trainer):
 rng=np.random.default_rng(94199);T,E,A=2,1,4
 obs=rng.normal(size=(T,E,A,52)).astype("f")
 obs[...,13]=1;obs[...,20]=0;obs[...,27]=1;obs[...,33]=1;obs[...,39]=0;obs[...,45]=1;obs[...,51]=0
 alive=np.ones((T,E,A),"f");raw=rng.normal(size=(T,E,A,3)).astype("f");actions=np.tanh(raw).astype("f")
 with torch.no_grad():
  ot=torch.as_tensor(obs,device=trainer.device);rt=torch.as_tensor(raw,device=trainer.device);at=torch.as_tensor(actions,device=trainer.device)
  old=trainer.actor._squashed_log_prob(trainer.actor.distribution(ot),rt,at).cpu().numpy()
 rewards=rng.normal(size=(T,E,A)).astype("f");ctx=np.zeros((T,E,0),"f")
 return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((T,E),"f"),alive,
  obs.copy(),alive.copy(),np.ones((T,E),dtype=np.int64),np.full((T,E),3,dtype=np.int64),ctx,ctx,
  episode_masks=np.ones((T,E),"f"))

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default="outputs/dev_grea_architecture_smoke")
 args=parser.parse_args();output=ROOT/args.output_dir
 if output.exists() and any(output.iterdir()):raise FileExistsError(f"smoke output is not fresh: {output}")
 if not torch.cuda.is_available():raise RuntimeError("CUDA is required for GREA smoke")
 output.mkdir(parents=True,exist_ok=True)
 env_config=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));records=[]
 for index,(method,path) in enumerate(METHODS.items()):
  config=load_config(ROOT/path);config=deepcopy(config);config["training"]["seed"]=SMOKE_SEED_BASE+index
  config["training"]["ppo_epochs"]=1;config["training"]["minibatch_size"]=8
  trainer=build_modular_mappo_trainer(config,"cuda",256,8)
  env=make_combat_environment(env_config);observation,_=env.reset(SMOKE_SEED_BASE+index)
  action,_=trainer.act(observation[None,...],env.red_alive_mask[None,...],deterministic=False)
  next_observation,reward,terminated,truncated,_=env.step(action[0])
  metrics=trainer.update(synthetic_rollout(trainer))
  if not all(np.isfinite(float(value)) for value in metrics.values()):raise FloatingPointError(f"{method} emitted non-finite metrics")
  checkpoint=output/f"{method}.pt";trainer.save(checkpoint)
  restored=build_modular_mappo_trainer(config,"cuda",256,8);restored.load(checkpoint)
  probe=torch.as_tensor(observation[None,...],dtype=torch.float32,device="cuda")
  with torch.no_grad():
   before=trainer.actor.distribution(probe);after=restored.actor.distribution(probe)
  if not (torch.equal(before.mean,after.mean) and torch.equal(before.stddev,after.stddev)):raise RuntimeError(f"{method} checkpoint round-trip changed output")
  records.append({"method":method,"smoke_seed":SMOKE_SEED_BASE+index,"environment_reset":True,"environment_steps":1,
   "ppo_updates":1,"checkpoint_round_trip":True,"diagnostics_finite":True,"terminated":bool(terminated),
   "truncated":bool(truncated),"reward_finite":bool(np.isfinite(reward).all()),"observation_finite":bool(np.isfinite(next_observation).all())})
 summary={"status":"GREA_ARCHITECTURE_SMOKE_PASS","device":torch.cuda.get_device_name(0),"reserved_ranges_touched":False,
          "validation_32m_touched":False,"future_final_33m_touched":False,"records":records}
 (output/"smoke_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()

