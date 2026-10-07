#!/usr/bin/env python3
"""Fail-closed preflight for matched MAPPO MLP-vs-attention critics."""
from __future__ import annotations
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256,runtime_source_manifest
from algorithm.mappo.trainer import MAPPOTrainer

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
MLP=ROOT/"configs/mappo_mlp_baseline_1p5m.yaml"
ATTN=ROOT/"configs/mappo_attention_baseline_1p5m.yaml"
OUTPUT=ROOT/"outputs/mappo_critic_baseline_1p5m_preflight.json"
RUNS=(ROOT/"outputs/mappo_mlp_seed5303_1p5m",ROOT/"outputs/mappo_attention_seed5303_1p5m")
def load(path):return yaml.safe_load(path.read_text(encoding="utf-8"))
def comparable(config):
 value=deepcopy(config);value["network"].pop("critic_type",None);return value
def build_seeded_trainer(config,device,seed):
 n=config["network"];t=config["training"];i=config["implementation"]
 return MAPPOTrainer(observation_dim=int(n["observation_dim"]),action_dim=int(n["action_dim"]),num_agents=int(n["num_agents"]),hidden_dim=int(n["actor_hidden_layers"][0]),attention_heads=int(n["attention_heads"]),actor_learning_rate=float(t["actor_learning_rate"]),critic_learning_rate=float(t["critic_learning_rate"]),gamma=float(t["gamma"]),gae_lambda=float(t["gae_lambda"]),clip_ratio=float(t["clip_ratio"]),value_loss_coefficient=float(t["value_loss_coefficient"]),entropy_coefficient=float(t["entropy_coefficient"]),max_grad_norm=float(t["max_grad_norm"]),ppo_epochs=int(t["ppo_epochs"]),minibatch_size=int(t["minibatch_size"]),normalize_advantages=bool(i["normalize_advantages"]),clip_value_loss=bool(i["clip_value_loss"]),device=device,seed=int(seed),actor_activation=str(i["actor_activation"]),critic_activation=str(i["critic_activation"]),log_std_min=float(i["log_std_min"]),log_std_max=float(i["log_std_max"]),critic_type=str(n["critic_type"]))
def validate_configs(mlp,attn):
 if comparable(mlp)!=comparable(attn):raise RuntimeError("baseline configs differ beyond critic_type")
 if mlp["network"].get("critic_type")!="mlp" or attn["network"].get("critic_type")!="attention":raise RuntimeError("critic_type identity mismatch")
 expected={"seed":5303,"total_sampled_steps":1_500_000,"num_train_envs":24,"rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"actor_learning_rate":.0003,"critic_learning_rate":.0003,"evaluation_episodes":50,"evaluation_interval_sampled_steps":100_000,"device":"cuda"}
 for name,cfg in (("MLP",mlp),("Attention",attn)):
  bad={k:(cfg["training"].get(k),v) for k,v in expected.items() if cfg["training"].get(k)!=v}
  if bad:raise RuntimeError(f"{name} training protocol mismatch: {bad}")
  n=cfg["network"];i=cfg["implementation"]
  if (n["observation_dim"],n["action_dim"],n["num_agents"],n["actor_hidden_layers"],n["critic_hidden_layers"])!=(52,3,4,[256,256],[256,256]):raise RuntimeError(f"{name} network mismatch")
  forbidden=("modules","wave_context","wave_index","mission_context","popart","recurrent_memory")
  if any(key in cfg or key in n for key in forbidden):raise RuntimeError(f"{name} includes research modules")
  if i!={"actor_activation":"relu","critic_activation":"relu","log_std_min":-5.0,"log_std_max":2.0,"normalize_advantages":True,"clip_value_loss":True,"evaluation_seed_base":46_000_000,"checkpoint_interval_sampled_steps":500_000}:raise RuntimeError(f"{name} implementation mismatch")
 return True
def build_report():
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 env,mlp,attn=load(ENV),load(MLP),load(ATTN);validate_configs(mlp,attn)
 if (env.get("environment_variant"),env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"])!=("persistent_wave_v2",3,3000,4):raise RuntimeError("environment identity mismatch")
 existing=[str(p) for p in RUNS if p.exists()]
 if existing:raise RuntimeError(f"formal output directories already exist: {existing}")
 m=build_seeded_trainer(mlp,"cuda",5303);a=build_seeded_trainer(attn,"cuda",5303)
 m_state,a_state=m.actor.state_dict(),a.actor.state_dict()
 actor_equal=set(m_state)==set(a_state) and all(torch.equal(m_state[key],a_state[key]) for key in m_state)
 if not actor_equal:raise RuntimeError("same-seed Actor initialization mismatch")
 manifest=runtime_source_manifest(ROOT)
 return {"status":"READY_FOR_MAPPO_CRITIC_BASELINE_1P5M","cuda":torch.cuda.get_device_name(0),"environment_config_sha256":config_sha256(env),"mlp_config_sha256":config_sha256(mlp),"attention_config_sha256":config_sha256(attn),"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],"actor_initialization_seed":5303,"actor_initial_state_bitwise_equal":True,"actor_parameter_count":sum(p.numel() for p in m.actor.parameters()),"mlp_critic_parameter_count":sum(p.numel() for p in m.critic.parameters()),"attention_critic_parameter_count":sum(p.numel() for p in a.critic.parameters()),"evaluation_seed_range":[46_000_000,46_000_049],"evaluation_provenance_persisted_by_runner":True,"uses_44m":False,"uses_45m":False,"formal_outputs_absent":"2/2"}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--print-only",action="store_true");parser.add_argument("--output",type=Path,default=OUTPUT);args=parser.parse_args();report=build_report()
 if not args.print_only:
  path=args.output if args.output.is_absolute() else ROOT/args.output
  if path.exists():raise FileExistsError(path)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2));print(report["status"])
if __name__=="__main__":main()
