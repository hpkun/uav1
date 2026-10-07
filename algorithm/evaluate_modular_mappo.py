"""Protocol-aware evaluation for self-describing modular MAPPO checkpoints."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import ENVIRONMENT_VERSION
from env.combat_env import MultiUAVCombatEnv
from env.observation import observation_dim_from_config
from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.evaluation import evaluate_modular
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_modular_checkpoint,is_formal_v2_checkpoint
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION

def resolved(value):
 p=Path(value);return p if p.is_absolute() else ROOT/p

def checkpoint_hidden_dim(state,extra):
 arch=extra.get("network_architecture",{})
 if "hidden_dim" in arch:return int(arch["hidden_dim"])
 legacy=state.get("actor",{}).get("backbone.0.weight")
 if legacy is None:raise RuntimeError("checkpoint lacks actor hidden-dimension metadata")
 return int(legacy.shape[0])

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--env-config",required=True);p.add_argument("--episodes",type=int,default=50);p.add_argument("--seed-base",type=int,default=10000000);p.add_argument("--device",default="cpu",choices=("cpu","cuda"));p.add_argument("--output");p.add_argument("--allow-cross-variant",action="store_true");a=p.parse_args()
 if a.episodes<=0:raise ValueError("episodes must be positive")
 checkpoint=resolved(a.checkpoint);state=torch.load(checkpoint,map_location="cpu",weights_only=False);extra=state.get("extra",{});config=extra.get("algorithm_config")
 checkpoint_version=state.get("modular_mappo_impl_version")
 if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}; v1 checkpoints are diagnostic-only and this formal evaluator refuses them")
 protocol_complete=is_formal_v2_checkpoint(state)
 required=("environment_version","environment_variant","environment_config_sha256","algorithm_config_sha256","network_architecture","observation_dim","action_dim","num_agents","training_seed","training_gamma","training_num_envs","training_total_sampled_steps")
 if config is None or any(key not in extra for key in required):protocol_complete=False
 if config is None:raise RuntimeError("legacy checkpoint lacks algorithm config; cannot reconstruct network")
 env_path=resolved(a.env_config);env=yaml.safe_load(env_path.read_text(encoding="utf-8"));source_variant=str(extra.get("environment_variant","unknown"));target_variant=str(env.get("environment_variant","direct_v2_3"));cross=source_variant!=target_variant
 if cross and not a.allow_cross_variant:raise RuntimeError("source/target environment variant mismatch; pass --allow-cross-variant explicitly")
 if str(env.get("environment_version",ENVIRONMENT_VERSION))!=str(extra.get("environment_version")):raise RuntimeError("target environment version mismatch")
 dimensions=(int(config["network"]["observation_dim"]),int(config["network"]["action_dim"]),int(config["network"]["num_agents"]));expected=(observation_dim_from_config(env.get("observation",{})),MultiUAVCombatEnv.action_dim,int(env.get("scenario",{}).get("team_size",MultiUAVCombatEnv.team_size)))
 if dimensions!=expected:raise RuntimeError("checkpoint static dimensions mismatch")
 if protocol_complete and not cross:validate_modular_checkpoint(state,env,config)
 hidden=checkpoint_hidden_dim(state,extra);trainer=build_modular_mappo_trainer(config,a.device,hidden,int(extra.get("training_total_sampled_steps",config["training"]["total_sampled_steps"])))
 if not protocol_complete:raise RuntimeError("v2 checkpoint metadata is incomplete; formal evaluation refused")
 trainer.load(checkpoint)
 seeds=list(range(a.seed_base,a.seed_base+a.episodes));assert seeds and all(b==a+1 for a,b in zip(seeds,seeds[1:]));metrics=evaluate_modular(trainer,env,seeds)
 pretraining=int(state.get("warm_start_provenance",{}).get("pretraining_sampled_steps",0));result={**metrics,"metadata":{"algorithm":state.get("algorithm"),"checkpoint_modular_impl_version":state.get("modular_mappo_impl_version"),"current_modular_impl_version":MODULAR_MAPPO_IMPL_VERSION,"modular_impl_version":state.get("modular_mappo_impl_version"),"baseline_mappo_impl_version":state.get("baseline_mappo_impl_version"),"baseline_impl_version":state.get("baseline_mappo_impl_version"),"checkpoint_sampled_steps":int(state.get("sampled_steps",0)),"training_seed":extra.get("training_seed"),"training_gamma":extra.get("training_gamma"),"training_num_envs":extra.get("training_num_envs"),"training_total_budget":extra.get("training_total_sampled_steps"),"source_environment_variant":source_variant,"target_environment_variant":target_variant,"cross_variant":cross,"enabled_modules":state.get("enabled_modules",[]),"module_config_sha256":state.get("module_config_sha256"),"source_environment_hash":extra.get("environment_config_sha256"),"target_environment_hash":config_sha256(env),"algorithm_hash":extra.get("algorithm_config_sha256"),"warm_start_provenance":state.get("warm_start_provenance",{}),"pretraining_sampled_steps":pretraining,"effective_experience_budget":pretraining+int(state.get("sampled_steps",0)),"evaluation_seed_base":seeds[0],"evaluation_seed_end":seeds[-1],"evaluation_seed_count":len(seeds),"protocol_complete":protocol_complete}}
 output=json.dumps(result,indent=2)
 if a.output:resolved(a.output).write_text(output,encoding="utf-8")
 print(output)
if __name__=="__main__":main()
