"""Static, fail-closed protocol preflight for HTA-MAPPO V1."""
from pathlib import Path
import json,sys
import torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture


def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for formal HTA preflight")
    cfg=load_config("configs/dev_hta_mappo_v1_3m.yaml")
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    trainer=build_modular_mappo_trainer(cfg,"cuda",256,3_000_000);arch=checkpoint_architecture(trainer)
    enabled={key for key,value in cfg["modules"].items() if isinstance(value,dict) and value.get("enabled",False)}
    manifest=json.loads((ROOT/"experiments/hta_mappo_v1_manifest.json").read_text(encoding="utf-8"))
    registry=json.loads((ROOT/"experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    plain_path=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_3000000.pt"
    if not plain_path.exists():raise FileNotFoundError("matched Plain exact-3M checkpoint is unavailable")
    plain=torch.load(plain_path,map_location="cpu",weights_only=False);embedded=plain.get("extra",{}).get("environment_config")
    if not isinstance(embedded,dict):raise RuntimeError("matched Plain checkpoint lacks embedded environment")
    t=cfg["training"];hta=cfg["modules"]["hierarchical_temporal_abstraction"]
    checks={"environment_hash_identical":config_sha256(env)==config_sha256(embedded),
      "persistent_wave_v2":env.get("environment_variant")=="persistent_wave_v2","three_waves":env["persistent_waves"]["total_waves"]==3,
      "max_steps":env["simulation"]["max_steps"]==3000,"observation_52":cfg["network"]["observation_dim"]==52,
      "action_3":cfg["network"]["action_dim"]==3,"num_envs_24":t["num_train_envs"]==24,"rollout_256":t["rollout_steps"]==256,
      "budget_3m":t["total_sampled_steps"]==3_000_000,"k16":hta["decision_interval_steps"]==16 and 256%16==0,
      "four_options":hta["num_options"]==4,"manager_feedforward":arch["manager_recurrent"] is False,
      "worker_zero_residual":arch["worker_option_conditioning"]=="zero_initialized_option_mean_residual",
      "shared_logstd":arch["worker_logstd"]=="shared_plain_logstd","critic_additive_zero":arch["tactical_critic_option_conditioning"]=="additive_zero_onehot",
      "raw_reward":arch["manager_reward"]=="discounted_raw_environment_reward","smdp":arch["manager_gae"]=="gamma_power_duration",
      "wave_nonterminal":arch["wave_transition_terminal"] is False,"rollout_bootstrap":arch["rollout_boundary_bootstrap"] is True,
      "rollout_no_trace":arch["rollout_boundary_trace_continuation"] is False,
      "enabled_modules_exact":enabled=={"actor_lr_decay","hierarchical_temporal_abstraction"},
      "legacy_credit_off":not any(cfg["modules"].get(name,{}).get("enabled",False) for name in ("inter_wave_credit","counterfactual_inter_wave_credit","boundary_redistributed_segment_credit")),
      "future_45m_untouched":manifest["future_final"]=={"seed_start":45000000,"seed_end":45000199,"executed":False} and
        registry["evaluation_ranges"]["45000000..45000199"].get("executed") is False and
        registry["evaluation_ranges"]["45000000..45000199"].get("status")=="CURRENT_FUTURE_FINAL_BLOCK"}
    failed=[key for key,value in checks.items() if not value]
    if failed:raise RuntimeError("HTA static preflight failed: "+", ".join(failed))
    print("HTA_STATIC_PREFLIGHT_PASS")


if __name__=="__main__":main()
