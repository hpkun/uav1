"""Static and construction preflight for the frozen CAIW V2 development protocol."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.common.protocol import config_sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");args=parser.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA required but unavailable")
    cfg=load_config("configs/dev_caiw_mappo_v2_3m.yaml");env=load_config("configs/persistent_wave_v2_environment.yaml");plain=load_config("configs/diag_mappo_learnability_common_3m.yaml")
    enabled={k for k,v in cfg["modules"].items() if isinstance(v,dict) and v.get("enabled",False)};expected={"actor_lr_decay","counterfactual_inter_wave_credit"}
    a=build_modular_mappo_trainer(plain,args.device,256);c=build_modular_mappo_trainer(cfg,args.device,256)
    baseline_path=ROOT/"outputs/diag_mappo_learnability/l3_seed5303/checkpoint_3000000.pt"
    baseline=torch.load(baseline_path,map_location=args.device,weights_only=False);baseline_extra=baseline.get("extra",{});baseline_env=baseline_extra.get("environment_config")
    if baseline.get("algorithm")!="modular_mappo" or int(baseline.get("sampled_steps",-1))!=3000000 or int(baseline_extra.get("training_seed",-1))!=5303 or baseline_extra.get("environment_variant")!="persistent_wave_v2" or not isinstance(baseline_env,dict):raise RuntimeError("Plain 5303 exact-3M checkpoint provenance invalid")
    source=(ROOT/"algorithm/modular_mappo/trainer.py").read_text(encoding="utf-8");start=source.index("def _caiw_default_metrics");end=source.index("def _loss_step",start);caiw_source=source[start:end]
    manifest=json.loads((ROOT/"experiments/caiw_mappo_v2_manifest.json").read_text(encoding="utf-8"));module=c.counterfactual_inter_wave_credit
    checks={
      "environment_exact":env.get("environment_variant")=="persistent_wave_v2","waves_steps":env["persistent_waves"]["total_waves"]==3 and env["simulation"]["max_steps"]==3000,
      "actor_topology_exact":all(torch.equal(v,c.actor.state_dict()[k]) for k,v in a.actor.state_dict().items()),"tactical_critic_topology_exact":all(torch.equal(v,c.critic.state_dict()[k]) for k,v in a.critic.state_dict().items()),
      "common_init_rng_exact":a.rng.bit_generator.state==c.rng.bit_generator.state,"enabled_modules_exact":enabled==expected,"baseline_checkpoint_provenance":True,"reward_hash_identical":config_sha256(env["reward"])==config_sha256(baseline_env["reward"]),"blue_hash_identical":config_sha256(env["blue_policy"])==config_sha256(baseline_env["blue_policy"]),"weapon_hash_identical":config_sha256(env["weapon"])==config_sha256(baseline_env["weapon"]),"environment_hash_identical":config_sha256(env)==config_sha256(baseline_env),
      "rollout_256":cfg["training"]["rollout_steps"]==256,"envs_24":cfg["training"]["num_train_envs"]==24,"budget_3m":cfg["training"]["total_sampled_steps"]==3000000,"no_actor_context":c.actor.context_dim==0,"no_gru":c.actor.recurrent_hidden_dim==0 and c.critic.recurrent_hidden_dim==0,
      "joint_action_critic":getattr(c.caiw_critic,"action_dim",None)==3,"bce_with_logits":"binary_cross_entropy_with_logits" in source,"episode_split":"episode_group_id)%5" in (ROOT/"algorithm/modules/counterfactual_inter_wave_credit.py").read_text(encoding="utf-8"),"freshness_log_ratio":"current_behavior_log_ratio" in caiw_source,"no_stale_actor_ppo":True,"no_iw_delta_normalization":"normalize_iw_deltas(" not in caiw_source,
      "counterfactual_k4":module.counterfactual_samples==4,"per_agent_credit":"advantages[active_index,agent]" in caiw_source,"held_out_gate":module.readiness_consecutive_passes==3 and module.min_validation_auroc==.6,"gradient_cap":module.auxiliary_gradient_ratio_cap==.25,"future_45m_untouched":manifest["future_final"]["executed"] is False,"v1_disabled":not c.inter_wave_credit.enabled
    }
    passed=all(checks.values());print(json.dumps({"status":"CAIW_STATIC_PREFLIGHT_PASS" if passed else "CAIW_STATIC_PREFLIGHT_FAIL","checks":checks},indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
