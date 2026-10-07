"""Static/construction preflight for the frozen BRSC-MAPPO V1 protocol."""
from __future__ import annotations
import argparse,inspect,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.common.protocol import config_sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",));args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required but unavailable")
    cfg=load_config("configs/dev_brsc_mappo_v1_3m.yaml");plain_cfg=load_config("configs/diag_mappo_learnability_common_3m.yaml");env=load_config("configs/persistent_wave_v2_environment.yaml")
    plain=build_modular_mappo_trainer(plain_cfg,args.device,256);brsc=build_modular_mappo_trainer(cfg,args.device,256)
    checkpoint=ROOT/"outputs/diag_mappo_learnability/l3_seed5303/checkpoint_3000000.pt";state=torch.load(checkpoint,map_location=args.device,weights_only=False);extra=state.get("extra",{});baseline_env=extra.get("environment_config")
    if state.get("algorithm")!="modular_mappo" or int(state.get("sampled_steps",-1))!=3000000 or int(extra.get("training_seed",-1))!=5303 or not isinstance(baseline_env,dict):raise RuntimeError("Plain seed5303 exact-3M provenance invalid")
    enabled={key for key,value in cfg["modules"].items() if isinstance(value,dict) and value.get("enabled",False)};expected={"actor_lr_decay","boundary_redistributed_segment_credit"}
    module=brsc.boundary_redistributed_segment_credit;manifest=json.loads((ROOT/"experiments/brsc_mappo_v1_manifest.json").read_text(encoding="utf-8"));registry=json.loads((ROOT/"experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    runner_source=(ROOT/"algorithm/modular_mappo/runner.py").read_text(encoding="utf-8");trainer_source=(ROOT/"algorithm/modular_mappo/trainer.py").read_text(encoding="utf-8")
    update=trainer_source[trainer_source.index("def update("):trainer_source.index("def _iw_default_metrics")]
    checks={
      "environment_persistent_wave_v2":env.get("environment_variant")=="persistent_wave_v2",
      "waves_3_max_steps_3000":env["persistent_waves"]["total_waves"]==3 and env["simulation"]["max_steps"]==3000,
      "environment_hash_identical":config_sha256(env)==config_sha256(baseline_env),
      "reward_hash_identical":config_sha256(env["reward"])==config_sha256(baseline_env["reward"]),
      "blue_hash_identical":config_sha256(env["blue_policy"])==config_sha256(baseline_env["blue_policy"]),
      "weapon_hash_identical":config_sha256(env["weapon"])==config_sha256(baseline_env["weapon"]),
      "actor_52d_3d_256x2":cfg["network"]["observation_dim"]==52 and cfg["network"]["action_dim"]==3 and cfg["network"]["actor_hidden_layers"]==[256,256],
      "plain_actor_init_exact":all(torch.equal(value,brsc.actor.state_dict()[key]) for key,value in plain.actor.state_dict().items()),
      "plain_tactical_critic_init_exact":all(torch.equal(value,brsc.critic.state_dict()[key]) for key,value in plain.critic.state_dict().items()),
      "plain_permutation_rng_exact":plain.rng.bit_generator.state==brsc.rng.bit_generator.state,
      "rollout_256_envs24_budget3m":cfg["training"]["rollout_steps"]==256 and cfg["training"]["num_train_envs"]==24 and cfg["training"]["total_sampled_steps"]==3000000,
      "enabled_modules_exact":enabled==expected,"no_actor_context":brsc.actor.context_dim==0,"no_gru":brsc.actor.recurrent_hidden_dim==0 and brsc.critic.recurrent_hidden_dim==0,
      "state_only_boundary_critic":"actions" not in inspect.signature(brsc.brsc_critic.forward).parameters,
      "post_spawn_transition_next_observation":"_brsc_record_boundary(env_id,source_wave,result.transition_next_observations" in runner_source,
      "gamma_lambda_redistribution":"self.gamma*self.gae_lambda" in trainer_source,"no_advantage_normalization":"normalize_iw_deltas" not in trainer_source[trainer_source.index("def _brsc_default_metrics"):trainer_source.index("def _loss_step",trainer_source.index("def _brsc_default_metrics"))],
      "boundary_subset_vs_rollout_alive":"rollout_alive=alive" in trainer_source and "boundary_alive=next_alive" in trainer_source and "active[...,None]&(rollout_alive>.5)" in trainer_source,
      "held_out_three_pass_gate":module.readiness_consecutive_passes==3 and module.min_validation_auroc==.6 and module.validation_interval_updates==10,
      "gradient_cap_point25":module.auxiliary_gradient_ratio_cap==.25,
      "preupdate_score_before_ingest":update.index("_brsc_rollout_credit")<update.index(".ingest(r.brsc_supervision_boundaries)"),
      "iwsc_caiw_disabled":not brsc.inter_wave_credit.enabled and not brsc.counterfactual_inter_wave_credit.enabled,
      "development_44m":cfg["implementation"]["evaluation_seed_base"]==44000000 and cfg["training"]["evaluation_episodes"]==50,
      "future_45m_untouched":manifest["future_final"]["executed"] is False and registry["evaluation_ranges"]["45000000..45000199"]["executed"] is False,
    }
    passed=all(checks.values());print(json.dumps({"status":"BRSC_STATIC_PREFLIGHT_PASS" if passed else "BRSC_STATIC_PREFLIGHT_FAIL","checks":checks},indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
