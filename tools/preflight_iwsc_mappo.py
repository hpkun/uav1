"""Static/synthetic preflight for IWSC-MAPPO. Does not step an environment."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.trainer import combine_iw_actor_loss
from tools.analyze_iwsc_mappo import exact_row
from algorithm.common.protocol import config_sha256

def main():
    config=load_config("configs/dev_iwsc_mappo_3m.yaml");plain=load_config("configs/diag_mappo_learnability_common_3m.yaml")
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));mods=config["modules"]
    baseline_env=yaml.safe_load((ROOT/"outputs/diag_mappo_learnability/l3_seed5301/env_config.yaml").read_text(encoding="utf-8"))
    trainer_source=(ROOT/"algorithm/modular_mappo/trainer.py").read_text(encoding="utf-8")
    runner_source=(ROOT/"algorithm/modular_mappo/runner.py").read_text(encoding="utf-8")
    p=ModularMAPPOTrainer(hidden_dim=256,seed=5301,modules_config=plain["modules"],gamma=.999)
    i=ModularMAPPOTrainer(hidden_dim=256,seed=5301,modules_config=mods,gamma=.999)
    provenance=json.loads((ROOT/"experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    surrogate=torch.cat((torch.ones(100,1),torch.full((10,1),10.0)));alive=torch.ones_like(surrogate)
    active=torch.ones(110,dtype=torch.bool);waves=torch.cat((torch.ones(100,dtype=torch.long),torch.full((10,),2,dtype=torch.long)))
    balanced=combine_iw_actor_loss(surrogate,alive,active,waves,True);pooled=combine_iw_actor_loss(surrogate,alive,active,waves,False)
    checks={
      "A_environment":env["environment_variant"]=="persistent_wave_v2","B_three_waves_max3000":env["persistent_waves"]["total_waves"]==3 and env["simulation"]["max_steps"]==3000,
      "C_actor_input_52":config["network"]["observation_dim"]==52,"D_actor_topology":list(p.actor.state_dict())==list(i.actor.state_dict()),
      "E_tactical_critic_topology":list(p.critic.state_dict())==list(i.critic.state_dict()),
      "F_common_initialization":all(torch.equal(v,i.actor.state_dict()[k]) for k,v in p.actor.state_dict().items()) and all(torch.equal(v,i.critic.state_dict()[k]) for k,v in p.critic.state_dict().items()),
      "G_enabled_modules":sorted(k for k,v in mods.items() if v.get("enabled",False))==["actor_lr_decay","inter_wave_credit"],
      "H_reward_hash":config_sha256(env["reward"])==config_sha256(baseline_env["reward"]),"I_blue_hash":config_sha256(env["blue_policy"])==config_sha256(baseline_env["blue_policy"]),
      "J_weapon_hash":config_sha256(env["weapon"])==config_sha256(baseline_env["weapon"]),"K_rollout":config["training"]["rollout_steps"]==256,
      "L_envs":config["training"]["num_train_envs"]==24,
      "M_ppo":all(config["training"][k]==v for k,v in {"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,"ppo_epochs":10,"minibatch_size":512}.items()),
      "N_45m_unused":provenance["evaluation_ranges"]["45000000..45000199"]["executed"] is False,
      "O_no_stale_actor_replay":"iw_supervision_segments" not in trainer_source.split("def _update_flat_iwsc",1)[1],
      "P_wave3_mask":"valid=(waves<=2)" in trainer_source and "for wave in (1,2)" in trainer_source,
      "Q_terminal_q_next_zero":"torch.where(terminal,torch.zeros_like(qn),qn)" in trainer_source,
      "R_source_wave_boundary":"credit_wave = 2" not in runner_source and "transition_next_observations" in runner_source,
      "S_independent_iw_rng":p.rng.bit_generator.state==i.rng.bit_generator.state and "self.iw_rng" in trainer_source,
      "T_cold_plain_path":"elif self.inter_wave_credit.enabled and iw_active is not None and bool(iw_active.any())" in trainer_source,
      "U_actor_balance_functional":balanced.item()==-5.5 and pooled.item()!=balanced.item(),
      "V_exact_3m_functional":int(exact_row([{"sampled_steps":"2900000"},{"sampled_steps":"3000000"}],3000000)["sampled_steps"])==3000000}
    if not all(checks.values()):raise RuntimeError({k:v for k,v in checks.items() if not v})
    print(json.dumps({"checks":checks,"result":"IWSC_STATIC_PREFLIGHT_PASS"},indent=2))
if __name__=="__main__":main()
