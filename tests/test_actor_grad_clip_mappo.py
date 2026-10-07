from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_actor_grad_clip_branch,validate_actor_grad_clip_config_pair

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"

def configs():return tuple(load_config(ROOT/name) for name in ("configs/dev_pwtr_plain_300k.yaml","configs/dev_actor_grad_clip_05_control_300k.yaml","configs/dev_actor_grad_clip_10_300k.yaml"))
def trainer(config):
 value=build_modular_mappo_trainer(config,"cpu",256,1_805_280)
 if SOURCE.is_file():value.load(SOURCE,strict_protocol=False,restore_rng=False)
 return value
def state(module):return {k:v.detach().clone() for k,v in module.state_dict().items()}
def equal(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def digest(value):
 h=hashlib.sha256()
 def add(item):
  if torch.is_tensor(item):h.update(item.detach().cpu().numpy().tobytes())
  elif isinstance(item,dict):
   for key in sorted(item,key=str):h.update(str(key).encode());add(item[key])
  elif isinstance(item,(list,tuple)):
   for child in item:add(child)
  else:h.update(repr(item).encode())
 add(value);return h.hexdigest()
def batch(reference):
 generator=torch.Generator().manual_seed(99);shape=(2,2);obs=torch.randn(*shape,4,52,generator=generator);alive=torch.ones(*shape,4);raw=torch.randn(*shape,4,3,generator=generator);actions=torch.tanh(raw)
 with torch.no_grad():
  flat_obs=obs.reshape(-1,4,52);flat_alive=alive.reshape(-1,4);dist=reference.actor.distribution_step(flat_obs,None,None,None,flat_alive)[0]
  oldlog=reference.actor._squashed_log_prob(dist,raw.reshape(-1,4,3),actions.reshape(-1,4,3)).reshape(*shape,4);oldvalue=reference.critic.forward_step(flat_obs,flat_alive,None,None,None)[0].reshape(*shape,4)
 return (obs,actions,raw,oldlog,alive,torch.randn(*shape,4,generator=generator),oldvalue,oldvalue+.2,torch.ones(*shape),torch.zeros(*shape,0))

def test_configs_are_single_variable_and_critic_limit_stays_half():
 _,control,treatment=configs();assert validate_actor_grad_clip_config_pair(control,treatment)
 assert control["training"]["max_grad_norm"]==treatment["training"]["max_grad_norm"]==.5
 assert control["modules"]["actor_gradient_clipping"]["actor_max_grad_norm"]==.5
 assert treatment["modules"]["actor_gradient_clipping"]["actor_max_grad_norm"]==1.

def test_disabled_helper_and_enabled_limits():
 plain,control,treatment=map(trainer,configs())
 assert plain._actor_grad_clip_limit()==.5 and control._actor_grad_clip_limit()==.5 and treatment._actor_grad_clip_limit()==1.
 assert control.max_grad_norm==treatment.max_grad_norm==.5

def test_clip05_is_bitwise_plain_noop_and_clip10_only_changes_actor():
 if not SOURCE.is_file():return
 plain_cfg,control_cfg,treatment_cfg=configs();plain,control,treatment=map(trainer,(plain_cfg,control_cfg,treatment_cfg));payload=batch(plain)
 rng=torch.get_rng_state();plain_metrics=plain._update_flat(*payload);torch.set_rng_state(rng);control_metrics=control._update_flat(*payload);torch.set_rng_state(rng);treatment_metrics=treatment._update_flat(*payload)
 assert equal(state(plain.actor),state(control.actor));assert equal(state(plain.critic),state(control.critic))
 assert digest(plain.actor_optimizer.state_dict())==digest(control.actor_optimizer.state_dict());assert digest(plain.critic_optimizer.state_dict())==digest(control.critic_optimizer.state_dict())
 assert digest(plain.capture_rng_state())==digest(control.capture_rng_state())
 assert plain.ppo_epochs==control.ppo_epochs==treatment.ppo_epochs==10
 assert plain.actor_update_count==control.actor_update_count and plain.critic_update_count==control.critic_update_count
 assert plain_metrics["ppo_epochs_executed"]==control_metrics["ppo_epochs_executed"]==treatment_metrics["ppo_epochs_executed"]==10
 assert plain_metrics["actor_loss"]==control_metrics["actor_loss"] and plain_metrics["value_loss"]==control_metrics["value_loss"]
 assert plain_metrics["actor_grad_norm"]==control_metrics["actor_grad_norm"] and plain_metrics["critic_grad_norm"]==control_metrics["critic_grad_norm"]
 assert not equal(state(control.actor),state(treatment.actor));assert equal(state(control.critic),state(treatment.critic))
 assert control_metrics["actor_grad_clip_limit"]==.5 and treatment_metrics["actor_grad_clip_limit"]==1.
 assert control_metrics["actor_grad_clip_critic_limit"]==treatment_metrics["actor_grad_clip_critic_limit"]==.5
 assert treatment_metrics["actor_grad_clip_exact_scale"]>control_metrics["actor_grad_clip_exact_scale"]

def test_checkpoint_strict_resume_and_cross_config_rejection(tmp_path):
 if not SOURCE.is_file():return
 _,control_cfg,treatment_cfg=configs();control=trainer(control_cfg);treatment=trainer(treatment_cfg)
 control_path=tmp_path/"control.pt";treatment_path=tmp_path/"treatment.pt";control.save(control_path);treatment.save(treatment_path)
 trainer(control_cfg).load(control_path,strict_protocol=True,restore_rng=False);trainer(treatment_cfg).load(treatment_path,strict_protocol=True,restore_rng=False)
 with pytest.raises(RuntimeError):trainer(control_cfg).load(treatment_path,strict_protocol=True,restore_rng=False)
 with pytest.raises(RuntimeError):trainer(treatment_cfg).load(control_path,strict_protocol=True,restore_rng=False)

def test_protocol_accepts_only_the_two_fixed_branches():
 if not SOURCE.is_file():return
 state_dict=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 for config,intervention in zip(configs()[1:],("actor_grad_clip_05_control","actor_grad_clip_10")):
  result=validate_actor_grad_clip_branch(state_dict,env,config,{"training_seed":5301,"training_num_envs":24,"training_smoke":False});assert result["intervention"]==intervention
