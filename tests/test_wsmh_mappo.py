from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.protocol import checkpoint_architecture,validate_wsmh_branch
from algorithm.modules.wave_specific_mean_heads import EXPECTED_WSMH_CONFIG

ROOT=Path(__file__).resolve().parents[1]

def make_trainer(device="cpu",epochs=1,minibatch=8):
 config=load_config(ROOT/"configs/dev_wsmh_300k.yaml")
 return ModularMAPPOTrainer(device=device,seed=7,modules_config=config["modules"],total_sampled_steps=1_805_280,ppo_epochs=epochs,minibatch_size=minibatch)

def snapshot(module):return {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}
def same(left,right):return left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)

def synthetic_update(trainer,wave):
 shape=(2,4);obs=torch.randn(*shape,4,52,device=trainer.device);alive=torch.ones(*shape,4,device=trainer.device)
 raw=torch.randn(*shape,4,3,device=trainer.device);act=torch.tanh(raw);waves=torch.full(shape,wave,device=trainer.device,dtype=torch.long)
 with torch.no_grad():
  fo=obs.reshape(-1,4,52);fm=alive.reshape(-1,4);fw=waves.reshape(-1);dist=trainer._wsmh_routed_distribution(fo,fm,fw)
  oldlog=trainer.actor._squashed_log_prob(dist,raw.reshape(-1,4,3),act.reshape(-1,4,3)).reshape(*shape,4)
  oldvalue=trainer.critic.forward_step(fo,fm,None,None,None)[0].reshape(*shape,4)
 return trainer._update_flat_wsmh(obs,act,raw,oldlog,alive,torch.randn(*shape,4,device=trainer.device),oldvalue,oldvalue+.1,torch.zeros(*shape,0,device=trainer.device),waves)

def test_wsmh_config_structure_routing_and_metadata():
 trainer=make_trainer();assert trainer.wave_specific_mean_heads.config==EXPECTED_WSMH_CONFIG
 assert trainer.wave2_mean is not trainer.actor.mean and trainer.wave3_mean is not trainer.actor.mean
 assert same(snapshot(trainer.actor.mean),snapshot(trainer.wave2_mean)) and same(snapshot(trainer.actor.mean),snapshot(trainer.wave3_mean))
 obs=torch.randn(3,4,52);alive=torch.ones(3,4);waves=torch.tensor([1,2,3])
 with torch.no_grad():
  trainer.wave2_mean.bias.add_(1);trainer.wave3_mean.bias.sub_(1);dist=trainer._wsmh_routed_distribution(obs,alive,waves)
  encoded=trainer.actor.backbone(obs);means=[head(encoded) for head in trainer._wsmh_means()]
 for i in range(3):assert torch.equal(dist.loc[i],means[i][i])
 metadata=checkpoint_architecture(trainer);assert metadata["base_actor_parameter_count"]==80902
 assert metadata["additional_mean_parameter_count"]==1542 and metadata["total_policy_parameter_count"]==82444
 assert abs(metadata["parameter_overhead_fraction"]-1542/80902)<1e-15

def test_wsmh_only_wave2_updates_shared_and_mean2_but_not_other_means():
 trainer=make_trainer();before=[snapshot(m) for m in (trainer.actor.backbone,trainer.actor.log_std,*trainer._wsmh_means())]
 metrics=synthetic_update(trainer,2);after=[snapshot(m) for m in (trainer.actor.backbone,trainer.actor.log_std,*trainer._wsmh_means())]
 assert not same(before[0],after[0]) and not same(before[1],after[1])
 assert same(before[2],after[2]) and not same(before[3],after[3]) and same(before[4],after[4])
 assert metrics["wsmh_wave1_mean_optimizer_steps"]==0 and metrics["wsmh_wave2_mean_optimizer_steps"]==1
 assert metrics["wsmh_global_actor_grad_norm_postclip"]<=.50001
 assert np.isfinite([v for v in metrics.values() if isinstance(v,(int,float))]).all()

def test_wsmh_branch_optimizer_clone_no_sample_noop_and_strict_resume(tmp_path):
 source=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
 if not source.is_file():return
 trainer=make_trainer();trainer.load(source,strict_protocol=False,restore_rng=False)
 assert same(snapshot(trainer.actor.mean),snapshot(trainer.wave2_mean)) and same(snapshot(trainer.actor.mean),snapshot(trainer.wave3_mean))
 for optimizer in (trainer.wave2_mean_optimizer,trainer.wave3_mean_optimizer):
  for source_parameter,destination_parameter in zip(trainer.actor.mean.parameters(),optimizer.param_groups[0]["params"]):
   for key in ("step","exp_avg","exp_avg_sq"):assert torch.equal(trainer.actor_optimizer.state[source_parameter][key],optimizer.state[destination_parameter][key])
 mean1=snapshot(trainer.actor.mean);state1=[{k:(v.clone() if torch.is_tensor(v) else v) for k,v in trainer.actor_optimizer.state[p].items()} for p in trainer.actor.mean.parameters()]
 synthetic_update(trainer,2);assert same(mean1,snapshot(trainer.actor.mean))
 for p,saved in zip(trainer.actor.mean.parameters(),state1):
  for key,value in saved.items():
   current=trainer.actor_optimizer.state[p][key];assert torch.equal(value,current) if torch.is_tensor(value) else value==current
 with torch.no_grad():trainer.wave2_mean.bias.add_(.25);trainer.wave3_mean.bias.sub_(.1)
 path=tmp_path/"wsmh.pt";trainer.save(path);restored=make_trainer();restored.load(path,strict_protocol=True,restore_rng=False)
 assert same(snapshot(trainer.wave2_mean),snapshot(restored.wave2_mean)) and same(snapshot(trainer.wave3_mean),snapshot(restored.wave3_mean))

def test_wsmh_protocol_validation():
 source=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
 if not source.is_file():return
 config=load_config(ROOT/"configs/dev_wsmh_300k.yaml");env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 state=torch.load(source,map_location="cpu",weights_only=False)
 result=validate_wsmh_branch(state,env,config,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
 assert result["intervention"]=="wave_specific_mean_heads"
