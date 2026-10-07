"""CUDA-only tiny matched-branch smoke; never launches the 300k screen."""
from __future__ import annotations
from copy import deepcopy
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.protocol import validate_team_credit_branch
from algorithm.mappo.trainer import compute_gae
from algorithm.modules import TeamMeanCreditModule

SOURCE=ROOT/'outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt'

def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def state_hash(value):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):h.update(str(v.dtype).encode()+str(tuple(v.shape)).encode()+v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode()+str(v.shape).encode()+np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:add(x)
  else:h.update(repr(v).encode())
 add(value);return h.hexdigest()

def fingerprint(trainer):
 rng=trainer.capture_rng_state()
 return {'actor':state_hash(trainer.actor.state_dict()),'critic':state_hash(trainer.critic.state_dict()),
  'actor_optimizer':state_hash(trainer.actor_optimizer.state_dict()),'critic_optimizer':state_hash(trainer.critic_optimizer.state_dict()),
  'sampled_steps':trainer.sampled_steps,'ppo_updates':trainer.ppo_update_count,'actor_updates':trainer.actor_update_count,'critic_updates':trainer.critic_update_count,
  'permutation_rng':state_hash(rng['trainer_permutation_rng_state']),'torch_cpu_rng':state_hash(rng['torch_cpu_rng_state']),
  'torch_cuda_rng':state_hash(rng['torch_cuda_rng_state_all'])}

def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='outputs/smoke_team_credit_screen_final');args=parser.parse_args()
 OUT=ROOT/args.output_dir
 if not torch.cuda.is_available():raise RuntimeError('CUDA is mandatory')
 if OUT.exists() and (OUT/'smoke_report.json').exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True,exist_ok=True);before=file_hash(SOURCE);source=torch.load(SOURCE,map_location='cpu',weights_only=False)
 env=yaml.safe_load((ROOT/'configs/persistent_wave_v2_environment.yaml').read_text());control_cfg=load_config(ROOT/'configs/dev_team_credit_control_300k.yaml');team_cfg=load_config(ROOT/'configs/dev_team_credit_teammean_300k.yaml')
 runtime={'training_seed':5302,'training_num_envs':24,'training_smoke':False}
 cv=validate_team_credit_branch(source,env,control_cfg,runtime);tv=validate_team_credit_branch(source,env,team_cfg,runtime)
 digest=file_hash(SOURCE);rollouts={};initial={};entry={}
 for name,cfg,val in [('control',control_cfg,cv),('treatment',team_cfg,tv)]:
  directory=OUT/name;runner=ModularMAPPOTrainingRunner(env,cfg,24,1_805_280,'cuda',5302,directory,False,resume_mode=True,branch_provenance={**val,'parent_checkpoint_sha256':digest})
  runner.branch_from(SOURCE,val['intervention'],digest);initial[name]=fingerprint(runner.trainer)
  rollouts[name]=runner.collect_rollout(2);entry[name]={k:v for k,v in runner.last_rollout_metrics.items() if k.startswith('natural_entry_')}
  runner.vector.close()
 if initial['control']!=initial['treatment']:raise RuntimeError('branch initial model/optimizer/RNG mismatch')
 fields=('observations','actions','raw_actions','old_log_probs','raw_environment_rewards','alive_masks','next_alive_masks','wave_indices','dones')
 parity={field:np.array_equal(getattr(rollouts['control'],field),getattr(rollouts['treatment'],field)) for field in fields}
 if not all(parity.values()):raise RuntimeError(f'first rollout mismatch: {parity}')
 if entry['control']!=entry['treatment']:raise RuntimeError('entry diagnostics changed the matched trajectory')
 # Recreate exact trainers and use one identical mixed-reward copy of the
 # matched rollout so the tiny test actually exercises redistribution.
 batch=deepcopy(rollouts['control']);batch.rewards=np.zeros_like(batch.rewards)
 batch.rewards[...,0]=10.0;batch.rewards[...,1]=-2.0;batch.rewards*=batch.alive_masks
 control=build_modular_mappo_trainer(control_cfg,'cuda',256,1_805_280);control.load(SOURCE,strict_protocol=False,restore_rng=True)
 team=build_modular_mappo_trainer(team_cfg,'cuda',256,1_805_280);team.load(SOURCE,strict_protocol=False,restore_rng=True)
 plain_cfg=deepcopy(control_cfg);plain_cfg['modules'].pop('team_mean_credit',None);plain=build_modular_mappo_trainer(plain_cfg,'cuda',256,1_805_280);plain.load(SOURCE,strict_protocol=False,restore_rng=True)
 # Updates use the same trainer permutation state and exact same global Torch RNG.
 saved=torch.get_rng_state();cuda_saved=torch.cuda.get_rng_state_all();mc=control.update(batch)
 control_rng={'cpu':torch.get_rng_state().clone(),'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()],'permutation':deepcopy(control.rng.bit_generator.state)}
 torch.set_rng_state(saved);torch.cuda.set_rng_state_all(cuda_saved);mp=plain.update(batch)
 plain_rng={'cpu':torch.get_rng_state().clone(),'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()],'permutation':deepcopy(plain.rng.bit_generator.state)}
 rng_exact=state_hash(control_rng)==state_hash(plain_rng)
 torch.set_rng_state(saved);torch.cuda.set_rng_state_all(cuda_saved);mt=team.update(batch)
 core=('actor_loss','value_loss','entropy','approx_kl','actor_optimizer_steps_this_update','critic_optimizer_steps_this_update','actor_learning_rate','critic_learning_rate')
 control_exact=(state_hash(control.actor.state_dict())==state_hash(plain.actor.state_dict()) and state_hash(control.critic.state_dict())==state_hash(plain.critic.state_dict()) and state_hash(control.actor_optimizer.state_dict())==state_hash(plain.actor_optimizer.state_dict()) and state_hash(control.critic_optimizer.state_dict())==state_hash(plain.critic_optimizer.state_dict()) and all(mc[k]==mp[k] for k in core) and rng_exact)
 if not control_exact:raise RuntimeError('Control is not bitwise Plain-equivalent')
 if mt['team_credit_mean_abs_redistribution']<=0 or mt['team_credit_sum_abs_error_max']>1e-6:raise RuntimeError('treatment transform inactive/non-conservative')
 if not all(np.isfinite(v) for v in mt.values() if isinstance(v,(int,float))):raise RuntimeError('non-finite treatment diagnostics')
 if mc['actor_optimizer_steps_this_update']!=mt['actor_optimizer_steps_this_update'] or mc['critic_optimizer_steps_this_update']!=mt['critic_optimizer_steps_this_update']:raise RuntimeError('treatment optimizer step count differs')
 if state_hash(control.actor.state_dict())==state_hash(team.actor.state_dict()) or state_hash(control.critic.state_dict())==state_hash(team.critic.state_dict()):raise RuntimeError('treatment update did not diverge')
 reward=torch.as_tensor(batch.rewards,device='cuda');alive=torch.as_tensor(batch.alive_masks,device='cuda');waves=torch.as_tensor(batch.wave_indices,device='cuda')
 transformed,_=TeamMeanCreditModule({'enabled':True}).transform(reward,alive,waves);zero=torch.zeros_like(reward);done=torch.as_tensor(batch.dones,device='cuda');next_alive=torch.as_tensor(batch.next_alive_masks,device='cuda')
 local_adv,local_ret=compute_gae(reward,zero,zero,done,alive,next_alive,.999,.95);team_adv,team_ret=compute_gae(transformed,zero,zero,done,alive,next_alive,.999,.95)
 gae_changed=not torch.equal(local_adv,team_adv) and not torch.equal(local_ret,team_ret)
 if not gae_changed:raise RuntimeError('treatment GAE/returns did not change')
 checkpoint=OUT/'treatment_roundtrip.pt';team.save(checkpoint,{'branch_provenance':tv});restored=build_modular_mappo_trainer(team_cfg,'cuda',256,1_805_280);restored.load(checkpoint,strict_protocol=True,restore_rng=True)
 roundtrip=state_hash(restored.actor.state_dict())==state_hash(team.actor.state_dict()) and state_hash(restored.critic.state_dict())==state_hash(team.critic.state_dict())
 if not roundtrip:raise RuntimeError('treatment checkpoint roundtrip failed')
 after=file_hash(SOURCE);result={'status':'MATCHED_TEAM_MEAN_CREDIT_CUDA_SMOKE_PASS','cuda':torch.cuda.get_device_name(0),'source_sha256':before,'source_unchanged':before==after,
  'initial_state_exact':initial['control']==initial['treatment'],'first_rollout_bitwise':parity,'control_plain_bitwise_parity':control_exact,'control_plain_rng_exact_after_update':rng_exact,
  'treatment_reward_sum_preserved':mt['team_credit_sum_abs_error_max']<=1e-6,'treatment_mean_abs_redistribution':mt['team_credit_mean_abs_redistribution'],
  'treatment_update_finite':True,'treatment_gae_and_returns_changed':gae_changed,'treatment_update_diverged':True,'optimizer_step_counts_matched':True,'natural_entry_diagnostics':entry,
  'checkpoint_roundtrip':roundtrip,'branch_provenance_valid':cv['matched_checkpoint_continuation'] and tv['matched_checkpoint_continuation'],
  'formal_training_started':False,'formal_evaluation_started':False,'45m_used':False}
 if not result['source_unchanged']:raise RuntimeError('source checkpoint mutated')
 (OUT/'smoke_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
