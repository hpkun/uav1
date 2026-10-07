"""Static/CUDA preflight for the matched 300k team-credit screen."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.common.protocol import runtime_source_manifest
from algorithm.modular_mappo.protocol import validate_team_credit_branch
from tools.analyze_team_credit_screen import MIN_W2_ENTRY_COUNT_PER_BRANCH

SEEDS=(5301,5302,5303);SOURCE_STEP=1_505_280;TARGET=1_805_280
ENV=ROOT/'configs/persistent_wave_v2_environment.yaml'
CONFIGS={'control':ROOT/'configs/dev_team_credit_control_300k.yaml','teammean':ROOT/'configs/dev_team_credit_teammean_300k.yaml'}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA is mandatory for team-credit preflight')
 manifest=runtime_source_manifest(ROOT)
 required_runtime_sources={
  'algorithm/modular_mappo/trainer.py','algorithm/modular_mappo/runner.py','algorithm/modular_mappo/protocol.py',
  'algorithm/train_modular_mappo.py','algorithm/modules/team_mean_credit.py','algorithm/mappo/trainer.py',
  'env/persistent_env.py','env/combat_env.py','env/reward.py','env/observation.py','env/weapon.py',
 }
 manifest_paths={item['path'] for item in manifest['runtime_source_manifest_files']}
 missing_runtime_sources=sorted(required_runtime_sources-manifest_paths)
 if missing_runtime_sources:raise RuntimeError(f'core runtime sources absent from manifest: {missing_runtime_sources}')
 env=yaml.safe_load(ENV.read_text());configs={k:load_config(v) for k,v in CONFIGS.items()}
 identity=(env['environment_variant'],env['persistent_waves']['total_waves'],env['simulation']['max_steps'],env['scenario']['team_size'])
 if identity!=('persistent_wave_v2',3,3000,4):raise RuntimeError(f'environment mismatch: {identity}')
 expected={'gamma':.999,'gae_lambda':.95,'clip_ratio':.2,'entropy_coefficient':.01,'value_loss_coefficient':.5,'max_grad_norm':.5,'rollout_steps':256,'ppo_epochs':10,'minibatch_size':512,'num_train_envs':24}
 for name,cfg in configs.items():
  if any(cfg['training'][k]!=v for k,v in expected.items()):raise RuntimeError(f'{name} training hyperparameter mismatch')
  if (cfg['network']['observation_dim'],cfg['network']['action_dim'],cfg['network']['num_agents'])!=(52,3,4):raise RuntimeError(f'{name} network mismatch')
  if (int(cfg['training']['evaluation_episodes']),int(cfg['training']['evaluation_interval_sampled_steps']),int(cfg['implementation']['evaluation_seed_base']))!=(50,100000,44000000):raise RuntimeError(f'{name} evaluation protocol mismatch')
  b=cfg['development_branch']
  if (cfg['training']['total_sampled_steps'],b['source_sampled_steps'],b['additional_sampled_steps'],b['target_sampled_steps'])!=(TARGET,SOURCE_STEP,300000,TARGET):raise RuntimeError(f'{name} budget mismatch')
 expected_modules={'control':['actor_lr_decay'],'teammean':['actor_lr_decay','team_mean_credit']}
 hashes={};validations={}
 for seed in SEEDS:
  path=ROOT/f'outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_{SOURCE_STEP}.pt'
  if not path.is_file():raise FileNotFoundError(path)
  state=torch.load(path,map_location='cpu',weights_only=False);extra=state['extra']
  if int(state['sampled_steps'])!=SOURCE_STEP or int(extra['training_seed'])!=seed:raise RuntimeError(f'source identity mismatch seed={seed}')
  if state.get('enabled_modules')!=['actor_lr_decay']:raise RuntimeError(f'source modules mismatch seed={seed}')
  if float(state['actor_optimizer']['param_groups'][0]['lr'])!=1e-4:raise RuntimeError(f'source actor LR mismatch seed={seed}')
  decay=extra['algorithm_config']['modules']['actor_lr_decay']
  if (decay['start_step'],decay['end_step'],decay['start_lr'],decay['end_lr'])!=(600000,900000,.0003,.0001):raise RuntimeError('actor LR schedule mismatch')
  hashes[str(seed)]={'absolute_path':str(path.resolve()),'sha256':digest(path)}
  validations[str(seed)]={name:validate_team_credit_branch(state,env,cfg,{'training_seed':seed,'training_num_envs':24,'training_smoke':False}) for name,cfg in configs.items()}
 for name,cfg in configs.items():
  enabled=sorted(k for k,v in cfg['modules'].items() if isinstance(v,dict) and v.get('enabled',False))
  if enabled!=expected_modules[name]:raise RuntimeError(f'{name} enabled modules mismatch: {enabled}')
 registry=json.loads((ROOT/'experiments/current_seed_provenance.json').read_text())['evaluation_ranges']['45000000..45000199']
 if registry.get('executed') is not False or registry.get('status')!='CURRENT_FUTURE_FINAL_BLOCK':raise RuntimeError('45M is not registered untouched')
 if MIN_W2_ENTRY_COUNT_PER_BRANCH!=30:raise RuntimeError('analyzer W2 entry threshold drifted from preregistered value 30')
 outputs=[ROOT/f'outputs/dev_team_credit_{kind}_seed{seed}_300k' for seed in SEEDS for kind in ('control','teammean')]
 existing=[str(p) for p in outputs if p.exists()]
 if existing:raise RuntimeError(f'formal output directories already exist: {existing}')
 result={'status':'READY_FOR_MATCHED_TEAM_MEAN_CREDIT_300K_SCREEN','cuda':torch.cuda.get_device_name(0),'runtime_source_manifest_sha256':manifest['runtime_source_manifest_sha256'],'runtime_source_manifest_file_count':manifest['runtime_source_manifest_file_count'],'runtime_source_core_files_verified':True,'source_checkpoint_hashes':hashes,'validations':validations,'target_sampled_steps':TARGET,'additional_sampled_steps':300000,'evaluation_seed_range':[44000000,44000049],'evaluation_episodes':50,'evaluation_interval_sampled_steps':100000,'min_W2_entry_count_per_branch':MIN_W2_ENTRY_COUNT_PER_BRANCH,'reserved_45m_executed':False,'formal_outputs_absent':True}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
