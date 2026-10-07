"""Strict static preflight for matched MAPPO versus WS-PBRS development."""
from __future__ import annotations
from copy import deepcopy
import json, sys
from pathlib import Path
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer

MANIFEST=ROOT/'experiments/ws_pbrs_development_manifest.json';SEEDS=(5101,5102,5103);EVAL=(39000000,39000049)
LAUNCHER=ROOT/'tools/run_ws_pbrs_development.sh'
OFF=('wave_context','recurrent_memory','popart','multi_wave_reward','wave_balancing','warm_start','curriculum','policy_anchor','entity_attention','advantage_priority','ppo_stabilization')

def validate_ws_pbrs_only_reward_shaping_diff(baseline, proposed):
    left=deepcopy(baseline);right=deepcopy(proposed)
    block={"enabled":"METHOD_SPECIFIC","coefficient":1.0,"potential":"progress_times_survival","terminal_zero":True}
    left['modules']['wave_survival_pbrs']=deepcopy(block);right['modules']['wave_survival_pbrs']=deepcopy(block)
    if left!=right:raise RuntimeError('resolved configs differ outside wave_survival_pbrs.enabled')

def validate(check_outputs=True,check_cuda=True):
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));runs=manifest['runs']
    if manifest['protocol_role']!='development_only' or len(runs)!=6:raise RuntimeError('manifest must define exactly six development runs')
    execution=manifest.get('execution_plan',{})
    if execution!={"mode":"strict_serial","max_parallel_processes":1,"within_seed_order":["MAPPO Baseline","MAPPO + WS-PBRS"],"seed_order":[5101,5102,5103],"shared_cuda_device":0}:raise RuntimeError('historical serial execution plan mismatch')
    expected={(m,s) for m in manifest['methods'] for s in SEEDS}
    if {(r['method'],r['training_seed']) for r in runs}!=expected:raise RuntimeError('matrix is not two methods x three paired seeds')
    if check_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    env=yaml.safe_load((ROOT/'configs/persistent_wave_v2_environment.yaml').read_text(encoding='utf-8'))
    if env.get('environment_variant')!='persistent_wave_v2':raise RuntimeError('environment must be persistent_wave_v2')
    configs={m:load_config(ROOT/next(r['config_path'] for r in runs if r['method']==m)) for m in manifest['methods']}
    validate_ws_pbrs_only_reward_shaping_diff(configs['MAPPO Baseline'],configs['MAPPO + WS-PBRS'])
    for method,cfg in configs.items():
        t=cfg['training']; expected_t={'actor_learning_rate':3e-4,'critic_learning_rate':3e-4,'gamma':.999,'gae_lambda':.95,'clip_ratio':.2,'value_loss_coefficient':.5,'entropy_coefficient':.01,'max_grad_norm':.5,'rollout_steps':256,'ppo_epochs':10,'minibatch_size':512,'num_train_envs':24,'total_sampled_steps':900000,'evaluation_episodes':50,'evaluation_interval_sampled_steps':100000,'device':'cuda'}
        if any(t.get(k)!=v for k,v in expected_t.items()):raise RuntimeError(f'{method} training protocol mismatch')
        if any(cfg['modules'].get(k,{}).get('enabled',False) for k in OFF):raise RuntimeError(f'{method} has forbidden module enabled')
        p=cfg['modules']['wave_survival_pbrs']; expected_enabled=method!='MAPPO Baseline'
        if p!={"enabled":expected_enabled,"coefficient":1.0,"potential":"progress_times_survival","terminal_zero":True}:raise RuntimeError(f'{method} PBRS block mismatch')
        lr=cfg['modules']['actor_lr_decay'];
        if lr!={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}:raise RuntimeError('actor LR schedule mismatch')
        if cfg['implementation']['evaluation_seed_base']!=EVAL[0]:raise RuntimeError('evaluation range mismatch')
    for seed in SEEDS:
        torch.manual_seed(seed);a=build_modular_mappo_trainer(configs['MAPPO Baseline'],'cuda' if check_cuda else 'cpu')
        torch.manual_seed(seed);b=build_modular_mappo_trainer(configs['MAPPO + WS-PBRS'],'cuda' if check_cuda else 'cpu')
        if any(not torch.equal(x,y) for x,y in zip(a.actor.state_dict().values(),b.actor.state_dict().values())):raise RuntimeError('paired actor initialization mismatch')
        if any(not torch.equal(x,y) for x,y in zip(a.critic.state_dict().values(),b.critic.state_dict().values())):raise RuntimeError('paired critic initialization mismatch')
    if manifest['reserved_untouched_future_final_test']['executed'] or manifest['evaluation']['seed_start']!=EVAL[0] or manifest['evaluation']['seed_end']!=EVAL[1]:raise RuntimeError('seed protocol mismatch')
    for run in runs:
        output=ROOT/run['output_dir']
        if check_outputs and output.exists() and any(output.iterdir()):raise FileExistsError(f'development output contains results: {output}')
    launcher=LAUNCHER.read_text(encoding='utf-8')
    required=('for seed in 5101 5102 5103','run_one baseline "$seed"','run_one pbrs "$seed"')
    if any(token not in launcher for token in required) or 'run_seed_pair' in launcher or 'MAX_PARALLEL' in launcher:raise RuntimeError('launcher is not the frozen serial plan')
    return {'status':'READY_FOR_WS_PBRS_DEVELOPMENT','runs':6,'training_seeds':list(SEEDS),'max_parallel_processes':1,'execution_mode':'strict_serial','within_seed_order':['MAPPO Baseline','MAPPO + WS-PBRS'],'evaluation_range':list(EVAL),'future_final_33m_executed':False,'evaluation_reward':'raw_environment_reward','primary_checkpoint':'latest.pt@900000'}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
