"""Read-only launch checks for the two experimental 433 MARC GRU screens."""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.protocol import validate_marc_gru_screen_config,validate_marc_gru_screen_environment
from algorithm.modular_mappo.factory import build_modular_mappo_trainer

CONTROL=ROOT/'configs/dev_marc_credit_balance_1m.yaml'
TREATMENT=ROOT/'configs/dev_marc_wsgru_v1_1m.yaml'
ENV=ROOT/'configs/persistent_wave_v2_blue433_environment.yaml'
OUTPUTS=[ROOT/'outputs/dev433_marc_credit_balance_seed5301_1m',ROOT/'outputs/dev433_marc_wsgru_v1_seed5301_1m']

def validate_pair(control,treatment,env):
    for cfg in (control,treatment):
        validate_marc_gru_screen_config(cfg);validate_marc_gru_screen_environment(cfg,env)
    left=deepcopy(control);right=deepcopy(treatment)
    for cfg in (left,right):
        cfg.pop('development_method');cfg['modules'].pop('recurrent_memory',None)
    if left!=right:raise RuntimeError('screen pair differs outside recurrent Actor/method identity')
    source=load_config(ROOT/'configs/dev_marc_mappo_v2_1m.yaml')
    for key in ('training','implementation','network','runtime_logging'):
        if control[key]!=source[key]:raise RuntimeError(f'formal common {key} changed')
    t=control['training'];i=control['implementation']
    if t['seed']!=5301 or t['total_sampled_steps']!=1000000 or t['num_train_envs']!=24 or t['ppo_epochs']!=10:
        raise RuntimeError('screen training seed/budget/envs/epochs mismatch')
    if i['evaluation_seed_base']!=44000000 or t['evaluation_episodes']!=50:
        raise RuntimeError('screen validation must be 44M/50 deterministic episodes')
    for cfg in (control,treatment):
        p=cfg['development_protocol']
        if p['validation']!={'seed_start':44000000,'seed_end':44000049,'episodes':50,'deterministic':True,'common_scenarios':True,'is_holdout':False}:
            raise RuntimeError('development validation protocol mismatch')
        if p['reserved_future_final_test']['executed'] is not False:raise RuntimeError('future holdout cannot be executed')
    return True

def matched_initialization(device='cuda',seed=5301):
    configs=[load_config(CONTROL),load_config(TREATMENT)]
    for cfg in configs:cfg['training']['seed']=seed
    control=build_modular_mappo_trainer(configs[0],device)
    rng=torch.get_rng_state().clone()
    treatment=build_modular_mappo_trainer(configs[1],device)
    checks={'backbone':all(torch.equal(v,treatment.actor.backbone.state_dict()[k]) for k,v in control.actor.backbone.state_dict().items()),
            'critic':all(torch.equal(v,treatment.critic.state_dict()[k]) for k,v in control.critic.state_dict().items()),
            'cpu_rng':torch.equal(rng,torch.get_rng_state()),
            'phase_zero':bool(torch.count_nonzero(treatment.actor.phase_initial_hidden)==0)}
    if not all(checks.values()):raise RuntimeError(f'matched initialization failed: {checks}')
    return checks

def preflight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    control,treatment,env=map(load_config,(CONTROL,TREATMENT,ENV))
    validate_pair(control,treatment,env)
    existing=[str(p) for p in OUTPUTS if p.exists()]
    if existing:raise RuntimeError(f'formal output already exists: {existing}')
    return {'status':'READY_FOR_MARC_WAVE_SEGMENTED_GRU_SCREEN','device':torch.cuda.get_device_name(0),
            'initialization':matched_initialization(), 'formal_outputs_present':'0/2',
            'validation':'44000000..44000049 deterministic development only',
            'future_holdout_executed':False,
            'environment_sha256':hashlib.sha256(ENV.read_bytes()).hexdigest(),
            'methods':[control['development_method'],treatment['development_method']]}

if __name__=='__main__':print(json.dumps(preflight(),indent=2))
