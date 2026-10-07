"""Read-only protocol/CUDA preflight for the isolated MARC-SM V1 screen."""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_marc_state_memory_config,validate_marc_gru_screen_environment,checkpoint_architecture

CONFIG=ROOT/'configs/dev_marc_state_memory_v1_2m.yaml'
CONTROL=ROOT/'configs/dev_marc_credit_balance_1m.yaml'
ENV=ROOT/'configs/persistent_wave_v2_blue433_environment.yaml'
OUTPUT=ROOT/'outputs/dev433_marc_state_memory_v1_seed5301_2m'

def validate_protocol(config,control,env):
    validate_marc_state_memory_config(config);validate_marc_gru_screen_environment(config,env)
    shadow=deepcopy(config)
    shadow['development_method']=control['development_method']
    shadow['modules']['recurrent_memory']=deepcopy(control['modules']['recurrent_memory'])
    shadow['training']['total_sampled_steps']=control['training']['total_sampled_steps']
    shadow['development_protocol']=deepcopy(control['development_protocol'])
    if shadow!=control:raise RuntimeError('MARC-SM differs from control outside its isolated Actor/method/budget/protocol role')
    if config['training']['total_sampled_steps']!=2000000:raise RuntimeError('MARC-SM default budget must be 2M')
    if config['training']['seed']!=5301:raise RuntimeError('MARC-SM screen training seed must be5301')
    validation=config['development_protocol']['validation']
    if validation!=control['development_protocol']['validation'] or validation['seed_start']!=44000000 or validation['seed_end']!=44000049 or validation['episodes']!=50 or not validation['deterministic'] or validation['is_holdout']:
        raise RuntimeError('MARC-SM must preserve deterministic 44M development bank')
    if config['implementation']['evaluation_seed_base']!=44000000 or config['training']['evaluation_episodes']!=50:
        raise RuntimeError('MARC-SM runtime development evaluation bank mismatch')
    if config['development_protocol']['reserved_future_final_test']['executed'] is not False:
        raise RuntimeError('future holdout cannot be executed')
    return True

def matched_critic(device='cuda',seed=5301):
    left,right=load_config(CONTROL),load_config(CONFIG)
    for cfg in (left,right):cfg['training']['seed']=seed
    control=build_modular_mappo_trainer(left,device);rng=torch.get_rng_state().clone()
    treatment=build_modular_mappo_trainer(right,device)
    checks={'critic_parameters_exact':all(torch.equal(v,treatment.critic.state_dict()[k]) for k,v in control.critic.state_dict().items()),
            'critic_context_dim_zero':treatment.critic.context_dim==0,
            'critic_feed_forward':treatment.critic.recurrent_hidden_dim==0,
            'post_init_rng_exact':torch.equal(rng,torch.get_rng_state()),
            'no_phase_parameters':not hasattr(treatment.actor,'phase_initial_hidden')}
    if not all(checks.values()):raise RuntimeError(f'MARC-SM matched critic isolation failed: {checks}')
    return checks

def preflight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    cfg,control,env=map(load_config,(CONFIG,CONTROL,ENV))
    validate_protocol(cfg,control,env)
    if OUTPUT.exists():raise RuntimeError(f'refuse occupied formal output: {OUTPUT}')
    checks=matched_critic()
    trainer=build_modular_mappo_trainer(cfg,'cuda')
    return {'status':'READY_FOR_MARC_STATE_MEMORY_V1','device':torch.cuda.get_device_name(0),
            'enabled_modules':trainer.module_protocol()['enabled_modules'],'architecture':checkpoint_architecture(trainer),
            'matched_critic':checks,'environment_sha256':hashlib.sha256(ENV.read_bytes()).hexdigest(),
            'validation':'44000000..44000049 development deterministic50','future_holdout_executed':False,
            'formal_output_exists':False,'formal_training_started':False}

if __name__=='__main__':print(json.dumps(preflight(),indent=2))
