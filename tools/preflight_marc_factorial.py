"""Read-only four-cell MARC protocol, initialization and optimization audit."""
from copy import deepcopy
from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import MARC_FACTORIAL_CELLS,validate_marc_factorial_config,checkpoint_architecture
from algorithm.modules import continuation_coefficients,tempered_wave_weights
from tools.smoke_marc_state_memory import state_memory_rollout

ENV=ROOT/'configs/persistent_wave_v2_blue433_environment.yaml'
SOURCE=ROOT/'configs/dev_marc_credit_balance_1m.yaml'
OUTPUTS={cell:ROOT/f'outputs/dev433_marc_factorial_{cell}_seed5301_3m' for cell in ('credit_only','balance_only')}
def config(cell):return load_config(ROOT/f'configs/dev_marc_factorial_{cell}_3m.yaml')

def matched_initialization(device,seed):
    states=[]
    for cell in MARC_FACTORIAL_CELLS:
        cfg=config(cell);cfg['training']['seed']=seed
        trainer=build_modular_mappo_trainer(cfg,device)
        states.append(({k:v.clone() for k,v in trainer.actor.state_dict().items()},
                       {k:v.clone() for k,v in trainer.critic.state_dict().items()},
                       torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()] if device=='cuda' else [],
                       deepcopy(trainer.rng.bit_generator.state)))
    for state in states[1:]:
        for index in (0,1):assert all(torch.equal(v,state[index][k]) for k,v in states[0][index].items())
        assert torch.equal(states[0][2],state[2]) and states[0][4]==state[4]
        assert all(torch.equal(a,b) for a,b in zip(states[0][3],state[3]))
    return True

def captured_update(trainer,batch):
    rows=[];original=trainer._loss_step
    def capture(*args,**kwargs):
        losses=original(*args,**kwargs)
        weights=kwargs.get('actor_weights')
        if weights is None:weights=torch.ones_like(args[8])
        rows.append([x.detach().clone() for x in (args[5],args[7],weights,*losses[:3])])
        return losses
    trainer._loss_step=capture
    try:metrics=trainer.update(batch)
    finally:trainer._loss_step=original
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['ppo_epochs_executed']==10
    return metrics,rows

def optimization_equivalence(cell,device):
    left=build_modular_mappo_trainer(config(cell),device)
    right_cfg=load_config(SOURCE)
    if cell=='none':
        right_cfg['development_method']='plain_factorial_audit'
        right_cfg['modules']['milestone_aware_retention_credit']['enabled']=False
    right=build_modular_mappo_trainer(right_cfg,device)
    batch=state_memory_rollout(left)
    # PPO's entropy estimate samples from the policy: match its RNG, too.
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    lm,lr=captured_update(left,batch)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    rm,rr=captured_update(right,batch)
    assert len(lr)==len(rr)==10
    differences={}
    for index,name in enumerate(('normalized_advantage','critic_target','actor_weight','actor_loss','value_loss','entropy')):
        differences[name]=max(float((a[index]-b[index]).abs().max()) for a,b in zip(lr,rr))
        assert all(torch.allclose(a[index],b[index],atol=2e-5,rtol=2e-5) for a,b in zip(lr,rr)),differences
    for name in ('actor','critic'):
        a=getattr(left,name).state_dict();b=getattr(right,name).state_dict()
        differences[name+'_post_update']=max(float((v-b[k]).abs().max()) for k,v in a.items())
        assert all(torch.allclose(v,b[k],atol=2e-5,rtol=2e-5) for k,v in a.items()),differences
    return differences

def preflight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    for output in OUTPUTS.values():
        if output.exists():raise RuntimeError(f'refuse occupied formal output: {output}')
    env=load_config(ENV);rows=[]
    init_checks={str(seed):matched_initialization('cuda',seed) for seed in (5301,5302)}
    for cell,(alpha,temp) in MARC_FACTORIAL_CELLS.items():
        cfg=config(cell);validate_marc_factorial_config(cfg,env)
        trainer=build_modular_mappo_trainer(cfg,'cuda');arch=checkpoint_architecture(trainer)
        assert arch['actor_input_dim']==52 and arch['actor_context_dim']==arch['critic_context_dim']==0
        assert arch['actor_gru_hidden_dim']==arch['critic_gru_hidden_dim']==0
        assert not trainer.milestone_aware_retention_credit.retention_active
        rows.append(dict(variant=cell,continuation_alpha=alpha,
            eta=continuation_coefficients(torch.tensor([1,2,3]),alpha=alpha).tolist(),
            wave_balance_temperature=temp,nonuniform_weights_expected=temp>0,retention=0,
            enabled_modules=trainer.module_protocol()['enabled_modules'],
            architecture=f"Actor52/256/256/Gaussian;Critic52/256/attention2;FF", 
            matched_initialization='5301/5302 PASS',
            evaluation='44000000..44000049 deterministic50',future_holdout_executed=False))
    return dict(status='MARC_FACTORIAL_ABLATION_READY',cells=rows,
        matched_initialization=init_checks,
        A0_plain_equivalence=optimization_equivalence('none','cuda'),
        A3_existing_core_equivalence=optimization_equivalence('full','cuda'),
        formal_outputs_present=0,formal_training_started=False)

if __name__=='__main__':
    result=preflight()
    print('variant | alpha | eta W1/W2/W3 | temp | nonuniform | retention | modules | architecture | matched initialization | evaluation | holdout executed')
    for row in result['cells']:print(' | '.join(str(row[key]) for key in ('variant','continuation_alpha','eta','wave_balance_temperature','nonuniform_weights_expected','retention','enabled_modules','architecture','matched_initialization','evaluation','future_holdout_executed')))
    print(json.dumps(result,indent=2))
