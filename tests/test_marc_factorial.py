from copy import deepcopy
import pytest
import torch
from tools.preflight_marc_factorial import *

@pytest.mark.parametrize('cell',MARC_FACTORIAL_CELLS)
def test_config_and_retention(cell):
    cfg=config(cell);validate_marc_factorial_config(cfg,load_config(ENV))
    trainer=build_modular_mappo_trainer(cfg,'cpu')
    assert not trainer.recurrent.enabled and not trainer.wave_context.enabled
    assert not trainer.milestone_aware_retention_credit.retention_active
    assert trainer.ppo_epochs==10

@pytest.mark.parametrize('seed',[5301,5302])
def test_exact_initialization(seed):assert matched_initialization('cpu',seed)

@pytest.mark.parametrize('alpha,expected',[(0,[1,1,1]),(1,[1/3,1/2,1])])
def test_eta(alpha,expected):
    assert torch.allclose(continuation_coefficients(torch.tensor([1,2,3]),alpha=alpha),torch.tensor(expected,dtype=torch.float32))

@pytest.mark.parametrize('cell',MARC_FACTORIAL_CELLS)
def test_actual_credit_and_balance(cell):
    cfg=config(cell);alpha,temp=MARC_FACTORIAL_CELLS[cell]
    trainer=build_modular_mappo_trainer(cfg,'cpu');batch=state_memory_rollout(trainer)
    from algorithm.mappo.trainer import compute_gae
    from algorithm.modules import compute_local_gae
    tt=lambda x:torch.as_tensor(x,dtype=torch.float32)
    obs,nobs=tt(batch.observations),tt(batch.next_observations)
    with torch.no_grad():values,nvalues=trainer._value_rollout(batch,obs,nobs)
    alive=tt(batch.alive_masks);waves=torch.as_tensor(batch.wave_indices)
    args=(tt(batch.rewards),values,nvalues,tt(batch.dones),alive,tt(batch.next_alive_masks))
    global_adv,target=compute_gae(*args,trainer.gamma,trainer.gae_lambda)
    local,_=compute_local_gae(*args,tt(batch.wave_transition_flags),trainer.gamma,trainer.gae_lambda)
    adv=local+continuation_coefficients(waves,alpha=alpha).unsqueeze(-1)*(global_adv-local)
    if alpha==0:assert torch.allclose(adv,global_adv,atol=1e-6)
    live=adv[alive>.5];adv=(adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8)*alive
    metrics,rows=captured_update(trainer,batch)
    assert torch.allclose(rows[0][0].sort(dim=0).values,adv.reshape(-1,4).sort(dim=0).values,atol=1e-6)
    assert torch.equal(rows[0][1].sort(dim=0).values,target.reshape(-1,4).sort(dim=0).values)
    waves=torch.tensor([[1],[1],[1],[1],[1],[1],[2],[2],[3]])
    alive=torch.ones(9,1,4)
    weights,_=tempered_wave_weights(waves,alive,temperature=temp)
    if temp==0:assert torch.equal(weights,torch.ones_like(weights))
    else:
        assert weights.min()>=.5 and weights.max()<=2
        assert weights[0]<weights[-1]
        assert ((weights.unsqueeze(-1)*alive).sum()/alive.sum()).item()==pytest.approx(1,abs=1e-6)

@pytest.mark.parametrize('cell',['none','full'])
def test_complete_update_equivalence(cell):optimization_equivalence(cell,'cpu')

@pytest.mark.parametrize('mutation', ['alpha','temperature','retention','recurrent','context','width','epochs','bank','holdout','reward','extra'])
def test_reject_protocol_changes(mutation):
    cfg=config('full');env=load_config(ENV);m=cfg['modules']['milestone_aware_retention_credit']
    if mutation=='alpha':m['continuation_alpha']=.5
    elif mutation=='temperature':m['wave_balance_temperature']=1
    elif mutation=='retention':m['deployment_distill_coefficient']=.05
    elif mutation=='recurrent':cfg['modules']['recurrent_memory']['enabled']=True
    elif mutation=='context':cfg['modules']['wave_context']['enabled']=True
    elif mutation=='width':cfg['network']['critic_hidden_layers']=[128,128]
    elif mutation=='epochs':cfg['training']['ppo_epochs']=1
    elif mutation=='bank':cfg['implementation']['evaluation_seed_base']=45000000
    elif mutation=='holdout':cfg['development_protocol']['reserved_future_final_test']['executed']=True
    elif mutation=='reward':env['reward']={'changed':True}
    else:cfg['modules']['popart']['enabled']=True
    with pytest.raises(ValueError):validate_marc_factorial_config(cfg,env)

@pytest.mark.parametrize('cell',MARC_FACTORIAL_CELLS)
def test_retention_does_not_ingest_or_sample(cell,monkeypatch):
    trainer=build_modular_mappo_trainer(config(cell),'cpu')
    module=trainer.milestone_aware_retention_credit
    rng=deepcopy(module.rng.bit_generator.state)
    def forbidden(*args,**kwargs):raise AssertionError('elite sampling must be inactive')
    monkeypatch.setattr(module,'sample_elite_balanced',forbidden)
    module.ingest_success_segments([{'deliberately_invalid_segment':True}],trainer.actor,trainer.device)
    loss,_=module.retention_loss(trainer.actor,trainer.device)
    assert loss.item()==0 and all(not bank for bank in module.banks.values())
    assert all(not bank for bank in module.elite_segments.values())
    assert rng==module.rng.bit_generator.state

def test_balance_only_does_not_weight_entropy_or_critic():
    left=build_modular_mappo_trainer(config('none'),'cpu')
    right=build_modular_mappo_trainer(config('balance_only'),'cpu')
    batch=state_memory_rollout(left);rng=torch.get_rng_state().clone()
    _,a=captured_update(left,batch);torch.set_rng_state(rng)
    _,b=captured_update(right,batch)
    # Before either optimizer has stepped, only the actor surrogate is weighted.
    assert torch.allclose(a[0][0],b[0][0],atol=1e-6)
    assert torch.equal(a[0][1],b[0][1])
    assert torch.equal(a[0][4],b[0][4]) and torch.equal(a[0][5],b[0][5])
