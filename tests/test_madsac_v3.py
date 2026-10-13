"""Implementation-v3 attention reconstruction and combat console contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pytest
import torch
from torch import nn
from algorithm.madsac.networks import CentralizedAttentionQCritic
from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.console import format_steps
from test_madsac_v326 import configs,metadata


def test_self_dead_mask_single_survivor_and_all_dead():
    torch.manual_seed(7)
    c=CentralizedAttentionQCritic(observation_dim=56,hidden_dim=16)
    obs=torch.randn(4,5,56);actions=torch.randn(4,5,3)
    alive=torch.tensor([[1,1,1,1,1],[1,0,1,0,1],[0,0,1,0,0],[0,0,0,0,0]],dtype=torch.float32)
    q,w=c(obs,actions,alive,True)
    assert torch.isfinite(q).all() and torch.isfinite(w).all()
    assert (w[0].diagonal(dim1=-2,dim2=-1)>0).all()
    for b in range(4):
        for i in range(5):
            if alive[b,i]:torch.testing.assert_close(w[b,:,i].sum(-1),torch.ones(2))
            else:assert not w[b,:,i].any() and q[b,i]==0
            if not alive[b,i]:assert not w[b,:,:,i].any()
    assert torch.equal(w[2,:,2,2],torch.ones(2))
    assert w[2].count_nonzero()==2
    assert c.q_network[0].in_features==16


def test_unscaled_dot_product_and_value_activation_context():
    c=CentralizedAttentionQCritic(observation_dim=1,action_dim=3,hidden_dim=4,attention_heads=2)
    c.embedding=nn.Identity()
    with torch.no_grad():
        c.wq.weight.copy_(torch.eye(4));c.wk.weight.copy_(torch.eye(4))
        c.wq.bias.zero_();c.wk.bias.zero_();c.wv.weight.zero_()
        c.wv.bias.copy_(torch.tensor([-1.,-2.,3.,4.]))
    embedding=torch.tensor([[[1.,0.,1.,0.],[2.,0.,2.,0.],[0.,1.,0.,1.],[-1.,0.,-1.,0.],[0.,0.,0.,0.]]])
    captured=[]
    hook=c.q_network.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    _,weights=c(embedding[:,:,:1],embedding[:,:,1:],torch.ones(1,5),True);hook.remove()
    heads=embedding.view(1,5,2,2).transpose(1,2);logits=heads@heads.transpose(-2,-1)
    expected=torch.softmax(logits,-1)
    torch.testing.assert_close(weights,expected)
    assert not torch.allclose(weights,torch.softmax(logits/np.sqrt(2),-1))
    expected_values=torch.tensor([-.01,-.02,3.,4.]).expand(1,5,4)
    torch.testing.assert_close(captured[0],expected_values)
    assert c.value_activation.negative_slope==.01


@pytest.mark.parametrize('forged',[False,True])
def test_old_checkpoint_implementation_or_self_shortcut_rejected(tmp_path,forged):
    assert MADSAC_IMPL_VERSION==3
    e,c=configs();trainer=build_madsac_trainer(c,'cuda',hidden_dim=32)
    state=trainer.checkpoint_state(metadata(e,c,True))
    if forged:state['critic1']['q_network.0.weight']=torch.zeros(32,64)
    else:state['implementation_version']=2
    path=tmp_path/'old.pt';torch.save(state,path)
    with pytest.raises(RuntimeError):validate_madsac_checkpoint(state,e,c)
    with pytest.raises(RuntimeError):trainer.load_for_evaluation(path)


@pytest.mark.parametrize('key,value',[('attention_self_inclusive',False),('attention_value_activation','relu'),
    ('attention_value_negative_slope',.2),('attention_logit_scaling','sqrt'),('critic_raw_self_shortcut',True)])
def test_critic_metadata_required(key,value):
    e,c=configs();trainer=build_madsac_trainer(c,'cuda',hidden_dim=32)
    extra=metadata(e,c,True);extra['network_architecture'][key]=value
    with pytest.raises(RuntimeError,match=key):validate_madsac_checkpoint(trainer.checkpoint_state(extra),e,c)


def runner():
    r=object.__new__(MADSACTrainingRunner);r.trainer=SimpleNamespace(sampled_steps=20000)
    r.total_sampled_steps=3000000;r.recent_episode_window=2;r.completed_records=[]
    r.evaluation_mode='stochastic'
    return r


def records():
    return [dict(red_success=1,red_losses=0,blue_losses=5,red_first_attempt_step=1,
        red_first_kill_step=2,termination_reason='red_win_elimination',episode_length=100,team_episode_return=500),
        dict(red_success=0,red_losses=5,blue_losses=0,red_first_attempt_step=None,
        red_first_fire_window_step=1,red_first_kill_step=None,termination_reason='blue_win_timeout',timeout=True,episode_length=1000,team_episode_return=-50),
        dict(red_success=1,red_losses=1,blue_losses=4,red_first_attempt_step=20,
        red_first_fire_window_step=1,red_first_kill_step=21,termination_reason='red_win_elimination',episode_length=200,team_episode_return=10)]


def test_recent_completed_window_episode_rates_and_train_format():
    r=runner();r.completed_records=records();m=r.recent_metrics()
    assert m==dict(red_win_rate=.5,average_red_loss=3.,average_blue_loss=2.,
        red_attempt_episode_rate=.5,red_kill_episode_rate=.5,timeout_rate=.5,
        average_episode_length=600.,average_return=-20.)
    assert r.train_log_line()=='[TRAIN] 20k/3.0M (0.7%) | ep=3 | win=50% | Rloss=3.00 | Bloss=2.00 | fire=50% | kill=50% | timeout=50% | len=600 | R=-20.00'
    assert all(term not in r.train_log_line() for term in ('replay','critic_updates','actor_updates','vector_steps'))


def test_no_completed_episodes_and_step_formats():
    r=runner();line=r.train_log_line()
    for key in ('win','Rloss','Bloss','fire','kill','timeout','len','R'):assert f'{key}=n/a' in line
    assert [format_steps(n) for n in (999,20000,500000,1500000)]==['999','20k','500k','1.50M']
    assert format_steps(3000000,total=True)=='3.0M'


def test_start_console_keeps_only_experiment_fields():
    r=runner()
    summary=dict(environment_version='3.26',num_agents=5,seed=1,device='cuda',num_envs=24,
        total_sampled_steps=3000000,hidden_dim=256,attention_heads=2,minibatch_size=1024,
        alpha=.1,evaluation_mode='stochastic')
    with patch.object(r,'startup_summary',return_value=summary):line=r.start_log_line()
    assert line=='[START] MADSAC-v3 | env=v3.26 | 5v5 | seed=1 | cuda | envs=24 | steps=3.0M | hidden=256 | heads=2 | batch=1024 | alpha=0.1 | eval=stochastic'


def test_eval_same_metrics_and_best_only_on_improvement(tmp_path,capsys):
    r=runner();r.output_dir=tmp_path;r.evaluation_history=[];r.best_evaluation=None;r.best_sampled_steps=None
    r.env_config={};r.evaluation_seeds=(1,);r.evaluation_policy_seed=1
    row=runner();row.completed_records=records();metrics=row.recent_metrics()
    metrics.update(win_rate=.5,average_red_loss=3.)
    with patch('algorithm.madsac.runner.evaluate_madsac',return_value=metrics),patch.object(r,'save_checkpoint') as save:
        r.evaluate();first=capsys.readouterr().out
        r.evaluate();second=capsys.readouterr().out
    assert first.startswith('[EVAL*] 20k | stochastic | win=50%')
    assert second.startswith('[EVAL] 20k | stochastic | win=50%')
    assert 'fire=50% | kill=50% | timeout=50% | len=600 | R=-20.00' in first
    assert save.call_count==1


def test_detailed_optimization_jsonl_preserved(tmp_path):
    r=runner();r.output_dir=tmp_path;r.trainer.vector_steps=4;r.replay=[1,2]
    fields=('critic1_loss','critic2_loss','actor_loss','entropy','mean_log_prob','q1_mean','q2_mean',
        'qmin_mean','target_q_mean','target_q_std','actor_gradient_norm','critic1_gradient_norm',
        'critic2_gradient_norm','action_abs_mean','action_saturation_fraction','actor_update_count','critic_update_count')
    metrics={key:float(i) for i,key in enumerate(fields)}
    r._write_optimization(metrics)
    data=json.loads((tmp_path/'optimization_metrics.jsonl').read_text())
    for key,value in metrics.items():assert data[key]==value
    assert data['replay_size']==2 and data['vector_steps']==4
