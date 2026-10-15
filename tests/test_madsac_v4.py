"""V5 vector-step delay regression (filename retained from v4). Algorithm-1 scheduling, scaled self-attention and strict checkpoint boundaries."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import torch
import yaml
from algorithm.madsac.networks import CentralizedAttentionQCritic
from algorithm.madsac.trainer import MADSACTrainer, MADSAC_IMPL_VERSION
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.replay_buffer import JointReplayBuffer
from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
from tools.validate_madsac_v4 import expected_counts


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(),'CUDA required'
    torch.set_num_threads(1)


def configs(version='337'):
    return tuple(yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in
        (f'configs/combat_environment_v{version}.yaml',f'configs/madsac_4v4_v{version}.yaml'))


def test_runtime_head_dim_scaling_and_self_key():
    c=CentralizedAttentionQCritic(observation_dim=45,hidden_dim=256,attention_heads=2).cuda()
    obs=torch.randn(3,4,45,device='cuda');act=torch.randn(3,4,3,device='cuda');alive=torch.ones(3,4,device='cuda')
    _,w=c(obs,act,alive,True)
    embedding=c.embedding(torch.cat((obs,act),-1))
    q=c.wq(embedding).view(3,4,2,128).transpose(1,2)
    k=c.wk(embedding).view(3,4,2,128).transpose(1,2)
    logits=(q@k.transpose(-2,-1))/math.sqrt(c.head_dim)
    expected=torch.softmax(logits,-1)
    torch.testing.assert_close(w,expected)
    assert c.head_dim==128 and (w.diagonal(dim1=-2,dim2=-1)>0).all()
    assert c.q_network[0].in_features==256 and c.value_activation.negative_slope==.01
    assert c.wq.bias is not None and c.wk.bias is not None and c.wv.bias is not None


def schedule_runner(tmp_path,ready=True):
    r=object.__new__(MADSACTrainingRunner)
    r.trainer=MADSACTrainer(observation_dim=45,num_agents=4,hidden_dim=32,device='cuda')
    r.replay=JointReplayBuffer(capacity=2048,num_agents=4,observation_dim=45)
    count=1024 if ready else 1000
    rng=np.random.default_rng(17)
    obs=rng.normal(size=(count,4,45)).astype(np.float32)
    r.replay.add_batch(obs,rng.uniform(-1,1,(count,4,3)).astype(np.float32),np.zeros((count,4),np.float32),
        obs,np.zeros(count,np.bool_),np.ones((count,4),np.float32),np.ones((count,4),np.float32))
    r.minibatch_size=1024;r.steps_per_update=100;r.num_updates=4;r.policy_delay=2
    r.first_update_sampled_steps=None;r.last_metrics={};r.output_dir=tmp_path
    return r


def test_threshold_readiness_four_rounds_and_reset(tmp_path):
    r=schedule_runner(tmp_path)
    r.trainer.transitions_since_update=99
    assert not r.maybe_update() and r.trainer.critic_update_count==0
    r.trainer.transitions_since_update=120;r.trainer.sampled_steps=1032;r.trainer.vector_steps=43
    before=deepcopy(r.trainer.actor.state_dict())
    with patch.object(r.replay,'sample',wraps=r.replay.sample) as sample:
        assert r.maybe_update()
        assert sample.call_count==4
        batches=[call for call in sample.call_args_list]
        assert all(call.args==(1024,) for call in batches)
    assert r.trainer.critic_update_count==4 and r.trainer.actor_update_count==0
    assert r.trainer.target_update_count==0 and r.trainer.transitions_since_update==0
    assert r.first_update_sampled_steps==1032
    assert all(torch.equal(v,r.trainer.actor.state_dict()[k]) for k,v in before.items())
    assert not r.maybe_update()


def test_replay_readiness_is_required_even_with_large_T(tmp_path):
    r=schedule_runner(tmp_path,False);r.trainer.transitions_since_update=9999
    assert not r.maybe_update() and r.trainer.transitions_since_update==9999


@pytest.mark.parametrize('cycles_before',[0,1])
@pytest.mark.parametrize('vector_steps',[100,101])
def test_delay_uses_vector_step_independent_of_update_cycle_parity(tmp_path,cycles_before,vector_steps):
    r=schedule_runner(tmp_path)
    r.trainer.update_cycle_count=cycles_before
    r.trainer.vector_steps=vector_steps
    r.trainer.sampled_steps=24*vector_steps  # always even: sampled_steps % d is also wrong
    r.trainer.transitions_since_update=120
    assert r.maybe_update()
    delayed=vector_steps % r.policy_delay == 0
    assert r.trainer.update_cycle_count==cycles_before+1
    assert r.trainer.critic_update_count==4
    assert r.trainer.actor_update_count==(4 if delayed else 0)
    assert r.trainer.target_update_count==(1 if delayed else 0)
    assert r.trainer.replay_sample_count_actor==(4 if delayed else 0)
    assert r.trainer.vector_steps==vector_steps


def test_count_reference_distinguishes_even_cycle_stride_from_vector_clock():
    # With readiness at vector43 and spacing4, all update steps are odd: 43,47,51.
    expected=expected_counts(24*51,steps_per_update=96)
    assert expected['update_cycle_count']==3
    assert expected['critic_update_count']==12
    assert expected['actor_update_count']==0 and expected['target_update_count']==0


def test_delayed_cycle_separate_batches_and_one_polyak_after_four_actor_steps(tmp_path):
    r=schedule_runner(tmp_path);events=[];samples=[]
    original=r.replay.sample
    def sampled(*args,**kwargs):
        b=original(*args,**kwargs);samples.append(b);events.append(('sample',id(b)));return b
    cq=r.trainer.update_critics;pi=r.trainer.update_actor;targets=r.trainer.update_targets
    def critic(b): events.append(('critic',id(b)));return cq(b)
    def actor(b): events.append(('actor',id(b)));return pi(b)
    def target(): events.append(('target',None));return targets()
    with patch.object(r.replay,'sample',side_effect=sampled),patch.object(r.trainer,'update_critics',side_effect=critic), \
         patch.object(r.trainer,'update_actor',side_effect=actor),patch.object(r.trainer,'update_targets',side_effect=target):
        r.trainer.transitions_since_update=1032;r.trainer.vector_steps=43;r.maybe_update()
        r.trainer.transitions_since_update=120;r.trainer.vector_steps=48;r.maybe_update()
    assert r.trainer.update_cycle_count==2 and r.trainer.critic_update_count==8
    assert r.trainer.actor_update_count==4 and r.trainer.target_update_count==1
    assert r.trainer.replay_sample_count_critic==8 and r.trainer.replay_sample_count_actor==4
    assert len(samples)==12 and len({id(b) for b in samples})==12
    assert events[-1][0]=='target'
    assert [event for event,_ in events if event!='sample']==['critic']*8+['actor']*4+['target']
    assert r.trainer.transitions_since_update==0


@pytest.mark.parametrize('version',['335','336','337'])
def test_formal_configs(version):
    e,c=configs(version);validate_madsac_config(e,c)
    t=c['training'];assert c['implementation_version']==5
    assert (t['steps_per_update'],t['num_updates'],t['policy_delay'])==(100,4,2)
    assert t['total_sampled_steps']==8400000 and t['num_train_envs']==24
    assert c['metadata']['project_implementation_choice']['policy_delay_clock']=='vector_steps'
    assert 'learning_starts' not in t and 'gradient_steps_per_vector_step' not in t
    for key in ('steps_per_update','num_updates','policy_delay'):
        assert key not in c['metadata']['paper_reported'] and key in c['metadata']['paper_unspecified']


@pytest.mark.parametrize('key,value',[('gradient_steps_per_vector_step',1),('learning_starts',10000),
                                    ('steps_per_update',0),('num_updates',0),('policy_delay',0)])
def test_reject_legacy_or_invalid_schedule(key,value):
    e,c=configs();c['training'][key]=value
    with pytest.raises(RuntimeError):validate_madsac_config(e,c)


def test_first_update_at_1032_without_10k_warmup():
    expected=expected_counts(4800)
    assert expected['first_update_sampled_steps']==1032
    assert expected['update_cycle_count']==32 and expected['critic_update_count']==128
    assert expected['actor_update_count']==64 and expected['target_update_count']==16
    assert expected['transitions_since_update']==48
    assert expected_counts(1008)['update_cycle_count']==0
    assert expected_counts(1032)['update_cycle_count']==1


@pytest.mark.parametrize('fault',['version3','version4','unscaled','head_dim','clock','target_schedule'])
def test_v3_or_forged_contract_rejected(tmp_path,fault):
    tr=MADSACTrainer(observation_dim=45,num_agents=4,hidden_dim=256,device='cuda')
    state=tr.checkpoint_state()
    if fault=='version3':state['implementation_version']=3
    elif fault=='version4':state['implementation_version']=4
    elif fault=='unscaled':state['network_architecture']['attention_logit_scaling']='none'
    elif fault=='head_dim':state['network_architecture']['attention_head_dim']=256
    elif fault=='clock':state['training_schedule']['update_cycle_clock']='completed_update_cycles_including_current_cycle_mod_policy_delay'
    else:state['training_schedule']['target_update_schedule']='per_actor_minibatch'
    path=tmp_path/'bad.pt';torch.save(state,path)
    with pytest.raises(RuntimeError):tr.load_for_evaluation(path)


def test_checkpoint_metadata_and_roundtrip(tmp_path):
    tr=MADSACTrainer(observation_dim=45,num_agents=4,hidden_dim=256,device='cuda')
    path=tmp_path/'good.pt';tr.save(path)
    state=torch.load(path,map_location='cuda',weights_only=False)
    assert MADSAC_IMPL_VERSION==state['implementation_version']==5
    assert state['network_architecture']['attention_head_dim']==128
    assert state['network_architecture']['attention_logit_scaling']=='sqrt_head_dim'
    assert state['training_schedule']['target_update_schedule']=='once_per_delayed_update_cycle'
    assert state['training_schedule']['policy_delay_clock']=='vector_steps'
    assert state['training_schedule']['delayed_policy_condition']=='vector_steps % policy_delay == 0'
    assert state['training_schedule']['update_cycle_clock']=='vector_environment_steps_mod_policy_delay'
    restored=MADSACTrainer(observation_dim=45,num_agents=4,hidden_dim=256,device='cuda',seed=97)
    restored.load_for_evaluation(path)
    for name in ('actor','critic1','critic2','target_actor','target_critic1','target_critic2'):
        for a,b in zip(getattr(tr,name).parameters(),getattr(restored,name).parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
