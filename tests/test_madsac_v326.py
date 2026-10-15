"""Formal v3.26 protocol, configured architecture and checkpoint isolation."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.common.protocol import config_sha256
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.madsac.critic_contract import CRITIC_ARCHITECTURE,schedule_contract


def configs():
    return (yaml.safe_load(Path('configs/combat_environment_v326.yaml').read_bytes()),
            yaml.safe_load(Path('configs/madsac_5v5_v326.yaml').read_bytes()))


def metadata(e,c,smoke=False):
    return dict(implementation_version=5,training_schedule=schedule_contract(c['training']['steps_per_update'],c['training']['num_updates'],c['training']['policy_delay']),environment_version='3.26',environment_config_sha256=config_sha256(e),
        algorithm_config_sha256=config_sha256(c),training_seed=c['training']['seed'],
        observation_dim=56,action_dim=3,num_agents=5,training_smoke=smoke,
        formal_exact_resume_supported=False,replay_buffer_in_checkpoint=False,
        network_architecture=dict(**CRITIC_ARCHITECTURE,shared_actor=True,twin_independent_critics=True,
            target_actor=True,target_critics=2,attention_heads=2,hidden_dim=32 if smoke else 256,
            attention_head_dim=16 if smoke else 128,
            observation_dim=56,action_dim=3,num_agents=5))


def test_formal_config_and_implementation_choice_metadata():
    e,c=configs();validate_madsac_config(e,c)
    assert c['environment']=={'environment_version':'3.26','team_size':5}
    assert c['network']==dict(observation_dim=56,action_dim=3,num_agents=5,
        actor_hidden_layers=[256,256],critic_hidden_layers=[256,256],attention_heads=2)
    for k,v in dict(actor_learning_rate=1e-4,critic_learning_rate=1e-4,gamma=.99,
        alpha=.1,tau=.001,replay_capacity=1000000,minibatch_size=1024,
        num_train_envs=24,total_sampled_steps=8400000,device='cuda',
        steps_per_update=100,policy_delay=2,num_updates=4).items():
        assert c['training'][k]==v
    for k in ('steps_per_update','policy_delay','num_updates'):
        assert k in c['metadata']['paper_unspecified']
        assert k not in c['metadata']['paper_reported']


@pytest.mark.parametrize('field,value',[('observation_dim',52),('observation_dim',56.5),
    ('observation_dim','56'),('observation_dim',56.0),('action_dim',4),('num_agents',4)])
def test_strict_dimensions(field,value):
    e,c=configs();c['network'][field]=value
    with pytest.raises(RuntimeError,match='exact dimensions'):validate_madsac_config(e,c)


@pytest.mark.parametrize('field,value',[('environment_version','3.25'),('team_size',4)])
def test_declared_environment_identity(field,value):
    e,c=configs();c['environment'][field]=value
    with pytest.raises(RuntimeError,match='identity'):validate_madsac_config(e,c)


@pytest.mark.parametrize('field,value',[('actor_hidden_layers',[128,128]),('critic_hidden_layers',[256,128]),('attention_heads',4)])
def test_formal_architecture_contract(field,value):
    e,c=configs();c['network'][field]=value
    with pytest.raises(RuntimeError,match='256/256'):validate_madsac_config(e,c)


def test_formal_256_networks_cuda_and_checkpoint(tmp_path):
    assert torch.cuda.is_available(),'CUDA required'
    e,c=configs();trainer=build_madsac_trainer(c,'cuda')
    obs=torch.randn(3,5,56,device='cuda');actions,raw,lp=trainer.actor.sample(obs)
    assert actions.shape==raw.shape==(3,5,3) and lp.shape==(3,5)
    assert torch.isfinite(lp).all() and (actions.abs()<=1).all()
    q,attention=trainer.critic1(obs,actions,torch.ones(3,5,device='cuda'),return_attention=True)
    assert q.shape==(3,5) and attention.shape==(3,2,5,5)
    for module in (trainer.actor,trainer.critic1,trainer.critic2):
        assert next(module.parameters()).shape[0]==256
    extra=metadata(e,c);path=tmp_path/'formal.pt';trainer.save(path,extra)
    state=torch.load(path,map_location='cpu',weights_only=False)
    validate_madsac_checkpoint(state,e,c)
    assert state['formal_exact_resume_supported'] is False
    assert state['replay_buffer_included'] is False
    restored=build_madsac_trainer(c,'cuda');restored.load_for_evaluation(path)
    x=np.ones((1,5,56),np.float32)
    np.testing.assert_array_equal(trainer.act(x,deterministic=True),restored.act(x,deterministic=True))


def test_old_4v4_checkpoint_rejected_even_if_metadata_forged():
    e,c=configs();old=MADSACTrainer(hidden_dim=256,device='cuda')
    extra=metadata(e,c);extra.update(environment_version='2.3',observation_dim=52,num_agents=4)
    state=old.checkpoint_state(extra)
    with pytest.raises(RuntimeError,match='environment_version'):validate_madsac_checkpoint(state,e,c)
    # Even new-looking identity cannot hide actual old actor input width.
    state['extra']=metadata(e,c)
    with pytest.raises(RuntimeError,match='actor architecture'):validate_madsac_checkpoint(state,e,c)


@pytest.mark.parametrize('field',['environment_config_sha256','algorithm_config_sha256','training_seed'])
def test_checkpoint_identity_stays_strict(field):
    e,c=configs();trainer=build_madsac_trainer(c,'cuda',hidden_dim=32)
    extra=metadata(e,c,True);extra[field]='wrong'
    with pytest.raises(RuntimeError,match=field):validate_madsac_checkpoint(trainer.checkpoint_state(extra),e,c)


@pytest.mark.parametrize('field,value',[('shared_actor',False),('twin_independent_critics',False),
    ('attention_heads',4),('hidden_dim',256),('num_agents',4)])
def test_smoke_architecture_provenance(field,value):
    e,c=configs();trainer=build_madsac_trainer(c,'cuda',hidden_dim=32)
    extra=metadata(e,c,True);extra['network_architecture'][field]=value
    with pytest.raises(RuntimeError):validate_madsac_checkpoint(trainer.checkpoint_state(extra),e,c)
