"""Identity-only v3.41 registrations; frozen configs and strict CUDA contracts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from env.config import load_config,environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from algorithm.maac import protocol as maac
from algorithm.eaa_mappo import protocol as attention

ROOT=Path(__file__).resolve().parents[1]
NAMES=('maac','ea_mappo','eaa_mappo')


def configs(name,version='341'):
    return load_config(ROOT/f'configs/combat_environment_v{version}.yaml'),yaml.safe_load(
        (ROOT/f'configs/{name}_4v4_v{version}.yaml').read_text(encoding='utf-8'))


def validate(name,e,c):
    return (maac.validate_config_pair if name=='maac' else attention.validate_config)(e,c)


@pytest.mark.parametrize('name',NAMES)
def test_exact_identity_only_config_delta(name):
    e,c=configs(name);_,old=configs(name,'339');baseline=deepcopy(c)
    baseline['environment']['environment_version']='3.39'
    assert baseline==old
    before=deepcopy(c);validate(name,e,c);assert c==before
    assert c['environment']==dict(environment_version='3.41',team_size=4)
    assert c['training']['total_sampled_steps']==3000000
    assert environment_dimensions(e)==(45,3,4)


@pytest.mark.parametrize('name',('mappo',)+NAMES)
def test_real_environment_and_dimensions(name,tmp_path):
    e,c=configs(name);env=MultiUAVCombatEnv(e);obs,_=env.reset(1)
    assert type(env).__name__=='CombatEnvironmentV341' and obs.shape==(4,45)
    assert tuple(c['network'][k] for k in ('observation_dim','action_dim','num_agents'))==(45,3,4)
    if name=='mappo':
        from algorithm.mappo.runner import MAPPOTrainingRunner
        with patch('algorithm.mappo.runner.MAPPOTrainer',side_effect=RuntimeError('identity passed')):
            with pytest.raises(RuntimeError,match='identity passed'):MAPPOTrainingRunner(e,c,output_dir=tmp_path)
    else:validate(name,e,c)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('version',['338','339'])
def test_historical_protocols_unchanged(name,version):
    e,c=configs(name,version);validate(name,e,c)
    if name!='maac':assert attention.checkpoint_family(e)=='eaa_mappo_v'+version


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('case',['algorithm','missing_identity','old_identity','wrong_team','obs','actions','agents','hidden','heads'])
def test_wrong_identity_or_network_rejected(name,case):
    e,c=configs(name)
    if case=='algorithm':c['algorithm']='invalid'
    elif case=='missing_identity':c.pop('environment')
    elif case=='old_identity':c['environment']['environment_version']='3.39'
    elif case=='wrong_team':c['environment']['team_size']=5
    else:
        key={'obs':'observation_dim','actions':'action_dim','agents':'num_agents','heads':'attention_heads',
             'hidden':'hidden_dim' if name=='maac' else 'actor_hidden_layers'}[case]
        c['network'][key]=[128,128] if key.endswith('layers') else c['network'][key]+1
    with pytest.raises((RuntimeError,ValueError)):validate(name,e,c)


FROZEN=[(section,k) for section in ('training','implementation','runtime_logging') for k in configs('ea_mappo')[1][section]]


@pytest.mark.parametrize('name',['ea_mappo','eaa_mappo'])
@pytest.mark.parametrize('section,key',FROZEN)
def test_all_attention_numerical_protocol_fields_frozen(name,section,key):
    e,c=configs(name);value=c[section][key]
    c[section][key]=not value if isinstance(value,bool) else value+1 if isinstance(value,(int,float)) else 'invalid'
    with pytest.raises(ValueError,match=section):attention.validate_config(e,c)


def test_families_and_formal_maac_parameters():
    assert attention.checkpoint_family(configs('ea_mappo')[0])=='eaa_mappo_v341'
    _,c=configs('maac');n,t=c['network'],c['training']
    assert n['hidden_dim']==128 and n['attention_heads']==4
    expected=dict(gamma=.99,alpha=.01,tau=.005,num_train_envs=12,replay_capacity=1000000,
        minibatch_size=1024,learning_starts=1024,sampled_steps_per_update=100,
        critic_updates_per_cycle=4,policy_updates_per_cycle=4,baseline_samples=16)
    assert all(t[k]==v for k,v in expected.items())
    assert c['implementation']['variant']==maac.VARIANT


@pytest.fixture(scope='module')
def cuda_trainers():
    assert torch.cuda.is_available(),'CUDA mandatory for network/checkpoint audit; no CPU fallback'
    torch.set_num_threads(1)
    from algorithm.maac.trainer import MAACTrainer
    from algorithm.eaa_mappo.factory import build_eaa_mappo_trainer
    result={}
    for name in NAMES:
        e,c=configs(name)
        result[name]=MAACTrainer(e,c,'cuda',seed=1,smoke=False) if name=='maac' else build_eaa_mappo_trainer(c,e,'cuda',1)
    return result


def test_actual_formal_network_classes_and_heads(cuda_trainers):
    for name,t in cuda_trainers.items():
        assert all(p.device.type=='cuda' for p in t.actor.parameters())
        if name=='maac':
            assert type(t.actor).__name__=='GaussianLocalActor' and type(t.critic).__name__=='CentralizedAttentionCritic'
            assert t.actor.backbone[0].out_features==128 and t.critic.heads==4
        else:
            assert type(t.actor).__name__=='EntityAttentionMAPPOActor'
            assert type(t.critic).__name__==('CentralizedMLPCritic' if name=='ea_mappo' else 'CentralizedValueCritic')
            assert t.actor.attention_heads==4 and t.actor.entity_embedding_dim==64
            assert t.actor_hidden_dim==t.critic_hidden_dim==256
            if name=='eaa_mappo':assert t.attention_heads==2


@pytest.fixture(scope='module')
def states(cuda_trainers,tmp_path_factory):
    directory=tmp_path_factory.mktemp('v341_config_checkpoints');result={}
    for name,t in cuda_trainers.items():
        if name=='maac':
            path=directory/'new_maac.pt';t.save(path);result[name]=torch.load(path,map_location='cuda',weights_only=False)
        else:result[name]=t.checkpoint_state()
    return result


@pytest.mark.parametrize('name',NAMES)
def test_matching_actual_checkpoint_accepted(name,states):
    e,c=configs(name)
    (maac.validate_checkpoint if name=='maac' else attention.validate_checkpoint)(states[name],e,c)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('version',['338','339'])
def test_actual_checkpoint_cross_environment_rejected_both_directions(name,version,states):
    old_e,old_c=configs(name,version);e,c=configs(name)
    check=maac.validate_checkpoint if name=='maac' else attention.validate_checkpoint
    with pytest.raises((RuntimeError,ValueError)):check(states[name],old_e,old_c)
    old=deepcopy(states[name])
    old['extra']['environment_version']=old_e['environment_version']
    if name!='maac':old['environment_version']=old_e['environment_version'];old['checkpoint_family']=attention.checkpoint_family(old_e)
    with pytest.raises((RuntimeError,ValueError)):check(old,e,c)


@pytest.mark.parametrize('name',['ea_mappo','eaa_mappo'])
def test_variants_cannot_impersonate_each_other(name,states):
    other='eaa_mappo' if name=='ea_mappo' else 'ea_mappo';e,c=configs(other)
    with pytest.raises(RuntimeError):attention.validate_checkpoint(states[name],e,c)
    forged=deepcopy(states[name]);forged['algorithm']=c['algorithm']
    forged['algorithm_variant']=forged['extra']['algorithm_variant']=c['algorithm']
    with pytest.raises(RuntimeError):attention.validate_checkpoint(forged,e,c)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('field',['environment_config_sha256','algorithm_config_sha256','network_architecture'])
def test_actual_checkpoint_fingerprint_and_architecture_protected(name,field,states):
    e,c=configs(name);s=deepcopy(states[name]);s['extra'][field]='invalid'
    if name!='maac':s[field]='invalid'
    with pytest.raises(RuntimeError):(maac.validate_checkpoint if name=='maac' else attention.validate_checkpoint)(s,e,c)


def test_maac_variant_implementation_and_resume_protected(states):
    e,c=configs('maac')
    for field in ('variant','training_smoke'):
        s=deepcopy(states['maac']);s['extra'][field]='invalid'
        with pytest.raises(RuntimeError):maac.validate_checkpoint(s,e,c)
    s=deepcopy(states['maac']);s['implementation_version']=2
    with pytest.raises(RuntimeError):maac.validate_checkpoint(s,e,c)
    with pytest.raises(RuntimeError,match='exact resume unsupported'):maac.validate_checkpoint_for_resume(states['maac'],e,c)
    assert not states['maac']['extra']['replay_buffer_in_checkpoint']


def test_ea_uses_current_attention_family_entry():
    source=(ROOT/'algorithm/train_eaa_mappo.py').read_text(encoding='utf-8')
    assert 'algorithm.eaa_mappo.entry import run_variant' in source
    assert 'train_ea_mappo' not in source


@pytest.mark.parametrize('name',['ea_mappo','eaa_mappo'])
def test_public_checkpoint_loader_dispatches_v341_attention_family(name,states,tmp_path,cuda_trainers):
    from tools.record_combat_episode import load_recording_policy
    e,c=configs(name);path=tmp_path/(name+'.pt');torch.save(states[name],path)
    with patch('algorithm.ea_mappo.protocol.validate_checkpoint',side_effect=AssertionError('legacy EA path')):
        loaded,_,_=load_recording_policy(path,e,c,'cuda')
    obs,_=MultiUAVCombatEnv(e).reset(3);alive=np.ones((1,4),dtype=np.float32)
    np.testing.assert_array_equal(loaded.act(obs[None],alive,deterministic=True),
                                  cuda_trainers[name].act(obs[None],alive,deterministic=True))
