"""Formal 8v8 compatibility, critic fairness and pre-change 5v5 regression."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.common.control_factory import trainer_kwargs
from algorithm.mappo.networks import CentralizedMLPCritic
from algorithm.rmappo.factory import build_rmappo_trainer
from algorithm.ea_mappo.factory import build_ea_mappo_trainer
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from algorithm.stea_mappo.networks import decompose_observations
from algorithm.rmappo.runner import RMAPPOTrainingRunner
from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
from env.config import environment_dimensions

ROOT=Path(__file__).resolve().parents[1]
NAMES={'mappo':'mappo_8v8_formal','rmappo':'rmappo_8v8',
       'ea_mappo':'ea_mappo_8v8','stea_mappo':'stea_mappo_8v8_formal'}
BUILD={'rmappo':build_rmappo_trainer,'ea_mappo':build_ea_mappo_trainer,
       'stea_mappo':build_stea_mappo_trainer}


def cfg(name,five=False):
    path=f'{name}_5v5' if five else NAMES[name]
    return yaml.safe_load((ROOT/f'configs/{path}.yaml').read_text())


def build(name,device='cpu',seed=0,five=False):
    config=cfg(name,five)
    # Formal MAPPO runner passes the explicit seed to this constructor; its
    # evaluation factory has no seed override and loads checkpoint weights.
    return (MAPPOTrainer(**trainer_kwargs(config,device,seed)) if name=='mappo'
            else BUILD[name](config,device,seed=seed))


def observations(*shape):
    obs=torch.randn(*shape,104)
    obs[...,7:56].reshape(*shape,7,7)[...,-1]=1
    obs[...,56:].reshape(*shape,8,6)[...,-1]=1
    return obs


def test_v24_dimensions_and_formal_configs_are_exact_five_v_five_protocol_copies():
    env=yaml.safe_load((ROOT/'configs/combat_environment_v24.yaml').read_text())
    assert environment_dimensions(env)==(104,3,8)
    for name in NAMES:
        new,old=cfg(name),cfg(name,True)
        expected=copy.deepcopy(old)
        expected['network'].update(observation_dim=104,action_dim=3,num_agents=8)
        assert new==expected
        assert new['training']['total_sampled_steps']==3000000
        assert new['training']['entropy_coefficient']==.001
        assert new['training']['target_kl']==.015
        assert new['implementation']['log_std_max']==.5
        assert new['implementation']['policy_std_mode']=='state_independent'
        assert new['network']['critic_type']=='mlp'


@pytest.mark.parametrize('seed',[0,3])
def test_all_four_8v8_critics_exactly_equal_at_initialization(seed):
    trainers=[build(name,seed=seed) for name in NAMES]
    reference=trainers[0].critic.state_dict()
    for trainer in trainers:
        assert type(trainer.critic) is CentralizedMLPCritic
        assert trainer.critic.value_network[0].in_features==936
        assert sum(p.numel() for p in trainer.critic.parameters())==305921
        assert trainer.critic.state_dict().keys()==reference.keys()
        for key,tensor in trainer.critic.state_dict().items():
            assert tensor.shape==reference[key].shape
            assert torch.equal(tensor,reference[key])


@pytest.mark.parametrize('name',list(NAMES))
def test_8v8_gaussian_and_actor_dimensions(name):
    trainer=build(name);actor=trainer.actor
    obs=observations(2,8);alive=torch.ones(2,8)
    assert actor.policy_std_mode=='state_independent'
    assert actor.log_std_parameter.shape==(3,)
    assert torch.equal(actor.log_std_parameter,torch.full((3,),-.5))
    assert (actor.log_std_min,actor.log_std_max)==(-5.,.5)
    if name in ('rmappo','stea_mappo'):
        actions,hidden=actor(obs,torch.zeros(2,8,128),alive,torch.ones(2))
        assert hidden.shape==(2,8,128)
    else: actions=actor(obs)
    assert actions.shape==(2,8,3) and torch.isfinite(actions).all()
    if name=='rmappo':
        assert actor.flat_encoder[0].in_features==104
        assert actor.gru.input_size==actor.gru.hidden_size==128
    if name=='stea_mappo':
        assert actor.gru.input_size==actor.gru.hidden_size==128
        assert trainer.network_architecture['actor_type']=='stea'
    assert sum(p.numel() for p in actor.parameters())=={
        'mappo':93446,'rmappo':129414,'ea_mappo':68038,'stea_mappo':167110}[name]


@pytest.mark.parametrize('name',['ea_mappo','stea_mappo'])
@pytest.mark.parametrize('side,start,end,slots,width',[('ally',7,56,7,7),('enemy',56,104,8,6)])
def test_8v8_entity_decomposition_dead_empty_and_permutation(name,side,start,end,slots,width):
    actor=build(name).actor;obs=observations(2,8);alive=torch.ones(2,8)
    own,allies,enemies=decompose_observations(obs)
    assert own.shape==(2,8,7) and allies.shape==(2,8,7,7) and enemies.shape==(2,8,8,6)
    encoded=actor.encode_entities(obs)
    assert [value.shape for value in encoded]==[(2,8,64),(2,8,7,64),(2,8,8,64)]
    entities=obs[...,start:end].reshape(2,8,slots,width);entities[...,0,-1]=0
    fused,diag=actor.spatial(obs,alive)
    assert diag[f'{side}_attention_weights'].shape==(2,8,2,slots)
    assert torch.count_nonzero(diag[f'{side}_attention_weights'][...,0])==0
    changed=obs.clone();changed[...,start:start+width-1]=1000
    assert torch.equal(fused,actor.spatial(changed,alive)[0])
    perm=list(reversed(range(slots)));changed=obs.clone()
    changed[...,start:end]=entities[...,perm,:].reshape(2,8,slots*width)
    permuted,pdiag=actor.spatial(changed,alive)
    torch.testing.assert_close(fused,permuted,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(diag[f'{side}_attention_weights'][...,perm],pdiag[f'{side}_attention_weights'])
    entities[...,-1]=0
    _,diag=actor.spatial(obs,alive)
    assert torch.count_nonzero(diag[f'{side}_attention_weights'])==0
    assert torch.count_nonzero(diag[f'{side}_context'])==0


def test_ea_stea_8v8_spatial_features_exactly_equal_with_same_weights():
    ea,stea=build('ea_mappo').actor,build('stea_mappo').actor
    for name in ('self_encoder','ally_encoder','enemy_encoder','ally_attention','enemy_attention','spatial_fusion'):
        getattr(ea,name).load_state_dict(getattr(stea,name).state_dict())
    obs=observations(2,8);alive=torch.ones(2,8);alive[0,1]=0
    a,ad=ea.spatial(obs,alive);s,sd=stea.spatial(obs,alive)
    assert torch.equal(a,s) and all(torch.equal(ad[k],sd[k]) for k in ad)


@pytest.mark.parametrize('name',list(NAMES))
def test_5v5_actor_weights_forward_and_deterministic_action_unchanged(name):
    # This fixture was captured before the dimension compatibility edits.
    fixture=torch.load(ROOT/'tests/fixtures/formal_5v5_before_8v8.pt',weights_only=True)
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer=build(name,five=True);actor=trainer.actor;ref=fixture['algorithms'][name]
        assert actor.state_dict().keys()==ref['state_dict'].keys()
        for key,value in actor.state_dict().items(): assert torch.equal(value,ref['state_dict'][key])
        for key,value in trainer.critic.state_dict().items(): assert torch.equal(value,ref['critic_state_dict'][key])
        assert getattr(trainer,'network_architecture',{})==ref['network_architecture']
        obs,alive,hidden,starts=(fixture[key] for key in ('observations','alive','hidden','starts'))
        with torch.no_grad():
            if name in ('rmappo','stea_mappo'):
                d,h,_=actor.distribution_step(obs,hidden,alive,starts)
                forward=actor(obs,hidden,alive,starts)
                actions=trainer.act(obs.numpy(),alive.numpy(),hidden.numpy(),starts.numpy(),deterministic=True)[0]
                assert torch.equal(h,ref['next_hidden'])
            else:
                d=actor.distribution(obs);forward=(actor(obs),)
                actions=trainer.act(obs.numpy(),alive.numpy(),deterministic=True)
        assert torch.equal(d.mean,ref['mean']) and torch.equal(d.scale,ref['scale'])
        assert all(torch.equal(a,b) for a,b in zip(forward,ref['forward']))
        assert torch.equal(torch.from_numpy(actions),ref['deterministic_action'])
    finally: torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('name',['rmappo','stea_mappo'])
def test_8v8_recurrent_ratio_replay_cuda(name):
    assert torch.cuda.is_available(),'CUDA mandatory'
    trainer=build(name,'cuda',seed=41)
    t,e,n=40,2,8
    obs=observations(t,e,n).numpy();alive=np.ones((t,e,n),np.float32)
    alive[5:11,0,1]=0
    starts=np.zeros((t,e),np.float32);starts[0]=1;starts[17,0]=1
    hidden=np.zeros((e,n,128),np.float32)
    actions,raws,logs,states=[],[],[],[]
    for step in range(t):
        states.append(hidden.copy())
        a,r,l,hidden=trainer.act(obs[step],alive[step],hidden,starts[step])
        actions.append(a);raws.append(r);logs.append(l)
        if step+1<t:
            hidden*=alive[step+1,...,None];hidden[starts[step+1]>.5]=0
    from algorithm.rmappo.trainer import RecurrentRolloutBatch as RB
    from algorithm.stea_mappo.trainer import RecurrentRolloutBatch as SB
    cls=RB if name=='rmappo' else SB
    dones=np.zeros((t,e),np.float32);dones[:-1]=starts[1:]
    batch=cls(obs,np.stack(actions),np.stack(raws),np.stack(logs),np.zeros((t,e,n),np.float32),
        dones,alive,obs+.01,np.concatenate((alive[1:],alive[-1:]),axis=0),np.stack(states),starts)
    assert trainer.audit_rollout_ratio(batch)['pre_update_ratio_max_abs_error']<1e-4
    batch.old_log_probs[1,0,0]+=.1
    with pytest.raises(RuntimeError,match='ratio mismatch'):trainer.audit_rollout_ratio(batch)


@pytest.mark.parametrize('name,cls',[('rmappo',RMAPPOTrainingRunner),('ea_mappo',EAMAPPOTrainingRunner)])
def test_control_runner_rejects_mixed_dimensions_and_unsupported_versions(name,cls,tmp_path):
    env=yaml.safe_load((ROOT/'configs/combat_environment_v24.yaml').read_text())
    with pytest.raises(ValueError,match='dimension'):
        cls(env,cfg(name,True),device='cuda',output_dir=tmp_path)
    env5=yaml.safe_load((ROOT/'configs/combat_environment_v25.yaml').read_text())
    with pytest.raises(ValueError,match='dimension'):
        cls(env5,cfg(name),device='cuda',output_dir=tmp_path)
    env['environment_version']='2.3'
    with pytest.raises(ValueError,match='2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2 or 3.3'):
        cls(env,cfg(name),device='cuda',output_dir=tmp_path)


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
def test_8v8_control_checkpoint_validation_accepts_v24_and_rejects_v25_config(name):
    from algorithm.common.protocol import config_sha256
    from algorithm.rmappo.protocol import validate_checkpoint as vr
    from algorithm.ea_mappo.protocol import validate_checkpoint as ve
    trainer=build(name);config=cfg(name)
    env=yaml.safe_load((ROOT/'configs/combat_environment_v24.yaml').read_text())
    ac=sum(p.numel() for p in trainer.actor.parameters());cc=sum(p.numel() for p in trainer.critic.parameters())
    state=trainer.checkpoint_state(dict(environment_version='2.4',observation_dim=104,action_dim=3,
        num_agents=8,training_seed=0,training_gamma=.99,training_num_envs=16,training_total_sampled_steps=3000000,
        training_smoke=False,effective_hidden_dim=256,critic_type='mlp',
        network_architecture=trainer.network_architecture,actor_parameter_count=ac,critic_parameter_count=cc,
        total_parameter_count=ac+cc,environment_config_sha256=config_sha256(env),
        algorithm_config_sha256=config_sha256(config),**{name+'_impl_version':1}))
    validate=vr if name=='rmappo' else ve
    validate(state,env,config)
    with pytest.raises(RuntimeError,match='dimension'):validate(state,env,cfg(name,True))
    env5=yaml.safe_load((ROOT/'configs/combat_environment_v25.yaml').read_text())
    with pytest.raises(RuntimeError,match='environment_version'):validate(state,env5,cfg(name,True))
