"""5v5 protocol and unchanged legacy/8v8 contracts."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.weapon import PairFireState, RearAspectWeaponEnvelope
from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.madsac.protocol import validate_madsac_config
from algorithm.stea_mappo.networks import decompose_observations, SpatioTemporalEntityAttentionActor
from tests.test_combat_v24 import state, plain

ROOT=Path(__file__).resolve().parents[1]
def config(version='2.5'):
    return load_config(ROOT/'configs'/('combat_environment.yaml' if version=='2.3' else f'combat_environment_v{version.replace(".", "")}.yaml'))

@pytest.mark.parametrize('size',[4,8])
def test_reject_wrong_team_size(size):
    cfg=config();cfg['scenario']['team_size']=size
    with pytest.raises(ValueError,match='requires team_size=5'):validate_config(cfg)

def test_config_and_reset():
    cfg=config();assert environment_dimensions(cfg)==(65,3,5)
    env=MultiUAVCombatEnv(cfg);obs,info=env.reset(31)
    assert obs.shape==(5,65)
    assert info['red_alive_mask'].shape==info['blue_alive_mask'].shape==(5,)
    assert isinstance(env.weapon,RearAspectWeaponEnvelope)
    for fire in (env.red_fire_states,env.blue_fire_states):
        assert isinstance(fire,PairFireState) and fire.armed.shape==(5,5) and fire.armed.all()
    bad=copy.deepcopy(cfg);bad['scenario']['formation_offsets'].pop()
    with pytest.raises(ValueError,match='formation_offsets'):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['weapon']['speed_gate']=True
    with pytest.raises(ValueError,match='weapon schema'):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['weapon'].pop('target_aspect_angle_max')
    with pytest.raises(ValueError,match='weapon schema'):validate_config(bad)
    inherited=config('2.4');inherited['environment_version']='2.5'
    inherited['scenario']['team_size']=5;inherited['scenario']['formation_offsets']=[-600.,-300.,0.,300.,600.]
    assert cfg==inherited

def test_formation_centroid_and_spacing():
    cfg=config()
    for key in ('altitude_perturbation_max','speed_perturbation_max','heading_perturbation_max'):cfg['scenario'][key]=0
    env=MultiUAVCombatEnv(cfg);_,info=env.reset(77)
    angle=info['radial_angle'];lateral=np.array([-np.sin(angle),np.cos(angle)])
    for team in (env.red,env.blue):
        xy=np.array([[a.x,a.y] for a in team]);offsets=xy@lateral
        np.testing.assert_allclose(offsets,[-600,-300,0,300,600],atol=1e-10)
        np.testing.assert_allclose(np.diff(offsets),300,atol=1e-10)
        assert abs(offsets.mean())<1e-10

def test_dead_entity_observation_slots():
    env=MultiUAVCombatEnv(config());env.reset(31)
    env.red[1].alive=False;env.blue[2].alive=False
    obs=env._observations()
    assert not obs[1].any()
    assert not obs[0,7:14].any()  # first ally is red[1]
    assert not obs[0,35+2*6:35+3*6].any()
    own,allies,enemies=decompose_observations(torch.tensor(obs))
    assert own.shape==(5,7) and allies.shape==(5,4,7) and enemies.shape==(5,5,6)

@pytest.mark.parametrize('speed',[200,250,300])
@pytest.mark.parametrize('target,expected',[(state(x=1000),True),(state(x=1000,psi=np.pi),False),
    (state(x=1000,psi=np.pi/3),False),(state(x=1000,z=-4000),False),(state(x=4001),False)])
def test_geometry_only_inherited(speed,target,expected):
    attacker=state(v=speed)
    for version in ('2.4','2.5'):
        assert MultiUAVCombatEnv(config(version))._in_fire_window(attacker,target)==expected

@pytest.mark.parametrize('side',['red','blue'])
def test_pair_lifecycle_symmetric(monkeypatch,side):
    env=MultiUAVCombatEnv(config());env.reset(31)
    attackers=[state(alive=False) for _ in range(5)];targets=[state(alive=False) for _ in range(5)]
    attackers[0]=state(v=200);targets[0]=state(x=1000);targets[1]=state(x=1500)
    fire=getattr(env,side+'_fire_states')
    monkeypatch.setattr(RearAspectWeaponEnvelope,'attempt_hit',lambda *args:False)
    run=lambda:env._entry_attempts(attackers,targets,fire,side)
    assert run()==[(0,0,False)] and fire.armed[0,1]
    assert run()==[(0,1,False)]
    assert run()==[]
    attackers[0].v=300;assert run()==[]
    targets[0].psi=np.pi;assert run()==[] and fire.armed[0,0]
    targets[0].psi=0;assert run()==[(0,0,False)]
    targets[0].alive=False;targets[1].psi=np.pi;assert run()==[]
    targets[1].psi=0;assert run()==[(0,1,False)]

@pytest.mark.parametrize('source,target',[('2.3','2.5'),('2.4','2.5'),('2.5','2.4')])
def test_environment_cross_version_rejected(source,target):
    obs,act,agents=environment_dimensions(config(source))
    metadata={'extra':dict(environment_version=source,observation_dim=obs,action_dim=act,num_agents=agents)}
    with pytest.raises(RuntimeError,match='environment_version'):validate_checkpoint_environment(metadata,config(target))

@pytest.mark.parametrize('version',['2.4','2.5'])
def test_madsac_remains_legacy_only(version):
    import yaml
    algo=yaml.safe_load((ROOT/'configs/madsac.yaml').read_text())
    with pytest.raises(RuntimeError,match='only v2.3'):validate_madsac_config(config(version),algo)

def test_stea_entity_slots_and_shared_parameters():
    actor=SpatioTemporalEntityAttentionActor();obs=torch.randn(2,5,65)
    own,allies,enemies=decompose_observations(obs)
    assert own.shape==(2,5,7) and allies.shape==(2,5,4,7) and enemies.shape==(2,5,5,6)
    dist,hidden=actor.distribution_step(obs,torch.zeros(2,5,128),torch.ones(2,5),torch.ones(2))[:2]
    assert dist.mean.shape==(2,5,3) and hidden.shape==(2,5,128)
    assert torch.isfinite(dist.mean).all() and torch.isfinite(hidden).all()
    assert sum(p.numel() for p in actor.parameters())==167494

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
def test_formal_config_only_dimensions_and_budget_change(name):
    import yaml
    old=yaml.safe_load((ROOT/f'configs/{name}_8v8.yaml').read_text())
    new=yaml.safe_load((ROOT/f'configs/{name}_5v5.yaml').read_text())
    old['network'].update(observation_dim=65,num_agents=5)
    old['network']['critic_type']='mlp'
    old['training']['total_sampled_steps']=3000000
    old['training'].update(entropy_coefficient=.001,target_kl=.015)
    old['implementation'].update(policy_std_mode='state_independent',log_std_init=-.5,
                                  mean_head_init_gain=.01,log_std_max=.5)
    assert old==new
    if name=='stea_mappo':
        from algorithm.stea_mappo.factory import validate_config as validate_algorithm
        validate_algorithm(new)

@pytest.mark.parametrize('case',json.loads((ROOT/'tests/fixtures/v24_final_reference.json').read_text()))
def test_old_protocol_full_episode_replay(case):
    env=MultiUAVCombatEnv(config(case['version']))
    assert plain(env.reset(case['seed']))==case['reset']
    rng=np.random.default_rng(case['seed']^0xBB);digest=hashlib.sha256()
    for _ in range(case['transitions']):
        action=env.fixed_policy.team_actions(env.red,env.blue) if case['mode']=='scripted' else rng.uniform(-1,1,(env.team_size,3)).astype(np.float32)
        digest.update(json.dumps(plain(env.step(action)),sort_keys=True,separators=(',',':')).encode())
    assert digest.hexdigest()==case['sha256']
