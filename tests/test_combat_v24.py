"""Version isolation, exact legacy replay, rear geometry and pair-entry lifecycle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.config import validate_config, environment_dimensions
from env.observation import OBSERVATION_DIM, observation_dim_for_team_size
from env.weapon import FireState, PairFireState, RearAspectWeaponEnvelope
from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.madsac.protocol import validate_madsac_config
from algorithm.stea_mappo.networks import decompose_observations, SpatioTemporalEntityAttentionActor

ROOT=Path(__file__).resolve().parents[1]

def config(version='2.4'):
    return yaml.safe_load((ROOT/'configs'/('combat_environment_v24.yaml' if version=='2.4' else 'combat_environment.yaml')).read_text())

def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value

@pytest.mark.parametrize('case',json.loads((ROOT/'tests/fixtures/v23_reference.json').read_text()))
def test_v23_exact_fixed_seed_replay(case):
    env=MultiUAVCombatEnv()
    assert plain(env.reset(case['seed'])) == case['reset']
    rng=np.random.default_rng(case['seed']^0xBB)
    digest=hashlib.sha256()
    for _ in range(case['transitions']):
        action=env.fixed_policy.team_actions(env.red,env.blue) if case['mode']=='scripted' else rng.uniform(-1,1,(4,3)).astype(np.float32)
        digest.update(json.dumps(plain(env.step(action)),sort_keys=True,separators=(',',':')).encode())
    assert digest.hexdigest() == case['sha256']
    assert isinstance(env.red_fire_states[0],FireState)

def state(x=0,y=0,z=-3000,v=250,psi=0,theta=0,alive=True):
    return AircraftState(x,y,z,v,theta,psi,alive)

@pytest.mark.parametrize('target,attacker,expected',[
    (state(x=1000),state(),True),
    (state(x=1000,psi=np.pi),state(),False),
    (state(x=1000,psi=np.pi/3),state(),False),
    (state(x=1000,v=251),state(),False),
    (state(x=1000,v=250),state(v=250),True),
    (state(x=1000,z=-4000),state(),False),
    (state(x=4001),state(),False),
    (state(x=4000),state(),True),
    (state(x=1000,alive=False),state(),False),
    (state(x=1000),state(alive=False),False),
])
def test_v24_joint_attack_geometry(target,attacker,expected):
    env=MultiUAVCombatEnv(config())
    geo=engagement_geometry(attacker,target)
    assert env.weapon.qualifies(geo,attacker.v,target.v) == (expected or not attacker.alive or not target.alive)
    assert env._in_fire_window(attacker,target) == expected

def test_head_on_accepted_only_by_legacy():
    a,b=state(),state(x=1000,psi=np.pi)
    geo=engagement_geometry(a,b)
    assert geo.target_aspect == pytest.approx(np.pi)
    assert MultiUAVCombatEnv()._in_fire_window(a,b)
    assert not MultiUAVCombatEnv(config())._in_fire_window(a,b)
    assert engagement_geometry(a,state(x=1000)).target_aspect == 0

def pair_env(monkeypatch):
    env=MultiUAVCombatEnv(config()); env.reset(31)
    env.red=[state(alive=False) for _ in range(8)]
    env.blue=[state(alive=False) for _ in range(8)]
    env.red[0]=state(); env.blue[0]=state(x=1000)
    monkeypatch.setattr(RearAspectWeaponEnvelope,'attempt_hit',lambda *args:False)
    return env

def fire(env):
    return env._entry_attempts(env.red,env.blue,env.red_fire_states,'red')

@pytest.mark.parametrize('invalid_condition',['range','cone','aspect','speed','target_dead','attacker_dead'])
def test_pair_invalid_then_reentry(monkeypatch,invalid_condition):
    env=pair_env(monkeypatch)
    assert fire(env)==[(0,0,False)]
    assert fire(env)==[]  # persistent valid window is not a repeated attack
    original=env.blue[0].copy()
    if invalid_condition=='range': env.blue[0].x=5000
    if invalid_condition=='cone': env.blue[0].y=3000
    if invalid_condition=='aspect': env.blue[0].psi=np.pi
    if invalid_condition=='speed': env.blue[0].v=251
    if invalid_condition=='target_dead': env.blue[0].alive=False
    if invalid_condition=='attacker_dead': env.red[0].alive=False
    assert fire(env)==[]
    assert env.red_fire_states.armed[0,0]
    env.blue[0]=original; env.red[0].alive=True
    assert fire(env)==[(0,0,False)]

def test_pair_nearest_only_selected_disarmed_and_switch(monkeypatch):
    env=pair_env(monkeypatch); env.blue[1]=state(x=1500)
    assert fire(env)==[(0,0,False)]
    assert not env.red_fire_states.armed[0,0]
    assert env.red_fire_states.armed[0,1]
    assert fire(env)==[(0,1,False)]
    assert fire(env)==[]

def test_pair_tie_index_and_one_attempt_per_attacker(monkeypatch):
    env=pair_env(monkeypatch); env.blue[1]=state(x=1000)
    env.red[1]=state(x=-100)
    assert fire(env)==[(0,0,False),(1,0,False)]
    assert env.red_fire_states.armed[0,1]
    assert env.red_fire_states.armed[1,1]

def test_pair_reset_and_sides_independent(monkeypatch):
    env=pair_env(monkeypatch); fire(env)
    assert env.blue_fire_states.armed.all()
    env.reset(31)
    assert isinstance(env.red_fire_states,PairFireState)
    assert env.red_fire_states.armed.shape==(8,8)
    assert env.red_fire_states.armed.all() and env.blue_fire_states.armed.all()

def test_simultaneous_mutual_kill_and_shared_credit(monkeypatch):
    env=pair_env(monkeypatch); env.red[1]=state()
    red,blue=env._resolve_combat([(0,0,True),(1,0,True)],[(0,0,True)])
    assert red=={0:[0,1]} and blue=={0:[0]}
    assert not env.red[0].alive and not env.blue[0].alive
    assert env.combat_counts['red']['attack_kills']==1

def test_v23_global_state_blocks_other_valid_target(monkeypatch):
    env=MultiUAVCombatEnv(); env.reset(31)
    env.red=[state(alive=False) for _ in range(4)]; env.red[0]=state()
    env.blue=[state(alive=False) for _ in range(4)]
    env.blue[0]=state(x=1000);env.blue[1]=state(x=1500)
    from env.weapon import WeaponEnvelope
    monkeypatch.setattr(WeaponEnvelope,'attempt_hit',lambda *args:False)
    assert fire(env)==[(0,0,False)]
    env.blue[0].alive=False
    assert fire(env)==[] # legacy attacker-global arming remains unchanged
    env.blue[1].x=5000; assert fire(env)==[]
    env.blue[1].x=1500; assert fire(env)==[(0,1,False)]

def test_v24_observation_slots_death_and_empty_context():
    env=MultiUAVCombatEnv(config());obs,info=env.reset(31)
    assert (env.team_size,env.observation_dim,env.action_dim)==(8,104,3)
    assert obs.shape==(8,104) and info['environment_version']=='2.4'
    assert OBSERVATION_DIM==52 and MultiUAVCombatEnv.team_size==4
    assert observation_dim_for_team_size(8)==7+7*7+8*6==104
    env.red[1].alive=False; env.blue[2].alive=False; obs=env._observations()
    assert not obs[1].any() and not obs[0,7:14].any() and not obs[0,68:74].any()
    env.red[1:]=[state(alive=False) for _ in range(7)]
    env.blue=[state(alive=False) for _ in range(8)]
    obs=env._observations(); assert not obs[0,7:].any()
    own,allies,enemies=decompose_observations(torch.from_numpy(obs))
    assert own.shape==(8,7) and allies.shape==(8,7,7) and enemies.shape==(8,8,6)
    actor=SpatioTemporalEntityAttentionActor()
    distribution,_,_=actor.distribution_step(torch.from_numpy(obs),torch.zeros(8,128),torch.from_numpy(env.red_alive_mask),torch.tensor(0.))
    assert torch.isfinite(distribution.mean).all()

@pytest.mark.parametrize('version,wrong_size',[('2.3',8),('2.4',4)])
def test_version_contract_rejects_wrong_size(version,wrong_size):
    cfg=config(version);cfg['scenario']['team_size']=wrong_size
    with pytest.raises(ValueError):validate_config(cfg)

def test_protocol_versions_and_madsac_rejection():
    assert environment_dimensions(config())==(104,3,8)
    with pytest.raises(RuntimeError,match='environment_version mismatch'):
        validate_checkpoint_environment({'extra':{'environment_version':'2.3'}},config())
    cfg=yaml.safe_load((ROOT/'configs/madsac.yaml').read_text())
    with pytest.raises(RuntimeError,match='unsupported protocol'):
        validate_madsac_config(config(),cfg)

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
def test_8v8_algorithm_preserves_optimizer_and_requested_runtime(name):
    old=yaml.safe_load((ROOT/f'configs/{name}.yaml').read_text())
    new=yaml.safe_load((ROOT/f'configs/{name}_8v8.yaml').read_text())
    old['network'].update(observation_dim=104,num_agents=8)
    old['training'].update(num_train_envs=16,total_sampled_steps=2000000)
    assert new==old

def test_v24_config_preserves_physics_policy_rewards():
    old=config('2.3');new=config()
    for section in ('aircraft','simulation','arena','action','blue_policy','reward','observation'):
        assert new[section]==old[section]
    for key,value in old['weapon'].items(): assert new['weapon'][key]==value
    assert new['weapon']['target_aspect_angle_max']==pytest.approx(np.pi/4)

def test_fast_qualification_matches_full_geometry():
    env=MultiUAVCombatEnv(config());rng=np.random.default_rng(81)
    for _ in range(1000):
        a=state(*rng.uniform(-4000,4000,3),v=rng.uniform(150,300),psi=rng.uniform(-np.pi,np.pi),theta=rng.uniform(-np.pi/3,np.pi/3))
        b=state(*rng.uniform(-4000,4000,3),v=rng.uniform(150,300),psi=rng.uniform(-np.pi,np.pi),theta=rng.uniform(-np.pi/3,np.pi/3))
        expected=env.weapon.qualifies(engagement_geometry(a,b),a.v,b.v)
        assert env._in_fire_window(a,b)==expected

@pytest.mark.parametrize('aspect',[np.pi/4,np.pi/4-1e-12,np.pi/4+1e-12])
def test_fast_qualification_aspect_boundary(aspect):
    env=MultiUAVCombatEnv(config());a=state();b=state(x=1000,psi=aspect)
    assert env._in_fire_window(a,b)==env.weapon.qualifies(engagement_geometry(a,b),a.v,b.v)

def test_qualification_symmetric_under_rotation_and_side_swap():
    env=MultiUAVCombatEnv(config())
    a,b=state(v=251),state(x=1000,y=250,v=250,psi=.2)
    assert env._in_fire_window(a,b)
    reflected_a=state(x=-a.x,y=-a.y,v=a.v,psi=a.psi+np.pi)
    reflected_b=state(x=-b.x,y=-b.y,v=b.v,psi=b.psi-np.pi)
    assert env._in_fire_window(reflected_a,reflected_b)
    env.red=[state(alive=False) for _ in range(8)]
    env.blue=[state(alive=False) for _ in range(8)]
    env.red[0],env.blue[0]=a,b
    assert env._window_pair_count(env.red,env.blue)==1
    env.red[0],env.blue[0]=reflected_b,reflected_a
    assert env._window_pair_count(env.blue,env.red)==1
