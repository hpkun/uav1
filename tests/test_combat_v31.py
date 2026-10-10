"""Initialization-only v3.1 and frozen legacy trajectory contracts."""
from pathlib import Path
import copy,json
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v30 import CombatEnvironmentV30
from env.combat_v31 import CombatEnvironmentV31
from env.config import load_config,validate_config,environment_dimensions
from env.math_utils import wrap_angle
from env.v31_scenario import sample_laterals
from tools._capture_v31_reference import capture

ROOT=Path(__file__).resolve().parents[1]
def cfg():return load_config(ROOT/'configs/combat_environment_v31.yaml')
def array(e):return np.array([[s.x,s.y,s.z,s.v,s.theta,s.psi,s.alive] for s in e.red+e.blue])

def test_legacy_bit_level_trajectories():
    assert capture()==json.loads((ROOT/'tests/v31_legacy_reference.json').read_text())

def test_shared_combat_and_frozen_configs():
    old=load_config(ROOT/'configs/combat_environment_v30.yaml');new=cfg()
    assert environment_dimensions(new)==(66,3,5)
    e=MultiUAVCombatEnv(new);assert type(e) is CombatEnvironmentV31
    o=MultiUAVCombatEnv(old);assert type(o) is CombatEnvironmentV30
    assert MultiUAVCombatEnv().environment_version=='2.3'
    for key in old:
        if key not in ('scenario','environment_version'):assert old[key]==new[key]
    for method in ('step','_observations','potential','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome','_advance'):
        assert getattr(CombatEnvironmentV31,method) is getattr(CombatEnvironmentV30,method)
    assert type(e.weapon) is type(o.weapon) and type(e.fixed_policy) is type(o.fixed_policy)
    assert 'formation_offsets' not in new['scenario']
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v31.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v30.yaml').read_bytes()

@pytest.mark.parametrize('seed',[1,31,32000000,32000127])
def test_geometry_and_independence(seed):
    e=MultiUAVCombatEnv(cfg());obs,info=e.reset(seed);assert obs.shape==(5,66)
    alpha=info['radial_angle'];long=np.array([np.cos(alpha),np.sin(alpha)]);lat=np.array([-np.sin(alpha),np.cos(alpha)])
    laterals=[];errors=[]
    for side,sign in ((e.red,-1),(e.blue,1)):
        p=np.array([[s.x,s.y] for s in side]);ys=p@lat;laterals.append(ys)
        np.testing.assert_allclose(p@long,sign*2500,atol=1e-10)
        assert np.max(np.abs(ys))<=2500+1e-10
        assert np.diff(np.sort(ys)).min()>=400-1e-10
        heading=np.array([wrap_angle(s.psi-alpha-(np.pi if sign==1 else 0)) for s in side]);errors.append(heading)
        assert np.max(np.abs(heading))<=np.deg2rad(20)+1e-14
        assert len(np.unique(heading))==5
        for s in side:
            assert 2900<=s.altitude<=3100 and 215<=s.v<=235
            assert s.theta==0 and s.alive and np.hypot(s.x,s.y)<5000
    assert not np.array_equal(laterals[0],laterals[1])
    assert not np.array_equal(np.sort(laterals[0]),np.sort(-laterals[1]))
    assert not np.array_equal(errors[0],errors[1])
    expected=array(e).copy();expected_obs=obs.copy();e.step(np.zeros((5,3)));obs,_=e.reset(seed)
    assert np.array_equal(array(e),expected) and np.array_equal(obs,expected_obs)
    assert not e.red_fire_states.previous_eligible.any() and (e.red_ammo==6).all()
    e.reset(seed+1);assert not np.array_equal(array(e),expected)

def test_index_ranks_vary_across_fixed_seeds():
    e=MultiUAVCombatEnv(cfg());ranks={s:[set() for _ in range(5)] for s in ('red','blue')}
    for seed in range(100):
        _,info=e.reset(seed);a=info['radial_angle'];lat=np.array([-np.sin(a),np.cos(a)])
        for side in ranks:
            y=np.array([[s.x,s.y] for s in getattr(e,side)])@lat
            for i,r in enumerate(np.argsort(np.argsort(y))):ranks[side][i].add(int(r))
    assert all(values==set(range(5)) for side in ranks.values() for values in side)

@pytest.mark.parametrize('key,value',[('center_radius',2600),('lateral_min',0),('lateral_max',3000),
    ('minimum_same_team_lateral_separation',200),('minimum_same_team_lateral_separation',450),
    ('heading_perturbation_max',.1),('speed_center',500),('altitude_center',7000),('team_size',True),('team_size',5.)])
def test_frozen_scenario_rejects_changes(key,value):
    c=cfg();c['scenario'][key]=value
    with pytest.raises(ValueError):validate_config(c)

def test_unknown_missing_fields_and_physical_ranges():
    c=cfg();c['scenario']['formation_offsets']=[0]*5
    with pytest.raises(ValueError):validate_config(c)
    c=cfg();del c['scenario']['lateral_min']
    with pytest.raises(ValueError):validate_config(c)
    for section,key,value in [('arena','radius',3000),('aircraft','v_max',220),('arena','altitude_max',3050)]:
        c=cfg();c[section][key]=value
        with pytest.raises(ValueError):validate_config(c)

def test_sampling_failure_is_explicit(monkeypatch):
    import env.v31_scenario as module
    class Impossible:
        def uniform(self,low,high,count):return np.zeros(count)
    monkeypatch.setattr(module,'MAX_SAMPLING_ATTEMPTS',2)
    with pytest.raises(RuntimeError,match='exhausted'):sample_laterals(Impossible(),5,-2500,2500,400)

def test_audit_probe_does_not_change_episode():
    from tools.audit_combat_v31_initialization import Probe
    left=MultiUAVCombatEnv(cfg());right=MultiUAVCombatEnv(cfg());left.reset(31);right.reset(31);Probe(right)
    for _ in range(50):
        a=left.step(np.zeros((5,3)));b=right.step(np.zeros((5,3)))
        assert np.array_equal(array(left),array(right)) and np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
        assert left.rng.bit_generator.state==right.rng.bit_generator.state

@pytest.mark.parametrize('algorithm',['rmappo','stea-mappo'])
def test_v31_cuda_recurrent_death_and_episode_reset(algorithm):
    import torch
    from algorithm.rmappo.networks import FlatRecurrentActor
    from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor
    assert torch.cuda.is_available(), 'v3.1 CUDA acceptance requires CUDA'
    model=(FlatRecurrentActor(observation_dim=66) if algorithm=='rmappo' else
        SpatioTemporalEntityAttentionActor(self_feature_dim=8,policy_std_mode='state_independent',log_std_max=.5)).cuda()
    e=MultiUAVCombatEnv(cfg());obs,_=e.reset(31)
    observation=torch.tensor(obs,device='cuda');hidden=torch.ones(5,128,device='cuda')
    alive=torch.ones(5,device='cuda');start=torch.tensor(0.,device='cuda')
    with torch.no_grad():
        alive[1]=0
        actions,next_hidden=model(observation,hidden,alive,start)
        assert torch.count_nonzero(actions[1])==0 and torch.count_nonzero(next_hidden[1])==0
        reset_actions,reset_hidden=model(observation,hidden,alive,torch.ones_like(start))
        zero_actions,zero_hidden=model(observation,torch.zeros_like(hidden),alive,start)
        assert torch.equal(reset_actions,zero_actions) and torch.equal(reset_hidden,zero_hidden)
