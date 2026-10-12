"""Isolated v3.17 repeat-fire baseline and weighted paper reward distances."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v317 import CombatEnvironmentV317
from env.combat_v319 import CombatEnvironmentV319
from env.config import load_config,validate_config,environment_dimensions
from env.models import AircraftState
from env.v319_reward import distance_weight,tactical_rewards
from tools.audit_combat_v317_v318 import body_pair,FixedNormal,window_env


def env(version='319',seed=31):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(seed);return e


def teams(distance=1000.,mode='offense',angle=0.):
    own=AircraftState(0,0,-3000,225,0,math.radians(angle) if mode=='offense' else 0)
    target=AircraftState(distance if mode=='offense' else -distance,0,-3000,225,0,0 if mode=='offense' else math.radians(angle))
    dead=lambda:AircraftState(0,0,-3000,225,0,0,False)
    return [own,*[dead() for _ in range(4)]],[target,*[dead() for _ in range(4)]]


def rewards(red,blue):
    return tactical_rewards(red,blue,load_config('configs/combat_environment_v319.yaml')['reward'])


@pytest.mark.parametrize('version,distances',[('317',(800,4000,4000)),('318',(800,4000,4000)),('319',(1000,3000,3000))])
def test_formal_version_values(version,distances):
    cfg=load_config(f'configs/combat_environment_v{version}.yaml')
    assert (cfg['weapon']['range_max'],cfg['reward']['guide_distance_min'],cfg['reward']['tactical_distance_max'])==distances


@pytest.mark.parametrize('version',['317','318','319'])
@pytest.mark.parametrize('section,field',[('weapon','range_max'),('reward','guide_distance_min'),('reward','tactical_distance_max')])
def test_wrong_formal_distance_rejected(version,section,field):
    cfg=load_config(f'configs/combat_environment_v{version}.yaml')
    cfg[section][field]=800 if version=='319' and section=='weapon' else 1000 if section=='weapon' else 4000 if version=='319' else 3000
    with pytest.raises(ValueError):validate_config(cfg)


def test_direct_baseline_only_dense_reward_override_and_frozen_config():
    old,new=env('317'),env()
    assert CombatEnvironmentV319.__bases__==(CombatEnvironmentV317,)
    assert set(CombatEnvironmentV319.__dict__)-{'__module__','__doc__'}=={'environment_version','dense_combat_reward'}
    for name in ('reset','step','_advance','_observations','build_weapon','attempt_hit_for_pair','_entry_attempts','_new_fire_states','_resolve_combat','_outcome','_resolve_noncombat_losses'):
        assert getattr(CombatEnvironmentV319,name) is getattr(CombatEnvironmentV317,name)
    cfg=deepcopy(new.config);unchanged=deepcopy(cfg);validate_config(cfg);assert cfg==unchanged
    cfg['environment_version']='3.17';cfg['weapon']['range_max']=800
    cfg['reward']['guide_distance_min']=4000;cfg['reward']['tactical_distance_max']=4000
    assert cfg==old.config and environment_dimensions(new.config)==(56,3,5)
    assert new.red_fire_states is None and new.blue_fire_states is None


@pytest.mark.parametrize('distance,expected',[(999.999,True),(1000.,True),(1000.000001,False)])
def test_weapon_distance_boundary(distance,expected):
    a,b=body_pair(distance=distance)
    assert env().weapon.qualifies(a,b,6)==expected


@pytest.mark.parametrize('az,el,expected',[(30,0,True),(0,-30,True),(31,0,False),(0,31,False),(25,25,True)])
def test_body_axes_unchanged(az,el,expected):
    a,b=body_pair(distance=500,azimuth=az,elevation=el)
    assert env().weapon.qualifies(a,b,6)==expected


@pytest.mark.parametrize('distance,expected',[(500,1),(1000,1),(1500,.75),(2000,.5),(2500,.25),(3000,0),(3500,0)])
def test_r4_weight(distance,expected):
    assert distance_weight(distance)==expected


def test_weight_bounds_and_finite():
    result=distance_weight(np.linspace(0,10000,1001))
    assert np.isfinite(result).all() and (result>=0).all() and (result<=1).all()


@pytest.mark.parametrize('distance,weight',[(1000,1),(1500,.75),(2000,.5),(2500,.25),(3000,0)])
@pytest.mark.parametrize('angle,level',[(0,.1),(10,.02),(20,.01)])
def test_r41_levels_and_distance_weights(distance,weight,angle,level):
    r,b=teams(distance,'offense',angle)
    assert rewards(r,b)['tactical_offense'][0]==pytest.approx(level*weight)


@pytest.mark.parametrize('distance,weight',[(1000,1),(1500,.75),(2000,.5),(2500,.25),(3000,0)])
@pytest.mark.parametrize('angle,level',[(0,-.15),(10,-.025),(20,-.015)])
def test_r42_reverse_levels_and_distance_weights(distance,weight,angle,level):
    r,b=teams(distance,'defense',angle)
    assert rewards(r,b)['tactical_defense'][0]==pytest.approx(level*weight)


@pytest.mark.parametrize('distance,expected',[(2999.999,0),(3000,.001),(3500,.001),(12000,.001)])
def test_r3_boundary_and_no_visibility_gate(distance,expected):
    r,b=teams(distance)
    assert rewards(r,b)['guide'][0]==expected


@pytest.mark.parametrize('case',['ATA','HA','red_dead','blue_dead'])
def test_r3_unqualified_or_dead_zero(case):
    r,b=teams(3500)
    if case=='ATA':r[0].psi=math.radians(31)
    if case=='HA':b[0].x=3500*math.cos(math.radians(31));b[0].z=-3000-3500*math.sin(math.radians(31))
    if case=='red_dead':r[0].alive=False
    if case=='blue_dead':b[0].alive=False
    assert rewards(r,b)['guide'][0]==0


def test_r3_any_target_not_sum():
    r,b=teams(3500);b[1]=AircraftState(3600,0,-3000,225,0,0)
    assert rewards(r,b)['guide'][0]==.001


def test_pair_weights_before_max_and_min_not_target_selection_or_sum():
    r,b=teams(2900)
    angle=math.radians(10)
    b[1]=AircraftState(1000*math.cos(angle),1000*math.sin(angle),-3000,225,0,angle)
    # Far high-level pair .1*.05=.005 loses to nearby .02*1=.02.
    result=rewards(r,b)
    assert result['tactical_offense'][0]==pytest.approx(.02)
    r,b=teams(2900,'defense')
    b[1]=AircraftState(-1000*math.cos(angle),-1000*math.sin(angle),-3000,225,0,0)
    # Far high-level threat -.15*.05=-.0075 loses to nearby -.025*1.
    result=rewards(r,b)
    assert result['tactical_defense'][0]==pytest.approx(-.025)
    np.testing.assert_array_equal(result['tactical'],result['guide']+result['tactical_offense']+result['tactical_defense'])


def test_aa_condition_and_dead_target_still_masked():
    r,b=teams(1000);b[0].psi=math.pi
    assert rewards(r,b)['tactical_offense'][0]==0
    b[0].psi=0;b[0].alive=False
    assert rewards(r,b)['tactical_offense'][0]==0


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_scenario_observation_control_dynamics_blue_exact(seed):
    old,new=env('317',seed),env('319',seed)
    oo,oi=old.reset(seed);no,ni=new.reset(seed)
    np.testing.assert_array_equal(oo,no);assert ni['nominal_anchor_separation']==oi['nominal_anchor_separation']==8000
    actions=np.tile([.4,.5,.8],(5,1))
    for side in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
        np.testing.assert_array_equal(old._advance(getattr(old,side),actions),new._advance(getattr(new,side),actions))
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
    np.testing.assert_array_equal(old._observations(),new._observations())
    np.testing.assert_array_equal(old.fixed_policy.team_actions(old.blue,old.red),new.fixed_policy.team_actions(new.blue,new.red))
    assert type(old.dynamics) is type(new.dynamics) and type(old.integrator) is type(new.integrator)


def test_repeat_fire_and_shared_sampler_unchanged():
    old,new=window_env('317'),env();new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    old.rng,new.rng=FixedNormal(4),FixedNormal(4)
    for _ in range(8):
        assert old._entry_attempts(old.red,old.blue,None,'red')==new._entry_attempts(new.red,new.blue,None,'red')
    assert new.red_ammo[0]==0 and new.rng.calls==old.rng.calls==6
    assert type(old.weapon) is type(new.weapon)
    for epsilon in (0,.5,-.5,2,-2):
        assert old.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1)==new.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1)


@pytest.mark.parametrize('distance',[100,2500,5000])
def test_full_step_r1_r2_safety_unchanged_and_dense_integrated(distance):
    old,new=env('317'),env();old.red,old.blue=teams(distance)
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    dense=rewards(new.red,new.blue)
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)),patch.object(old,'_entry_attempts',return_value=[(0,0,True)]),patch.object(new,'_entry_attempts',return_value=[(0,0,True)]):
        _,_,to,xo,io=old.step(np.zeros((5,3)),np.zeros((5,3)))
        _,rn,tn,xn,inn=new.step(np.zeros((5,3)),np.zeros((5,3)))
    assert (to,xo)==(tn,xn)
    for key in ('r1_rewards','r2_rewards','safety_rewards'):np.testing.assert_array_equal(io[key],inn[key])
    np.testing.assert_array_equal(inn['r3_rewards'],dense['guide'])
    np.testing.assert_array_equal(inn['r4_rewards'],dense['tactical_offense']+dense['tactical_defense'])
    np.testing.assert_array_equal(rn,sum((inn[f'r{i}_rewards'] for i in range(1,5)))+inn['safety_rewards'])


@pytest.mark.parametrize('case',['blue_dead','red_dead','timeout','boundary','ground','ceiling'])
def test_termination_and_noncombat_unchanged(case):
    old,new=env('317'),env()
    for e in (old,new):
        if case.endswith('_dead'):
            for s in getattr(e,case.split('_')[0]):s.alive=False
        if case=='boundary':e.red[0].x=5001
        if case=='ground':e.red[0].z=0
        if case=='ceiling':e.red[0].z=-6001
    assert old._resolve_noncombat_losses()==new._resolve_noncombat_losses()
    assert old._outcome(case=='timeout')==new._outcome(case=='timeout')


def test_mappo_config_and_version_logging():
    assert Path('configs/mappo_5v5_v319.yaml').read_bytes()==Path('configs/mappo_5v5_v317.yaml').read_bytes()
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from types import SimpleNamespace
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=env().config
    runner.trainer=SimpleNamespace(sampled_steps=0);runner.evaluation_seeds=[10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):assert runner._evaluation_record()['environment_version']=='3.19'


def test_recording_dispatch_and_old_checkpoint_version_rejection():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
    import yaml
    e=RecordingCombatEnv('configs/combat_environment_v319.yaml');assert isinstance(e,CombatEnvironmentV319)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v319.yaml').read_bytes())
    with pytest.raises(RuntimeError,match='environment_version'):
        validate_checkpoint_for_evaluation({'algorithm':'MAPPO','extra':{'environment_version':'3.17'}},e.config,cfg)
