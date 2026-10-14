"""Pure v3.32 initialization ablation: geometry, dynamic selection and frozen mechanics."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v332 import CombatEnvironmentV332
from env.combat_v333 import CombatEnvironmentV333
from env.combat_v334 import CombatEnvironmentV334
from env.v334_config import FORMATION_LOCAL_OFFSETS
from env.v334_scenario import random_combat_states_v334
from env.config import validate_config, environment_dimensions
from env.fixed_policy import NearestTargetPursuitPolicy
from env.v310_policy import PaperNearestTargetBluePolicy
from env.geometry import engagement_geometry


def env(v='334'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{v}.yaml');e.reset(31);return e


def positions(e):
    return np.array([s.as_array() for s in e.red]),np.array([s.as_array() for s in e.blue])


def indices(e):
    return [e.fixed_policy.nearest_target_index(b,e.red) for b in e.blue]


def test_identity_config_and_all_mechanics_inherited():
    e=env();old=env('332')
    assert CombatEnvironmentV334.__bases__==(CombatEnvironmentV332,)
    assert CombatEnvironmentV333 not in CombatEnvironmentV334.__mro__
    assert type(e) is CombatEnvironmentV334 and environment_dimensions(e.config)==(45,3,4)
    assert e._observations().shape==(4,45)
    c=deepcopy(e.config);c['environment_version']='3.32';c['scenario'].pop('formation_local_offsets');c['scenario']['formation_offsets']=[-300.,-100.,100.,300.]
    assert c==old.config
    for name in ('_advance','step','reset','dense_combat_reward','_observations','_outcome','build_weapon','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','reward_aliases'):
        assert getattr(CombatEnvironmentV334,name) is getattr(CombatEnvironmentV332,name)
    assert e.weapon==old.weapon and e.weapon.range_max==1000
    assert type(e.fixed_policy) is type(old.fixed_policy) is PaperNearestTargetBluePolicy
    assert type(e.fixed_policy).nearest_target_index is NearestTargetPursuitPolicy.nearest_target_index
    assert not hasattr(e,'dynamics') and not hasattr(e,'integrator')
    np.testing.assert_array_equal(np.sum(FORMATION_LOCAL_OFFSETS,axis=0),[0,0])


@pytest.mark.parametrize('alpha',[0,math.pi/4,math.pi/2,-math.pi/3])
def test_axis_projection_centroids_symmetry(alpha):
    class Axis:
        def uniform(self,low,high):return alpha
    c=env().config['scenario'];red,blue,a=random_combat_states_v334(Axis(),**c)
    u=np.array([np.cos(a),np.sin(a)]);t=np.array([-u[1],u[0]])
    r=np.array([s.as_array() for s in red]);b=np.array([s.as_array() for s in blue])
    np.testing.assert_allclose(r[:,:2].mean(0),-4000*u,atol=1e-10)
    np.testing.assert_allclose(b[:,:2].mean(0),4000*u,atol=1e-10)
    np.testing.assert_array_equal(b[:,:2],-r[:,:2])
    np.testing.assert_allclose(np.column_stack(((r[:,:2]+4000*u)@u,(r[:,:2]+4000*u)@t)),FORMATION_LOCAL_OFFSETS,atol=1e-10)
    np.testing.assert_allclose(np.column_stack(((b[:,:2]-4000*u)@(-u),(b[:,:2]-4000*u)@(-t))),FORMATION_LOCAL_OFFSETS,atol=1e-10)
    assert np.all(np.linalg.norm(np.vstack((r[:,:2],b[:,:2])),axis=1)<5000)


def test_200_seeds_distance_topology_and_frozen_attitude():
    e=env();old=env('332');reference=None
    for seed in range(200):
        _,info=e.reset(seed);old.reset(seed);r,b=positions(e);ro,bo=positions(old)
        alpha=info['radial_angle'];u=np.array([np.cos(alpha),np.sin(alpha)])
        np.testing.assert_allclose(r[:,:2].mean(0),-4000*u,atol=1e-9)
        np.testing.assert_allclose(b[:,:2].mean(0),4000*u,atol=1e-9)
        np.testing.assert_array_equal(b[:,:2],-r[:,:2])
        np.testing.assert_array_equal(r[:,2:],ro[:,2:]);np.testing.assert_array_equal(b[:,2:],bo[:,2:])
        assert np.all(np.linalg.norm(np.vstack((r[:,:2],b[:,:2])),axis=1)<5000)
        d=np.linalg.norm(b[:,None,:3]-r[None,:,:3],axis=-1)
        assert indices(e)==[0,0,0,0]
        assert [e.fixed_policy.nearest_target_index(x,e.blue) for x in e.red]==[0,0,0,0]
        assert np.all(np.sort(d,axis=1)[:,1]-d[:,0]>300)
        if reference is None:reference=d
        else:np.testing.assert_allclose(d,reference,atol=1e-9)


def test_immediate_actual_step_switch_and_surviving_reselection():
    e=env();assert indices(e)==[0,0,0,0]
    # Move R1 near B2 using true state coordinates, without modifying policy.
    e.red[1].x=e.blue[2].x+50;e.red[1].y=e.blue[2].y+25
    expected=indices(e);assert expected[2]==1
    e.fixed_policy.team_actions(e.blue,e.red);assert e.fixed_policy.last_targets==expected
    with patch.object(e,'_advance',return_value=np.zeros(4)):
        result=e.step(np.zeros((4,3)))
    assert result[4]['blue_pursuit_target_indices'].tolist()==expected
    e.red[0].alive=False
    expected=indices(e);assert 0 not in expected
    with patch.object(e,'_advance',return_value=np.zeros(4)):
        result=e.step(np.zeros((4,3)))
    assert result[4]['blue_pursuit_target_indices'].tolist()==expected
    assert e.fixed_policy.last_targets==expected


@pytest.mark.parametrize('section,key,value',[
    ('scenario','center_radius',4800),('scenario','team_size',5),
    ('scenario','formation_local_offsets',[[300,0],[-100,250],[-100,0],[-100,-250]]),
    ('scenario','formation_offsets',[-300,-100,100,300]),('scenario','speed_center',230),
    ('weapon','range_max',4000),('weapon','ammo_per_aircraft',7),
    ('reward','guide_distance_min',4000),('reward','tactical_distance_max',4000),
    ('reward','offense_reward_5deg',.2),('reward','tactical_aspect_max',math.pi/2),
    ('action','command',{}),('sensor','range_max',9000),('observation','speed_scale',50),
    ('aircraft','v_max',350),('arena','radius',6000),('blue_policy','desired_speed',250),
    ('fire_control','cooldown_seconds',2)])
def test_validator_rejects_any_nonformation_drift(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('seed',[0,31,199])
def test_exact_reward_weapon_motion_and_transition_on_same_states(seed):
    a,b=env('332'),env();b.reset(seed)
    a.red=deepcopy(b.red);a.blue=deepcopy(b.blue);a.rng=np.random.default_rng(seed);b.rng=np.random.default_rng(seed)
    # Identical state must produce identical observations, reward, combat and motion.
    np.testing.assert_array_equal(a._observations(),b._observations())
    policy=NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    for _ in range(200):
        acts=policy.team_actions(a.red,a.blue)
        x=a.step(acts);y=b.step(acts)
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        assert x[2:4]==y[2:4] and a.red==b.red and a.blue==b.blue
        for i in range(1,5):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
        for side in ('red','blue'):
            np.testing.assert_array_equal(getattr(a,side+'_ammo'),getattr(b,side+'_ammo'))
        if x[2] or x[3]:break
    assert x[2] or x[3]


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_formal_algorithm_configs_equal_v332(name):
    a=yaml.safe_load(Path(f'configs/{name}_4v4_v332.yaml').read_text())
    b=yaml.safe_load(Path(f'configs/{name}_4v4_v334.yaml').read_text())
    if 'environment' in b:b['environment']['environment_version']='3.32'
    assert a==b


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.30','3.31','3.32','3.33'])
def test_checkpoint_version_isolation(name,previous):
    ec=env().config;cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v334.yaml').read_text())
    state=dict(algorithm={'mappo':'MAPPO','madsac':'madsac','maac':'MAAC'}[name],implementation_version=3,
               extra=dict(environment_version=previous,training_smoke=True))
    if name=='mappo':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
        checks=(validate_checkpoint_for_evaluation,validate_checkpoint_for_resume)
    elif name=='madsac':
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        checks=(validate_madsac_checkpoint,)
    else:
        from algorithm.maac.protocol import validate_checkpoint
        checks=(validate_checkpoint,)
    for check in checks:
        with pytest.raises(RuntimeError,match='environment_version'):check(state,ec,cfg)


def test_r4_funnel_is_read_only_and_distinguishes_aspect_from_alignment():
    from tools.audit_combat_v334_initialization import funnel
    from tests.test_combat_v332 import state
    e=env();e.red=[state()]+[state(alive=False) for _ in range(3)]
    e.blue=[state(2000,psi=np.pi)]+[state(alive=False) for _ in range(3)]
    before=deepcopy((e.red,e.blue,e.config));f=funnel(e)
    assert f['live_pairs']==f['distance_le_tactical']==1
    assert f['body_alignment_le_5']==1 and f['actual_r4_offense_positive']==0
    assert f['distance_and_aa_le_90']==f['distance_and_aa_le_60']==f['distance_and_aa_le_30']==0
    assert (e.red,e.blue,e.config)==before
    e.blue[0].psi=0;f=funnel(e)
    assert f['actual_r4_offense_positive']==1
    e.blue[0].x=3000;assert funnel(e)['actual_r4_offense_positive']==0
