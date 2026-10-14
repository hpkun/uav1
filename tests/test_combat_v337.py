"""Single-variable reward angles, reverse geometry and frozen transitions."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import inspect
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v336 import CombatEnvironmentV336
from env.combat_v337 import CombatEnvironmentV337
from env.config import validate_config
from env.geometry import engagement_geometry
from env.v336_weapon import PaperATAHAWeapon
from env import v337_reward
from env.v323_reward import tactical_rewards_v323
from env.v322_reward import distance_weights
from tests.test_combat_v336 import pair
from tools.audit_combat_v336_weapon_geometry import policy_actions
from env.fixed_policy import NearestTargetPursuitPolicy


def env(version='337'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(31)
    return e


def geometry_pair(d=2000,ata=0,ha=0,theta=0,aa=0):
    a,b = pair(d,ata,ha,theta)
    b.psi = math.radians(ata+aa)
    return a,b


def rewards(a,b,reverse=False,old=False):
    e=env();fn=tactical_rewards_v323 if old else v337_reward.tactical_rewards_v337
    return fn([b] if reverse else [a],[a] if reverse else [b],e.config['reward'],e.config['weapon'])


def test_inheritance_config_and_entire_weapon_frozen():
    a,b=env('336'),env()
    assert CombatEnvironmentV337.__bases__==(CombatEnvironmentV336,)
    assert set(CombatEnvironmentV337.__dict__) - {'__module__','__doc__'} == {'environment_version','dense_combat_reward'}
    c=deepcopy(b.config);c['environment_version']='3.36';assert c==a.config
    assert type(b.weapon) is PaperATAHAWeapon and b.weapon==a.weapon
    for name in ('step','reset','_advance','initial_states','_observations','_outcome','build_weapon','attempt_hit_for_pair','_entry_attempts'):
        assert getattr(CombatEnvironmentV337,name) is getattr(CombatEnvironmentV336,name)
    assert b.observation_dim==45 and b.weapon.range_max==1000
    assert b.weapon.ammo_per_aircraft==6 and b.cooldown_steps==10
    assert b.weapon.effective_hit_distance==558.1106265512473


def test_only_authoritative_geometry_and_explicit_reverse_calls():
    for fn in (v337_reward.paper_pair_geometry,v337_reward.paper_levels,v337_reward.tactical_pair_rewards_v337):
        s=inspect.getsource(fn)
        for forbidden in ('boresight_azimuth_error','boresight_elevation_error','off_boresight','combat_geometry'):
            assert forbidden not in s
    a,b=geometry_pair(2000,10,35,20)
    with patch.object(v337_reward,'engagement_geometry',wraps=engagement_geometry) as wrapped:
        rewards(a,b)
    assert wrapped.call_args_list[0].args==(a,b)
    assert wrapped.call_args_list[1].args==(b,a)
    g=v337_reward.paper_pair_geometry([a],[b])
    assert set(g)=={'distance','ata','ha','aa'} and math.degrees(g['ha'][0,0])==pytest.approx(35)


@pytest.mark.parametrize('component,reverse',[('guide',False),('tactical_offense',False),('tactical_defense',True)])
def test_ha35_theta20_zeroes_each_paper_path(component,reverse):
    a,b=geometry_pair(2000,0,35,20)
    g=engagement_geometry(a,b)
    assert math.degrees(g.ha)==pytest.approx(35)
    assert math.degrees(g.boresight_elevation_error)==pytest.approx(15)
    assert rewards(a,b,reverse,old=True)[component][0] != 0
    assert rewards(a,b,reverse)[component][0] == 0


@pytest.mark.parametrize('component',['guide','tactical_offense'])
def test_ata31_body_az_under30_changes_shaping(component):
    a,b=geometry_pair(2000,31,20,20)
    g=engagement_geometry(a,b)
    assert math.degrees(g.ata)==pytest.approx(31)
    assert abs(math.degrees(g.boresight_azimuth_error))<30
    assert rewards(a,b,old=True)[component][0]>0
    assert rewards(a,b)[component][0]==0


@pytest.mark.parametrize('component,reverse',[('guide',False),('tactical_offense',False),('tactical_defense',True)])
def test_reward_pitch_independent(component,reverse):
    values=[]
    for theta in (-20,0,20):
        values.append(rewards(*geometry_pair(2000,10,10,theta),reverse)[component][0])
    assert values[0]==values[1]==values[2]


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('aa,active',[(29.999,True),(30,True),(30.001,False),(-30,True),(-30.001,False)])
def test_aa_outer_gate_and_guide_does_not_use_aa(reverse,aa,active):
    a,b=geometry_pair(2000,0,0,0,aa)
    out=rewards(a,b,reverse)
    component='tactical_defense' if reverse else 'tactical_offense'
    assert (out[component][0] != 0)==active
    if not reverse:assert out['guide'][0]==.0005


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('axis',['ata','ha','both'])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('angle,level',[(30,0),(15,1),(5,2),(30.00001,None),(15.00001,0),(5.00001,1)])
def test_all_level_boundaries_override_wider_levels(reverse,axis,sign,angle,level):
    ata=sign*angle if axis in ('ata','both') else 0
    ha=sign*angle if axis in ('ha','both') else 0
    out=rewards(*geometry_pair(500,ata,ha),reverse)
    component='tactical_defense' if reverse else 'tactical_offense'
    coefficients=(-.015,-.025,-.15) if reverse else (.01,.02,.1)
    expected=0 if level is None else coefficients[level]
    assert out[component][0]==pytest.approx(expected,abs=1e-15)


@pytest.mark.parametrize('d,w3,w4',[(500,0,1),(1000,0,1),(1500,.25,.75),(2000,.5,.5),(2500,.75,.25),(3000,1,0),(3500,1,0)])
def test_exact_frozen_continuous_distance_weights(d,w3,w4):
    e=env();a,b=geometry_pair(d)
    actual=distance_weights(d,e.config['reward'],e.config['weapon'])
    assert actual==(w3,w4)
    out=rewards(a,b)
    assert out['guide'][0]==.001*w3 and out['tactical_offense'][0]==.1*w4
    assert rewards(a,b,reverse=True)['tactical_defense'][0]==-.15*w4
    old=rewards(a,b,old=True)
    for k in out:np.testing.assert_array_equal(out[k],old[k])


def test_max_max_min_live_masks_and_dead_targets():
    e=env();a,b=geometry_pair(2000)
    dead=b.copy();dead.alive=False
    out=v337_reward.tactical_rewards_v337([a],[b,dead],e.config['reward'],e.config['weapon'])
    one=rewards(a,b)
    for k in out:np.testing.assert_array_equal(out[k],one[k])
    a.alive=False
    out=rewards(a,b)
    assert all(not x.any() for x in out.values())


@pytest.mark.parametrize('section,key,value',[('reward','guide_reward',.002),('reward','tactical_distance_max',4000),
    ('reward','guide_distance_min',4000),('reward','offense_reward_5deg',.2),('reward','threat_penalty_5deg',-.3),
    ('reward','win_reward',10),('weapon','range_max',4000),('weapon','effective_hit_distance',2232.442506204989),
    ('fire_control','cooldown_seconds',2),('scenario','center_radius',4800),('sensor','range_max',9000)])
def test_config_drift_strictly_rejected(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_hyperparameters_frozen(name):
    a=yaml.safe_load(Path(f'configs/{name}_4v4_v336.yaml').read_text());b=yaml.safe_load(Path(f'configs/{name}_4v4_v337.yaml').read_text())
    if 'environment' in b:b['environment']['environment_version']='3.36'
    assert a==b


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.30','3.31','3.32','3.33','3.34','3.35','3.36'])
def test_old_checkpoints_rejected(name,previous):
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.madsac.protocol import validate_madsac_checkpoint
    from algorithm.maac.protocol import validate_checkpoint
    checks={'mappo':(validate_checkpoint_for_evaluation,validate_checkpoint_for_resume),'madsac':(validate_madsac_checkpoint,),'maac':(validate_checkpoint,)}
    state=dict(algorithm={'mappo':'MAPPO','madsac':'madsac','maac':'MAAC'}[name],implementation_version=3,extra=dict(environment_version=previous,training_smoke=True))
    ac=yaml.safe_load(Path(f'configs/{name}_4v4_v337.yaml').read_text())
    for check in checks[name]:
        with pytest.raises(RuntimeError,match='environment_version'):check(state,env().config,ac)


def test_200_resets_identical():
    a,b=env('336'),env()
    for seed in range(200):
        x,_=a.reset(seed);y,_=b.reset(seed)
        assert a.red==b.red and a.blue==b.blue
        np.testing.assert_array_equal(x,y)
        assert [b.fixed_policy.nearest_target_index(t,b.red) for t in b.blue]==[0]*4


@pytest.mark.parametrize('mode',['fixed','pitch_probe'])
@pytest.mark.parametrize('seed',[0,1,2])
def test_full_episode_weapon_trajectory_rng_event_zero_difference(seed,mode):
    a,b=env('336'),env();a.reset(seed);b.reset(seed)
    p=NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    while True:
        actions=policy_actions(a,p,mode)
        x,y=a.step(actions),b.step(actions.copy())
        assert a.red==b.red and a.blue==b.blue and x[2:4]==y[2:4]
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        assert a.combat_counts==b.combat_counts
        np.testing.assert_array_equal(x[0],y[0])
        for side in ('red','blue'):
            for suffix in ('_ammo','_fire_cooldown_steps'):
                np.testing.assert_array_equal(getattr(a,side+suffix),getattr(b,side+suffix))
        for field in ('r1_rewards','r2_rewards','outcome_rewards','safe_rewards','adv_rewards'):
            np.testing.assert_array_equal(x[4][field],y[4][field])
        if x[2] or x[3]:break
