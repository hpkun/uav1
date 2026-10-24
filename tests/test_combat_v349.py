"""Explicit R4 hypothesis; all physical behavior/R1-R3 must stay identical."""
from copy import deepcopy
import math
import numpy as np
import pytest
from env.config import load_config,validate_config,environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v343_reward import continuous_tactical_rewards
from env.v349_config import CANDIDATE_REWARD
from tests.test_combat_v342 import pair


@pytest.mark.parametrize('distance',[0,500,2000,3000,4000,4000.001,8000])
@pytest.mark.parametrize('angle',[0,30,90,180])
def test_continuous_formula(distance,angle):
    a,b=pair(heading=angle,distance=distance)
    r=continuous_tactical_rewards([a],[b],CANDIDATE_REWARD)
    from env.geometry import engagement_geometry
    g=engagement_geometry(a,b);reverse=engagement_geometry(b,a)
    w=np.clip((4000-g.distance)/2000,0,1)
    expected=.02*w*np.clip(1-g.off_boresight/math.pi,0,1)
    threat=-.01*w*np.clip(1-reverse.off_boresight/math.pi,0,1)
    assert r['tactical_offense'][0]==pytest.approx(expected,abs=1e-12)
    assert r['tactical_defense'][0]==pytest.approx(threat,abs=1e-12)
    assert r['tactical'][0]==pytest.approx(expected+threat,abs=1e-12)
    assert -.01<=r['tactical'][0]<=.02


def test_no_aspect_gate_or_offense_precedence_dead_filter():
    a,b=pair(distance=1500);off=[]
    for yaw in (0,math.pi/2,math.pi):
        b.psi=yaw;r=continuous_tactical_rewards([a],[b],CANDIDATE_REWARD);off.append(r['tactical_offense'][0])
    assert off==[.02]*3
    assert r['tactical_defense'][0]==-.01 and r['tactical'][0]==.01
    b.alive=False;assert not continuous_tactical_rewards([a],[b],CANDIDATE_REWARD)['tactical'].any()


def test_schema_and_full_fixed_actions_physics_events_r1_r3_unchanged():
    a,b=[MultiUAVCombatEnv(load_config(f'configs/combat_environment_v{v}.yaml')) for v in ('348','349')]
    assert environment_dimensions(b.config)==(45,3,4)
    c=deepcopy(b.config);c.pop('candidate_reward');c['environment_version']='3.48';assert c==a.config
    bad=deepcopy(b.config);bad['candidate_reward']['offense_base']=.1
    with pytest.raises(ValueError):validate_config(bad)
    a.reset(12);b.reset(12)
    while True:
        action=a.fixed_policy.team_actions(a.red,a.blue)
        x,y=a.step(action),b.step(action)
        np.testing.assert_array_equal(x[0],y[0]);assert x[2:4]==y[2:4]
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        for i in range(1,4):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
        for side in ('red','blue'):
            np.testing.assert_array_equal(getattr(a,side+'_ammo'),getattr(b,side+'_ammo'))
            for key in ('fire_attempts','weapon_hits','attack_kills'):assert x[4][side+'_'+key]==y[4][side+'_'+key]
        np.testing.assert_allclose(y[1],sum(y[4][f'r{i}_rewards'] for i in range(1,5)),rtol=0,atol=1e-12)
        assert y[4]['environment_version']=='3.49' and y[4]['dwell_metrics_applicable'] is False
        if x[2] or x[3]:break
