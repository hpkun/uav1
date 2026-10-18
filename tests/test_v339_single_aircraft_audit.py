"""CPU-only invariants and independent formulas for current v3.39 motion."""
import math
from unittest.mock import patch
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v332 import CombatEnvironmentV332
from env.combat_v339 import CombatEnvironmentV339
from env.v311_kinematics import advance_aircraft
from tools.audit_v339_single_aircraft import (config,state,trajectory,independent_series,
    fit_circle,angle_error,full_boundary_case,boundary_flags,isolated_env)

@pytest.mark.parametrize('speed',[150.,225.,300.])
@pytest.mark.parametrize('channel',[0,1,2])
@pytest.mark.parametrize('magnitude',[-2.,-1.,-.5,0.,.5,1.,2.])
def test_single_step_mapping_independent_formula(speed,channel,magnitude):
    cfg=config();a=np.zeros(3);a[channel]=magnitude;s=state(speed)
    actual=trajectory(cfg,s,a,.1)[-1];c=np.clip(magnitude,-1,1)
    expected_psi=(math.pi/40*c) if channel==0 else 0.
    expected_theta=(math.pi/60*c) if channel==1 else 0.
    expected_v=min(300,max(150,speed+2.5*c)) if channel==2 else speed
    expect=np.array([expected_v*math.cos(expected_theta)*math.cos(expected_psi)*.1,
        expected_v*math.cos(expected_theta)*math.sin(expected_psi)*.1,
        -3000-expected_v*math.sin(expected_theta)*.1,expected_v,expected_theta,expected_psi])
    np.testing.assert_allclose(actual,expect,atol=1e-11,rtol=0)

@pytest.mark.parametrize('speed',[150.,225.,300.])
@pytest.mark.parametrize('magnitude',[.5,1.])
def test_actual_circle_fit_and_left_right_symmetry(speed,magnitude):
    cfg=config();left=trajectory(cfg,state(speed),[-magnitude,0,0],32);right=trajectory(cfg,state(speed),[magnitude,0,0],32)
    omega=math.pi/4*magnitude;expected=speed*.1/(2*math.sin(omega*.1/2))
    for tr in [left,right]:
        radius,residual=fit_circle(tr);assert abs(radius-expected)<1e-7 and residual<1e-7
        assert np.all(tr[:,3]==speed) and np.all(tr[:,2]==-3000) and np.all(tr[:,4]==0)
        heading=abs(np.unwrap(tr[:,5]));assert heading[round((math.pi/2/omega)/.1)]==pytest.approx(math.pi/2,abs=1e-11)
    np.testing.assert_allclose(left[:,0],right[:,0],atol=1e-9,rtol=0)
    np.testing.assert_allclose(left[:,1],-right[:,1],atol=1e-9,rtol=0)

@pytest.mark.parametrize('pitch_action',[-1.,-.5,.5,1.])
@pytest.mark.parametrize('speed',[150.,225.,300.])
def test_pitch_limit_climb_descent_constant_speed_and_heading(speed,pitch_action):
    cfg=config();tr=trajectory(cfg,state(speed),[0,pitch_action,0],8)
    t60=2/abs(pitch_action);step=round(t60/.1)
    assert tr[step,4]==pytest.approx(math.copysign(math.pi/3,pitch_action),abs=1e-12)
    np.testing.assert_allclose(tr[step:,4],tr[-1,4],rtol=0,atol=1e-12)
    assert np.all(tr[:,3]==speed) and np.all(tr[:,5]==0)
    assert np.sign(tr[-1,2]+3000)==-np.sign(pitch_action)
    assert -(tr[-1,2]-tr[-2,2])/.1==pytest.approx(speed*math.sin(math.copysign(math.pi/3,pitch_action)),abs=1e-8)

@pytest.mark.parametrize('speed',[150.,200.,225.,250.,300.])
@pytest.mark.parametrize('av',[-1.,-.5,0.,.5,1.])
def test_speed_saturation_monotonic_and_no_pitch_heading_drift(speed,av):
    tr=trajectory(config(),state(speed),[0,0,av],20)
    expected=np.clip(speed+np.arange(201)*2.5*av,150,300)
    np.testing.assert_array_equal(tr[:,3],expected)
    assert np.all(tr[:,4:]==0) and np.all(tr[:,2]==-3000)

@pytest.mark.parametrize('action',[[1,1,0],[1,-1,0],[-1,1,0],[1,1,1],[1,1,-1],[.5,.5,.5],[0,0,0]])
@pytest.mark.parametrize('speed',[150.,225.,300.])
def test_combination_closed_form_independent_channels_and_bounded_steps(speed,action):
    cfg=config();tr=trajectory(cfg,state(speed),action,20);expected=independent_series(cfg,state(speed),action,200)
    np.testing.assert_allclose(tr[:,:5],expected[:,:5],rtol=0,atol=1e-8)
    assert abs(angle_error(tr[:,5],expected[:,5])).max()<1e-11
    assert np.linalg.norm(np.diff(tr[:,:3],axis=0),axis=1).max()<=30+1e-10

@pytest.mark.parametrize('height,ap,cause',[(100.,-1.,'ground'),(5900.,1.,'ceiling'),(3000.,-1.,'ground'),(3000.,1.,'ceiling')])
def test_full_transition_boundary_with_no_blue_policy_or_weapons(height,ap,cause):
    cfg=config()
    with patch('env.v339_policy.DistanceTierNearestTargetBluePolicy.team_actions',side_effect=AssertionError('automatic Blue called')):
        result=full_boundary_case(cfg,state(225,height),[0,ap,0])
    assert result['flags'][cause] and result['counts_match'] and result['no_fire'] and result['terminated']
    assert result['prediction_error']<1e-7
    assert result['red_ground_losses']+result['red_boundary_exits']==1
    assert result['red_ceiling_losses']==int(cause=='ceiling')

@pytest.mark.parametrize('speed',[150.,225.,300.])
def test_full_horizontal_exit_time(speed):
    result=full_boundary_case(config(),state(speed),[0,0,0])
    assert result['flags']['horizontal'] and result['no_fire']
    assert result['time_s']==pytest.approx((math.floor(5000/(speed*.1))+1)*.1,abs=1e-10)

def test_current_dispatch_no_legacy_control_dead_freeze_and_determinism():
    cfg=config();e=isolated_env(cfg,state());assert type(e) is CombatEnvironmentV339
    assert e._advance.__func__ is CombatEnvironmentV332._advance
    assert not hasattr(e,'dynamics') and not hasattr(e,'integrator')
    with (patch('env.combat_v329.action_to_control_v329',side_effect=AssertionError('controller')),
          patch('env.integrator.RK4Integrator.step',side_effect=AssertionError('RK4'))):
        first=trajectory(cfg,state(),[1,1,1],10);second=trajectory(cfg,state(),[1,1,1],10)
    np.testing.assert_array_equal(first,second)
    e.red=[state(alive=False) for _ in range(4)];before=[s.as_array() for s in e.red];e._advance(e.red,np.ones((4,3)))
    np.testing.assert_array_equal(before,[s.as_array() for s in e.red])
    e.red=[state() for _ in range(4)];e.blue=[s.copy() for s in e.red]
    a=np.tile([.5,-.5,1.],(4,1));e._advance(e.red,a);e._advance(e.blue,a)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])

@pytest.mark.parametrize('psi',[-math.pi,math.pi,math.pi-.001,-math.pi+.001])
@pytest.mark.parametrize('av',[-1.,1.])
def test_signed_wrap_is_not_a_large_physical_jump(psi,av):
    tr=trajectory(config(),state(heading=psi),[av,0,0],.1)
    assert -math.pi<=tr[-1,5]<math.pi
    assert angle_error(tr[-1,5],psi)==pytest.approx(av*math.pi/40,abs=1e-12)

@pytest.mark.parametrize('bad',[[1,2],[1,2,3,4],[[0,0,0]],[np.nan,0,0],[0,np.inf,0],[0,0,-np.inf]])
def test_invalid_live_commands_rejected(bad):
    cfg=config();e=MultiUAVCombatEnv(cfg)
    with pytest.raises(ValueError):advance_aircraft(state(),bad,.1,cfg['action']['command'],e.spec)

@pytest.mark.parametrize('height,alive,ceiling',[(0,False,False),(1e-7,True,False),(6000,True,False),(6000+1e-7,False,True)])
def test_exact_altitude_thresholds_both_sides(height,alive,ceiling):
    cfg=config();e=MultiUAVCombatEnv(cfg);e.red=[state(height=height)]+[state(alive=False) for _ in range(3)];e.blue=[s.copy() for s in e.red]
    e._resolve_noncombat_losses()
    for side in ['red','blue']:
        assert getattr(e,side)[0].alive==alive
        assert e.combat_counts[side]['ceiling_losses']==int(ceiling)


@pytest.mark.parametrize('pitch',[-math.pi/9,math.pi/9])
def test_zero_action_holds_nonlevel_attitude_without_altitude_hold(pitch):
    tr=trajectory(config(),state(pitch=pitch,heading=math.pi/2),[0,0,0],10)
    np.testing.assert_array_equal(tr[:,4],np.full(len(tr),pitch))
    assert -tr[-1,2]==pytest.approx(3000+225*math.sin(pitch)*10,abs=1e-8)
