"""Shared 10g envelope and Blue 225m/s; frozen v3.26 target semantics."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v326 import CombatEnvironmentV326
from env.combat_v328 import CombatEnvironmentV328
from env.config import validate_config,environment_dimensions
from env.models import AircraftState
from env.v320_control import action_to_control_v320
from tools.audit_combat_v317_v318 import FixedNormal
from tools.audit_combat_v328 import turn_response


def env(v='328'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{v}.yaml');e.reset(31);return e


def dead():return AircraftState(0,0,-3000,225,0,0,False)


def fixture():
    e=env();e.blue=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
    e.red=[AircraftState(0,400,-3000,225,0,0),AircraftState(700,0,-3000,225,0,0)]+[dead() for _ in range(3)]
    e.rng=FixedNormal(4);return e


def step(e):
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        return e.step(np.zeros((5,3)))


def test_direct_v326_baseline_and_only_three_config_changes():
    assert CombatEnvironmentV328.__bases__==(CombatEnvironmentV326,)
    assert set(CombatEnvironmentV328.__dict__)-{'__module__','__doc__'}=={'environment_version'}
    for name in ('__init__','reset','step','_advance','_observations','dense_combat_reward','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV328,name) is getattr(CombatEnvironmentV326,name)
    e=env();assert not hasattr(e,'locked_target_indices')
    cfg=deepcopy(e.config);validate_config(cfg);assert cfg==e.config
    assert environment_dimensions(cfg)==(56,3,5)
    cfg['environment_version']='3.26';cfg['action']['controller']['normal_load_max']=8.
    cfg['blue_policy']['desired_speed']=250.
    assert cfg==env('326').config
    assert type(e.fixed_policy) is type(env('326').fixed_policy)
    assert Path('configs/mappo_5v5_v328.yaml').read_bytes()==Path('configs/mappo_5v5_v326.yaml').read_bytes()
    assert Path('configs/madsac_5v5_v328.yaml').read_bytes().replace(b'3.28',b'3.26',1)==Path('configs/madsac_5v5_v326.yaml').read_bytes()


@pytest.mark.parametrize('section,field,value',[('controller','normal_load_max',8.),('blue_policy','desired_speed',250.),('weapon','range_max',800.),('reward','guide_reward',.002)])
def test_strict_frozen_parameters(section,field,value):
    cfg=deepcopy(env().config)
    if section=='controller':cfg['action']['controller'][field]=value
    else:cfg[section][field]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_no_lock_nearer_live_target_switch_and_off_target_suppression():
    e=fixture();first=step(e)
    assert first[4]['blue_pursuit_target_indices'][0]==0
    assert e.weapon.qualifies(e.blue[0],e.red[1],6)
    assert first[4]['blue_step_fire_attempts']==0
    assert e.blue_ammo[0]==6 and e.blue_fire_cooldown_steps[0]==0 and e.rng.calls==0
    e.red[1].x=200
    with patch.object(e,'attempt_hit_for_pair',wraps=e.attempt_hit_for_pair) as hit:second=step(e)
    assert e.red[0].alive and second[4]['blue_pursuit_target_indices'][0]==1
    assert second[4]['blue_step_fire_attempts']==1 and hit.call_args.args[1] is e.red[1]


@pytest.mark.parametrize('speed',[150,225,250,300])
def test_shared_ten_g_control_and_identical_motion(speed):
    e=env();state=AircraftState(0,0,-3000,speed,0,0)
    c=action_to_control_v320(state,[1,0,0],e.config['action'],e.spec)
    assert c.nz==pytest.approx(10.) and c.nz>8
    e.red=[state.copy()]+[dead() for _ in range(4)];e.blue=deepcopy(e.red)
    actions=np.tile([1,0,0],(5,1))
    np.testing.assert_array_equal(e._advance(e.red,actions),e._advance(e.blue,actions))
    assert e.red==e.blue


def test_load_budget_combined_and_pitch_boundaries():
    e=env()
    for theta in np.linspace(-math.pi/3,math.pi/3,7):
        for heading in (-1,-.5,0,.5,1):
            for pitch in (-1,-.5,0,.5,1):
                c=action_to_control_v320(AircraftState(0,0,-3000,225,theta,0),[heading,pitch,0],e.config['action'],e.spec)
                assert 0<=c.nz<=10+1e-12


@pytest.mark.parametrize('speed',[200,250])
def test_blue_target_speed_225_and_convergence(speed):
    e=env();own=AircraftState(0,0,-3000,speed,0,0);target=AircraftState(10000,0,-3000,225,0,0)
    action=e.fixed_policy.action(own,[target])
    assert action[0]==action[1]==0 and own.v+50*action[2]==225
    c=action_to_control_v320(own,action,e.config['action'],e.spec)
    assert c.nx==pytest.approx((225-speed)/2/9.81)
    for _ in range(200):
        action=e.fixed_policy.action(own,[target])
        c=action_to_control_v320(own,action,e.config['action'],e.spec)
        own=e.integrator.step(own,c,e.dynamics,e.spec)
    assert own.v==pytest.approx(225,abs=.002)


@pytest.mark.parametrize('speed',[150,225,300])
@pytest.mark.parametrize('av',[-1,1])
def test_red_speed_semantics_unchanged(speed,av):
    e=env();old=env('326');s=AircraftState(0,0,-3000,speed,.2,0)
    a=action_to_control_v320(s,[.5,.5,av],e.config['action'],e.spec)
    b=action_to_control_v320(s,[.5,.5,av],old.config['action'],old.spec)
    assert a.nx==b.nx
    assert e.spec.v_min==150 and e.spec.v_max==300
    target=np.clip(speed+50*av,150,300)
    assert e.dynamics.derivatives(s,a)[3]==pytest.approx((target-speed)/2)


@pytest.mark.parametrize('speed',[150,225,250,300])
def test_turn_audit_rk4_and_physical_trend(speed):
    old,new=turn_response('326',speed),turn_response('328',speed)
    assert new['yaw_rate_deg_s']>old['yaw_rate_deg_s']
    assert new['turn_90_s']<old['turn_90_s'] and new['turn_180_s']<old['turn_180_s']
    assert new['approximate_radius_m']<old['approximate_radius_m']
    assert new['rk4_circle_radius_m']==pytest.approx(new['approximate_radius_m'],rel=1e-6)


def test_reset_observation_rewards_and_weapons_frozen():
    old,new=env('326'),env()
    assert old.red==new.red and old.blue==new.blue
    np.testing.assert_array_equal(old._observations(),new._observations())
    for name in ('reward','weapon','observation','scenario','fire_control','aircraft','arena'):
        assert old.config[name]==new.config[name]
    a,da=old.dense_combat_reward();b,db=new.dense_combat_reward()
    np.testing.assert_array_equal(a,b)
    for name in da:np.testing.assert_array_equal(da[name],db[name])
    assert vars(old.weapon)==vars(new.weapon)
    for epsilon in (0,.5,-.5,2,-2):
        assert old.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1)==new.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1)


def test_algorithm_protocol_checkpoint_and_recording():
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
    from algorithm.madsac.factory import build_madsac_trainer
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv('configs/combat_environment_v328.yaml');assert isinstance(e,CombatEnvironmentV328)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v328.yaml').read_bytes())
    for func in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            func({'algorithm':'MAPPO','extra':{'environment_version':'3.26'}},e.config,cfg)
    cfg=yaml.safe_load(Path('configs/madsac_5v5_v328.yaml').read_bytes());validate_madsac_config(e.config,cfg)
    trainer=build_madsac_trainer(cfg,'cuda',hidden_dim=32)
    with pytest.raises(RuntimeError,match='environment_version'):
        validate_madsac_checkpoint(trainer.checkpoint_state({'environment_version':'3.26'}),e.config,cfg)
