"""v3.0 physical, information, fire-entry, reward and algorithm contracts."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config,environment_dimensions,validate_config
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.sensor import is_detected,tactical_angles_3d,sensor_geometry
from env.v30_reward import distance_potential,potentials,potential_shaping,safety_rewards
from env.v30_policy import SensorLimitedPursuitPolicy
from env.v30_weapon import FiniteAmmoWeapon

ROOT=Path(__file__).resolve().parents[1]


def cfg(): return load_config(ROOT/'configs/combat_environment_v30.yaml')


def state(x=0,y=0,z=-3000,v=225,theta=0,psi=0,alive=True):
    return AircraftState(x,y,z,v,theta,psi,alive)


def env():
    e=MultiUAVCombatEnv(cfg());e.reset(30)
    return e


def isolated():
    e=env()
    e.red=[state(alive=False) for _ in range(5)];e.blue=[state(alive=False) for _ in range(5)]
    return e


class Draw:
    def __init__(self,value): self.value=value
    def random(self): return self.value


def test_dimensions_schemas_and_frozen_algorithm_configs():
    c=cfg();assert environment_dimensions(c)==(66,3,5)
    assert type(env().fixed_policy) is SensorLimitedPursuitPolicy
    assert type(env().weapon) is FiniteAmmoWeapon
    for version in ('27','29'):
        assert environment_dimensions(load_config(ROOT/f'configs/combat_environment_v{version}.yaml'))==(65,3,5)
    assert MultiUAVCombatEnv().environment_version=='2.3'
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        old=yaml.safe_load((ROOT/f'configs/{name}_5v5.yaml').read_text())
        new=yaml.safe_load((ROOT/f'configs/{name}_5v5_v30.yaml').read_text())
        expected=copy.deepcopy(old);expected['network']['observation_dim']=66
        if name in ('ea_mappo','stea_mappo'):expected['network']['self_feature_dim']=8
        assert new==expected


@pytest.mark.parametrize('section,key,value',[('weapon','range_max',2000),('weapon','range_min',1),
    ('weapon','hit_probability',1.1),('weapon','ammo_per_aircraft',True),('weapon','ammo_per_aircraft',6.),
    ('sensor','range_max',0),('sensor','off_boresight_angle_max',4),('arena','altitude_max',-1),
    ('reward','safe_distance',0),('reward','potential_gamma',1.1),('blue_policy','guard_radius',6000)])
def test_invalid_v30_values_rejected(section,key,value):
    c=cfg();c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_unknown_legacy_fields_and_missing_fields_rejected():
    for section,field in [('weapon','target_aspect_angle_max'),('reward','approach_reward'),('sensor','noise')]:
        c=cfg();c[section][field]=1
        with pytest.raises(ValueError):validate_config(c)
    c=cfg();del c['sensor']
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('target,visible',[ (state(2999),True),(state(3001),False),
    (state(100,z=-4000),True),(state(-1,z=-4000),False),(state(100,alive=False),False)])
def test_sensor_range_angle_and_zero_slots(target,visible):
    e=isolated();e.red[0]=state();e.blue[0]=target
    assert is_detected(e.red[0],target,e.config['sensor'])==visible
    assert engagement_geometry(e.red[0],target).off_boresight<=np.pi/2 if visible else True
    slot=e._observations()[0,36:42]
    assert bool(slot[-1])==visible
    if not visible:np.testing.assert_array_equal(slot,np.zeros(6))


def test_high_altitude_3d_sensor_disagrees_with_horizontal_ata():
    observer=state(theta=-np.pi/3)
    target=state(100,z=-4000)
    geometry=engagement_geometry(observer,target)
    assert abs(geometry.ata)<1e-10 and geometry.off_boresight>np.pi/2
    assert not is_detected(observer,target,cfg()['sensor'])


def test_hidden_enemy_observation_and_potential_no_leak():
    e=isolated();e.red[0]=state()
    result=[]
    for hidden in (state(-900,z=-3900,v=150,psi=1),state(4000,z=-6000,v=300,psi=-2)):
        e.blue[0]=hidden
        assert not is_detected(e.red[0],hidden,e.config['sensor'])
        obs=e._observations();phi,diag=e.potential()
        assert phi[0]==0 and diag['distance'][0]==diag['angle'][0]==0
        np.testing.assert_array_equal(obs[0,36:],np.zeros(30))
        result.append((obs,potential_shaping(phi,phi,e.config['reward'])))
    for a,b in zip(result[0],result[1]):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('target,eligible',[(state(999,psi=np.pi),True),(state(999,psi=0),True),
    (state(1001),False),(state(-1),False),(state(100,z=-4000),False)])
def test_weapon_without_aspect_gate(target,eligible):
    e=env();assert e.weapon.qualifies(state(),target,6)==eligible
    assert not e.weapon.qualifies(state(),target,0)


def test_weapon_true3d_angle_not_horizontal():
    assert not env().weapon.qualifies(state(theta=-np.pi/3),state(100,z=-3500),6)


@pytest.mark.parametrize('draw,hit',[(0.,True),(.699999,True),(.7,False),(.999,False)])
def test_fixed_bernoulli_and_ammo_consumed(draw,hit):
    e=isolated();e.red[0]=state();e.blue[0]=state(500);e.rng=Draw(draw)
    attempts=e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
    assert attempts==[(0,0,hit)] and e.red_ammo[0]==5


def test_pair_entry_consumes_unselected_pairs_reentry_and_zero_ammo():
    e=isolated();e.red[0]=state();e.blue[0]=state(500);e.blue[1]=state(500);e.rng=Draw(.9)
    fire=lambda:e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
    assert fire()==[(0,0,False)] # index tie-break
    assert e.red_fire_states.previous_eligible[0,:2].all()
    assert fire()==[]
    e.blue[0].x=e.blue[1].x=1500;assert fire()==[]
    e.blue[0].x=700;e.blue[1].x=400
    assert fire()==[(0,1,False)]
    for _ in range(10):
        e.blue[0].x=e.blue[1].x=1500;fire()
        e.blue[0].x=e.blue[1].x=500;fire()
    assert e.red_ammo[0]==0 and fire()==[]
    assert e.combat_counts['red']['fire_attempts']==e.combat_counts['red']['ammo_used']==6


def test_simultaneous_kills_and_shared_actual_attacker_credit():
    e=isolated();e.red[0]=state();e.blue[0]=state(500,psi=np.pi);e.rng=Draw(0)
    a=e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
    b=e._entry_attempts(e.blue,e.red,e.blue_fire_states,'blue')
    rk,bk=e._resolve_combat(a,b)
    assert not e.red[0].alive and not e.blue[0].alive
    event,boundary=e._event_reward_components([],[],rk,bk)
    assert event[0]==0 and boundary.sum()==0 # +10 and -10 in same step
    e=isolated();e.blue[0]=state(500);e.red[0]=state();e.red[1]=state(0,10)
    rk,bk=e._resolve_combat([(0,0,True),(1,0,True)],[])
    event,_=e._event_reward_components([],[],rk,bk)
    np.testing.assert_array_equal(event,[5,5,0,0,0])


def test_blue_hidden_search_visible_nearest_and_guard():
    e=env();p=e.fixed_policy;b=state(1000)
    hidden=[state(-100),state(5000,v=300)]
    a=p.action(b,hidden)
    np.testing.assert_array_equal(a,p.action(b,[state(-2000),state(7000,v=150)]))
    targets=[state(1900),state(1500,100),state(950)] # closer hidden behind
    np.testing.assert_array_equal(p.action(b,targets),p.action(b,[targets[1]]))
    for own in (state(4500),state(0,z=-500),state(0,z=-5500)):
        expected=p.action_toward(own,np.arctan2(-own.y,-own.x),np.arctan2(own.z+3000,np.hypot(own.x,own.y)),250)
        np.testing.assert_array_equal(p.action(own,[state(own.x+100)]),expected)


@pytest.mark.parametrize('aircraft,cause',[(state(5001),'boundary_exits'),(state(z=0),'ground_losses'),(state(z=-6001),'ceiling_losses')])
def test_physical_boundaries_symmetric_and_no_blue_kill_reward(aircraft,cause):
    e=isolated();e.red[0]=aircraft.copy();e.blue[0]=aircraft.copy()
    exits,blue_exits,ground,blue_ground=e._resolve_noncombat_losses()
    assert not e.red[0].alive and not e.blue[0].alive
    assert e.combat_counts['red'][cause]==e.combat_counts['blue'][cause]==1
    event,boundary=e._event_reward_components(exits,ground,{},{} )
    assert event.sum()+boundary.sum()==-10


@pytest.mark.parametrize('reds,blues,truncated,win,loss,draw,reason',[
    (4,3,True,True,False,False,'red_win_timeout_survivors'),
    (2,4,True,False,True,False,'blue_win_timeout_survivors'),
    (3,3,True,False,False,True,'draw_timeout_equal_survivors'),
    (1,0,False,True,False,False,'red_win_elimination'),
    (0,1,False,False,True,False,'blue_win_elimination'),
    (0,0,False,False,False,True,'draw_mutual_destruction')])
def test_outcomes_and_all_slot_terminal_rewards(reds,blues,truncated,win,loss,draw,reason):
    e=isolated()
    for i in range(reds):e.red[i]=state(-2000,i*300)
    for i in range(blues):e.blue[i]=state(2000,i*300,psi=np.pi)
    assert e._outcome(truncated)==(win,loss,draw,reason)
    if truncated:e.steps=e.max_steps-1
    _,_,t,tr,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert t or tr
    assert info['termination_reason']==reason and info['red_success']==win
    np.testing.assert_array_equal(info['outcome_rewards'],np.full(5,2 if win else -2 if loss else 0))


@pytest.mark.parametrize('d,value',[(0,-1),(100,0),(200,1),(500,1),(1000,1),(2000,.5),(3000,0),(4000,0)])
def test_distance_potential(d,value):assert distance_potential(d)==value


def test_angles_are_true3d_and_pbrs_formula():
    a=state();tail=state(500);head=state(500,psi=np.pi)
    for target,value in ((tail,1),(head,0)):
        ata,aa=tactical_angles_3d(a,target)
        assert (np.cos(ata)+np.cos(aa))/2==pytest.approx(value)
        assert ata==pytest.approx(engagement_geometry(a,target).off_boresight)
    ata,aa=tactical_angles_3d(state(theta=np.pi/4),state(500,z=-3500,theta=np.pi/4))
    assert ata==aa==pytest.approx(0,abs=2e-8)
    np.testing.assert_allclose(potential_shaping(np.array([.2,0]),np.array([.8,0]),cfg()['reward']),[.99*.8-.2,0])


@pytest.mark.parametrize('d,value',[(300,0),(200,0),(100,-.1),(.001,-.199999)])
def test_safety_true3d_and_dead_entities_ignored(d,value):
    own=state();other=state(z=-3000-d)
    assert safety_rewards([own],[other],cfg()['reward'])[0]==pytest.approx(value)
    other.alive=False;assert safety_rewards([own],[other],cfg()['reward'])[0]==0
    own.alive=False;assert safety_rewards([own],[state()],cfg()['reward'])[0]==0


def test_step_pbrs_final_observation_state_and_reset():
    e=env();e.red[0]=state();e.blue[0]=state(2000)
    before,_=e.potential()
    obs,rewards,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    after,_=e.potential()
    np.testing.assert_array_equal(info['phi_next'],after)
    np.testing.assert_allclose(info['adv_rewards'],.99*after-before)
    np.testing.assert_allclose(rewards,sum(info[f'{n}_rewards'] for n in ('event','outcome','adv','safe')),rtol=1e-6,atol=1e-6)
    assert obs.shape==(5,66) and np.isfinite(obs).all()
    np.testing.assert_allclose(obs[:,7],e.red_ammo/6)
    e.red_ammo[:]=0;e.red_fire_states.previous_eligible[:]=True
    e.reset(30)
    assert (e.red_ammo==6).all() and (e.blue_ammo==6).all()
    assert not e.red_fire_states.previous_eligible.any()
    assert all(value==0 for counts in e.combat_counts.values() for value in counts.values())


@pytest.mark.parametrize('algorithm',['mappo','rmappo','ea_mappo','stea_mappo'])
def test_v30_trainers_and_entity_hidden_masks(algorithm):
    c=yaml.safe_load((ROOT/f'configs/{algorithm}_5v5_v30.yaml').read_text())
    if algorithm=='mappo':
        from algorithm.common.control_factory import trainer_kwargs
        from algorithm.mappo.trainer import MAPPOTrainer
        trainer=MAPPOTrainer(**trainer_kwargs(c,'cpu',31))
    else:
        import importlib
        trainer=getattr(importlib.import_module(f'algorithm.{algorithm}.factory'),f'build_{algorithm}_trainer')(c,'cpu',seed=31)
    assert trainer.critic.value_network[0].in_features==396
    obs=torch.tensor(env().reset(31)[0]).unsqueeze(0);alive=torch.ones(1,5)
    if algorithm in ('ea_mappo','stea_mappo'):
        from algorithm.stea_mappo.networks import decompose_observations
        own,allies,enemies=decompose_observations(obs)
        assert own.shape==(1,5,8) and allies.shape==(1,5,4,7) and enemies.shape==(1,5,5,6)
        assert trainer.actor.self_encoder[0].in_features==8
        _,diagnostic=trainer.actor.spatial(obs,alive)
        assert torch.count_nonzero(diagnostic['enemy_attention_weights'])==0
    if algorithm in ('rmappo','stea_mappo'):
        actions,hidden=trainer.actor(obs,torch.zeros(1,5,128),alive,torch.ones(1))
        assert hidden.shape==(1,5,128)
        reset_actions,reset_hidden=trainer.actor(obs,torch.ones_like(hidden)*7,alive,torch.ones(1))
        torch.testing.assert_close(reset_actions,actions,rtol=0,atol=0)
        torch.testing.assert_close(reset_hidden,hidden,rtol=0,atol=0)
        dead_mask=alive.clone();dead_mask[0,2]=0
        death_actions,death_hidden=trainer.actor(obs,hidden,dead_mask,torch.zeros(1))
        assert torch.count_nonzero(death_actions[0,2])==0
        assert torch.count_nonzero(death_hidden[0,2])==0
    else:actions=trainer.actor(obs)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize('algorithm',['madsac','maddpg'])
def test_unrelated_algorithm_support_unchanged(algorithm):
    import importlib
    c=yaml.safe_load((ROOT/f'configs/{"maddpg_5v5" if algorithm=="maddpg" else "madsac"}.yaml').read_text())
    module=importlib.import_module(f'algorithm.{algorithm}.protocol')
    with pytest.raises(RuntimeError):getattr(module,'validate_config' if algorithm=='maddpg' else 'validate_madsac_config')(cfg(),c)


def test_seed_reproducible_real_combat_and_no_noncombat_kill_credit():
    sequences=[]
    for _ in range(2):
        e=env();rows=[]
        for step in range(10):
            e.red[0]=state();e.blue[0]=state(700 if step%2==0 else 2000)
            _,r,_,_,i=e.step(np.zeros((5,3)),np.zeros((5,3)))
            rows.append((r.tolist(),e.red_ammo.tolist(),e.blue_ammo.tolist(),i['red_weapon_hits']))
        sequences.append(rows)
    assert sequences[0]==sequences[1]
    e=isolated();e.red[0]=state();e.blue[0]=state(5001)
    _,_,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert info['event_rewards'].sum()==0 and info['red_attack_kills']==0
