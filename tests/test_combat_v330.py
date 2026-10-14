"""Actual four-agent environment, frozen v3.29 mechanics and slot contracts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v329 import CombatEnvironmentV329
from env.combat_v330 import CombatEnvironmentV330
from env.config import environment_dimensions,validate_config
from env.models import AircraftState
from env.v329_control import action_to_control_v329
from env.v330_observation import build_team_observations_v330
from tools.audit_combat_v317_v318 import FixedNormal


def env(v='330'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{v}.yaml');e.reset(31);return e


def state(x=0,y=0,h=3000,alive=True):return AircraftState(x,y,-h,225,0,0,alive)
def dead():return state(alive=False)


def test_real_four_aircraft_reset_step_arrays_and_config_diff():
    e=env();assert type(e) is CombatEnvironmentV330 and CombatEnvironmentV330.__bases__==(CombatEnvironmentV329,)
    assert environment_dimensions(e.config)==(45,3,4) and len(e.red)==len(e.blue)==4
    obs,info=e.reset(31);assert obs.shape==(4,45) and info['nominal_anchor_separation']==8000
    assert np.linalg.norm(np.mean([s.as_array()[:3] for s in e.red],axis=0)-np.mean([s.as_array()[:3] for s in e.blue],axis=0))==pytest.approx(8000)
    assert np.linalg.norm(np.diff([s.as_array()[:3] for s in e.red],axis=0),axis=1).tolist()==pytest.approx([200]*3)
    obs,r,_,_,info=e.step(np.zeros((4,3)));assert obs.shape==(4,45) and r.shape==(4,)
    for name in ('red_ammo','blue_ammo','red_fire_cooldown_steps','blue_fire_cooldown_steps','blue_step_target_indices','red_alive_mask','blue_alive_mask'):
        assert getattr(e,name).shape==(4,)
    assert e.red_fire_states is None and e.blue_fire_states is None
    assert all(v.shape==(4,) for v in e.episode_reward_components.values())
    assert not hasattr(e,'locked_target_indices')
    with pytest.raises(ValueError):e.step(np.zeros((5,3)))
    cfg=deepcopy(e.config);cfg['environment_version']='3.29';cfg['scenario']['team_size']=5;cfg['scenario']['formation_offsets']=[-400.,-200.,0.,200.,400.]
    assert cfg==env('329').config


@pytest.mark.parametrize('section,key,value',[('weapon','range_max',800),('reward','guide_reward',.002),('blue_policy','desired_speed',250),('scenario','center_radius',4800),('scenario','team_size',5),('scenario','formation_offsets',[-400,-200,0,200,400])])
def test_validator_rejects_drift(section,key,value):
    cfg=deepcopy(env().config);cfg[section][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_observation_exact_layout_order_and_dead_slots():
    e=env();e.red=[state(10*i,20*i,h=3000+i) for i in range(4)];e.blue=[state(100+30*i,-40*i,h=3100+i) for i in range(4)]
    o=e._observations();old=env('329');old.red=deepcopy(e.red)+[dead()];old.blue=deepcopy(e.blue)+[dead()]
    reference=old._observations()
    for i in range(4):
        np.testing.assert_array_equal(o[i,:7],reference[i,:7])
        np.testing.assert_array_equal(o[i,7:25],reference[i,7:25])
        np.testing.assert_array_equal(o[i,25:45],reference[i,31:51])
    e.red[1].alive=False;e.blue[2].alive=False;o=e._observations()
    assert (o[0,7:13]==0).all() and (o[:,35:40]==0).all() and (o[1]==0).all()
    assert build_team_observations_v330(e.blue,e.red,e.config['observation']).shape==(4,45)


def test_blue_nearest_3d_tie_switch_and_off_target_suppression():
    e=env();e.blue=[state()]+[dead() for _ in range(3)]
    e.red=[state(0,400),state(700)]+[dead() for _ in range(2)];e.rng=FixedNormal(4)
    with patch.object(e,'_advance',return_value=np.zeros(4)):
        _,_,_,_,info=e.step(np.zeros((4,3)))
        assert info['blue_pursuit_target_indices'].tolist()==[0,-1,-1,-1]
        assert e.weapon.qualifies(e.blue[0],e.red[1],6) and info['blue_step_fire_attempts']==0
        e.red[1].x=200;_,_,_,_,info=e.step(np.zeros((4,3)))
        assert info['blue_pursuit_target_indices'][0]==1 and info['blue_step_fire_attempts']==1
    e.red=[state(100),state(-100),dead(),dead()]
    assert e.fixed_policy.nearest_target_index(e.blue[0],e.red)==0
    e.red[0].z=-5000;assert e.fixed_policy.nearest_target_index(e.blue[0],e.red)==1


def test_red_four_candidates_cooldown_and_simultaneous_resolution():
    e=env();e.red=[state()]+[dead() for _ in range(3)];e.blue=[state(500),state(300),state(400),state(200)]
    e.rng=FixedNormal(4);a=e._entry_attempts(e.red,e.blue,None,'red');assert a[0][1]==3
    assert e.red_ammo[0]==5 and e.red_fire_cooldown_steps[0]==10
    for _ in range(9):assert not e._entry_attempts(e.red,e.blue,None,'red')
    assert e._entry_attempts(e.red,e.blue,None,'red')[0][1]==3
    e=env();e.red=[state()]+[dead() for _ in range(3)];e.blue=[state(500)]+[dead() for _ in range(3)];e.blue[0].psi=math.pi;e.rng=FixedNormal(0)
    with patch.object(e,'_advance',return_value=np.zeros(4)):
        _,r,terminated,_,info=e.step(np.zeros((4,3)))
    assert terminated and not e.red[0].alive and not e.blue[0].alive
    assert info['red_attack_kills']==info['blue_attack_kills']==1
    assert not info['red_win'] and not info['blue_win'] and info['draw']
    assert info['termination_reason']=='draw_mutual_destruction'


@pytest.mark.parametrize('kind',['ground','horizontal','ceiling'])
def test_noncombat_loss_symmetry(kind):
    e=env();s=state(h=0 if kind=='ground' else 6001 if kind=='ceiling' else 3000)
    if kind=='horizontal':s.x=5001
    e.red=[s.copy()]+[dead() for _ in range(3)];e.blue=deepcopy(e.red)
    result=e._resolve_noncombat_losses();assert not e.red[0].alive and not e.blue[0].alive
    assert result[0]==result[1] and result[2]==result[3]
    assert e.combat_counts['red']==e.combat_counts['blue']


@pytest.mark.parametrize('red_alive,blue_alive,truncated',[
    (False,False,False),(False,False,True),(True,False,False),
    (False,True,False),(True,True,True),(True,True,False)])
def test_outcome_only_mutual_destruction_changes(red_alive,blue_alive,truncated):
    e=env();old=env('329')
    for instance in (e,old):
        instance.red=[state(alive=red_alive) for _ in range(instance.team_size)]
        instance.blue=[state(alive=blue_alive) for _ in range(instance.team_size)]
    if not red_alive and not blue_alive:
        assert e._outcome(truncated)==(False,False,True,'draw_mutual_destruction')
        assert old._outcome(truncated)==(True,False,False,'red_win_elimination')
    else:
        assert e._outcome(truncated)==old._outcome(truncated)


def test_mutual_destruction_reward_and_transition_regression():
    current,previous=env(),env()
    for e in (current,previous):
        e.red=[state()]+[dead() for _ in range(3)]
        e.blue=[state(500)]+[dead() for _ in range(3)]
        e.blue[0].psi=math.pi;e.rng=FixedNormal(0)
    with patch.object(current,'_advance',return_value=np.zeros(4)), \
         patch.object(previous,'_advance',return_value=np.zeros(4)), \
         patch.object(previous,'_outcome',side_effect=lambda truncated: CombatEnvironmentV329._outcome(previous,truncated)):
        after=current.step(np.zeros((4,3)));before=previous.step(np.zeros((4,3)))
    for i in range(4):np.testing.assert_equal(after[i],before[i])
    assert after[2] and not after[3]
    changed={'red_success','red_win','blue_win','draw','termination_reason'}
    assert after[4].keys()==before[4].keys()
    for k in after[4].keys()-changed:np.testing.assert_equal(after[4][k],before[4][k])
    # blue_win remains False; all other outcome aliases change as intended.
    assert after[4]['red_success']==after[4]['red_win']==False
    assert not after[4]['blue_win'] and after[4]['draw']
    assert before[4]['red_win'] and not before[4]['draw']
    for i in range(1,5):
        np.testing.assert_array_equal(after[4][f'r{i}_rewards'],before[4][f'r{i}_rewards'])
        np.testing.assert_array_equal(after[4][f'episode_r{i}_rewards'],before[4][f'episode_r{i}_rewards'])
    assert after[4]['red_attack_kills']==after[4]['blue_attack_kills']==1
    assert current.red==previous.red and current.blue==previous.blue
    from algorithm.common.evaluator import aggregate_combat_records
    record=dict(after[4],episode_return=float(after[1].sum()),mean_agent_episode_return=float(after[1].mean()))
    metrics=aggregate_combat_records([record])
    assert metrics['win_rate']==metrics['blue_win_rate']==0 and metrics['draw_rate']==1


@pytest.mark.parametrize('a',[[0,0,0],[.1,.1,.1],[1,-1,1],[-1,1,-1]])
def test_shared_controller_and_paper_action_frozen(a):
    e=env();old=env('329');s=state()
    c=action_to_control_v329(s,a,e.config['action'],e.spec)
    assert c==action_to_control_v329(s,a,old.config['action'],old.spec) and c.nz<=10+1e-12
    e.red=[s.copy() for _ in range(4)];e.blue=deepcopy(e.red);actions=np.tile(a,(4,1))
    np.testing.assert_array_equal(e._advance(e.red,actions),e._advance(e.blue,actions));assert e.red==e.blue
    assert CombatEnvironmentV330._advance is CombatEnvironmentV329._advance


def test_frozen_v329_transition_for_four_live_aircraft():
    e,old=env(),env('329');old.red=deepcopy(e.red)+[dead()];old.blue=deepcopy(e.blue)+[dead()]
    assert environment_dimensions(old.config)==(56,3,5)
    assert old._observations().shape==(5,56)
    for name in ('dense_combat_reward','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_resolve_noncombat_losses'):
        assert getattr(type(e),name) is getattr(type(old),name)
    rng=np.random.default_rng(5)
    for _ in range(40):
        a=rng.uniform(-1,1,(4,3));x=e.step(a);y=old.step(np.concatenate([a,np.zeros((1,3))]))
        np.testing.assert_array_equal(x[1],y[1][:4]);assert e.red==old.red[:4] and e.blue==old.blue[:4]
        assert e.rng.bit_generator.state==old.rng.bit_generator.state
        for i in range(1,5):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'][:4])
        for side in ('red','blue'):
            np.testing.assert_array_equal(getattr(e,side+'_ammo'),getattr(old,side+'_ammo')[:4])


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_configs_only_dimension_identity_change(name):
    old=yaml.safe_load(Path(f'configs/{name}_5v5_v329.yaml').read_text());new=yaml.safe_load(Path(f'configs/{name}_4v4_v330.yaml').read_text())
    new['network']['observation_dim']=56;new['network']['num_agents']=5
    if 'environment' in new:new['environment']=old['environment']
    assert new==old


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_actual_four_agent_algorithm_networks(name):
    import torch
    cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v330.yaml').read_text());ec=env().config
    observations=np.zeros((3,4,45),dtype=np.float32);alive=np.ones((3,4),dtype=np.float32)
    o=torch.as_tensor(observations,device='cuda');m=torch.as_tensor(alive,device='cuda');a=torch.zeros(3,4,3,device='cuda')
    if name=='mappo':
        from algorithm.mappo.factory import build_mappo_trainer
        t=build_mappo_trainer(cfg,'cuda',32,32)
        assert t.critic(o,m).shape==(3,4)
        assert t.critic.value_network[0].in_features==225  # joint4 + local1, all45
    elif name=='madsac':
        from algorithm.madsac.factory import build_madsac_trainer
        from algorithm.madsac.protocol import validate_madsac_config
        validate_madsac_config(ec,cfg);t=build_madsac_trainer(cfg,'cuda',32)
        for critic in (t.critic1,t.critic2,t.target_critic1,t.target_critic2):
            q,weights=critic(o,a,m,True);assert q.shape==(3,4) and weights.shape==(3,2,4,4)
    else:
        from algorithm.maac.trainer import MAACTrainer
        from algorithm.maac.replay import JointReplayBuffer
        t=MAACTrainer(ec,cfg,'cuda',1,True)
        q,weights=t.critic(o,a,m,True);assert q.shape==(3,4,1) and weights.shape==(3,4,4,4)
        assert len(t.critic.encoders)==len(t.critic.q_heads)==4
        replay=JointReplayBuffer(64,observation_dim=45,num_agents=4)
        replay.add_batch(observations,np.zeros((3,4,3)),np.ones((3,4)),observations,np.zeros(3),alive,alive)
        b=replay.sample(3);assert b.observations.shape==(3,4,45) and b.rewards.shape==(3,4)
        assert np.isfinite(list(t.update(b).values())).all()
    assert t.act(observations,alive,deterministic=True).shape==(3,4,3)


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_old_checkpoint_version_rejected(name):
    cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v330.yaml').read_text());ec=env().config
    s=dict(algorithm='MAPPO' if name=='mappo' else 'MAAC' if name=='maac' else 'madsac',
        implementation_version=3,extra=dict(environment_version='3.29',training_smoke=True))
    if name=='mappo':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
        for fn in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
            with pytest.raises(RuntimeError,match='environment_version'):fn(s,ec,cfg)
    elif name=='madsac':
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        with pytest.raises(RuntimeError,match='environment_version'):validate_madsac_checkpoint(s,ec,cfg)
    else:
        from algorithm.maac.protocol import validate_checkpoint
        with pytest.raises(RuntimeError,match='environment_version'):validate_checkpoint(s,ec,cfg)
