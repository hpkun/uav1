"""Attacker-level entry windows, target switches, rearm, and real step cadence."""
from copy import deepcopy
import numpy as np
import pytest
from env.combat_v317 import CombatEnvironmentV317
from env.combat_v318 import CombatEnvironmentV318
from env.weapon import FireState
from tools.audit_combat_v317_v318 import window_env


def fire(e,side='red',attackers=None,targets=None):
    attackers=e.red if attackers is None else attackers
    targets=e.blue if targets is None else targets
    return e._entry_attempts(attackers,targets,getattr(e,f'{side}_fire_states'),side)


def test_only_cadence_overridden_and_configs_equal_except_version():
    e,old=window_env(),window_env('317')
    assert type(e) is CombatEnvironmentV318
    assert set(CombatEnvironmentV318.__dict__)-{'__module__','__doc__'}=={'environment_version','_new_fire_states','_entry_attempts'}
    for name in ('__init__','reset','step','build_weapon','attempt_hit_for_pair','_advance','_observations','dense_combat_reward','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV318,name) is getattr(CombatEnvironmentV317,name)
    cfg=deepcopy(e.config);cfg['environment_version']='3.17';assert cfg==old.config
    assert type(e.weapon) is type(old.weapon)


def test_twenty_continuous_eligible_real_steps_one_attempt_ammo_and_rng():
    e=window_env()
    for _ in range(20):
        _,_,term,trunc,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
        assert not term and not trunc
    assert info['red_fire_attempts']==1 and info['red_ammo_used']==1
    assert e.red_ammo[0]==5 and e.rng.calls==1 and not e.red_fire_states[0].armed


@pytest.mark.parametrize('switch',['target_dead','target_not_nearest'])
def test_new_target_with_nonempty_window_does_not_rearm(switch):
    e=window_env();assert fire(e)[0][1]==0
    if switch=='target_dead':e.blue[0].alive=False
    else:e.blue[0].x=3800
    assert fire(e)==[] and e.red_ammo[0]==5 and e.rng.calls==1
    assert not e.red_fire_states[0].armed


def test_empty_window_rearms_and_next_entry_fires_without_timer():
    e=window_env();assert len(fire(e))==1
    e.blue[0].x=e.blue[1].x=5000
    assert fire(e)==[] and e.red_fire_states[0].armed and e.rng.calls==1
    e.blue[1].x=3500
    assert fire(e)[0][1]==1 and e.red_ammo[0]==4 and e.rng.calls==2


def test_zero_ammo_never_attempts_or_draws():
    e=window_env();e.red_ammo[0]=0
    for _ in range(20):assert fire(e)==[]
    assert e.rng.calls==0 and e.red_ammo[0]==0


def test_reset_has_ten_distinct_armed_fire_states():
    e=window_env()
    for state in e.red_fire_states+e.blue_fire_states:state.armed=False
    e.red_ammo[0]=0;e.reset(31)
    states=e.red_fire_states+e.blue_fire_states
    assert len(states)==10 and len({id(s) for s in states})==10
    assert all(type(s) is FireState and s.armed for s in states)
    assert (e.red_ammo==6).all() and (e.blue_ammo==6).all()


def test_red_blue_cadence_symmetric():
    e=window_env();a,b=deepcopy(e.red),deepcopy(e.blue)
    assert fire(e,'red',a,b)==fire(e,'blue',deepcopy(a),deepcopy(b))
    assert fire(e,'red',a,b)==fire(e,'blue',deepcopy(a),deepcopy(b))==[]
    np.testing.assert_array_equal(e.red_ammo,e.blue_ammo)
    assert [s.armed for s in e.red_fire_states]==[s.armed for s in e.blue_fire_states]
    assert e.rng.calls==2


def test_one_attempt_nearest_and_index_tie_break():
    e=window_env();e.blue[1].x=e.blue[0].x
    shots=fire(e);assert len(shots)==1 and shots[0][:2]==(0,0)
    assert e.red_ammo[0]==5 and e.rng.calls==1


def test_fire_states_are_attacker_local_not_team_shared():
    e=window_env();e.red[1]=e.red[0].copy()
    shots=fire(e)
    assert [a for a,_,_ in shots]==[0,1] and e.rng.calls==2
    assert not e.red_fire_states[0].armed and not e.red_fire_states[1].armed
    assert fire(e)==[]


def test_v317_keeps_repeat_fire_and_ammo_six():
    e=window_env('317')
    assert sum(len(fire(e)) for _ in range(20))==6
    assert e.red_ammo[0]==0 and e.rng.calls==6
