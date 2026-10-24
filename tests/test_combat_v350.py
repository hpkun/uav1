"""Paper reward boundaries, exclusive branches, version isolation and physics parity."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pytest
import yaml

from env.combat_env import MultiUAVCombatEnv
from env.config import load_config, validate_config, environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v338_observation import build_team_observations_v338
from env.v350_reward import paper_pair_rewards, nearest_target_rewards
from tools.combat_visualization import RecordingCombatEnv


def config(version='350'):
    return load_config(f'configs/combat_environment_v{version}.yaml')


def state(x=0., y=0., altitude=3000., psi=0., theta=0., alive=True):
    return AircraftState(x=x, y=y, z=-altitude, v=225., psi=psi, theta=theta, alive=alive)


def geometry(distance=4000., ata=0., ha=0., aa=0., off=math.pi):
    g = engagement_geometry(state(), state(x=distance))
    return replace(g, distance=distance, ata=math.radians(ata), ha=math.radians(ha),
                   aa=math.radians(aa), off_boresight=off)


def frozen(version='350'):
    e = RecordingCombatEnv(config(version)); e.reset(7)
    e._advance = lambda states, actions: np.zeros(4)
    e.red = [state(x=20*i) for i in range(4)]
    e.blue = [state(x=1000+20*i, psi=math.pi) for i in range(4)]
    return e


ZERO = np.zeros((4, 3))


def test_schema_frozen_networks_and_no_pbrs_calls(monkeypatch):
    c, old = config(), config('348')
    assert environment_dimensions(c) == (45, 3, 4)
    for key in old:
        if key not in ('environment_version', 'reward', 'research_reward'):
            assert c[key] == old[key]
    reward = deepcopy(c['reward']); reward['guide_distance_min'] = 3000.
    assert reward == old['reward']
    for algorithm in ('mappo', 'maac'):
        a, b = [yaml.safe_load(Path(f'configs/{algorithm}_4v4_v{v}.yaml').read_text(encoding='utf-8'))
                for v in ('348', '350')]
        b['environment'] = a['environment']; assert a == b
    def forbidden(*args):
        raise AssertionError('PBRS/continuous candidate reward must not execute')
    import env.combat_v343 as module
    monkeypatch.setattr(module, 'state_potential', forbidden)
    monkeypatch.setattr(module, 'continuous_tactical_rewards', forbidden)
    # Also forbid the old gamma difference/terminal compensation hooks.
    from env.combat_v343 import CombatEnvironmentV343
    monkeypatch.setattr(CombatEnvironmentV343, 'advantage_reward', forbidden)
    monkeypatch.setattr(CombatEnvironmentV343, 'shaping_next_potential', forbidden)
    e = MultiUAVCombatEnv(c); obs, info = e.reset(7)
    assert not info['pbrs_enabled'] and 'research_reward' not in c
    for _ in range(3):
        obs, r, _, _, info = e.step(ZERO)
        np.testing.assert_array_equal(obs, build_team_observations_v338(e.red, e.blue, c['observation']))
        assert not info['pbrs_enabled'] and not info['full_paper_environment_replication']
        assert not any(k.startswith('episode_research') for k in info)
        assert not info['phi_current'].any() and not info['phi_next'].any()


@pytest.mark.parametrize('section,key,value', [
    ('reward','guide_distance_min',3000.), ('reward','offense_reward_5deg',.2),
    ('reward','win_reward',10.), ('reward','safety_penalty_scale',.1),
    ('reward','team_casualty_penalty',-1.), ('weapon','range_max',4000.),
    ('observation','extra',True), ('simulation','dt',.2)])
def test_frozen_config(section, key, value):
    c = config(); c[section][key] = value
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('extra', ['research_reward','candidate_reward'])
def test_forbid_research_config(extra):
    c = config(); c[extra] = {}
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('distance', [3999.999999,4000.,4000.000001,8000.])
@pytest.mark.parametrize('ata', [-30.000001,-30.,0.,30.,30.000001])
@pytest.mark.parametrize('ha', [-30.000001,-30.,0.,30.,30.000001])
def test_r3_exact_boundaries(distance, ata, ha):
    r3, _, _ = paper_pair_rewards(geometry(distance,ata,ha), geometry(aa=180.), config()['reward'])
    expected = .001 if distance >= 4000. and abs(ata) <= 30 and abs(ha) <= 30 else 0.
    assert r3 == expected


@pytest.mark.parametrize('branch', ['offense','defense'])
@pytest.mark.parametrize('distance', [3999.999999,4000.,4000.000001])
@pytest.mark.parametrize('aa', [-30.000001,-30.,0.,30.,30.000001])
@pytest.mark.parametrize('angle', [-30.000001,-30.,-15.,-5.,0.,5.,5.000001,15.,15.000001,30.,30.000001])
@pytest.mark.parametrize('axis', ['ata','ha'])
def test_r4_all_bins_and_gates(branch, distance, aa, angle, axis):
    kwargs = {axis:angle, 'distance':distance, 'aa':aa}
    active, inactive = geometry(**kwargs), geometry(aa=180.)
    rg, bg = (active,inactive) if branch=='offense' else (inactive,active)
    _, offense, defense = paper_pair_rewards(rg,bg,config()['reward'])
    expected = 0.
    if distance <= 4000. and abs(aa) <= 30:
        for threshold, pos, neg in ((5,.1,-.15),(15,.02,-.025),(30,.01,-.015)):
            if abs(angle) <= threshold:
                expected = pos if branch=='offense' else neg; break
    assert offense+defense == expected
    assert (defense if branch=='offense' else offense) == 0.


def test_exclusive_outer_branches_including_zero_offense():
    c=config()['reward']
    assert paper_pair_rewards(geometry(ata=31.),geometry(),c)[1:] == (0.,0.)
    assert paper_pair_rewards(geometry(),geometry(),c)[1:] == (.1,0.)
    assert paper_pair_rewards(geometry(aa=31.),geometry(),c)[1:] == (0.,-.15)
    assert paper_pair_rewards(geometry(aa=31.),geometry(aa=31.),c)[1:] == (0.,0.)


@pytest.mark.parametrize('ata,ha,expected', [(5,15,.02),(15,5,.02),(15,30,.01),(30,15,.01),(0,31,0.)])
def test_both_ata_and_ha_must_meet_same_bin(ata,ha,expected):
    r3,offense,defense=paper_pair_rewards(geometry(ata=ata,ha=ha),geometry(aa=180.),config()['reward'])
    assert offense==expected and defense==0.
    assert r3==(.001 if max(abs(ata),abs(ha))<=30 else 0.)


def test_real_geometry_uses_paper_ata_ha_not_3d_angle():
    # Pitch is deliberately opposed to LOS; paper ATA/HA stay zero.
    red, blue = state(theta=math.radians(60)), state(x=4500.)
    g=engagement_geometry(red,blue);assert g.off_boresight > math.pi/6
    result,_=nearest_target_rewards([red],[blue],config()['reward'])
    assert result['guide'][0] == .001
    blue.x=1000.
    result,_=nearest_target_rewards([red],[blue],config()['reward'])
    assert result['tactical_offense'][0] == .1


def test_one_nearest_alive_target_not_max_reward_and_ties():
    red=state(); near=state(x=-1000.); far=state(x=2000.)
    reward=config()['reward']
    result,indices=nearest_target_rewards([red],[near,far],reward)
    assert indices.tolist()==[0]
    assert result['tactical_offense'][0]==0.  # far aligned target would pay +.1
    near.alive=False
    result,indices=nearest_target_rewards([red],[near,far],reward)
    assert indices.tolist()==[1] and result['tactical_offense'][0]==.1
    near.alive=True;far.x=1000.
    assert nearest_target_rewards([red],[near,far],reward)[1].tolist()==[0]
    near.alive=far.alive=False
    result,indices=nearest_target_rewards([red],[near,far],reward)
    assert indices.tolist()==[-1] and not result['tactical'].any()
    red.alive=False;far.alive=True
    assert nearest_target_rewards([red],[near,far],reward)[1].tolist()==[-1]
    # R3 cannot separately select a farther guide-eligible Blue.
    red.alive=True;near=state(x=1000.);far=state(x=4500.)
    result,indices=nearest_target_rewards([red],[near,far],reward)
    assert indices.tolist()==[0] and result['guide'][0]==0.


def test_r1_unique_kill_shared_credit_and_single_loss():
    e=frozen()
    red_kills,blue_kills=e._resolve_combat([(0,0,True),(1,0,True)],[(0,2,True),(1,2,True)])
    r1,r2=e._event_reward_components([],[],red_kills,blue_kills)
    np.testing.assert_array_equal(r1,[5.,5.,-10.,0.]);assert not r2.any()
    assert e.combat_counts['red']['attack_kills']==e.combat_counts['blue']['attack_kills']==1


@pytest.mark.parametrize('cause', ['ground','ceiling','horizontal','ground_and_horizontal'])
def test_noncombat_loss_single_penalty_in_real_step(cause):
    e=frozen();e.red_ammo.fill(0);e.blue_ammo.fill(0)
    if 'ground' in cause:e.red[0].z=0.
    if 'horizontal' in cause:e.red[0].x=5001.
    if cause=='ceiling':e.red[0].z=-6001.
    _,r,_,_,info=e.step(ZERO,ZERO)
    is_ground='ground' in cause
    assert info['r1_rewards'][0]==(-10. if is_ground else 0.)
    assert info['r2_rewards'][0]==(0. if is_ground else -10.)
    assert r[0]==-10. and not e.red[0].alive
    assert info['red_ground_losses']==int(is_ground)
    assert info['red_boundary_exits']==int(not is_ground)
    _,r,_,_,info=e.step(ZERO,ZERO)
    assert r[0]==0.  # no repeat punishment of an already dead aircraft


@pytest.mark.parametrize('red,blue,timeout,expected', [
    (4,0,False,(True,False,False,'red_win_elimination')),
    (0,3,False,(False,True,False,'blue_win_elimination')),
    (0,0,False,(True,False,False,'red_win_mutual_destruction')),
    (1,3,True,(False,True,False,'blue_win_timeout')),
    (4,1,True,(False,True,False,'blue_win_timeout')),
    (2,2,True,(False,True,False,'blue_win_timeout')),
    (4,4,False,(False,False,False,'ongoing'))])
def test_outcome(red,blue,timeout,expected):
    e=frozen()
    for side,count in (('red',red),('blue',blue)):
        for i,s in enumerate(getattr(e,side)):s.alive=i<count
    assert e._outcome(timeout)==expected


def test_real_mutual_destruction_and_timeout_ledger():
    for version in ('348','349','350'):
        e=frozen(version);e.attempt_hit_for_pair=lambda *args:True
        e.red=[state(x=20*i,y=1000*i) for i in range(4)]
        e.blue=[state(x=1000+20*i,y=1000*i,psi=math.pi) for i in range(4)]
        _,r,term,trunc,info=e.step(ZERO,ZERO)
        assert term and not trunc and not e.red_alive_mask.any() and not e.blue_alive_mask.any()
        assert info['red_win']==(version=='350') and info['draw']==(version!='350')
        assert not info['outcome_rewards'].any() and not info['r1_rewards'].any()
        assert info['termination_reason']==('red_win_mutual_destruction' if version=='350' else 'draw_mutual_destruction')
        e=frozen(version);e.red_ammo.fill(0);e.blue_ammo.fill(0);e.blue[1].alive=False
        e.steps=999
        _,r,term,trunc,info=e.step(ZERO,ZERO)
        assert not term and trunc and info['blue_win'] and not info['draw']
        assert info['termination_reason']=='blue_win_timeout'
        assert not info['outcome_rewards'].any()


@pytest.mark.parametrize('red,blue,timeout', [(4,0,False),(0,3,False),(0,0,False),(1,3,True),(4,1,True),(2,2,True),(4,4,False)])
def test_step_and_evaluation_share_outcome(red,blue,timeout):
    from algorithm.common.evaluator import aggregate_combat_records
    e=frozen();e.red_ammo.fill(0);e.blue_ammo.fill(0)
    for side,count in (('red',red),('blue',blue)):
        for i,s in enumerate(getattr(e,side)):s.alive=i<count
    e.steps=999 if timeout else 0
    _,r,term,trunc,info=e.step(ZERO,ZERO)
    assert term==(red==0 or blue==0)
    assert trunc==(timeout and not term)
    expected_win=blue==0
    expected_loss=not expected_win and (red==0 or timeout)
    assert info['red_win']==info['red_success']==expected_win
    assert info['blue_win']==expected_loss and not info['draw']
    assert not info['outcome_rewards'].any()
    row=dict(info,episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean()))
    result=aggregate_combat_records([row])
    assert result['win_rate']==result['red_win_rate']==float(expected_win)
    assert result['blue_win_rate']==float(expected_loss) and result['draw_rate']==0.
    assert result['mutual_annihilation_draw_rate']==0.
    assert result['mutual_destruction_red_win_rate']==float(red==blue==0)
    assert not result['pbrs_enabled']
    assert result['average_episode_paper_reward_total']==pytest.approx(float(r.sum()))


@pytest.mark.parametrize('seed', [1,7,12,20000071])
def test_v348_v350_exact_states_events_rng_observations_and_reward_ledger(seed):
    old,new=[RecordingCombatEnv(config(v)) for v in ('348','350')]
    np.testing.assert_array_equal(old.reset(seed)[0],new.reset(seed)[0])
    totals=np.zeros((4,4));saw_shots=False
    while True:
        actions=old.fixed_policy.team_actions(old.red,old.blue)
        a,b=old.step(actions),new.step(actions)
        np.testing.assert_array_equal(a[0],b[0]);assert a[2:4]==b[2:4]
        assert old.events==new.events and old.combat_counts==new.combat_counts
        assert old.rng.bit_generator.state==new.rng.bit_generator.state
        for side in ('red','blue'):
            np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)], [s.as_array() for s in getattr(new,side)])
            for suffix in ('ammo','fire_cooldown_steps','alive_mask'):
                np.testing.assert_array_equal(getattr(old,side+'_'+suffix),getattr(new,side+'_'+suffix))
        info=b[4];components=np.stack([info[f'r{i}_rewards'] for i in range(1,5)])
        totals+=components
        np.testing.assert_array_equal(b[1],sum(components))
        np.testing.assert_array_equal(info['local_rewards'],b[1])
        for i in range(1,5):
            np.testing.assert_allclose(info[f'episode_r{i}_rewards'],totals[i-1],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(info['r1_rewards'],a[4]['r1_rewards'])
        np.testing.assert_array_equal(info['r2_rewards'],a[4]['r2_rewards'])
        assert not info['safe_rewards'].any() and not info['outcome_rewards'].any()
        assert not info['team_casualty_rewards'].any()
        assert not info['phi_next_for_shaping'].any()
        saw_shots |= bool(info['red_fire_attempts']+info['blue_fire_attempts'])
        if b[2] or b[3]:break
    assert saw_shots
    np.testing.assert_allclose(info['episode_paper_reward_rewards'],totals.sum(axis=0),rtol=0,atol=1e-11)
