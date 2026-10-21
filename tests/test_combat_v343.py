"""Candidate research reward mathematics and authoritative transition accounting."""
from copy import deepcopy
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.config import validate_config, environment_dimensions
from env.v343_reward import state_potential, continuous_tactical_rewards, RESEARCH_REWARD
from env.v343_config import validate_shaping_gamma
from env.v338_observation import build_team_observations_v338
from env.geometry import engagement_geometry
from tests.test_combat_v342 import config, pair

C = RESEARCH_REWARD


@pytest.mark.parametrize('distance',[0,1500,4000,8000,10000])
@pytest.mark.parametrize('angle',[0,30,90,180])
def test_potential_formula(distance,angle):
    a,t=pair(heading=angle,distance=distance)
    phi,diag=state_potential([a],[t],C)
    alpha=engagement_geometry(a,t).off_boresight
    assert phi[0] == pytest.approx(.1*(.5*(1-np.clip(distance/8000,0,1))+.5*(1-alpha/np.pi)))
    assert diag['target_indices'][0]==0
    assert 0<=phi[0]<=.1


def test_target_death_switch_no_target_and_self_death():
    a,t=pair(); _,far=pair(distance=2000)
    x,d=state_potential([a],[t,far],C); assert d['target_indices'][0]==0
    t.alive=False
    y,d=state_potential([a],[t,far],C); assert d['target_indices'][0]==1
    assert y[0]<x[0]
    far.alive=False
    assert state_potential([a],[t,far],C)[0][0]==0
    t.alive=True; a.alive=False
    assert state_potential([a],[t],C)[0][0]==0
    a.alive=True
    _,d=state_potential([a],[t,t.copy()],C); assert d['target_indices'][0]==0
    assert state_potential([a],[],C)[0][0]==0


@pytest.mark.parametrize('distance',[0.,1999.999,2000.,2000.001,3000.,3999.999,4000.,4000.001])
@pytest.mark.parametrize('angle',[0.,30.,59.999999,60.,60.000001,180.])
def test_r4_formula_bounds_continuity(distance,angle):
    a,t=pair(heading=angle,distance=distance)
    g=engagement_geometry(a,t); reverse=engagement_geometry(t,a)
    r=continuous_tactical_rewards([a],[t],C)
    w=np.clip((4000-distance)/2000,0,1)
    off=w*np.clip(1-g.off_boresight/(np.pi/3),0,1)*(.005+.015*(1+np.cos(g.aa))/2)
    de=-.01*w*np.clip(1-reverse.off_boresight/(np.pi/3),0,1)
    assert r['tactical_offense'][0]==pytest.approx(off)
    assert r['tactical_defense'][0]==pytest.approx(de)
    assert -.01<=r['tactical'][0]<=.02
    # Perturbing direction and distance across every knot must have a small effect.
    if distance>0:
        _,u=pair(heading=angle+1e-7,distance=distance+1e-6)
        s=continuous_tactical_rewards([a],[u],C)
        assert abs(s['tactical'][0]-r['tactical'][0])<1e-8


def test_aggregation_masks_and_reordering():
    a,t=pair(); _,u=pair(heading=180,distance=1000)
    u.psi=0  # rear threat points at Red
    dead=t.copy(); dead.alive=False
    one=continuous_tactical_rewards([a],[t],C)
    other=continuous_tactical_rewards([a],[u],C)
    for targets in ([t,u,dead],[dead,u,t],[u,t,dead]):
        r=continuous_tactical_rewards([a],targets,C)
        assert r['tactical_offense'][0]==max(one['tactical_offense'][0],other['tactical_offense'][0])
        assert r['tactical_defense'][0]==min(one['tactical_defense'][0],other['tactical_defense'][0])
        assert r['tactical_defense'][0]<0
    a.alive=False
    assert continuous_tactical_rewards([a],[t,u],C)['tactical'][0]==0
    assert continuous_tactical_rewards([a],[],C)['tactical'][0]==0


@pytest.mark.parametrize('mode',['ongoing','kill','self_death','ground','boundary','no_targets','truncated','mutual'])
def test_actual_step_order_and_ledger(mode):
    e=MultiUAVCombatEnv(config('343'));e.reset(9)
    a,t=pair(distance=1500)
    e.red=[a.copy() for _ in range(4)]; e.blue=[t.copy() for _ in range(4)]
    # Freeze motion to isolate state transitions from events.
    e._advance=lambda states,actions:np.zeros(4)
    e.red_ammo.fill(0);e.blue_ammo.fill(0)
    if mode=='truncated': e.max_steps=1
    if mode=='no_targets':
        for b in e.blue:b.alive=False
    if mode=='ground':e.red[0].z=0
    if mode=='boundary':e.red[0].x=5001
    before,_=e.potential()
    if mode in ('kill','self_death','mutual'):
        def resolve(red_attempts,blue_attempts):
            red_kills={};blue_kills={}
            if mode in ('kill','mutual'):
                for b in e.blue:b.alive=False
                red_kills={j:[0] for j in range(4)}
            if mode in ('self_death','mutual'):
                for a in e.red:a.alive=False
                blue_kills={j:[0] for j in range(4)}
            return red_kills,blue_kills
        e._resolve_combat=resolve
    _,reward,terminated,truncated,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
    after,_=e.potential()
    expected_next=np.zeros(4) if terminated or truncated else after
    np.testing.assert_allclose(info['r3_rewards'],.99*expected_next-before,rtol=0,atol=1e-15)
    np.testing.assert_allclose(info['phi_next_actual'],after,rtol=0,atol=0)
    np.testing.assert_allclose(info['phi_next_for_shaping'],expected_next,rtol=0,atol=0)
    np.testing.assert_array_equal(reward,sum(info[f'r{k}_rewards'] for k in range(1,5)))
    np.testing.assert_allclose(reward,info['event_rewards']+info['adv_rewards'],atol=1e-14)
    assert info['episode_research_reward_total']==pytest.approx(sum(info[f'episode_r{k}_total'] for k in range(1,5)))
    assert info['pbrs_enabled'] and not any('paper_reward' in k or k.startswith('madsac_r') for k in info)
    if mode in ('ground','self_death','mutual'):assert info['r1_rewards'][0] <= -10 or mode=='mutual'
    if mode=='boundary':assert info['r2_rewards'][0]==-10


@pytest.mark.parametrize('seed',[0,5,19])
def test_full_episode_discounted_telescoping_and_accounting(seed):
    e=MultiUAVCombatEnv(config('343'));e.reset(seed)
    phi,_=e.potential(); discounted=np.zeros(4); total=np.zeros(4)
    sums=np.zeros((4,4))
    while True:
        _,r,term,trunc,info=e.step(np.zeros((4,3)))
        discounted += .99**(e.steps-1)*info['r3_rewards']
        total += r
        for k in range(4):sums[k]+=info[f'r{k+1}_rewards']
        np.testing.assert_array_equal(r,sum(info[f'r{k}_rewards'] for k in range(1,5)))
        if term or trunc:break
    np.testing.assert_allclose(discounted,-phi,atol=2e-15)
    np.testing.assert_allclose(total,sums.sum(axis=0),atol=1e-12)
    for k in range(4):np.testing.assert_array_equal(sums[k],info[f'episode_r{k+1}_rewards'])
    assert info['episode_research_reward_total']==pytest.approx(total.sum(),abs=1e-12)


def test_stationary_potential_is_not_positive_annuity():
    e=MultiUAVCombatEnv(config('343'));e.reset(4)
    a,t=pair(heading=180,distance=6000)
    a.x+=3000; t.x+=3000
    e.red=[a.copy() for _ in range(4)];e.blue=[t.copy() for _ in range(4)]
    e._advance=lambda states,actions:np.zeros(4)
    p,_=e.potential(); total=np.zeros(4)
    for _ in range(1000):
        _,r,term,trunc,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
        assert np.all(info['r3_rewards']<=0) and np.all(info['r4_rewards']==0)
        total+=r
        if term or trunc:break
    assert trunc and not term
    np.testing.assert_allclose(total,-(1+.01*999)*p,atol=1e-13)


def test_49d_prefix_dead_slots_and_gamma():
    c=config('343');snapshot=deepcopy(c)
    validate_config(c);assert c==snapshot and environment_dimensions(c)==(49,3,4)
    e=MultiUAVCombatEnv(c);obs,_=e.reset(8)
    np.testing.assert_array_equal(obs[:,:45],build_team_observations_v338(e.red,e.blue,c['observation']))
    for i in range(4):
        for j in range(4):assert obs[i,45+j]==engagement_geometry(e.red[i],e.blue[j]).off_boresight/np.pi
    e.red[0].alive=False;e.blue[1].alive=False
    obs=e._observations();assert np.all(obs[0]==0) and np.all(obs[:,46]==0) and np.all(obs[:,30:35]==0)
    assert np.isfinite(obs).all() and np.all((obs[:,45:]>=0)&(obs[:,45:]<=1))
    validate_shaping_gamma(c,{'training':{'gamma':.99}})
    with pytest.raises(ValueError):validate_shaping_gamma(c,{'training':{'gamma':.95}})


@pytest.mark.parametrize('algo',['mappo','ippo'])
@pytest.mark.parametrize('version,dim',[('342',45),('343',49)])
def test_only_necessary_algorithm_fields_changed(algo,version,dim):
    def read(v):
        with open(f'configs/{algo}_4v4_v{v}.yaml') as f:return yaml.safe_load(f)
    old,new=read('341'),read(version)
    assert old['training']==new['training'] and old['implementation']==new['implementation']
    new['environment']=old['environment'];new['network']['observation_dim']=45
    assert new==old


def test_fixed_observation_diagnostic_49d_and_version(tmp_path):
    from types import SimpleNamespace
    from algorithm.ippo.diagnostics import save_observation_bank, rollout_action_diagnostics
    observations=np.zeros((2,1,4,49),np.float32)
    batch=SimpleNamespace(observations=observations,next_observations=observations.copy(),
        actions=np.zeros((2,1,4,3),np.float32),alive_masks=np.ones((2,1,4),np.float32),
        next_alive_masks=np.ones((2,1,4),np.float32),dones=np.zeros((2,1),np.float32))
    assert 'v3.43' in rollout_action_diagnostics(batch,'3.43')['flight_state_source']
    source=dict(split='training',seed_provenance=[0],environment_version='3.43')
    save_observation_bank(tmp_path/'49.npz',observations,source)
    with np.load(tmp_path/'49.npz') as bank:assert bank['observations'].shape[-1]==49
    with pytest.raises(ValueError):save_observation_bank(tmp_path/'bad.npz',observations[...,:45],source)


def test_mixed_version_summary_does_not_mislabel_research_as_paper():
    from algorithm.common.evaluator import aggregate_combat_records
    rows=[]
    for version in ('342','343'):
        e=MultiUAVCombatEnv(config(version));e.reset(0);e.max_steps=1
        _,r,_,_,info=e.step(np.zeros((4,3)))
        rows.append(dict(info,episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean())))
    result=aggregate_combat_records(rows)
    assert 'average_episode_r3_total' in result and 'average_episode_extended_reward_total' in result
    assert 'average_episode_paper_reward_total' not in result
    assert 'average_episode_research_reward_total' not in result
    assert 'average_madsac_r3_total' not in result
