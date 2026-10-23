"""Isolated probabilistic ablation: geometry, sampling, resources and rewards."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v343 import CombatEnvironmentV343
from env.combat_v346 import CombatEnvironmentV346
from env.combat_v347 import CombatEnvironmentV347
from env.config import load_config, validate_config, environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v338_observation import build_team_observations_v338
from env.v342_weapon import ConeWeapon
from tests.test_combat_v342 import pair
from tests.test_combat_v344 import frozen, state


def config(v='348'):
    return load_config(f'configs/combat_environment_v{v}.yaml')


def env():
    e = MultiUAVCombatEnv(config()); e.reset(7); return e


def fixed(sharing=False):
    e = frozen('348', sharing)
    e.red_ammo.fill(6); e.blue_ammo.fill(6)
    e.red_fire_cooldown_steps.fill(0); e.blue_fire_cooldown_steps.fill(0)
    return e


def test_config_observation_reward_implementation_identity():
    c, old = config(), config('347')
    before = deepcopy(c); validate_config(c); assert c == before
    assert environment_dimensions(c) == (45, 3, 4)
    for key in old:
        if key not in ('environment_version', 'weapon', 'fire_control'):
            assert c[key] == old[key]
    e = env(); assert not hasattr(e, 'dwell') and e.cooldown_steps == 10
    reference=MultiUAVCombatEnv(old)
    np.testing.assert_array_equal(e._observations(),reference.reset(7)[0])
    for name in ('potential', 'shaping_next_potential', 'advantage_reward', 'reward_aliases', '_reward_diagnostics'):
        assert getattr(type(e), name) is getattr(CombatEnvironmentV347, name)
    assert type(e).dense_combat_reward is CombatEnvironmentV346.dense_combat_reward
    for a in ('mappo', 'maac'):
        x, y = [yaml.safe_load(Path(f'configs/{a}_4v4_v{v}.yaml').read_text()) for v in ('347', '348')]
        y['environment'] = x['environment']; assert x == y
    for _ in range(5):
        obs, r, _, _, info = e.step(np.zeros((4, 3)))
        np.testing.assert_array_equal(obs, build_team_observations_v338(e.red, e.blue, c['observation']))
        np.testing.assert_array_equal(r, sum(info[f'r{i}_rewards'] for i in range(1, 5)))
        assert info['dwell_metrics_applicable'] is False
    e.red[0].alive = False
    assert not e._observations()[0].any()
    e.red_ammo.fill(0); e.red_fire_cooldown_steps.fill(10)
    e.reset(7); assert np.all(e.red_ammo == 6) and not e.red_fire_cooldown_steps.any()


@pytest.mark.parametrize('section,key,value', [
    ('weapon', 'ammo_per_aircraft', 7), ('weapon', 'effective_hit_distance', 1116.),
    ('weapon', 'azimuth_noise_scale', 1.), ('weapon', 'off_boresight_max', math.pi/4),
    ('fire_control', 'cooldown_seconds', .1), ('fire_control', 'dwell_seconds', .5),
    ('reward', 'kill_reward', 11), ('observation', 'extra', True),
    ('simulation', 'dt', .2), ('research_reward', 'gamma', .98)])
def test_frozen_schema(section, key, value):
    c = config(); c[section][key] = value
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('distance,valid', [(0, True), (1999.999, True), (2000, True), (2000.001, False)])
@pytest.mark.parametrize('angle,inside', [(0, True), (29.999999, True), (30, True), (30.000001, False), (40, False)])
@pytest.mark.parametrize('axis', ['heading', 'elevation'])
def test_distance_cone_boundaries(distance, valid, angle, inside, axis, monkeypatch):
    a, b = pair(distance=distance, **{axis: angle})
    # Isolate literal range bounds from roundoff in trigonometric construction.
    from dataclasses import replace
    import env.v342_weapon as module
    original = module.engagement_geometry
    monkeypatch.setattr(module, 'engagement_geometry',
                        lambda a,b: replace(original(a,b), distance=distance))
    assert ConeWeapon(**config()['weapon']).qualifies(a, b, 6) == (valid and (inside or distance == 0))


@pytest.mark.parametrize('az,el', [(0, 0), (20, 10), (-20, 10), (25, -25), (-25, -25)])
def test_original_gaussian_formula_probability_and_sampling(az, el):
    w = env().weapon
    assert w.attempt_hit.__func__ is ConeWeapon.attempt_hit
    az, el = np.deg2rad([az, el]); probabilities = []
    for distance in (0, 500, 1000, 1500, 2000):
        threshold = math.pi*math.exp(-distance/1116.2212531024945)
        scale = .9984711359155588
        low=max((-threshold-az)/scale, (-threshold-el)/scale)
        high=min((threshold-az)/scale, (threshold-el)/scale)
        expected=0 if low > high else .5*(math.erf(high/math.sqrt(2))-math.erf(low/math.sqrt(2)))
        assert w.expected_hit_probability(distance, az, el) == pytest.approx(expected, abs=1e-15)
        probabilities.append(expected)
    assert all(x >= y for x, y in zip(probabilities, probabilities[1:]))
    class Rng:
        calls = 0
        def normal(self, loc, std):
            assert (loc, std) == (0, 1); self.calls += 1; return .41
    rng=Rng(); hit=w.attempt_hit(rng, 1800, az, el)
    threshold=math.pi*math.exp(-1800/1116.2212531024945)
    assert hit == (abs(az+scale*.41) <= threshold and abs(el+scale*.41) <= threshold)
    assert rng.calls == 1
    x=np.random.default_rng(19); y=np.random.default_rng(19)
    hits=[w.attempt_hit(x,1800,az,el) for _ in range(1000)]
    assert hits == [w.attempt_hit(y,1800,az,el) for _ in range(1000)]
    assert 0 < sum(hits) < len(hits)


def test_failed_shots_symmetric_cooldown_exhaustion_and_reset():
    e=fixed(); e.attempt_hit_for_pair=lambda *args: False
    for k in range(1, 72):
        _, r, term, trunc, info=e.step(np.zeros((4,3)),np.zeros((4,3)))
        expected=min(6,(k-1)//10+1)
        for side in ('red','blue'):
            assert np.all(getattr(e,side+'_ammo') == 6-expected)
            assert info[side+'_fire_attempts'] == 4*expected
            assert info[side+'_ammo_used'] == 4*expected and info[side+'_weapon_hits'] == 0
            assert info[side+'_step_fire_attempts'] == (4 if k in (1,11,21,31,41,51) else 0)
        assert not term and not trunc
    assert e.red_alive_mask.sum() == e.blue_alive_mask.sum() == 4
    assert not e.red_fire_cooldown_steps.any()


@pytest.mark.parametrize('side', ['red', 'blue'])
@pytest.mark.parametrize('invalid', ['ammo', 'attacker_dead', 'target_dead', 'out_of_range', 'out_of_cone'])
def test_invalid_targets_never_sample(side, invalid):
    e=env(); a,b=pair(); attackers=[a]+[state(alive=False) for _ in range(3)]
    targets=[b]+[state(alive=False) for _ in range(3)]
    if invalid=='ammo': getattr(e,side+'_ammo').fill(0)
    if invalid=='attacker_dead': a.alive=False
    if invalid=='target_dead': b.alive=False
    if invalid=='out_of_range': b.x=2100
    if invalid=='out_of_cone': b.x=-1500
    e.blue_step_target_indices.fill(0)
    e.attempt_hit_for_pair=lambda *args: pytest.fail('invalid attack sampled RNG')
    assert e._entry_attempts(attackers,targets,None,side) == []


def test_synchronous_hits_mutual_destruction_no_dwell():
    e=fixed(); e.attempt_hit_for_pair=lambda *args: True
    obs,r,term,trunc,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
    assert term and not trunc and info['termination_reason']=='draw_mutual_destruction'
    assert not obs.any()
    for side in ('red','blue'):
        assert info[side+'_fire_attempts']==info[side+'_weapon_hits']==info[side+'_attack_kills']==4
    np.testing.assert_array_equal(info['r1_rewards'],np.zeros(4))


def test_shared_unique_victim_and_reward_credit():
    e=fixed(True); e.attempt_hit_for_pair=lambda *args: True
    e.blue_ammo.fill(0)
    _,r,term,_,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
    assert term and info['red_win']
    assert info['red_weapon_hits']==4 and info['red_attack_kills']==1
    np.testing.assert_array_equal(info['r1_rewards'],[2.5]*4)
    np.testing.assert_array_equal(r,sum(info[f'r{i}_rewards'] for i in range(1,5)))


def test_blue_step_start_target_binding_and_red_nearest_eligible():
    e=env(); e.red=[state(x=1500,psi=np.pi),state(x=1600,psi=np.pi)]+[state(alive=False) for _ in range(2)]
    e.blue=[state()]+[state(alive=False) for _ in range(3)]
    def advance(states, actions):
        if states is e.red: states[0].x=1900; states[1].x=1000
        return np.zeros(4)
    e._advance=advance; fired=[]
    e.attempt_hit_for_pair=lambda a,b: fired.append((a,b)) or False
    info=e.step(np.zeros((4,3)),np.zeros((4,3)))[4]
    assert info['blue_pursuit_target_indices'][0]==0
    assert fired[-1][1] is e.red[0]


def test_red_nearest_qualified_target_and_no_rng_during_cooldown():
    e=env(); e.red=[state()]+[state(alive=False) for _ in range(3)]
    e.blue=[state(x=-100),state(x=1800),state(x=1200),state(x=1200)]
    fired=[]
    e.attempt_hit_for_pair=lambda a,b: fired.append(b) or False
    attempts=e._entry_attempts(e.red,e.blue,None,'red')
    assert attempts==[(0,2,False)] and fired==[e.blue[2]]
    for _ in range(9): assert e._entry_attempts(e.red,e.blue,None,'red')==[]
    assert len(fired)==1
    assert e._entry_attempts(e.red,e.blue,None,'red')==[(0,2,False)]


def test_real_sampling_uses_body_errors_and_environment_rng_only():
    e=env();a,b=pair(heading=15,elevation=-10)
    g=engagement_geometry(a,b)
    duplicate=np.random.default_rng();duplicate.bit_generator.state=deepcopy(e.rng.bit_generator.state)
    expected=e.weapon.attempt_hit(duplicate,g.distance,g.boresight_azimuth_error,g.boresight_elevation_error)
    assert e.attempt_hit_for_pair(a,b)==expected
    assert e.rng.bit_generator.state==duplicate.bit_generator.state


@pytest.mark.parametrize('algorithm', ['mappo','maac'])
def test_algorithm_config_rejects_wrong_dimensions_and_identity(algorithm,tmp_path):
    c=config();ac=yaml.safe_load(Path(f'configs/{algorithm}_4v4_v348.yaml').read_text())
    if algorithm=='maac':
        from algorithm.maac.protocol import validate_config_pair
        validate=lambda a:validate_config_pair(c,a)
    else:
        from algorithm.mappo.runner import MAPPOTrainingRunner
        validate=lambda a:MAPPOTrainingRunner(c,a,output_dir=tmp_path,device='cuda')
    bad=deepcopy(ac);bad['network']['observation_dim']=49
    with pytest.raises((ValueError,RuntimeError)): validate(bad)
    bad=deepcopy(ac);bad['environment']['environment_version']='3.47'
    with pytest.raises((ValueError,RuntimeError)): validate(bad)


def test_complete_fixed_seed_trajectory_and_reward_invariance_without_hits():
    a,b=env(),env()
    for k in range(150):
        action=np.full((4,3),.1*np.sin(k*.05))
        x,y=a.step(action),b.step(action)
        for i in (0,1): np.testing.assert_array_equal(x[i],y[i])
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        for side in ('red','blue'):
            np.testing.assert_array_equal(getattr(a,side+'_ammo'),getattr(b,side+'_ammo'))
        if x[2] or x[3]: break
    a,b=fixed(),frozen('347')
    a.attempt_hit_for_pair=lambda *args: False
    b._entry_attempts=lambda *args: []
    a.max_steps=b.max_steps=3
    for _ in range(3):
        x=a.step(np.zeros((4,3)),np.zeros((4,3)))
        y=b.step(np.zeros((4,3)),np.zeros((4,3)))
        for i in range(1,5): np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
    assert x[3] and not x[4]['phi_next_for_shaping'].any()


def test_full_dynamic_trajectory_rewards_match_v347_with_weapon_events_disabled():
    a,b=env(),MultiUAVCombatEnv(config('347'));b.reset(7)
    for e in (a,b): e._entry_attempts=lambda *args: []
    while True:
        action=np.zeros((4,3));action[:,0]=.2*np.sin(a.steps*.05)
        x,y=a.step(action),b.step(action)
        for i in (0,1): np.testing.assert_array_equal(x[i],y[i])
        np.testing.assert_array_equal([s.as_array() for s in a.red+a.blue],
                                      [s.as_array() for s in b.red+b.blue])
        assert x[2:4]==y[2:4]
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        for i in range(1,5): np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
        if x[2] or x[3]: break
