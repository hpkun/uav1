"""ATA/HA weapon boundaries, pitch independence and frozen v3.35 mechanisms."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import inspect
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v335 import CombatEnvironmentV335
from env.combat_v336 import CombatEnvironmentV336
from env.config import validate_config, environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v317_weapon import BodyFrameGaussianWeapon
from env.v336_weapon import PaperATAHAWeapon


def env(version='336'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def pair(distance=500, ata=0, ha=0, theta=0):
    ata, ha, theta = np.deg2rad([ata, ha, theta])
    a = AircraftState(0, 0, -3000, 225, theta, 0)
    b = AircraftState(distance*np.cos(ha)*np.cos(ata), distance*np.cos(ha)*np.sin(ata),
                      -3000-distance*np.sin(ha), 225, 0, math.pi)
    return a, b


def test_inheritance_exact_numerical_config_and_frozen_methods():
    e, old = env(), env('335')
    assert CombatEnvironmentV336.__bases__ == (CombatEnvironmentV335,)
    assert type(e) is CombatEnvironmentV336
    assert type(e.weapon) is PaperATAHAWeapon and type(old.weapon) is BodyFrameGaussianWeapon
    assert environment_dimensions(e.config) == (45, 3, 4)
    c = deepcopy(e.config)
    c['environment_version'] = '3.35'
    c['weapon']['boresight_azimuth_max'] = c['weapon'].pop('ata_max')
    c['weapon']['boresight_elevation_max'] = c['weapon'].pop('ha_max')
    assert c == old.config
    for name in ('__init__','initial_states','reset','step','_advance','dense_combat_reward',
                 '_resolve_combat','_outcome','_observations'):
        assert getattr(CombatEnvironmentV336, name) is getattr(CombatEnvironmentV335, name)
    assert (e.weapon.range_min, e.weapon.range_max, e.weapon.ammo_per_aircraft) == (0, 1000, 6)
    assert e.weapon.effective_hit_distance == 558.1106265512473
    assert e.weapon.azimuth_noise_scale == e.weapon.elevation_noise_scale == .9984711359155588
    assert e.cooldown_steps == 10 and e.config['fire_control']['cooldown_seconds'] == 1
    for method in (PaperATAHAWeapon.eligibility, CombatEnvironmentV336.attempt_hit_for_pair):
        source = inspect.getsource(method)
        assert 'boresight' not in source and 'off_boresight' not in source
    assert list(inspect.signature(PaperATAHAWeapon.expected_hit_probability).parameters) == ['self','distance','ata','ha']


@pytest.mark.parametrize('distance,ata,ha,expected',[
    (0,0,0,True),(500,0,0,True),(500,30,0,True),(500,-30,0,True),
    (500,30.00001,0,False),(500,-30.00001,0,False),
    (500,0,30,True),(500,0,-30,True),(500,0,30.00001,False),
    (500,0,-30.00001,False),(1000,0,0,True),(1000.000001,0,0,False)])
def test_eligibility_inclusive_boundaries(distance, ata, ha, expected):
    assert env().weapon.qualifies(*pair(distance, ata, ha), 6) is expected


@pytest.mark.parametrize('attacker_alive,target_alive,ammo,expected',[
    (True,True,6,True),(False,True,6,False),(True,False,6,False),(True,True,0,False)])
def test_live_ammo_and_aa_independence(attacker_alive,target_alive,ammo,expected):
    a,b = pair()
    a.alive, b.alive = attacker_alive,target_alive
    # Head-on AA=180 never participates in qualification.
    assert abs(engagement_geometry(a,b).aa) == math.pi
    assert env().weapon.qualifies(a,b,ammo) is expected


@pytest.mark.parametrize('ata,ha,expected',[(0,0,True),(20,20,True),(31,0,False),(0,35,False)])
def test_pitch_independent_geometry_and_qualification(ata,ha,expected):
    w = env().weapon
    angles = []
    for pitch in (0,20,-20):
        a,b = pair(500,ata,ha,pitch)
        g = engagement_geometry(a,b)
        angles.append((g.ata,g.ha))
        assert w.qualifies(a,b,6) is expected
    assert angles[0] == angles[1] == angles[2]


def test_ha35_pitch20_distinguishes_old_body_elevation():
    a,b = pair(500,0,35,20)
    g = engagement_geometry(a,b)
    assert math.degrees(g.ha) == pytest.approx(35)
    assert math.degrees(g.boresight_elevation_error) == pytest.approx(15)
    assert env('335').weapon.qualifies(a,b,6)
    assert not env().weapon.qualifies(a,b,6)


def test_nonzero_pitch_body_azimuth_distinguishes_ata():
    # Elevation satisfies both limits: azimuth alone causes the disagreement.
    a,b = pair(500,31,20,20)
    g = engagement_geometry(a,b)
    assert math.degrees(g.ata) == pytest.approx(31)
    assert abs(math.degrees(g.boresight_azimuth_error)) < 30
    assert abs(math.degrees(g.boresight_elevation_error)) < 30
    assert env('335').weapon.qualifies(a,b,6)
    assert not env().weapon.qualifies(a,b,6)


@pytest.mark.parametrize('seed',range(10))
def test_environment_hit_entry_pitch_independent_and_rng_state_identical(seed):
    outcomes, states = [], []
    for theta in (0,20,-20):
        e = env();e.rng = np.random.default_rng(seed)
        a,b = pair(750,10,20,theta)
        outcomes.append([e.attempt_hit_for_pair(a,b) for _ in range(30)])
        states.append(e.rng.bit_generator.state)
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert states[0] == states[1] == states[2]


@pytest.mark.parametrize('epsilon',[-3,-1,-.5,0,.5,1,3])
def test_exact_hit_inputs_and_single_shared_draw(epsilon):
    class Noise:
        calls = 0
        def normal(self, mean, std):
            assert (mean,std) == (0,1)
            self.calls += 1
            return epsilon
    e = env(); e.rng = Noise()
    a,b = pair(900,10,-20,40)
    g = engagement_geometry(a,b)
    t = math.pi*math.exp(-g.distance/e.weapon.effective_hit_distance)
    result = e.attempt_hit_for_pair(a,b)
    assert result == (abs(g.ata+e.weapon.azimuth_noise_scale*epsilon)<=t
                      and abs(g.ha+e.weapon.elevation_noise_scale*epsilon)<=t)
    assert e.rng.calls == 1


def test_hit_uses_paper_angles_even_when_body_hit_would_miss():
    class ZeroNoise:
        def normal(self,*args):return 0.
    a,b = pair(1000,0,0,60)
    e = env(); e.rng = ZeroNoise()
    assert e.attempt_hit_for_pair(a,b)
    old = env('335');old.rng = ZeroNoise()
    assert not old.attempt_hit_for_pair(a,b)


@pytest.mark.parametrize('d,p',[(0,.9983470796),(250,.9556082264),(500,.8010377188),(750,.5882005801),(1000,.4)])
def test_probability_calibration_frozen(d,p):
    assert env().weapon.expected_hit_probability(d,0,0) == pytest.approx(p,abs=5e-11,rel=0)


@pytest.mark.parametrize('d,ata,ha',[(500,.2,-.3),(1000,0,0),(1000,2,-2),(750,.5,.1)])
def test_expected_probability_intersects_shared_noise_intervals(d,ata,ha):
    w = env().weapon
    # Same scalar inputs have exactly the same numerical hit function as v3.35.
    assert w.expected_hit_probability(d,ata,ha) == env('335').weapon.expected_hit_probability(d,ata,ha)


@pytest.mark.parametrize('section,key,value',[
    ('weapon','range_max',4000),('weapon','range_min',1),('weapon','ata_max',.6),('weapon','ha_max',.6),
    ('weapon','boresight_azimuth_max',math.pi/6),('weapon','effective_hit_distance',2232.442506204989),
    ('weapon','azimuth_noise_scale',1),('weapon','elevation_noise_scale',1),('weapon','ammo_per_aircraft',7),
    ('fire_control','cooldown_seconds',2),('reward','tactical_aspect_max',math.pi/2),
    ('reward','guide_reward',.002),('scenario','center_radius',4800),('sensor','range_max',9000),
    ('blue_policy','desired_speed',250),('observation','speed_scale',50)])
def test_validator_rejects_any_config_drift(section,key,value):
    c = deepcopy(env().config);c[section][key] = value
    with pytest.raises(ValueError):validate_config(c)


def test_red_nearest_eligible_and_blue_pursuit_binding_stay_inherited():
    for version in ('335','336'):
        e = env(version)
        a,_ = pair()
        e.red = [a.copy() for _ in range(4)]
        e.blue = [pair(d)[1] for d in (800,500,600,700)]
        with patch.object(e,'attempt_hit_for_pair',return_value=False):
            attempts = e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
        assert [j for _,j,_ in attempts] == [1,1,1,1]
        np.testing.assert_array_equal(e.red_ammo,[5]*4)
        assert not e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
        # Blue's current pursuit target is closer but outside the firing cone.
        e.reset(31)
        e.blue = [a.copy() for _ in range(4)]
        e.red = [pair(400,40)[1],pair(500,0)[1],pair(600,0)[1],pair(700,0)[1]]
        e.blue_step_target_indices[:] = 0
        assert e.weapon.qualifies(e.blue[0],e.red[1],6)
        assert not e._entry_attempts(e.blue,e.red,e.blue_fire_states,'blue')
        assert e.blue_ammo.tolist() == [6]*4


def test_diagnostic_counters_are_paper_geometry_without_added_rng_draws():
    e = env();a,b = pair(500,0,35,20)
    e.red = [a.copy() for _ in range(4)];e.blue = [b.copy() for _ in range(4)]
    before = deepcopy(e.rng.bit_generator.state)
    assert e._entry_attempts(e.red,e.blue,e.red_fire_states,'red') == []
    assert e.firing_geometry_diagnostics == dict(live=4,range=4,range_az=4,geometry=0,eligible=0)
    assert e.rng.bit_generator.state == before


def test_200_reset_symmetry_topology_and_no_shot_transition_reward_freeze():
    old,new = env('335'),env()
    for seed in range(200):
        x,_ = old.reset(seed);y,_ = new.reset(seed)
        assert old.red == new.red and old.blue == new.blue
        np.testing.assert_array_equal(x,y)
        assert [new.fixed_policy.nearest_target_index(b,new.red) for b in new.blue] == [0]*4
    with patch.object(old,'_entry_attempts',return_value=[]),patch.object(new,'_entry_attempts',return_value=[]):
        for _ in range(160):
            actions = np.tile([.02,.01,0],(4,1))
            x,y = old.step(actions),new.step(actions)
            assert old.red == new.red and old.blue == new.blue and x[2:4] == y[2:4]
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            for i in range(1,5):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_formal_hyperparameters_frozen(name):
    old = yaml.safe_load(Path(f'configs/{name}_4v4_v335.yaml').read_text())
    new = yaml.safe_load(Path(f'configs/{name}_4v4_v336.yaml').read_text())
    if 'environment' in new:new['environment']['environment_version'] = '3.35'
    assert old == new


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.30','3.31','3.32','3.33','3.34','3.35'])
def test_historical_checkpoint_strictly_rejected(name,previous):
    ec = env().config; ac = yaml.safe_load(Path(f'configs/{name}_4v4_v336.yaml').read_text())
    state = dict(algorithm={'mappo':'MAPPO','madsac':'madsac','maac':'MAAC'}[name],implementation_version=5 if name == 'madsac' else 3,
                 extra=dict(environment_version=previous,training_smoke=True))
    if name == 'mappo':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
        checks = (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume)
    elif name == 'madsac':
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        checks = (validate_madsac_checkpoint,)
    else:
        from algorithm.maac.protocol import validate_checkpoint
        checks = (validate_checkpoint,)
    for check in checks:
        with pytest.raises(RuntimeError,match='environment_version'):check(state,ec,ac)
