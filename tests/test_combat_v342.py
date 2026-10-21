"""3-D research geometry, shared noise, role symmetry and frozen baseline."""
from copy import deepcopy
import math
import numpy as np
import pytest
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.sensor import is_detected
from env.v336_weapon import PaperATAHAWeapon
from env.v342_weapon import ConeWeapon


def config(v='342'):
    return load_config(f'configs/combat_environment_v{v}.yaml')


def pair(heading=0., elevation=0., psi=0., theta=0., distance=1500.):
    a = AircraftState(0., 0., -3000., 225., np.deg2rad(theta), np.deg2rad(psi))
    h, e = np.deg2rad([heading, elevation])
    t = AircraftState(distance*np.cos(e)*np.cos(h), distance*np.cos(e)*np.sin(h),
                      a.z-distance*np.sin(e), 225., 0., h)
    return a, t


@pytest.mark.parametrize('heading,elevation,psi,theta,expected', [
    (0,-20,0,50,70), (0,-20,0,-20,0), (0,-40,0,0,40),
    (45,20,0,20,None), (179,0,-179,0,2), (-179,0,179,0,2),
    (120,30,120,30,0), (-120,-30,-120,-30,0),
    (0,30-1e-6,0,0,30-1e-6), (0,30,0,0,30), (0,30+1e-6,0,0,30+1e-6)])
def test_geometry(heading,elevation,psi,theta,expected):
    a,t = pair(heading,elevation,psi,theta)
    g = engagement_geometry(a,t)
    los = (t.as_array()[:3]-a.as_array()[:3])/g.distance
    direct = math.acos(np.clip(np.dot(a.velocity_vector()/a.v,los),-1,1))
    assert g.off_boresight == pytest.approx(direct,abs=3e-8)
    if expected is not None:
        assert np.rad2deg(g.off_boresight) == pytest.approx(expected,abs=2e-6)
    w = ConeWeapon(**config()['weapon'])
    assert w.qualifies(a,t,6) == (g.off_boresight <= np.pi/6+1e-12)


def test_video_reproduction_and_replacement_not_intersection():
    old = PaperATAHAWeapon(**config('341')['weapon'])
    new = ConeWeapon(**config()['weapon'])
    a,t = pair(elevation=-20,theta=50)
    g = engagement_geometry(a,t)
    assert np.rad2deg([g.ata,g.ha,g.off_boresight]) == pytest.approx([0,-20,70])
    assert old.qualifies(a,t,6) and not new.qualifies(a,t,6)
    a.theta = np.deg2rad(-20)
    assert new.qualifies(a,t,6)
    a,t = pair(elevation=50,theta=50)
    assert new.qualifies(a,t,6) and not old.qualifies(a,t,6)


@pytest.mark.parametrize('distance',[0.,1e-10,100.,1500.,2000.,2000.001])
@pytest.mark.parametrize('angle',[-180.,-60.,-30.,0.,30.,60.,180.])
def test_finite_sensor_containment_and_symmetric_roles(distance,angle):
    a,t = pair(heading=angle,distance=distance)
    w = ConeWeapon(**config()['weapon'])
    g = engagement_geometry(a,t)
    assert np.isfinite(list(vars(g).values())).all()
    # Changing the labels of identical states must never change qualification.
    assert w.qualifies(a,t,6) == bool(w.eligibility([a],[t],[6])[0][0,0])
    if w.qualifies(a,t,6):
        assert is_detected(a,t,config()['sensor'])
    assert not w.qualifies(a,t,0)
    t.alive = False
    assert not w.qualifies(a,t,6)


@pytest.mark.parametrize('az,el',[(0,0),(15,10),(-15,10),(25,-15),(-25,-15)])
def test_probability_draw_count_shared_noise_and_seed(az,el):
    old = PaperATAHAWeapon(**config('341')['weapon'])
    new = ConeWeapon(**config()['weapon'])
    x,y = np.deg2rad([az,el])
    assert old.expected_hit_probability(1500,x,y) == new.expected_hit_probability(1500,x,y)
    class Rng:
        calls=0
        def normal(self,loc,scale):
            self.calls+=1
            assert (loc,scale)==(0,1)
            return .31
    rng = Rng()
    result = new.attempt_hit(rng,1500,x,y)
    threshold = np.pi*np.exp(-1500/new.effective_hit_distance)
    assert result == (abs(x+new.azimuth_noise_scale*.31)<=threshold and abs(y+new.elevation_noise_scale*.31)<=threshold)
    assert rng.calls == 1
    r1,r2 = np.random.default_rng(19),np.random.default_rng(19)
    assert [old.attempt_hit(r1,1500,x,y) for _ in range(30)] == [new.attempt_hit(r2,1500,x,y) for _ in range(30)]
    assert r1.bit_generator.state == r2.bit_generator.state


@pytest.mark.parametrize('side',['red','blue'])
def test_entry_roles_cooldown_rng_and_probability_inputs(side):
    e = MultiUAVCombatEnv(config()); e.reset(7)
    a,t = pair(elevation=-20,theta=50)
    attackers,targets = getattr(e,side),getattr(e,'blue' if side=='red' else 'red')
    attackers[:] = [a.copy() for _ in range(4)]
    targets[:] = [t.copy() for _ in range(4)]
    e.blue_step_target_indices.fill(0)
    before = deepcopy(e.rng.bit_generator.state)
    assert e._entry_attempts(attackers,targets,None,side) == []
    assert e.rng.bit_generator.state == before
    for s in attackers:s.theta=np.deg2rad(-20)
    captured=[]
    # Capture the actual shared hook without mutating the frozen dataclass.
    e.attempt_hit_for_pair=lambda a,t: captured.append(engagement_geometry(a,t)) or False
    assert len(e._entry_attempts(attackers,targets,None,side)) == 4
    assert all(g.off_boresight < 3e-8 for g in captured)
    assert np.all(getattr(e,side+'_ammo')==5)
    assert e._entry_attempts(attackers,targets,None,side) == []


def test_isolated_schema_and_unchanged_reward_observation():
    c = config(); snapshot=deepcopy(c)
    assert validate_config(c) == snapshot
    assert environment_dimensions(c)==(45,3,4)
    a,b = MultiUAVCombatEnv(config('341')),MultiUAVCombatEnv(c)
    oa,_=a.reset(31); ob,_=b.reset(31)
    np.testing.assert_array_equal(oa,ob)
    for key in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(a,key)],[s.as_array() for s in getattr(b,key)])
    np.testing.assert_array_equal(a.dense_combat_reward()[0],b.dense_combat_reward()[0])
    c['weapon']['ata_max']=np.pi/6
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('psi',[-179.,-90.,0.,90.,179.])
@pytest.mark.parametrize('theta',[-50.,0.,50.])
def test_body_angles_actual_probability_hook_and_sensor(psi,theta,monkeypatch):
    a,_=pair(psi=psi,theta=theta)
    forward=a.velocity_vector()/a.v
    right=np.array([-np.sin(a.psi),np.cos(a.psi),0.])
    up=np.cross(right,forward)
    az,el=np.deg2rad([15.,-10.])
    los=np.cos(el)*np.cos(az)*forward+np.cos(el)*np.sin(az)*right+np.sin(el)*up
    t=a.copy();t.x,t.y,t.z=a.as_array()[:3]+1500.*los
    g=engagement_geometry(a,t)
    np.testing.assert_allclose([g.boresight_azimuth_error,g.boresight_elevation_error],[az,el],atol=1e-15)
    captured=[]
    def attempt(w,rng,d,x,y):
        captured.append((d,x,y));return False
    monkeypatch.setattr(ConeWeapon,'attempt_hit',attempt)
    e=MultiUAVCombatEnv(config());e.reset(2)
    e.attempt_hit_for_pair(a,t)
    np.testing.assert_allclose(captured,[[1500.,az,el]],atol=1e-12)
    assert e.weapon.qualifies(a,t,6) and is_detected(a,t,e.config['sensor'])


def test_seeded_rotated_sensor_containment():
    rng=np.random.default_rng(342)
    w=ConeWeapon(**config()['weapon'])
    contained=0
    for _ in range(1000):
        a,t=pair(heading=rng.uniform(-180,180),elevation=rng.uniform(-90,90),
                 psi=rng.uniform(-180,180),theta=rng.uniform(-60,60),distance=rng.uniform(0,10000))
        g=engagement_geometry(a,t)
        probability=w.expected_hit_probability(g.distance,g.boresight_azimuth_error,g.boresight_elevation_error)
        assert np.isfinite(probability) and 0<=probability<=1
        if w.qualifies(a,t,6):
            assert is_detected(a,t,config()['sensor']);contained+=1
    assert contained>0


def test_real_event_draws_and_no_extra_randomness_on_cooldown():
    e=MultiUAVCombatEnv(config());e.reset(3)
    a,t=pair();t.psi=np.pi
    e.red=[a.copy() for _ in range(4)];e.blue=[t.copy() for _ in range(4)]
    e._advance=lambda states,actions:np.zeros(4)
    class Noise:
        calls=0
        def normal(self,*args):self.calls+=1;return 1e6 # deterministic misses
    e.rng=Noise()
    for step in range(1,13):
        e.step(np.zeros((4,3)))
        assert e.rng.calls==(8 if step<11 else 16)
    assert np.all(e.red_ammo==4) and np.all(e.blue_ammo==4)


@pytest.mark.parametrize('version',['342','343'])
def test_seed_reproducible_event_trajectories(version):
    x,y=MultiUAVCombatEnv(config(version)),MultiUAVCombatEnv(config(version))
    for e in (x,y):
        e.reset(7)
        a,t=pair();t.psi=np.pi
        e.red=[a.copy() for _ in range(4)];e.blue=[t.copy() for _ in range(4)]
    for _ in range(20):
        a,b=x.step(np.zeros((4,3))),y.step(np.zeros((4,3)))
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        assert a[2:4]==b[2:4]
        assert x.rng.bit_generator.state==y.rng.bit_generator.state
        np.testing.assert_array_equal(x.red_ammo,y.red_ammo)
        if a[2] or a[3]:break
