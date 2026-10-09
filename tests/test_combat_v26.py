"""Isolated v2.6 geometry, stateful Blue, and version compatibility checks."""
import copy
from pathlib import Path
import numpy as np
import pytest
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import SternConversionPolicy, SternPhase, NearestTargetPursuitPolicy
from env.models import AircraftState
from env.weapon import PairFireState, RearAspectWeaponEnvelope
from env.math_utils import wrap_angle
from tools.validate_stern_conversion import run_trial

ROOT = Path(__file__).resolve().parents[1]


def config(version='26'):
    return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')


def policy(n=2):
    cfg = config()
    return SternConversionPolicy(cfg['blue_policy'], cfg['action'], n)


def aircraft(x=0, y=0, psi=0, v=225, z=-3000, alive=True):
    return AircraftState(x, y, z, v, 0, psi, alive)


def test_protocol_and_only_two_config_differences():
    cfg, old = config(), config('24')
    assert environment_dimensions(cfg) == (104,3,8)
    for key in old:
        if key not in ('environment_version','blue_policy'):
            assert cfg[key] == old[key]
    assert cfg['blue_policy'] == dict(desired_speed=250., turn_angle=np.pi/12,
        turn_range=6500., required_lateral_displacement=600., conversion_range=2500.)
    env = MultiUAVCombatEnv(cfg)
    assert isinstance(env.weapon, RearAspectWeaponEnvelope)
    assert isinstance(env.red_fire_states, PairFireState)
    assert env.red_fire_states.armed.shape == env.blue_fire_states.armed.shape == (8,8)
    assert type(env.fixed_policy) is SternConversionPolicy
    a,b=env.reset(42),MultiUAVCombatEnv(old).reset(42)
    np.testing.assert_array_equal(a[0],b[0])
    assert MultiUAVCombatEnv().environment_version == '2.3'


@pytest.mark.parametrize('field,value',[
    ('desired_speed',149),('desired_speed',301),('turn_angle',0),('turn_angle',np.pi/2),
    ('turn_range',0),('required_lateral_displacement',0),('conversion_range',600),
    ('conversion_range',4001),('turn_range',float('nan')),('turn_angle',float('inf'))])
def test_strict_blue_validation(field,value):
    cfg=config();cfg['blue_policy'][field]=value
    with pytest.raises(ValueError): validate_config(cfg)


def test_blue_schema_missing_unknown_and_team_size():
    for edit in ('missing','extra','team'):
        cfg=config()
        if edit=='missing': del cfg['blue_policy']['turn_angle']
        elif edit=='extra': cfg['blue_policy']['random_turn']=True
        else: cfg['scenario']['team_size']=5
        with pytest.raises(ValueError): validate_config(cfg)


def test_nearest_3d_lock_and_reacquisition_clears_conversion():
    p=policy();own=aircraft()
    targets=[aircraft(100,z=-3500),aircraft(200),aircraft(1,alive=False)]
    p.action(own,targets)
    assert p.states[0].locked_target_index==1
    targets[0]=aircraft(1)
    p.action(own,targets)
    assert p.states[0].locked_target_index==1
    assert p.states[0].phase is SternPhase.RELATIVE_BEARING
    targets[1].alive=False
    p.action(own,targets)
    assert p.states[0].locked_target_index==0
    assert p.states[0].phase is SternPhase.PURE_PURSUIT
    assert p.states[0].relative_bearing_heading is p.states[0].turn_side is None
    targets[0].alive=False
    np.testing.assert_array_equal(p.action(own,targets),np.zeros(3))


@pytest.mark.parametrize('psi',[0,np.pi/2,-1.2])
def test_target_coordinate_signs(psi):
    t=aircraft(31,44,psi)
    for longitudinal,lateral in [(300,700),(-300,-700)]:
        b=aircraft(t.x+longitudinal*np.cos(psi)-lateral*np.sin(psi),
                   t.y+longitudinal*np.sin(psi)+lateral*np.cos(psi))
        np.testing.assert_allclose(policy().target_coordinates(b,t),(longitudinal,lateral))


def test_four_phase_conditions_and_heading_semantics():
    p=policy();t=aircraft();b=aircraft(8000,psi=np.pi)
    def expected(h):
        return p.action_toward(b,h,0,250)
    np.testing.assert_array_equal(p.action(b,[t]),expected(np.pi))
    b.x=6500
    np.testing.assert_allclose(p.action(b,[t]),expected(wrap_angle(np.pi+np.pi/12)))
    s=p.states[0];assert s.phase is SternPhase.RELATIVE_BEARING
    stored=s.relative_bearing_heading
    b.y=200;t.y=-100;b.psi=1
    p.action(b,[t]);assert s.relative_bearing_heading==stored
    b.y=500 # lateral exactly 600
    np.testing.assert_allclose(p.action(b,[t]),expected(wrap_angle(t.psi+np.pi)))
    assert s.phase is SternPhase.OFFSET_RECIPROCAL
    b.x=1000
    p.action(b,[t]);assert s.phase is SternPhase.OFFSET_RECIPROCAL # in front
    b.x=-3000
    p.action(b,[t]);assert s.phase is SternPhase.OFFSET_RECIPROCAL # too far
    b.x=-1000
    np.testing.assert_allclose(p.action(b,[t]),expected(np.arctan2(t.y-b.y,t.x-b.x)))
    assert s.phase is SternPhase.CONVERT
    b.x=9000;p.action(b,[t]);assert s.phase is SternPhase.CONVERT


@pytest.mark.parametrize('index,lateral,side',[(0,0,1),(1,0,-1),(0,10,1),(1,-10,-1)])
def test_deterministic_turn_side(index,lateral,side):
    p=policy();b=aircraft(6000,lateral,np.pi);t=aircraft()
    p.action(b,[t],index);p.action(b,[t],index)
    assert p.states[index].turn_side==side


@pytest.mark.parametrize('phase',list(SternPhase))
def test_all_phases_keep_speed_and_altitude_controller(phase):
    p=policy();b=aircraft(7000,psi=.2,v=230,z=-3050);t=aircraft(z=-3000)
    p.action(b,[t]);s=p.states[0];s.phase=phase;s.relative_bearing_heading=.4
    heading={SternPhase.PURE_PURSUIT:np.pi,SternPhase.RELATIVE_BEARING:.4,
             SternPhase.OFFSET_RECIPROCAL:wrap_angle(np.pi),SternPhase.CONVERT:np.pi}[phase]
    np.testing.assert_allclose(p.action(b,[t]),p.action_toward(b,heading,np.arctan2(-50,7000),250))


def test_reset_and_external_override():
    env=MultiUAVCombatEnv(config());env.reset(12)
    env.fixed_policy.team_actions(env.blue,env.red)
    for s in env.fixed_policy.states:
        s.phase=SternPhase.CONVERT;s.relative_bearing_heading=.2;s.turn_side=-1
    explicit=np.zeros((8,3),dtype=np.float32)
    before=[s.copy() for s in env.blue]
    # An override must never even call the internal policy.
    env.fixed_policy.team_actions=lambda *args: (_ for _ in ()).throw(AssertionError('policy called'))
    expected=MultiUAVCombatEnv(config());expected.reset(12)
    expected._advance(before,explicit)
    env.step(explicit,blue_actions=explicit)
    for a,b in zip(env.blue,before): np.testing.assert_array_equal(a.as_array(),b.as_array())
    env.reset(12)
    for s in env.fixed_policy.states:
        assert s.locked_target_index is None and s.phase is SternPhase.PURE_PURSUIT
        assert s.relative_bearing_heading is s.turn_side is None


def test_independent_instances_deterministic_and_dead_blue_does_not_act():
    p,q=policy(),policy();targets=[aircraft()]
    for x,y in [(8000,0),(6400,0),(5000,300),(2000,601),(-2000,601)]:
        team=[aircraft(x,y,np.pi),aircraft(x,-y,np.pi)]
        np.testing.assert_array_equal(p.team_actions(team,targets),q.team_actions(team,targets))
        assert p.states==q.states
    assert p.states[0].phase is SternPhase.CONVERT
    assert p.states[1].phase is SternPhase.CONVERT
    np.testing.assert_array_equal(p.action(aircraft(alive=False),targets),np.zeros(3))


@pytest.mark.parametrize('version',['23','24','25'])
def test_old_policy_unchanged(version):
    cfg=load_config(ROOT/('configs/combat_environment.yaml' if version=='23' else f'configs/combat_environment_v{version}.yaml'))
    env=MultiUAVCombatEnv(cfg);env.reset(92)
    assert type(env.fixed_policy) is NearestTargetPursuitPolicy
    reference=NearestTargetPursuitPolicy(cfg['blue_policy'],cfg['action'])
    np.testing.assert_array_equal(env.fixed_policy.team_actions(env.blue,env.red),reference.team_actions(env.blue,env.red))


def test_geometric_trial_repeatable_and_straight_target():
    a=run_trial(config(),synthetic=True,trace=True)
    assert a==run_trial(config(),synthetic=True,trace=True)
    for row in a['trace']:
        assert row['target_heading']==0
        assert row['target_position']==[-4000+22.5*row['step'],0.,-3000.]
    assert [r['phase'] for r in a['transitions']]==['PURE_PURSUIT','RELATIVE_BEARING','OFFSET_RECIPROCAL','CONVERT']
    assert a['fire_window_success'] and a['phase_completion'] and not a['blue_arena_exit']
    reciprocal=a['transitions'][2]
    convert=a['transitions'][3]
    assert reciprocal['longitudinal']>0 and convert['longitudinal']<0
    assert abs(wrap_angle(convert['blue_heading']-np.pi))<np.pi/12
    assert a['trace'][a['first_fire_window_step']]['longitudinal']<0


def test_madsac_still_rejects_v26():
    from algorithm.madsac.protocol import validate_madsac_checkpoint
    with pytest.raises(RuntimeError,match='environment_version'):
        validate_madsac_checkpoint({},config(),{'algorithm':'madsac'})


@pytest.mark.parametrize('name',['mappo','rmappo','ea-mappo','stea-mappo'])
def test_formal_configs_accept_v26_dimensions_and_remain_3m(name):
    import yaml
    from tools.smoke_formal_8v8 import ALGORITHMS
    from algorithm.common.control_factory import trainer_kwargs
    from algorithm.mappo.trainer import MAPPOTrainer
    from algorithm.rmappo.factory import build_rmappo_trainer
    from algorithm.ea_mappo.factory import build_ea_mappo_trainer
    from algorithm.stea_mappo.factory import build_stea_mappo_trainer
    stem=ALGORITHMS[name][0]
    cfg=yaml.safe_load((ROOT/f'configs/{stem}.yaml').read_text())
    assert tuple(cfg['network'][k] for k in ('observation_dim','action_dim','num_agents'))==environment_dimensions(config())
    assert cfg['training']['total_sampled_steps']==3000000
    factories={'rmappo':build_rmappo_trainer,'ea-mappo':build_ea_mappo_trainer,'stea-mappo':build_stea_mappo_trainer}
    trainer=(MAPPOTrainer(**trainer_kwargs(cfg,'cpu',31)) if name=='mappo' else factories[name](cfg,'cpu',seed=31))
    assert trainer.critic.value_network[0].in_features==936


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
def test_formal_protocol_allows_v26_but_still_rejects_v23(name):
    import importlib
    import yaml
    validate=importlib.import_module(f'algorithm.{name}.protocol').validate_checkpoint
    cfg=yaml.safe_load((ROOT/f'configs/{name}_8v8.yaml').read_text())
    # Empty checkpoint reaches the identity check for supported v2.6.
    with pytest.raises(RuntimeError,match='checkpoint is not'):
        validate({},config(),cfg)
    old=load_config(ROOT/'configs/combat_environment.yaml')
    with pytest.raises(RuntimeError,match='formal control requires'):
        validate({},old,cfg)


def test_spawn_worker_policy_reset_matches_fresh_local_episode():
    import multiprocessing as mp
    from env.process_worker import combat_environment_worker
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    process=ctx.Process(target=combat_environment_worker,args=(child,config()),daemon=True)
    process.start();child.close()
    try:
        status,metadata=parent.recv()
        assert status=='ready' and metadata['fixed_policy_class']=='SternConversionPolicy'
        trajectories=[]
        for episode in range(2):
            parent.send(('reset',71));status,payload=parent.recv();assert status=='ok'
            local=MultiUAVCombatEnv(config());obs,_=local.reset(71)
            np.testing.assert_array_equal(payload[0],obs)
            rows=[]
            actions=np.zeros((8,3),dtype=np.float32)
            for _ in range(80):
                parent.send(('step',actions));status,payload=parent.recv();assert status=='ok'
                expected=local.step(actions)
                np.testing.assert_array_equal(payload[0],expected[0])
                np.testing.assert_array_equal(payload[1],expected[1])
                rows.append(payload[0])
            assert any(s.phase is not SternPhase.PURE_PURSUIT for s in local.fixed_policy.states)
            trajectories.append(np.stack(rows))
        np.testing.assert_array_equal(*trajectories)
    finally:
        if process.is_alive():
            parent.send(('close',None));parent.recv()
        parent.close();process.join(timeout=5)
        if process.is_alive(): process.terminate();process.join()
