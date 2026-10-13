"""Persistent Blue locks; only target persistence differs from v3.26."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v326 import CombatEnvironmentV326
from env.combat_v327 import CombatEnvironmentV327
from env.models import AircraftState
from env.config import validate_config,environment_dimensions
from tools.audit_combat_v317_v318 import FixedNormal


def env(v='327'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{v}.yaml');e.reset(31);return e


def dead():return AircraftState(0,0,-3000,225,0,0,False)


def fixture(v='327',off=False):
    e=env(v);e.blue=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
    e.red=[dead(),AircraftState(0 if off else 500,400 if off else 0,-3000,225,0,0),dead(),AircraftState(700,0,-3000,225,0,0),dead()]
    e.rng=FixedNormal(4);return e


def step(e,external=False):
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        return e.step(np.zeros((5,3)),np.zeros((5,3)) if external else None)


def test_minimal_inheritance_config_algorithm_and_state_identity():
    assert CombatEnvironmentV327.__bases__==(CombatEnvironmentV326,)
    assert set(CombatEnvironmentV327.__dict__)-{'__module__','__doc__'}=={'environment_version','__init__','reset','step'}
    for name in ('_advance','_observations','dense_combat_reward','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV327,name) is getattr(CombatEnvironmentV326,name)
    cfg=deepcopy(env().config);validate_config(cfg);assert cfg==env().config
    assert environment_dimensions(cfg)==(56,3,5)
    cfg['environment_version']='3.26';assert cfg==env('326').config
    assert Path('configs/mappo_5v5_v327.yaml').read_bytes()==Path('configs/mappo_5v5_v326.yaml').read_bytes()
    assert Path('configs/madsac_5v5_v327.yaml').read_bytes().replace(b'3.27',b'3.26',1)==Path('configs/madsac_5v5_v326.yaml').read_bytes()
    from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
    assert MADSAC_IMPL_VERSION==3
    e=env();assert e.locked_target_indices.shape==(5,) and e.locked_target_indices.dtype==np.int64
    assert e.fixed_policy.step_targets is e.blue_step_target_indices


@pytest.mark.parametrize('external',[False,True])
def test_initial_lock_nearer_target_does_not_switch_and_diagnostics(external):
    e=fixture(off=True);first=step(e,external)
    assert first[4]['blue_pursuit_target_indices'][0]==1
    assert e.locked_target_indices[0]==1 and first[4]['blue_target_switches']==0
    e.red[3].x=100
    with patch.object(e.fixed_policy,'action_toward',wraps=e.fixed_policy.action_toward) as encode:
        second=step(e,external)
    assert second[4]['blue_pursuit_target_indices'][0]==1
    assert e.locked_target_indices[0]==1 and second[4]['blue_target_switches']==0
    if not external:assert encode.call_args.args[1]==pytest.approx(math.pi/2)
    assert second[4]['blue_step_fire_attempts']==0
    assert e.rng.calls==0 and e.blue_ammo[0]==6
    e.locked_target_indices[0]=-1
    assert first[4]['blue_locked_target_indices'][0]==1  # diagnostic copies


def test_off_target_suppression_then_locked_target_enters_window():
    e=fixture(off=True);step(e)
    assert e.locked_target_indices[0]==1
    assert e.weapon.qualifies(e.blue[0],e.red[3],6)
    assert e.blue_ammo[0]==6 and e.rng.calls==0 and e.blue_fire_cooldown_steps[0]==0
    e.red[1].x=600;e.red[1].y=0;e.red[3].x=300
    with patch.object(e,'attempt_hit_for_pair',wraps=e.attempt_hit_for_pair) as hit:result=step(e)
    assert result[4]['blue_step_fire_attempts']==1
    assert hit.call_args.args[1] is e.red[1]
    assert e.blue_ammo[0]==5 and e.blue_fire_cooldown_steps[0]==10


@pytest.mark.parametrize('cause',['ground','horizontal','ceiling'])
def test_target_dies_this_step_next_step_relocks(cause):
    e=fixture(off=True)
    def motion(states,actions):
        if states is e.red:
            if cause=='ground':e.red[1].z=1
            if cause=='horizontal':e.red[1].x=6000
            if cause=='ceiling':e.red[1].z=-7000
        return np.zeros(5)
    with patch.object(e,'_advance',side_effect=motion):first=e.step(np.zeros((5,3)))
    assert first[4]['blue_pursuit_target_indices'][0]==first[4]['blue_locked_target_indices'][0]==1
    assert first[4]['blue_step_fire_attempts']==0 and first[4]['blue_target_switches']==0
    assert e.rng.calls==0 and e.blue_ammo[0]==6
    second=step(e)
    assert second[4]['blue_pursuit_target_indices'][0]==3
    assert second[4]['blue_target_switches']==1 and second[4]['blue_step_fire_attempts']==1


def test_other_blue_kills_locked_target_no_same_step_relock():
    e=fixture();e.blue[0].psi=math.pi
    e.blue[1]=AircraftState(400,0,-3000,225,0,0);e.rng=FixedNormal(0)
    first=step(e)
    assert not e.red[1].alive
    np.testing.assert_array_equal(first[4]['blue_pursuit_target_indices'][:2],[1,1])
    assert e.blue_ammo[0]==6 and e.blue_ammo[1]==5
    assert first[4]['blue_locked_target_indices'][0]==1
    second=step(e)
    np.testing.assert_array_equal(second[4]['blue_pursuit_target_indices'][:2],[3,3])
    assert second[4]['blue_target_switches']==2


def test_repeat_cadence_six_rounds_without_window_reentry():
    e=fixture();shots=[]
    for n in range(1,62):
        if step(e)[4]['blue_step_fire_attempts']:shots.append(n)
    assert shots==[1,11,21,31,41,51] and e.blue_ammo[0]==0 and e.rng.calls==6


def test_target_death_switch_preserves_cooldown_and_ammo():
    e=fixture();step(e);e.red[1].alive=False
    for n in range(1,10):
        result=step(e)
        assert result[4]['blue_pursuit_target_indices'][0]==3
        assert e.blue_fire_cooldown_steps[0]==10-n and e.blue_ammo[0]==5
        assert result[4]['blue_step_fire_attempts']==0 and e.rng.calls==1
    assert step(e)[4]['blue_step_fire_attempts']==1 and e.blue_ammo[0]==4


def test_dead_blue_no_targets_reset_and_three_dimensional_tie_rules():
    e=fixture();e.red[1]=AircraftState(100,0,-4000,225,0,0)
    assert step(e)[4]['blue_pursuit_target_indices'][0]==3
    e.reset(31);assert np.all(e.locked_target_indices==-1) and e.fixed_policy.target_switches==0
    e=fixture();e.red[1]=e.red[3].copy()
    assert step(e)[4]['blue_pursuit_target_indices'][0]==1
    e.blue[0].alive=False
    assert step(e)[4]['blue_locked_target_indices'][0]==-1
    e=fixture();e.red[1].alive=False;e.red[3].alive=False
    assert step(e)[4]['blue_pursuit_target_indices'][0]==-1


def test_red_firing_regression_including_kill_resolution():
    old,new=env('326'),env()
    for e in (old,new):
        e.red=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
        e.blue=[AircraftState(500,0,-3000,225,0,0),AircraftState(700,0,-3000,225,0,0)]+[dead() for _ in range(3)]
        e.rng=FixedNormal(4)
    for n in range(61):
        if n==1:
            for e in (old,new):e.blue[1].x=300
        a=old._entry_attempts(old.red,old.blue,None,'red');b=new._entry_attempts(new.red,new.blue,None,'red')
        assert a==b and old.rng.calls==new.rng.calls
        np.testing.assert_array_equal(old.red_ammo,new.red_ammo)
        np.testing.assert_array_equal(old.red_fire_cooldown_steps,new.red_fire_cooldown_steps)
    for e in (old,new):e.red_ammo[0]=1;e.red_fire_cooldown_steps[0]=0;e.rng=FixedNormal(0)
    a=old._entry_attempts(old.red,old.blue,None,'red');b=new._entry_attempts(new.red,new.blue,None,'red')
    assert a==b
    assert old._resolve_combat(a,[])==new._resolve_combat(b,[])


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_no_switch_fixed_actions_transition_exact(seed):
    old,new=fixture('326'),fixture();old.reset(seed);new.reset(seed)
    # One live Red makes nearest/persistent motion identical; death still occurs normally.
    for e in (old,new):
        for i in range(1,5):e.red[i].alive=False
    for _ in range(60):
        action=np.tile([.2,0,0],(5,1));a=old.step(action);b=new.step(action)
        assert old.red==new.red and old.blue==new.blue and a[2:4]==b[2:4]
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        assert old.combat_counts==new.combat_counts
        for side in ('red','blue'):
            np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
            np.testing.assert_array_equal(getattr(old,side+'_fire_cooldown_steps'),getattr(new,side+'_fire_cooldown_steps'))
        if a[2] or a[3]:break


def test_mappo_madsac_checkpoint_and_recording_isolation():
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
    from algorithm.madsac.factory import build_madsac_trainer
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv('configs/combat_environment_v327.yaml');assert isinstance(e,CombatEnvironmentV327)
    mappo=yaml.safe_load(Path('configs/mappo_5v5_v327.yaml').read_bytes())
    for func in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            func({'algorithm':'MAPPO','extra':{'environment_version':'3.26'}},e.config,mappo)
    cfg=yaml.safe_load(Path('configs/madsac_5v5_v327.yaml').read_bytes());validate_madsac_config(e.config,cfg)
    trainer=build_madsac_trainer(cfg,'cuda',hidden_dim=32)
    with pytest.raises(RuntimeError,match='environment_version'):
        validate_madsac_checkpoint(trainer.checkpoint_state({'environment_version':'3.26'}),e.config,cfg)
