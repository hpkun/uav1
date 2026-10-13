"""Single-variable Blue pursuit/weapon target coupling contracts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v325 import CombatEnvironmentV325
from env.combat_v326 import CombatEnvironmentV326
from env.config import validate_config,environment_dimensions
from env.models import AircraftState
from tools.audit_combat_v317_v318 import FixedNormal


def env(version='326'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(31);return e


def dead():return AircraftState(0,0,-3000,225,0,0,False)


def fixture(version='326',off_target=False):
    e=env(version)
    e.blue=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
    # All Red face away, so Red cannot consume RNG or fire in these cases.
    e.red=[AircraftState(0 if off_target else 500,400 if off_target else 0,-3000,225,0,0),
           AircraftState(700,0,-3000,225,0,0)]+[dead() for _ in range(3)]
    e.rng=FixedNormal(4)
    return e


def frozen_step(e):
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        return e.step(np.zeros((5,3)))


def test_baseline_and_only_coupling_methods():
    assert CombatEnvironmentV326.__bases__==(CombatEnvironmentV325,)
    assert set(CombatEnvironmentV326.__dict__)-{'__module__','__doc__'}=={'environment_version','__init__','reset','step','_entry_attempts'}
    for name in ('_advance','_observations','dense_combat_reward','build_weapon','attempt_hit_for_pair','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV326,name) is getattr(CombatEnvironmentV325,name)
    e=env();cfg=deepcopy(e.config);validate_config(cfg);assert cfg==e.config
    cfg['environment_version']='3.25';assert cfg==env('325').config
    assert environment_dimensions(e.config)==(56,3,5)
    assert Path('configs/combat_environment_v326.yaml').read_bytes().replace(b'3.26',b'3.25',1)==Path('configs/combat_environment_v325.yaml').read_bytes()
    assert Path('configs/mappo_5v5_v326.yaml').read_bytes()==Path('configs/mappo_5v5_v325.yaml').read_bytes()
    assert e.blue_step_target_indices.dtype==np.int64 and e.blue_step_target_indices.shape==(5,)


def test_movement_and_weapon_share_pre_step_target():
    e=fixture()
    with patch.object(e.fixed_policy,'action_toward',wraps=e.fixed_policy.action_toward) as encode,patch.object(e,'attempt_hit_for_pair',wraps=e.attempt_hit_for_pair) as hit:
        _,_,_,_,info=frozen_step(e)
    np.testing.assert_array_equal(info['blue_pursuit_target_indices'],[0,-1,-1,-1,-1])
    assert encode.call_count==1 and encode.call_args.args[1:]==(0.,0.,250.)
    assert hit.call_count==1 and hit.call_args.args==(e.blue[0],e.red[0])
    assert e.blue_ammo[0]==5 and e.rng.calls==1
    # info owns a snapshot, not a view that changes in a later step.
    e.blue_step_target_indices[0]=1
    assert info['blue_pursuit_target_indices'][0]==0


def test_off_target_suppression_against_v325():
    old,new=fixture('325',True),fixture(off_target=True)
    assert new.fixed_policy.nearest_target_index(new.blue[0],new.red)==0
    assert not new.weapon.qualifies(new.blue[0],new.red[0],6)
    assert new.weapon.qualifies(new.blue[0],new.red[1],6)
    with patch.object(old,'attempt_hit_for_pair',wraps=old.attempt_hit_for_pair) as old_hit:
        old_result=frozen_step(old)
    new_result=frozen_step(new)
    assert old_result[4]['blue_step_fire_attempts']==1
    assert old_hit.call_args.args[1] is old.red[1]
    assert new_result[4]['blue_step_fire_attempts']==0
    assert new.blue_ammo[0]==6 and new.blue_fire_cooldown_steps[0]==0 and new.rng.calls==0
    assert new_result[4]['blue_fire_window_pairs']>0  # eligibility metric remains unchanged


def test_dynamics_cannot_reselect_weapon_target_until_next_step():
    e=fixture(off_target=True)
    def move(states,actions):
        if states is e.red:e.red[0].y=900;e.red[1].x=300
        return np.zeros(5)
    with patch.object(e,'_advance',side_effect=move):
        first=e.step(np.zeros((5,3)))
    assert first[4]['blue_pursuit_target_indices'][0]==0
    assert e.fixed_policy.nearest_target_index(e.blue[0],e.red)==1
    assert first[4]['blue_step_fire_attempts']==0 and e.rng.calls==0
    with patch.object(e,'attempt_hit_for_pair',wraps=e.attempt_hit_for_pair) as hit:
        second=frozen_step(e)
    assert second[4]['blue_pursuit_target_indices'][0]==1
    assert second[4]['blue_step_fire_attempts']==1 and hit.call_args.args[1] is e.red[1]


@pytest.mark.parametrize('cause',['ground','horizontal','ceiling'])
def test_target_dies_before_fire_no_fallback_then_next_step_switch(cause):
    e=fixture(off_target=True)
    def move(states,actions):
        if states is e.red:
            if cause=='ground':e.red[0].z=1
            if cause=='horizontal':e.red[0].x=6000
            if cause=='ceiling':e.red[0].z=-7000
        return np.zeros(5)
    with patch.object(e,'_advance',side_effect=move):first=e.step(np.zeros((5,3)))
    assert not e.red[0].alive and first[4]['blue_pursuit_target_indices'][0]==0
    assert first[4]['blue_step_fire_attempts']==0
    assert e.blue_ammo[0]==6 and e.blue_fire_cooldown_steps[0]==0 and e.rng.calls==0
    second=frozen_step(e)
    assert second[4]['blue_pursuit_target_indices'][0]==1 and second[4]['blue_step_fire_attempts']==1


@pytest.mark.parametrize('hit',[False,True])
def test_target_switch_does_not_reset_cooldown_or_ammo(hit):
    e=fixture();e.rng=FixedNormal(0 if hit else 4)
    first=frozen_step(e);assert first[4]['blue_step_fire_attempts']==1
    assert e.blue_fire_cooldown_steps[0]==10 and e.blue_ammo[0]==5
    e.red[0].alive=False # hit kills it too; next target stays alive during waiting.
    for n in range(1,10):
        result=frozen_step(e)
        assert result[4]['blue_pursuit_target_indices'][0]==1
        assert result[4]['blue_step_fire_attempts']==0
        assert e.blue_fire_cooldown_steps[0]==10-n and e.blue_ammo[0]==5 and e.rng.calls==1
    result=frozen_step(e)
    assert result[4]['blue_step_fire_attempts']==1 and e.blue_ammo[0]==4 and e.rng.calls==2


def test_six_rounds_one_second_cadence():
    e=fixture();steps=[]
    for step in range(1,62):
        result=frozen_step(e)
        if result[4]['blue_step_fire_attempts']:steps.append(step)
    assert steps==[1,11,21,31,41,51] and e.blue_ammo[0]==0 and e.rng.calls==6


def test_selector_3d_tie_dead_blue_no_target_and_reset():
    e=fixture()
    # Nearest horizontal target is farther in 3D; reuse authoritative selector.
    e.red[0]=AircraftState(100,0,-4000,225,0,0)
    assert frozen_step(e)[4]['blue_pursuit_target_indices'][0]==1
    e.red[0]=e.red[1].copy();e.blue_fire_cooldown_steps[0]=0
    assert frozen_step(e)[4]['blue_pursuit_target_indices'][0]==0
    e.blue[0].alive=False
    assert frozen_step(e)[4]['blue_pursuit_target_indices'][0]==-1
    e.blue[0].alive=True
    for s in e.red:s.alive=False
    assert frozen_step(e)[4]['blue_pursuit_target_indices'][0]==-1
    e.reset(31);np.testing.assert_array_equal(e.blue_step_target_indices,np.full(5,-1))


def test_red_firing_exact_regression_for_multiple_targets_and_steps():
    old,new=env('325'),env()
    for e in (old,new):
        e.red=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
        e.blue=[AircraftState(500,0,-3000,225,0,0),AircraftState(700,0,-3000,225,0,0)]+[dead() for _ in range(3)]
        e.rng=FixedNormal(4)
    for step in range(61):
        if step==1:
            for e in (old,new):e.blue[1].x=400
        ao=old._entry_attempts(old.red,old.blue,None,'red');an=new._entry_attempts(new.red,new.blue,None,'red')
        assert ao==an
        if an:assert an[0][1]==(0 if step==0 else 1)
        np.testing.assert_array_equal(old.red_ammo,new.red_ammo)
        np.testing.assert_array_equal(old.red_fire_cooldown_steps,new.red_fire_cooldown_steps)
        assert old.rng.calls==new.rng.calls


def compare_transition(old,new,a,b):
    oo,ro,to,xo,io=a;no,rn,tn,xn,inn=b
    np.testing.assert_array_equal(oo,no);np.testing.assert_array_equal(ro,rn)
    assert (to,xo)==(tn,xn) and old.red==new.red and old.blue==new.blue
    assert old.combat_counts==new.combat_counts
    for key in ('r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards','outcome_rewards','red_step_fire_attempts','blue_step_fire_attempts','red_step_weapon_hits','blue_step_weapon_hits','termination_reason'):
        np.testing.assert_array_equal(io[key],inn[key])
    for side in ('red','blue'):
        for suffix in ('_ammo','_fire_cooldown_steps'):
            np.testing.assert_array_equal(getattr(old,side+suffix),getattr(new,side+suffix))


@pytest.mark.parametrize('hit',[False,True])
def test_no_off_target_window_full_transition_identity(hit):
    old,new=fixture('325'),fixture()
    for e in (old,new):e.red[1].alive=False;e.rng=FixedNormal(0 if hit else 4)
    for _ in range(61):
        a,b=frozen_step(old),frozen_step(new);compare_transition(old,new,a,b)
        if a[2] or a[3]:break
    assert old.rng.calls==new.rng.calls


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_fixed_action_natural_transition_without_fire_windows(seed):
    old,new=env('325'),env();old.reset(seed);new.reset(seed)
    for _ in range(50):
        actions=np.tile([.2,0,0],(5,1))
        a,b=old.step(actions),new.step(actions)
        compare_transition(old,new,a,b)
        assert a[4]['blue_fire_window_pairs']==0


def test_recording_and_checkpoint_isolation_and_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e=RecordingCombatEnv('configs/combat_environment_v326.yaml');assert isinstance(e,CombatEnvironmentV326)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v326.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.25'}},e.config,cfg)
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=e.config
    runner.trainer=SimpleNamespace(sampled_steps=0);runner.evaluation_seeds=[10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):assert runner._evaluation_record()['environment_version']=='3.26'
