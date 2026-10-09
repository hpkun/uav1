"""Strict single-variable v2.8 contract and conversion threshold regression."""
import copy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.config import load_config, environment_dimensions, validate_config
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import SternConversionPolicy, SternPhase
from env.models import AircraftState
from env.weapon import PairFireState, RearAspectWeaponEnvelope

ROOT=Path(__file__).resolve().parents[1]
STEMS={'mappo':'mappo_5v5','rmappo':'rmappo_5v5','ea_mappo':'ea_mappo_5v5','stea_mappo':'stea_mappo_5v5'}


def cfg(version='28'):
    return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')


def differences(a,b,prefix=''):
    result=set()
    for key in a.keys() | b.keys():
        path=prefix+key
        if key not in a or key not in b: result.add(path)
        elif isinstance(a[key],dict) and isinstance(b[key],dict):
            result.update(differences(a[key],b[key],path+'.'))
        elif a[key]!=b[key]: result.add(path)
    return result


def test_exact_two_leaf_differences_and_frozen_protocol():
    old,new=cfg('27'),cfg()
    assert differences(old,new)=={'environment_version','blue_policy.conversion_range'}
    assert old['blue_policy']['conversion_range']==2500 and new['blue_policy']['conversion_range']==2000
    assert new['scenario']['team_size']==5
    assert new['scenario']['formation_offsets']==[-600.,-300.,0.,300.,600.]
    assert environment_dimensions(new)==(65,3,5)
    a,b=MultiUAVCombatEnv(old),MultiUAVCombatEnv(new)
    assert type(a.fixed_policy) is type(b.fixed_policy) is SternConversionPolicy
    assert type(a.weapon) is type(b.weapon) is RearAspectWeaponEnvelope
    assert a.weapon==b.weapon
    assert new['reward']==old['reward']
    assert type(b.red_fire_states) is type(b.blue_fire_states) is PairFireState
    assert b.red_fire_states.armed.shape==b.blue_fire_states.armed.shape==(5,5)
    np.testing.assert_array_equal(a.reset(28)[0],b.reset(28)[0])
    assert MultiUAVCombatEnv().environment_version=='2.3'


def test_conversion_threshold_is_actual_3d_distance_not_legacy_constant():
    policies=[MultiUAVCombatEnv(cfg(v)).fixed_policy for v in ('27','28')]
    target=AircraftState(0,0,-3000,225,0,0)
    own=AircraftState(-2200,0,-3000,225,0,np.pi)
    for p in policies:
        p.action(own,[target]);p.states[0].phase=SternPhase.OFFSET_RECIPROCAL
        p.action(own,[target])
    assert policies[0].states[0].phase is SternPhase.CONVERT
    assert policies[1].states[0].phase is SternPhase.OFFSET_RECIPROCAL
    own.x=-1900;own.z=-3700 # horizontal <2000, but 3-D distance >2000
    policies[1].action(own,[target])
    assert policies[1].states[0].phase is SternPhase.OFFSET_RECIPROCAL
    own.x=-2000;own.z=-3000
    policies[1].action(own,[target])
    assert policies[1].states[0].phase is SternPhase.CONVERT


def test_reset_outward_turn_and_five_state_clear():
    env=MultiUAVCombatEnv(cfg());env.reset(5)
    own=AircraftState(6000,100,-3000,225,0,np.pi)
    target=AircraftState(0,0,-3000,225,0,0)
    for i in range(5):
        env.fixed_policy.action(own,[target],i);env.fixed_policy.action(own,[target],i)
        s=env.fixed_policy.states[i]
        assert np.sin(s.relative_bearing_heading)>0
        assert s.phase is SternPhase.RELATIVE_BEARING
    env.reset(5)
    for s in env.fixed_policy.states:
        assert s.locked_target_index is None and s.phase is SternPhase.PURE_PURSUIT
        assert s.relative_bearing_heading is s.turn_side is None


@pytest.mark.parametrize('ending',['normal','timeout','red_dead','blue_dead'])
def test_external_actions_reward_and_termination_match_v27(ending):
    envs=[MultiUAVCombatEnv(cfg(v)) for v in ('27','28')]
    for e in envs:
        e.reset(73)
        if ending=='timeout': e.steps=e.max_steps-1
        elif ending.endswith('_dead'):
            for state in getattr(e,ending.split('_')[0]): state.alive=False
        # Prove that explicit overrides never invoke the Stern policy.
        e.fixed_policy.team_actions=lambda *args: (_ for _ in ()).throw(AssertionError('policy called'))
    actions=np.zeros((5,3),dtype=np.float32)
    a,b=[e.step(actions,blue_actions=actions) for e in envs]
    for x,y in zip(a[:2],b[:2]): np.testing.assert_array_equal(x,y)
    assert a[2:4]==b[2:4]
    for key in ('terminated','truncated','winner','termination_reason'):
        assert a[4].get(key)==b[4].get(key)


@pytest.mark.parametrize('name',list(STEMS))
def test_formal_five_v_five_trainer_and_eight_v_eight_dimension_guard(name,tmp_path,monkeypatch):
    from algorithm.common.control_factory import trainer_kwargs
    from algorithm.mappo.trainer import MAPPOTrainer
    c=yaml.safe_load((ROOT/f'configs/{STEMS[name]}.yaml').read_text())
    trainer=(MAPPOTrainer(**trainer_kwargs(c,'cpu',31)) if name=='mappo' else
             getattr(importlib.import_module(f'algorithm.{name}.factory'),f'build_{name}_trainer')(c,'cpu',seed=31))
    assert trainer.critic.value_network[0].in_features==390
    with torch.no_grad():
        assert torch.isfinite(trainer.critic(torch.zeros(2,5,65),torch.ones(2,5))).all()
    stem={'mappo':'mappo_8v8_formal','rmappo':'rmappo_8v8',
          'ea_mappo':'ea_mappo_8v8','stea_mappo':'stea_mappo_8v8_formal'}[name]
    wrong=yaml.safe_load((ROOT/f'configs/{stem}.yaml').read_text())
    module=importlib.import_module(f'algorithm.{name}.runner')
    # Only the early dimension guard is exercised here; no update/worker is started.
    if name!='mappo': monkeypatch.setattr(module,'require_cuda',lambda device:None)
    cls={'mappo':'MAPPOTrainingRunner','rmappo':'RMAPPOTrainingRunner',
         'ea_mappo':'EAMAPPOTrainingRunner','stea_mappo':'STEAMAPPOTrainingRunner'}[name]
    with pytest.raises(ValueError,match='dimension'):
        getattr(module,cls)(cfg(),wrong,device='cuda',output_dir=tmp_path)
    if name!='mappo':
        validate=importlib.import_module(f'algorithm.{name}.protocol').validate_checkpoint
        with pytest.raises(RuntimeError,match='checkpoint is not'): validate({},cfg(),c)
        with pytest.raises(RuntimeError,match='dimensions mismatch'): validate({},cfg(),wrong)


@pytest.mark.parametrize('algorithm',['madsac','maddpg'])
def test_other_baseline_support_is_not_expanded(algorithm):
    if algorithm=='madsac':
        from algorithm.madsac.protocol import validate_madsac_config
        with pytest.raises(RuntimeError,match='environment_version'):
            validate_madsac_config(cfg(),{'algorithm':'madsac'})
    else:
        from algorithm.maddpg.protocol import validate_config
        c=yaml.safe_load((ROOT/'configs/maddpg_5v5.yaml').read_text())
        with pytest.raises(RuntimeError,match='MADDPG requires'): validate_config(cfg(),c)


def test_shared_schema_accepts_2000_and_rejects_unknown_fields():
    assert validate_config(cfg())['blue_policy']['conversion_range']==2000
    c=copy.deepcopy(cfg());c['blue_policy']['delay_timer']=1
    with pytest.raises(ValueError,match='schema'): validate_config(c)
