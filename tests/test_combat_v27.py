"""v2.7 changes team size only, reusing the frozen Stern policy."""
import copy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import SternConversionPolicy, SternPhase
from env.models import AircraftState
from env.weapon import PairFireState, RearAspectWeaponEnvelope

ROOT=Path(__file__).resolve().parents[1]
STEMS={'mappo':'mappo_5v5','rmappo':'rmappo_5v5',
       'ea_mappo':'ea_mappo_5v5','stea_mappo':'stea_mappo_5v5'}


def cfg(version='27'):
    return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')


def test_exact_config_differences_and_protocol():
    current,old,stern=cfg(),cfg('25'),cfg('26')
    assert environment_dimensions(current)==(65,3,5)
    for key in old:
        if key not in ('environment_version','blue_policy'):
            assert current[key]==old[key]
    expected=copy.deepcopy(stern)
    expected['environment_version']='2.7'
    expected['scenario']['team_size']=5
    expected['scenario']['formation_offsets']=old['scenario']['formation_offsets']
    assert current==expected
    assert current['scenario']['formation_offsets']==[-600.,-300.,0.,300.,600.]
    assert current['blue_policy']==stern['blue_policy']
    env=MultiUAVCombatEnv(current)
    assert type(env.fixed_policy) is SternConversionPolicy
    assert type(env.weapon) is RearAspectWeaponEnvelope
    assert type(env.red_fire_states) is type(env.blue_fire_states) is PairFireState
    assert env.red_fire_states.armed.shape==env.blue_fire_states.armed.shape==(5,5)
    np.testing.assert_array_equal(env.reset(29)[0],MultiUAVCombatEnv(old).reset(29)[0])
    assert MultiUAVCombatEnv().environment_version=='2.3'


def test_five_locks_reacquisition_and_reset():
    env=MultiUAVCombatEnv(cfg());env.reset(31);p=env.fixed_policy
    own=AircraftState(6000,100,-3000,225,0,np.pi)
    targets=[AircraftState(0,0,-3000,225,0,0),AircraftState(-1000,0,-3000,225,0,0)]
    for index in range(5):
        p.action(own,targets,index);p.action(own,targets,index)
        assert p.states[index].locked_target_index==0
        assert p.states[index].phase is SternPhase.RELATIVE_BEARING
        assert np.sin(p.states[index].relative_bearing_heading)>0
    targets[1].x=5900
    for index in range(5):
        p.action(own,targets,index)
        assert p.states[index].locked_target_index==0
    targets[0].alive=False
    for index in range(5):
        p.action(own,targets,index)
        s=p.states[index]
        assert s.locked_target_index==1 and s.phase is SternPhase.PURE_PURSUIT
        assert s.relative_bearing_heading is s.turn_side is None
    env.reset(31)
    assert len(p.states)==5
    for s in p.states:
        assert s.locked_target_index is None and s.phase is SternPhase.PURE_PURSUIT
        assert s.relative_bearing_heading is s.turn_side is None


@pytest.mark.parametrize('field,value',[('turn_angle',0),('turn_range',0),
    ('required_lateral_displacement',0),('conversion_range',600),('conversion_range',4001),
    ('desired_speed',301)])
def test_shared_stern_validation(field,value):
    for version in ('26','27'):
        c=cfg(version);c['blue_policy'][field]=value
        with pytest.raises(ValueError,match='blue_policy'): validate_config(c)


def test_schema_rejects_missing_extra_weapon_aspect_and_wrong_team():
    for field in ('missing','extra','weapon','team'):
        c=cfg()
        if field=='missing': del c['blue_policy']['turn_range']
        elif field=='extra': c['blue_policy']['random_turn']=True
        elif field=='weapon': del c['weapon']['target_aspect_angle_max']
        else: c['scenario']['team_size']=8
        with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('name',list(STEMS))
def test_five_v_five_formal_trainer_and_eight_v_eight_rejection(name,tmp_path,monkeypatch):
    from algorithm.common.control_factory import trainer_kwargs
    from algorithm.mappo.trainer import MAPPOTrainer
    c=yaml.safe_load((ROOT/f'configs/{STEMS[name]}.yaml').read_text())
    assert tuple(c['network'][k] for k in ('observation_dim','action_dim','num_agents'))==environment_dimensions(cfg())
    trainer=(MAPPOTrainer(**trainer_kwargs(c,'cpu',31)) if name=='mappo' else
             getattr(importlib.import_module(f'algorithm.{name}.factory'),f'build_{name}_trainer')(c,'cpu',seed=31))
    assert trainer.critic.value_network[0].in_features==390
    with torch.no_grad():
        assert torch.isfinite(trainer.critic(torch.zeros(2,5,65),torch.ones(2,5))).all()
    stem={'mappo':'mappo_8v8_formal','rmappo':'rmappo_8v8',
          'ea_mappo':'ea_mappo_8v8','stea_mappo':'stea_mappo_8v8_formal'}[name]
    wrong=yaml.safe_load((ROOT/f'configs/{stem}.yaml').read_text())
    module=importlib.import_module(f'algorithm.{name}.runner')
    # Exercise only the pre-construction dimension guard; no trainer/update
    # or simulated CPU training is run through this path.
    if name!='mappo': monkeypatch.setattr(module,'require_cuda',lambda device:None)
    runner={'mappo':'MAPPOTrainingRunner','rmappo':'RMAPPOTrainingRunner',
            'ea_mappo':'EAMAPPOTrainingRunner','stea_mappo':'STEAMAPPOTrainingRunner'}[name]
    with pytest.raises(ValueError,match='dimension'):
        getattr(module,runner)(cfg(),wrong,device='cuda',output_dir=tmp_path)
    if name in ('rmappo','ea_mappo','stea_mappo'):
        validate=importlib.import_module(f'algorithm.{name}.protocol').validate_checkpoint
        with pytest.raises(RuntimeError,match='checkpoint is not'):
            validate({},cfg(),c)
        with pytest.raises(RuntimeError,match='dimensions mismatch'):
            validate({},cfg(),wrong)


def test_madsac_rejects_v27():
    from algorithm.madsac.protocol import validate_madsac_config
    with pytest.raises(RuntimeError,match='environment_version'):
        validate_madsac_config(cfg(),{'algorithm':'madsac'})
