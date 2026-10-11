"""56D layout, common combat geometry and observation-only isolation."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v311 import CombatEnvironmentV311
from env.combat_v312 import CombatEnvironmentV312
from env.config import load_config, environment_dimensions, validate_config
from env.models import AircraftState
from env.fixed_policy import NearestTargetPursuitPolicy
from env.sensor import sensor_geometry
from env.observation import flight_path_frame, _relative_position
from env.v36_geometry import combat_geometry
from env.v312_observation import build_team_observations_v312
from algorithm.stea_mappo.networks import decompose_observations, entity_valid_mask
from algorithm.ea_mappo.networks import EntityAttentionActor
from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor
from algorithm.common.control_factory import trainer_kwargs
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.rmappo.factory import build_rmappo_trainer
from algorithm.ea_mappo.factory import build_ea_mappo_trainer
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from tools._prepare_v312 import capture311, actor_reference
from tools._capture_v33_reference import canonical, states
from tools.combat_visualization import RecordingCombatEnv

CONFIG = 'configs/combat_environment_v312.yaml'


def state(x=0., y=0., altitude=3000., v=225., psi=0., theta=0., alive=True):
    return AircraftState(x, y, -altitude, v, theta, psi, alive)


def env():
    e = MultiUAVCombatEnv(CONFIG)
    e.reset(1)
    return e


def test_exact_self7_no_ammo_and_zero_phi():
    e = env()
    e.red[2] = state(x=2500., y=-1000., altitude=1500., v=300., psi=np.pi/2, theta=-np.pi/6)
    e.red_last_executed_phi.fill(1.2)  # builder cannot resurrect a bank feature
    obs = e._observations()
    np.testing.assert_array_equal(obs[2,:7], [.5, -.2, .25, 1., 0., .5, -.5])
    assert obs.shape == (5,56)
    e.red_ammo[:] = [0,1,2,3,4]
    e.blue_ammo.fill(0)
    np.testing.assert_array_equal(e._observations(), obs)
    assert not obs[:,4].any()
    assert obs[2,7] == _relative_position(e.red[2],e.red[0],flight_path_frame(e.red[2]),10000.)[0]


@pytest.mark.parametrize('psi,displacement,expected', [
    (0., [1000,0,0], [.1,0,0]), (0., [0,1000,0], [0,.1,0]),
    (0., [0,0,-1000], [0,0,.1]), (np.pi/2, [0,1000,0], [.1,0,0]),
    (np.pi/2, [-1000,0,0], [0,.1,0]), (np.pi/2, [0,0,-1000], [0,0,.1])])
def test_ally_body_frame_and_absolute_attitude(psi, displacement, expected):
    e = env()
    e.red[0] = state(psi=psi)
    x,y,z = displacement
    e.red[1] = state(x=x, y=y, altitude=3000-z, v=150., psi=-np.pi/2, theta=np.pi/6)
    slot = e._observations()[0,7:13]
    np.testing.assert_allclose(slot[:3], expected, atol=1e-14)
    np.testing.assert_array_equal(slot[3:], [-1.,-.5,.5])
    assert slot.shape == (6,)


@pytest.mark.parametrize('target', [state(x=1000.), state(y=1000.), state(y=-1000.),
    state(altitude=4000.), state(altitude=2000.), state(x=1000.,psi=np.pi),
    state(x=1000.,psi=0.), state(x=2500.,y=-1234.,altitude=4100.,psi=.3,theta=.2)])
def test_enemy_geometry_exact_combat_output(target):
    e = env()
    e.red[0] = state(psi=.13,theta=-.25)
    e.blue[0] = target
    g = combat_geometry(e.red, e.blue)
    slot = e._observations()[0,31:36]
    np.testing.assert_allclose(slot[[0,2,3,4]]*[10000,np.pi,np.pi,np.pi],
        [g['distance'][0,0],g['aa'][0,0],g['ata'][0,0],g['ha'][0,0]], rtol=0, atol=1e-8)
    assert slot[1] == (target.v-225.)/75.
    eligibility = e.weapon.eligibility(e.red, e.blue, e.red_ammo)[0][0,0]
    assert eligibility == (g['distance'][0,0] <= 4000 and g['ata'][0,0] <= np.pi/6+1e-12 and g['ha'][0,0] <= np.pi/6+1e-12)


@pytest.mark.parametrize('target', [state(x=-1000.,v=250.), state(x=11000.,v=250.),
    state(y=5000.,altitude=4500.,v=250.)])
def test_radar_invisible_enemy_stays_observable_and_builder_never_calls_sensor(target):
    e = env()
    e.red[0] = state()
    e.blue[0] = target
    assert not sensor_geometry(e.red, e.blue, e.config['sensor'])[0][0,0]
    with (patch('env.sensor.sensor_geometry', side_effect=RuntimeError('radar called')),
          patch('env.geometry.engagement_geometry', side_effect=RuntimeError('legacy geometry called'))):
        obs = e._observations()
    assert obs[0,31] > 0 and obs[0,32] != 0
    assert np.isfinite(obs).all()
    before = obs.copy()
    e.config['sensor'] = dict(range_max=0., off_boresight_angle_max=0.)
    np.testing.assert_array_equal(e._observations(), before)


def test_fixed_identity_order_and_dead_zeroing_without_repacking():
    e = env()
    e.red = [state(x=100.*i,v=150.+30*i,psi=.1*i,theta=.02*i) for i in range(5)]
    e.blue = [state(x=5000.-900*i,v=160.+25*i,psi=.2*i,theta=-.02*i) for i in range(5)]
    before = e._observations()[2]
    for slot, index in enumerate((0,1,3,4)):
        assert before[7+6*slot+3] == (e.red[index].v-225)/75
    for j in range(5):
        assert before[31+5*j+1] == (e.blue[j].v-225)/75
    e.red[1].alive = False
    e.blue[0].alive = False
    e.blue[3].alive = False
    after = e._observations()
    assert not after[2,13:19].any()
    assert not after[2,31:36].any() and not after[2,46:51].any()
    for start,end in ((7,13),(19,31),(36,46),(51,56)):
        np.testing.assert_array_equal(after[2,start:end], before[start:end])
    assert not after[1].any()


def test_normalization_finite_at_legal_extremes_and_no_clipping():
    e = env()
    e.red[0] = state(x=-4999.,altitude=5999.,v=150.,psi=-np.pi,theta=-np.pi/3)
    e.blue[0] = state(x=4999.,altitude=1.,v=300.,psi=np.pi,theta=np.pi/3)
    obs = e._observations()
    assert np.isfinite(obs).all()
    assert np.abs(obs[:,:7]).max() <= 1
    assert obs[0,31] > 1.  # legal long diagonal preserved
    assert obs[0,32] == 1.
    assert 0 <= obs[0,35] <= 1.
    assert obs.dtype == np.float64


@pytest.mark.parametrize('prefix', [(5,), (2,5), (2,3,5)])
def test_entity_decomposition_batched_shapes_and_zero_slot_masks(prefix):
    obs = torch.zeros(*prefix,56)
    obs[...,7] = .1
    obs[...,31] = .2
    own, ally, enemy = decompose_observations(obs)
    assert own.shape == (*prefix,7) and ally.shape == (*prefix,4,6) and enemy.shape == (*prefix,5,5)
    alive = torch.ones(prefix)
    assert entity_valid_mask(ally, alive, 56)[...,0].all()
    assert not entity_valid_mask(ally, alive, 56)[...,1:].any()
    assert entity_valid_mask(enemy, alive, 56)[...,0].all()  # HA=0 is not a dead flag
    alive.fill_(0.)
    assert not entity_valid_mask(enemy, alive, 56).any()


@pytest.mark.parametrize('kind', ['ea','stea'])
def test_attention_masks_whole_slots_and_encoder_widths(kind):
    actor = (EntityAttentionActor if kind == 'ea' else SpatioTemporalEntityAttentionActor)(
        self_feature_dim=7, observation_layout='madsac56')
    assert [actor.self_encoder[0].in_features,actor.ally_encoder[0].in_features,actor.enemy_encoder[0].in_features] == [7,6,5]
    obs = torch.zeros(5,56)
    obs[:,7] = .1
    obs[:,31] = .2
    _, attention = actor.spatial(obs,torch.ones(5))
    for side in ('ally','enemy'):
        w = attention[f'{side}_attention_weights']
        assert torch.all(w[...,0] == 1.) and not w[...,1:].any()
    _, empty = actor.spatial(torch.zeros_like(obs),torch.ones(5))
    assert not empty['ally_context'].any() and not empty['enemy_context'].any()


@pytest.mark.parametrize('name', ['mappo','rmappo','ea_mappo','stea_mappo'])
def test_four_formal_network_inputs_and_only_layout_config_changes(name):
    cfg = yaml.safe_load(Path(f'configs/{name}_5v5_v312.yaml').read_text())
    previous = yaml.safe_load(Path(f'configs/{name}_5v5_v311.yaml').read_text())
    adjusted = deepcopy(cfg)
    adjusted['network']['observation_dim'] = 66
    if name in ('ea_mappo','stea_mappo'):
        adjusted['network']['self_feature_dim'] = 8
        adjusted['network'].pop('observation_layout')
    assert adjusted == previous
    trainer = (MAPPOTrainer(**trainer_kwargs(cfg,'cpu',31)) if name == 'mappo' else
        {'rmappo':build_rmappo_trainer,'ea_mappo':build_ea_mappo_trainer,
         'stea_mappo':build_stea_mappo_trainer}[name](cfg,'cpu',seed=31))
    assert trainer.observation_dim == 56 and trainer.num_agents == 5
    assert trainer.critic.value_network[0].in_features == 336
    if name == 'mappo':
        assert trainer.actor.backbone[0].in_features == 56
    elif name == 'rmappo':
        assert trainer.actor.flat_encoder[0].in_features == 56
    else:
        assert trainer.actor.observation_layout == 'madsac56'
        assert trainer.network_architecture['observation_layout'] == 'madsac56'


@pytest.mark.parametrize('mode,seed', [(p,s) for p in ('ZERO','PURSUIT','RANDOM') for s in (1,31,10000000)])
def test_v311_state_action_reward_weapon_rng_and_outcome_identical(mode, seed):
    old = MultiUAVCombatEnv('configs/combat_environment_v311.yaml')
    new = env()
    a, ia = old.reset(seed)
    b, ib = new.reset(seed)
    assert a.shape == (5,66) and b.shape == (5,56)
    def strip(info):
        return canonical({k:v for k,v in info.items() if k != 'environment_version'})
    assert strip(ia) == strip(ib)
    rng = np.random.default_rng(417)
    pursuit = NearestTargetPursuitPolicy(old.config['blue_policy'],old.config['action'])
    while True:
        action = (np.zeros((5,3)) if mode == 'ZERO' else
                  pursuit.team_actions(old.red,old.blue) if mode == 'PURSUIT' else rng.uniform(-1,1,(5,3)))
        for side in ('red','blue'):
            np.testing.assert_array_equal(old.weapon.eligibility(getattr(old,side),getattr(old,'blue' if side=='red' else 'red'),getattr(old,side+'_ammo'))[0],
                new.weapon.eligibility(getattr(new,side),getattr(new,'blue' if side=='red' else 'red'),getattr(new,side+'_ammo'))[0])
        ra = old.step(action)
        rb = new.step(action)
        np.testing.assert_array_equal(ra[1],rb[1])
        assert ra[2:4] == rb[2:4] and strip(ra[4]) == strip(rb[4])
        assert canonical(states(old)) == canonical(states(new))
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        if ra[2] or ra[3]:
            break


def test_environment_class_only_replaces_observation_and_legacy_mask_stays():
    assert set(CombatEnvironmentV312.__dict__)-{'__module__','__doc__'} == {'environment_version','_observations'}
    for method in ('__init__','step','reset','initial_states','_advance','_entry_attempts','_resolve_combat',
                   '_resolve_noncombat_losses','dense_combat_reward','_outcome','build_weapon'):
        assert getattr(CombatEnvironmentV312,method) is getattr(CombatEnvironmentV311,method)
    old = MultiUAVCombatEnv('configs/combat_environment_v311.yaml')
    old.reset(1)
    old.red[0] = state()
    old.blue[0] = state(x=-1000.)
    obs = old._observations()
    assert obs.shape == (5,66) and obs[0,7] == 1.
    assert not obs[0,36:42].any()
    assert environment_dimensions(load_config(CONFIG)) == (56,3,5)
    assert environment_dimensions(old.config) == (66,3,5)
    cfg = deepcopy(load_config(CONFIG))
    cfg['weapon']['range_max'] = 2000.
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_replay_environment_dispatches_56d_without_changing_events():
    recorder = RecordingCombatEnv(CONFIG)
    normal = env()
    obs, _ = recorder.reset(1)
    assert obs.shape == (5,56) and recorder.observation_dim == 56
    while True:
        a = recorder.step(np.zeros((5,3)))
        b = normal.step(np.zeros((5,3)))
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
        assert recorder.rng.bit_generator.state == normal.rng.bit_generator.state
        if a[2] or a[3]:
            break
    assert any(event['type'] == 'fire_attempt' for event in recorder.events)


def test_v311_full_reference_and_legacy_entity_actor_exact():
    assert capture311() == json.loads(Path('tests/v312_v311_reference.json').read_text())
    assert actor_reference() == json.loads(Path('tests/v312_legacy_actor_reference.json').read_text())
