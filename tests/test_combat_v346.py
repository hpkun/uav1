"""Paper Eq. (25) bins plus explicit engineering precedence and isolation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pytest
import yaml
from env.config import load_config, validate_config, environment_dimensions, SUPPORTED_ENVIRONMENT_VERSIONS
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.v346_reward import PAPER_R4, paper_branch, tactical_rewards, tactical_pair_rewards
from env.v343_reward import continuous_tactical_rewards, RESEARCH_REWARD
from env.v343_config import validate_shaping_gamma
from tools.audit_v339_attack_geometry import pair, states
from tests.test_combat_v344 import frozen


@pytest.mark.parametrize('prefix,values',[
    ('offense_reward',(.1,.02,.01)), ('threat_penalty',(-.15,-.025,-.015))])
@pytest.mark.parametrize('ata,ha,aa,index',[
    (3,4,0,0),(10,12,0,1),(25,20,0,2),
    (5,5,30,0),(15,15,-30,1),(30,30,30,2),
    (-5,-5,-30,0),(-15,-15,0,1),(-30,-30,0,2),
    (30.000001,0,0,None),(0,30.000001,0,None),(0,0,30.000001,None),
    (5.000001,0,0,1),(15.000001,0,0,2),(3,12,0,1),(20,4,0,2)])
def test_all_paper_bins_and_inclusive_angles(prefix,values,ata,ha,aa,index):
    a,b=pair(distance=2500,ata=ata,ha=ha,aa=aa)
    g=engagement_geometry(a,b)
    assert paper_branch(g,PAPER_R4,prefix)==(0 if index is None else values[index])


@pytest.mark.parametrize('distance,expected',[(0,.1),(1e-10,.1),(500,.1),
    (1999,.1),(2000,.1),(3000,.1),(4000,.1),(4000+1e-8,0)])
def test_range_gate_no_continuous_weight(distance,expected):
    a,b=pair(distance=distance)
    assert tactical_rewards([a],[b],PAPER_R4)['tactical'][0]==expected


@pytest.mark.parametrize('ata,ha',[(30,0),(0,30),(30,30)])
def test_distance_boundary_with_angular_geometry(ata,ha):
    a,b=pair(distance=4000,ata=ata,ha=ha)
    g=engagement_geometry(a,b)
    assert abs(g.distance-4000)<1e-11
    # Isolate literal distance boundary from trig construction roundoff.
    assert paper_branch(replace(g,distance=4000.),PAPER_R4,'offense_reward')==.01
    assert paper_branch(replace(g,distance=4000.+1e-8),PAPER_R4,'offense_reward')==0


@pytest.mark.parametrize('ata,ha,value',[(3,4,-.15),(10,12,-.025),(25,20,-.015),(31,0,0),(0,31,0)])
def test_reverse_geometry_uses_blue_as_attacker(ata,ha,value):
    # Construct BLUE's 3/4/0 geometry; the Red offense geometry is ineligible.
    blue,red=pair(distance=3000,ata=ata,ha=ha)
    off,de=tactical_pair_rewards([red],[blue],PAPER_R4)
    assert off[0,0]==0 and de[0,0]==value
    assert tactical_rewards([red],[blue],PAPER_R4)['tactical'][0]==value


def test_independent_ata_ha_not_alpha_and_no_tail_function():
    a,b=pair(distance=3000,ata=25,ha=25)
    assert engagement_geometry(a,b).off_boresight>np.pi/6
    assert tactical_rewards([a],[b],PAPER_R4)['tactical'][0]==.01
    a,b=pair(distance=3000,pitch=50,ha=50)
    assert engagement_geometry(a,b).off_boresight<1e-6
    assert tactical_rewards([a],[b],PAPER_R4)['tactical'][0]==0
    values=[];old=[]
    for aa in (0,10,29):
        a,b=pair(distance=2500,ata=3,ha=4,aa=aa)
        values.append(tactical_rewards([a],[b],PAPER_R4)['tactical'][0])
        old.append(continuous_tactical_rewards([a],[b],RESEARCH_REWARD)['tactical'][0])
    assert values==[.1]*3 and len(set(old))==3


def test_engineering_multitarget_max_min_offense_precedence_and_slot_invariance():
    red,weak=pair(distance=2500,ata=25,ha=20)
    _,strong=pair(distance=2500,ata=3,ha=4)
    # Blue behind Red: reverse geometry perfect; forward geometry invalid.
    threat=red.copy();threat.x=-2500
    threat_weak=threat.copy();threat_weak.psi=np.deg2rad(12)
    for targets in ([weak,strong,threat,threat_weak],[threat_weak,threat,strong,weak]):
        off,de=tactical_pair_rewards([red],targets,PAPER_R4)
        assert off.max()==.1 and de.min()==-.15
        r=tactical_rewards([red],targets,PAPER_R4)
        assert r['tactical'][0]==.1 and r['tactical_defense'][0]==0
    assert tactical_rewards([red],[threat_weak,threat],PAPER_R4)['tactical'][0]==-.15
    strong.alive=False
    assert tactical_rewards([red],[weak,strong,threat],PAPER_R4)['tactical'][0]==.01
    weak.alive=False;threat.alive=False
    assert tactical_rewards([red],[weak,strong,threat],PAPER_R4)['tactical'][0]==0
    red.alive=False
    assert tactical_rewards([red],[threat_weak],PAPER_R4)['tactical'][0]==0
    assert tactical_rewards([red],[],PAPER_R4)['tactical'][0]==0


@pytest.mark.parametrize('sharing',[False,True])
def test_dwell_simultaneous_death_and_shared_credit_identical(sharing):
    a,b=frozen('344',sharing),frozen('346',sharing)
    for _ in range(5):
        x=a.step(np.zeros((4,3)),np.zeros((4,3)))
        y=b.step(np.zeros((4,3)),np.zeros((4,3)))
        np.testing.assert_array_equal(states(a),states(b))
        for k in range(1,4):np.testing.assert_array_equal(x[4][f'r{k}_rewards'],y[4][f'r{k}_rewards'])
    assert x[2:4]==y[2:4]==(True,False)
    assert x[4]['termination_reason']==y[4]['termination_reason']
    for side in ('red','blue'):
        assert x[4][side+'_dwell_completed']==y[4][side+'_dwell_completed']
        assert x[4][side+'_dwell_neutralizations']==y[4][side+'_dwell_neutralizations']
    # Existing temporal semantics: a lethal transition may contain pre-event R4;
    # already-dead entities produce no new reward on subsequent computation.
    assert not b.dense_combat_reward()[0].any()


def test_config_baseline_identity_single_r4_source_and_algorithm_hyperparameters():
    c=load_config('configs/combat_environment_v346.yaml')
    old=load_config('configs/combat_environment_v344.yaml')
    assert environment_dimensions(c)==(49,3,4)
    for key in old:
        if key not in ('environment_version','research_reward'):assert c[key]==old[key]
    assert c['research_reward']=={k:v for k,v in old['research_reward'].items()
                                if k=='gamma' or k.startswith('potential_')}
    for algo in ('mappo','ippo','maac'):
        a=yaml.safe_load(Path(f'configs/{algo}_4v4_v346.yaml').read_text())
        b=yaml.safe_load(Path(f'configs/{algo}_4v4_v344.yaml').read_text())
        validate_shaping_gamma(c,a)
        a['environment']=b['environment'];assert a==b
        a['training']['gamma']=.98
        with pytest.raises(ValueError,match='gamma'):validate_shaping_gamma(c,a)
    for key in set(RESEARCH_REWARD)-set(c['research_reward']):
        bad=deepcopy(c);bad['research_reward'][key]=RESEARCH_REWARD[key]
        with pytest.raises(ValueError):validate_config(bad)
    for key in PAPER_R4:
        bad=deepcopy(c);bad['reward'][key]+=1e-5
        with pytest.raises(ValueError):validate_config(bad)
    for section,key,value in [('weapon','off_boresight_max',math.pi/4),
                              ('fire_control','dwell_seconds',.6),('simulation','dt',.2),
                              ('reward','win_reward',10),('reward','kill_reward',11)]:
        bad=deepcopy(c);bad[section][key]=value
        with pytest.raises(ValueError):validate_config(bad)
    # Historical schemas still require their full continuous research reward.
    old['research_reward']=c['research_reward']
    with pytest.raises(ValueError):validate_config(old)


def test_retired_version_not_registered_or_constructible():
    assert '3.45' not in SUPPORTED_ENVIRONMENT_VERSIONS
    c=load_config('configs/combat_environment_v344.yaml');c['environment_version']='3.45'
    with pytest.raises(ValueError):MultiUAVCombatEnv(c)
    for name in ('combat_v345.py','v345_config.py'):assert not (Path('env')/name).exists()
    for name in ('combat_environment','mappo_4v4','ippo_4v4','maac_4v4'):
        assert not Path(f'configs/{name}_v345.yaml').exists()
