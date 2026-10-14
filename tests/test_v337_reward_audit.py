"""Audit reconstruction and non-mutating same-state comparisons."""
from copy import deepcopy
import numpy as np
import pytest
from tools.audit_combat_v337_reward_geometry import compare_rewards,assert_frozen_transition,geometry_pair
from tests.test_combat_v337 import env
from env.v337_reward import tactical_rewards_v337,tactical_pair_rewards_v337


@pytest.mark.parametrize('reverse',[False,True])
def test_controlled_audit_rows_forward_and_blue_centered_defense(reverse):
    e=env();a,b=geometry_pair(2000,0,35,20)
    red,blue=([b],[a]) if reverse else ([a],[b]);before=deepcopy((red,blue));rows=[]
    c,old,new=compare_rewards(red,blue,e.config['reward'],e.config['weapon'],rows.append,source='test',seed=0,step=0)
    assert (red,blue)==before
    name='tactical_defense' if reverse else 'tactical_offense'
    assert c[name+'_pair_changed']==1 and new[name][0]==0 and old[name][0]!=0
    row=next(r for r in rows if r['component']==name)
    assert row['attacker_side']==('blue' if reverse else 'red')
    assert row['ha_deg']==pytest.approx(35) and row['body_elevation_deg']==pytest.approx(15)


def test_multiple_targets_max_max_min_and_audit_reconstruction():
    e=env();a,b=geometry_pair(1500,15);_,c=geometry_pair(2500)
    pair=tactical_pair_rewards_v337([a],[b,c],e.config['reward'],e.config['weapon'])
    out=tactical_rewards_v337([a],[b,c],e.config['reward'],e.config['weapon'])
    assert out['guide'][0]==.00075 and out['tactical_offense'][0]==.025
    assert out['guide'][0]==max(pair['guide'][0])
    assert out['tactical_offense'][0]==max(pair['tactical_offense'][0])
    # Reverse both pairs so two independently aligned Blue threats compete.
    red=[a];a.psi=np.pi;b.psi=15*np.pi/180+np.pi;c.psi=np.pi
    out=tactical_rewards_v337(red,[b,c],e.config['reward'],e.config['weapon'])
    pair=tactical_pair_rewards_v337(red,[b,c],e.config['reward'],e.config['weapon'])
    assert out['tactical_defense'][0]==min(pair['tactical_defense'][:,0])<0
    compare_rewards(red,[b,c],e.config['reward'],e.config['weapon'])


def test_comparison_adds_no_rng_or_state_mutation_on_real_reset():
    e=env();before=deepcopy((e.red,e.blue,e.rng.bit_generator.state,e.red_ammo,e.blue_ammo))
    c,old,new=compare_rewards(e.red,e.blue,e.config['reward'],e.config['weapon'])
    assert e.red==before[0] and e.blue==before[1] and e.rng.bit_generator.state==before[2]
    np.testing.assert_array_equal(e.red_ammo,before[3]);np.testing.assert_array_equal(e.blue_ammo,before[4])
    assert c['total_tactical_delta_sum']==0
