"""Read-only audit forwarding and same-state window accounting."""
from copy import deepcopy
import numpy as np
import pytest
from tools.audit_combat_v336_weapon_geometry import compare_snapshot,GeometryObserver,policy_actions
from env.fixed_policy import NearestTargetPursuitPolicy
from tests.test_combat_v336 import env,pair


def test_disagreement_row_is_same_pair_and_expected_probabilities_are_diagnostic():
    old,new = env('335'),env()
    a,b = pair(500,0,35,20)
    before = (a.copy(),b.copy());rows = []
    counts = compare_snapshot([a],[b],[6],old.weapon,new.weapon,rows.append,source='test',seed=0,side='red',step=0)
    assert a == before[0] and b == before[1]
    assert counts['live_pair_steps'] == counts['range_qualified'] == counts['range_ata_qualified'] == 1
    assert counts['range_ata_ha_qualified'] == counts['v336_only'] == counts['agreement'] == 0
    assert counts['v335_only'] == 1
    assert rows[0]['ha_deg'] == pytest.approx(35)
    assert rows[0]['body_elevation_deg'] == pytest.approx(15)


def test_v336_only_ammo_and_dead_denominators():
    old,new = env('335'),env()
    a,b = pair(500,0,0,40)
    c = compare_snapshot([a],[b],[6],old.weapon,new.weapon)
    assert c['v336_only'] == 1 and c['v335_only'] == 0
    c = compare_snapshot([a],[b],[0],old.weapon,new.weapon)
    assert c['both_ineligible'] == c['range_ata_ha_qualified'] == 1
    b.alive = False
    assert sum(compare_snapshot([a],[b],[6],old.weapon,new.weapon).values()) == 0


@pytest.mark.parametrize('version',['335','336'])
@pytest.mark.parametrize('mode',['fixed','pitch_probe'])
def test_observer_preserves_actions_states_rewards_rng_ammo_and_cooldown(version,mode):
    a,b = env(version),env(version)
    body,paper = env('335').weapon,env().weapon
    policy = NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    assert '_entry_attempts' not in b.__dict__
    with GeometryObserver(b,body,paper) as observer:
        for _ in range(190):
            actions = policy_actions(a,policy,mode)
            x,y = a.step(actions),b.step(actions.copy())
            assert a.red == b.red and a.blue == b.blue
            assert a.rng.bit_generator.state == b.rng.bit_generator.state
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            assert x[2:4] == y[2:4]
            assert a.combat_counts == b.combat_counts
            for side in ('red','blue'):
                for suffix in ('_ammo','_fire_cooldown_steps'):
                    np.testing.assert_array_equal(getattr(a,side+suffix),getattr(b,side+suffix))
            if x[2] or x[3]:break
    assert '_entry_attempts' not in b.__dict__
    assert observer.counts['red']['live_pair_steps'] > 0


def test_observer_restores_existing_instance_hook():
    e = env();hook = e._entry_attempts;e._entry_attempts = hook
    with GeometryObserver(e,env('335').weapon,e.weapon):pass
    assert e.__dict__['_entry_attempts'] is hook
