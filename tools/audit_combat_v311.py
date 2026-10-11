"""Bounded v3.11-only sanity, never a training or parameter-search driver."""
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from tools.audit_combat_v310 import expected_targets
from tools.audit_combat_v39 import check_decomposition

SEEDS = (1, 31, 10000000)


def trial(policy, seed):
    e = MultiUAVCombatEnv('configs/combat_environment_v311.yaml')
    obs, initial = e.reset(seed)
    assert obs.shape == (5, 66) and np.isfinite(obs).all()
    assert initial['initial_red_fire_eligible_pairs'] == initial['initial_blue_fire_eligible_pairs'] == 0
    pursuit = NearestTargetPursuitPolicy(e.config['blue_policy'], e.config['action'])
    accumulated = np.zeros(5)
    decisions = []
    while True:
        expected = expected_targets(e.blue, e.red)
        action = np.zeros((5, 3)) if policy == 'ZERO' else pursuit.team_actions(e.red, e.blue)
        obs, reward, term, trunc, info = e.step(action)
        assert info['blue_last_target_indices'] == expected
        assert np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(reward).all()
        for s in e.red + e.blue:
            assert np.isfinite(s.as_array()).all()
            assert 150. <= s.v <= 300. and -np.pi/3 <= s.theta <= np.pi/3
        assert not e.red_last_executed_phi.any() and not e.blue_last_executed_phi.any()
        assert ((e.red_ammo >= 0) & (e.red_ammo <= 6)).all()
        assert ((e.blue_ammo >= 0) & (e.blue_ammo <= 6)).all()
        accumulated += reward
        check_decomposition(reward, info, accumulated)
        decisions.append(dict(step=e.steps, selected_targets=expected,
            red_attempts=info['red_step_fire_attempts'], blue_attempts=info['blue_step_fire_attempts'],
            red_ammo=e.red_ammo.tolist(), blue_ammo=e.blue_ammo.tolist()))
        if term or trunc:
            break
    assert e.steps > 1 and not info['draw']
    result = dict(policy=policy, seed=seed,
        initial_center_distance=initial['nominal_anchor_separation'],
        initial_min_inter_team_distance=initial['minimum_inter_team_distance'],
        initial_max_inter_team_distance=initial['maximum_inter_team_distance'],
        initial_red_visible_pairs=initial['initial_red_visible_pairs'],
        initial_blue_visible_pairs=initial['initial_blue_visible_pairs'],
        episode_length=e.steps, red_boundary_losses=info['red_boundary_exits'],
        red_ground_losses=info['red_ground_losses'], blue_boundary_returns=info['blue_boundary_returns'],
        red_survivors=info['red_survivors'], blue_survivors=info['blue_survivors'],
        red_ammo_used=info['red_ammo_used'], blue_ammo_used=info['blue_ammo_used'],
        red_remaining_ammo=e.red_ammo.tolist(), blue_remaining_ammo=e.blue_ammo.tolist(),
        outcome=info['termination_reason'], episode_return=float(accumulated.sum()),
        finite_observation_action_reward_state=True, target_selection_violations=0)
    for event in ('fire_window', 'attempt', 'kill'):
        values = [info[f'{side}_first_{event}_step'] for side in ('red', 'blue')
                  if info[f'{side}_first_{event}_step'] is not None]
        result[f'first_{event}_step'] = min(values) if values else None
        for side in ('red', 'blue'):
            result[f'{side}_first_{event}_step'] = info[f'{side}_first_{event}_step']
    return result, decisions


def main():
    out = Path('outputs/v311_validation')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in ('ZERO', 'PURSUIT'):
        for seed in SEEDS:
            result, decisions = trial(policy, seed)
            rows.append(result)
            (out / f'decisions_{policy.lower()}_{seed}.json').write_text(json.dumps(decisions))
    result = dict(seeds=SEEDS, rows=rows, formal_training_started=False, parameters_tuned=False)
    (out / 'fixed_policy_sanity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
