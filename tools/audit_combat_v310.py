"""v3.10-only fixed-policy sanity; verify every automatic Blue decision."""
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from tools.audit_combat_v39 import check_decomposition

SEEDS = (1, 31, 10000000)


def expected_targets(blue, red):
    # Independent implementation used only by the audit, never by policy control.
    result = []
    for own in blue:
        candidates = [((r.x-own.x)**2 + (r.y-own.y)**2 + (r.z-own.z)**2, i)
            for i, r in enumerate(red) if r.alive] if own.alive else []
        result.append(min(candidates)[1] if candidates else None)
    return result


def trial(policy, seed):
    e = MultiUAVCombatEnv('configs/combat_environment_v310.yaml')
    e.reset(seed)
    pursuit = NearestTargetPursuitPolicy(e.config['blue_policy'], e.config['action'])
    rows = []
    total = np.zeros(5)
    violations = 0
    while True:
        expected = expected_targets(e.blue, e.red)
        previous = list(e.fixed_policy.last_targets)
        pre_pursuit, pre_switches = e.fixed_policy.pursuit_agent_steps, e.fixed_policy.target_switches
        red_action = np.zeros((5, 3)) if policy == 'ZERO' else pursuit.team_actions(e.red, e.blue)
        _, reward, term, trunc, info = e.step(red_action)
        selected = info['blue_last_target_indices']
        violations += sum(a != b for a, b in zip(expected, selected))
        assert selected == expected
        assert info['blue_paper_nearest_agent_steps'] == pre_pursuit + sum(i is not None for i in expected)
        switches = sum(target is not None and i < len(previous) and previous[i] is not None
            and previous[i] != target for i, target in enumerate(expected))
        assert info['blue_target_switches'] == pre_switches + switches
        total += reward
        check_decomposition(reward, info, total)
        assert not hasattr(e.fixed_policy, 'rng')
        assert not any(word in k for k in info for word in ('escape', 'guerrilla', 'centripetal'))
        rows.append(dict(step=e.steps, selected_targets=selected,
            switches_this_step=switches, switches_total=info['blue_target_switches']))
        if term or trunc:
            break
    result = dict(policy=policy, seed=seed, episode_length=e.steps,
        termination_reason=info['termination_reason'], blue_target_switches=info['blue_target_switches'],
        blue_paper_nearest_agent_steps=info['blue_paper_nearest_agent_steps'],
        blue_last_target_indices=info['blue_last_target_indices'], target_selection_violations=violations,
        red_boundary_exits=info['red_boundary_exits'], blue_boundary_returns=info['blue_boundary_returns'],
        red_survivors=info['red_survivors'], blue_survivors=info['blue_survivors'])
    for event in ('fire_window', 'attempt', 'kill'):
        values = [info[f'{side}_first_{event}_step'] for side in ('red', 'blue')
            if info[f'{side}_first_{event}_step'] is not None]
        result[f'first_{event}_step'] = min(values) if values else None
    return result, rows


def main():
    out = Path('outputs/v310_validation')
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for policy in ('ZERO', 'PURSUIT'):
        for seed in SEEDS:
            summary, decisions = trial(policy, seed)
            summaries.append(summary)
            (out / f'blue_decisions_{policy.lower()}_{seed}.json').write_text(json.dumps(decisions))
    result = dict(seeds=SEEDS, rows=summaries, target_selection_violations=0,
        comparator_experiments_run=False, formal_training_started=False,
        target_geometry_phase='pre-dynamics state at the start of each step')
    (out / 'fixed_policy_sanity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
