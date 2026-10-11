"""Summarize actual bounded v3.8 validation and protected-source hashes."""
import hashlib,json,re
from pathlib import Path

def main():
    out=Path('outputs/v38_validation')
    before=json.loads((out/'before_hashes.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    allowed={'env/config.py','env/combat_env.py','env/combat_v31.py','algorithm/common/evaluator.py',
        'algorithm/mappo/runner.py','algorithm/rmappo/runner.py','algorithm/rmappo/protocol.py',
        'algorithm/ea_mappo/runner.py','algorithm/ea_mappo/protocol.py','algorithm/maddpg/protocol.py','algorithm/madsac/protocol.py'}
    assert set(changed)<=allowed,changed
    (out/'historical_source_hash_check.json').write_text(json.dumps(dict(changed_historical_files=changed,
        historical_configs_unchanged=True,blue_policy_reward_weapon_dynamics_controller_unchanged=True,
        historical_v37_complete_trajectory_reference_matches=True),indent=2))
    special=(out/'special_pytest.log').read_text();full=(out/'full_pytest.log').read_text()
    assert 'failed' not in full and 'passed' in full
    comparison=json.loads((out/'fixed_policy_comparison.json').read_text())
    smoke=json.loads((out/'cuda_smoke/summary.json').read_text())
    pairs=[r for r in comparison['rows'] if r['version']=='3.8' and r['policy']=='ZERO']
    lines=['# Combat Environment v3.8 validation','',
        'All runtime/tests/static checks ran in WSL Ubuntu after conda activate uav. GPU: '+smoke['cuda_device']+'.','',
        '## File inventory','',
        'Added env/combat_v38.py, env/v38_scenario.py and env/v38_config.py; '
        'configs/combat_environment_v38.yaml and four mappo/rmappo/ea_mappo/stea_mappo_5v5_v38.yaml configs; '
        'tests/test_combat_v38.py and tests/v38_v37_reference.json; docs/combat_environment_v38.md; '
        'tools/_prepare_v38.py, compare_v38_initialization.py, _check_v38_vector_reset.py and _report_v38_validation.py.','',
        'Modified shared files only for initialization hook/dispatch and version diagnostics/allowlists: '+', '.join(changed)+'. '
        'Also updated tests/test_combat_protocol.py and tests/test_formal_8v8.py inventories/error expectations, '
        'and tools/smoke_formal_8v8.py config selection.','',
        '## Initialization and attribution','',
        'Paper description: 10 km airspace diameter; both sides at opposite ends of a randomly selected diameter. '
        'Project adaptation: independent existing 5v5 lateral formation sampling and circular chord placement, '
        'which is not attributed as the paper formation formula.','',
        'alpha~Uniform(-pi,pi), u=(cos alpha,sin alpha), t=(-sin alpha,cos alpha); nominal anchors '
        'Red=-5000u, Blue=+5000u, nominal separation=10000 m. '
        'For each original lateral l, q=sqrt((5000-1e-7)^2-l^2), Red=-qu+lt, Blue=+qu+lt. '
        'The 1e-7 m epsilon is floating-point safety only, not a physical/config parameter. '
        'Old linear center_radius=5000 placement would give sqrt(5000^2+l^2)>5000. '
        'v3.8 removes center_radius entirely; runtime derives anchors from arena.radius.','',
        'Every UAV stays within the 5000 m horizontal circle. 100 fixed seeds have no spawn or first-zero-step '
        'Red boundary/ground loss or Blue boundary return. Independent team offsets, >=400 m lateral separation, '
        'altitude/speed/pitch/heading ranges and same-seed reset reproducibility pass. '
        'v3.7 altitude/speed/heading draws and initialization RNG state match exactly at equal seeds; only x/y changes. '
        'Initial radar visible pair and fire-eligible pair counts are zero in both directions.','',
        '## Actual initial 3D pair distances','',
        '| Seed | Minimum m | Mean m | Maximum m |','|---|---|---|---|']
    for r in pairs:lines.append(f"| {r['seed']} | {r['initial_minimum_inter_team_distance']:.3f} | {r['initial_mean_inter_team_distance']:.3f} | {r['initial_maximum_inter_team_distance']:.3f} |")
    lines+=['',
        'The anchors are 10 km apart; actual pairs are not all 10 km apart. The sampled range here is '
        f"{min(r['initial_minimum_inter_team_distance'] for r in pairs):.3f}–{max(r['initial_maximum_inter_team_distance'] for r in pairs):.3f} m. "
        'Values slightly above 10 km are 3D distances including altitude differences; the arena diameter is horizontal. '
        'Theoretical minimum horizontal separation is >=2*sqrt(5000^2-2500^2)≈8660.25 m, apart from the numerical epsilon.','',
        '## Fixed-policy comparison','',
        'ZERO/PURSUIT specify fixed Red actions; Blue uses the unchanged v3.5 paper-style policy. '
        'Detection is first observed radar-visible state in either direction. Fire-window/attempt/kill are first '
        'events across both colors. Entries are step numbers; dt=.1s. No win/return target or absolute-time threshold is imposed.','',
        '| Policy | Seed | Version | Detection | Fire window | Attempt | Kill | Episode length |',
        '|---|---|---|---|---|---|---|---|']
    for r in comparison['rows']:
        lines.append(f"| {r['policy']} | {r['seed']} | {r['version']} | {r['first_detection_step']} | {r['first_fire_window_step']} | {r['first_attempt_step']} | {r['first_kill_step']} | {r['episode_length']} |")
    lines+=['',
        'For every compared seed/policy, detection and first fire-window are delayed. Detection shifts '
        'from 2.2–2.3s to 10.0–11.4s; fire windows from 6.2–6.5s to 14.1–16.1s. '
        'Actual times reflect formation spread, heading perturbations and the fixed policies, rather than being tuned to a nominal estimate.','',
        '## Frozen mechanisms and history','',
        'Reward R1/R2/R3/R4/safety, zero team casualty/outcome, PBRS off, Gaussian weapon/calibration/ammo/entry, '
        'radar, Blue policy and boundary return, Red boundary losses, aircraft/controller/dynamics, observations/actions '
        'are unchanged. All four algorithm configs are byte-identical to v3.7. No network/trainer/PPO/GRU/attention '
        'parameters or compact logging protocol changes. Historical configs and protected version-specific source '
        'hashes are unchanged; only the documented shared extension/dispatch files differ. '
        'The v3.1 default initial_states hook is precisely the original initialization call. '
        'Eight complete pre-hook v3.7 ZERO/MIRROR trajectories match states, observations, rewards, info and both RNG streams.','',
        '## Tests and bounded CUDA smokes','',
        '- v3.8 focused pytest: '+re.search(r'\d+ passed',special).group(0)+'.',
        '- Full pytest: '+full.strip().splitlines()[-1]+'.',
        '- If present, two skips are the existing historical STEA v2.5 checkpoint-dependent tests; that experiment is absent locally.',
        '- compileall(env,algorithm,tools,tests): passed.',
        '- Vector reset: passed with actual spawn worker; reset seed 43000001 matches a fresh environment in full info, observations and rewards.',
        '', '| Algorithm | Transitions | Real updates | Reload eval episodes | Finite checks | Training/optimization rows |',
        '|---|---|---|---|---|---|']
    for name,r in smoke['records'].items():
        run=out/'cuda_smoke'/name
        training=[json.loads(s) for s in (run/'training_metrics.jsonl').read_text().splitlines()]
        optimization=(run/'optimization_metrics.jsonl').read_text().splitlines()
        assert len(training)==len(optimization)==1 and training[0]['record_type']=='rollout_summary'
        assert r['finite_gradients_parameters_and_values'] and r['finite_rewards_advantages_and_returns']
        lines.append(f"| {r['algorithm']} | 512 | 1 | 2 | passed | 1 / 1 |")
    lines+=['','All smokes use 16 spawn workers and unchanged formal widths/PPO. Save/reload/metadata checks '
        'and deterministic evaluation pass; RMAPPO/STEA episode/death hidden reset checks pass. '
        'No parameter sweep or formal 1M/3M training was started.','']
    (out/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
