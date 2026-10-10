"""Render v3.6 evidence without parameter selection or follow-up tuning."""
import json,hashlib,re
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/v36_validation'
    log=(out/'full_pytest.log').read_text();match=re.search(r'(\d+) passed in [^\n]+',log)
    if not match or 'FAILED' in log:raise RuntimeError('complete passing pytest required')
    assert '47 passed' in (out/'special_pytest.log').read_text()
    assert (out/'compileall.log').exists() and not (out/'compileall.log').read_text()
    a=json.loads((out/'audit/summary.json').read_text());smoke=json.loads((out/'cuda_smoke/summary.json').read_text())
    before=json.loads((out/'before_hashes.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    for p in ('env/combat_v31.py','env/combat_v32.py','env/combat_v33.py','env/combat_v34.py','env/combat_v35.py','env/v30_policy.py','env/v34_policy.py','env/v35_policy.py','env/v30_reward.py','env/sensor.py','env/v30_weapon.py','env/dynamics.py','env/control.py','env/v31_scenario.py'):
        assert p not in changed
    assert not any(p.startswith('configs/') for p in changed)
    lines=[(root/'docs/combat_environment_v36.md').read_text(),'\n## Completed acceptance\n',
        f'- Full WSL Ubuntu/uav pytest: **{match.group(0)}**.',
        '- V36 focused suite: **47 passed**. Includes distance 1999/2000/2001, ATA/HA 30/30.1, dead/ammo, rear/head-on/crossing/NED/wrap/coincidence geometry, exact reward levels, max/min/bounded aggregation, actual invisible defensive threat, precombat reward cache surviving a kill, PBRS disabled, component totals, both-color strict entry/ammo regression and physical/sensor/weapon symmetry.',
        '- compileall: passed for env/algorithm/tools/tests, exit 0 and empty diagnostics.',
        '- Old full-info reference hashes: 24 v3.0/v3.1/v3.2 +8 v3.3 +8 v3.4 +8 v3.5 complete episodes, all unchanged. V3.5 references were captured before adding the hooks and include the independent Blue RNG as well as environment RNG.',
        '- Default dense hook returns an empty neutral result and emits no diagnostics; the default advantage method executes the exact old potential_shaping expression. No zero-array addition changes old rounding. The new weapon construction hook builds the exact old weapon class/config for historical versions.',
        '- Physical symmetry: three independent same-state/same-action seeds give exactly identical Red/Blue x/y/z/v/theta/psi after dynamics. One shared spec/controller/dynamics path, one radar config and one weapon instance serve both colors. Mirrored eligibility, identical ammunition/p_hit and sensor matrices passed. No aircraft/hardware asymmetry exists.',
        '- Existing color-specific boundary semantics remain V35: Red boundary death and Blue forced return. This is inherited game logic, not a different aircraft performance parameter.',
        '- Actual spawn vector reset: '+json.dumps(json.loads((out/'vector_reset.json').read_text()))+'.',
        '\n## Bounded CUDA algorithm checks\n',f'Device: {smoke["cuda_device"]}. Four algorithms each use 16 workers, 512 transitions, formal network widths and unchanged optimizer/PPO settings, a real update, finite loss/gradient/parameter/reward/GAE/return checks, strict checkpoint save/reload and two deterministic Red evaluation episodes. RMAPPO/STEA recurrence/reset checks pass. Common initial critics are identical under the common seed.',
        '', '| Algorithm | sampled transitions | workers | real finite update | checkpoint save/reload | eval episodes |', '|---|---:|---:|---|---|---:|']
    for name,r in smoke['records'].items():
        assert r['finite_gradients_parameters_and_values'] and r['finite_rewards_advantages_and_returns']
        lines.append(f'| {name} | {r["sampled_steps"]} | {r["num_envs"]} | passed | passed | 2 |')
    lines+=['\n## Fixed policy audit\n',f'Four groups each use {a["common_seeds"]} common environment seeds {a["seed_base"]}–{a["seed_base"]+999}; 4000 physical episodes. The same Red pursuit implementation/thresholds are used, with the shared version-specific sensor input. All rates are fractions; lengths/times are simulation steps. First event times exclude missing events and report occurrence counts. No learned policy or formal training is used.',
        '', '| Group | Red win | Blue win | draw | timeout | mean length | median length | Red losses | Blue losses | Red kills | Blue kills | bilateral | same-step first |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in a['groups'].items():
        keys=('red_win_rate','blue_win_rate','draw_rate','timeout_rate','average_episode_length','median_episode_length','average_red_loss','average_blue_loss','red_attack_kills','blue_attack_kills','bilateral_fire_window_episode_rate','same_step_first_fire_window_rate')
        lines.append('| '+label+' | '+' | '.join(f'{g[k]:.6f}' for k in keys)+' |')
    lines+=['','| Group | Red attempts | Blue attempts | Red hits | Blue hits | Red ammo used | Blue ammo used |', '|---|---:|---:|---:|---:|---:|---:|']
    for label,g in a['groups'].items():
        vals=[g[f'{side}_{key}'] for key in ('fire_attempts','weapon_hits','ammo_used') for side in ('red','blue')]
        lines.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','Conditional event timing (occurrence count / mean / median):','', '| Group / side | radar | fire window | attempt | kill |', '|---|---|---|---|---|']
    for label,g in a['groups'].items():
        for side in ('red','blue'):
            cells=[g[f'{side}_first_radar_detection_stats'],g[f'{side}_first_fire_window_step_stats'],g[f'{side}_first_attempt_stats'],g[f'{side}_first_kill_step_stats']]
            lines.append('| '+label+'/'+side+' | '+' | '.join(f'{x["count"]} / {x["mean"]} / {x["median"]}' for x in cells)+' |')
    lines+=['','| Group / side | eligible entries | ATA<=30 fraction | HA<=30 fraction | both<=30 fraction | first-fire mean ATA deg | HA deg | AA deg |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in a['groups'].items():
        for side in ('red','blue'):
            entry=g[f'{side}_entry_geometry'];angles=g[f'{side}_first_fire_mean_degrees']
            if label.startswith('v36'):assert entry['total']==entry['ata30']==entry['ha30']==entry['both30']
            vals=[entry['total'],entry['ata30_fraction'],entry['ha30_fraction'],entry['both30_fraction'],angles['ata'],angles['ha'],angles['aa']]
            lines.append('| '+label+'/'+side+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','First-fire geometry means are weighted across actual attempted pairs on each episode\'s first firing step. Entry counts include unselected pairs, consistent with strict pair-entry state consumption.',
        '', '| Group | event | team casualty (included in event) | outcome | guide | offense | defense | tactical/adv | safe | mean return |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in a['groups'].items():
        tactical=g.get('average_episode_tactical_total',g['average_episode_adv_total'])
        vals=[g['average_episode_event_total'],g['average_episode_team_casualty_total'],g['average_episode_outcome_total'],g.get('average_episode_guide_total',0),g.get('average_episode_tactical_offense_total',0),g.get('average_episode_tactical_defense_total',0),tactical,g['average_episode_safe_total'],g['average_return']]
        lines.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','V35 adv in this table is its original PBRS; V35 does not have guide/offense/defense. V36 adv is the tactical total; PBRS is exactly disabled with zero diagnostic phi arrays and no potential parameters in its schema. Tactical values are cached pre-hit, and can still be delivered on a terminal/death transition.',
        '', '| Group | detected pursuit | detected escape | guerrilla | centripetal | escape episode | guerrilla episode | centripetal episode | boundary return episode |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in a['groups'].items():
        vals=[g['blue_mode_agent_step_fractions'][k] for k in ('detected_pursuit','detected_escape','guerrilla','centripetal')]+[g[f'blue_{k}_episode_rate'] for k in ('escape','guerrilla','centripetal','boundary_return')]
        lines.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','## Observations and limitations','',
        'All V36 eligible entries on both colors satisfy ATA/HA <=30 degrees (zero geometry violations). Bilateral fire-window and same-step first-window rates remain 100% in these scripted comparisons. ZERO offense reward totals are zero; PURSUIT offense totals are small, consistent with the rear-aspect requirement, not a promise about learned-policy behavior. Blue hardware is symmetric and its 1800 m close-distance policy remains unchanged even though fire range is 2000 m. These outcomes are reported without thresholds, performance targets or tuning.',
        '', 'No old-version regression or unintended reward mechanism change was found. V36 intentionally replaces PBRS and changes hardware; automatic trajectories, event timing, outcomes and total returns can therefore change. This is not a learned-policy performance assessment.',
        '', '## File changes and preservation','',
        'New: `env/combat_v36.py`, `env/v36_geometry.py`, `env/v36_weapon.py`, `env/v36_reward.py`, `env/v36_config.py`; five V36 configs; `tests/test_combat_v36.py`, `tests/v36_v35_reference.json`; preparation/compatibility/vector/reset/report tools; `tools/audit_combat_v36.py`; `docs/combat_environment_v36.md`.',
        '', 'Changed pre-existing env/algorithm/config files:']
    lines+=['- `'+p+'`' for p in changed]
    lines+=['', 'Pre-existing changes are minimal shared-step/weapon/advantage hooks, factory/validator/dimension dispatch, allowlists and new diagnostic aggregation. Existing config inventory/error-message tests and smoke version selection were updated. Prior recorder compatibility changes predate this task. Old YAML files, old Blue policy source, old weapon/reward implementations, dynamics/controller/init/sensor implementations and V31–V35 class source retain their hashes. All four V36 algorithm YAML files retain V35 bytes.',
        '', 'Evidence: `before_hashes.json`, `special_pytest.log`, `full_pytest.log`, `compileall.log`, `vector_reset.json`, `cuda_smoke/summary.json`, `audit/summary.json` and four raw audit episode files. No 1M/3M/8M training or parameter sweep was started.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
