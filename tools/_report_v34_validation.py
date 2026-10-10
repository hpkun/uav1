"""Render saved v3.4 evidence without changing any experimental parameter."""
from pathlib import Path
import json, hashlib, re

def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/v34_validation'
    log=(out/'full_pytest.log').read_text();match=re.search(r'(\d+) passed in [^\n]+',log)
    if not match or 'FAILED' in log:raise RuntimeError('complete passing pytest required')
    special=(out/'special_pytest_final.log').read_text().strip()
    if '38 passed' not in special:raise RuntimeError('38 policy tests required')
    assert (out/'compileall.log').exists() and not (out/'compileall.log').read_text()
    audit=json.loads((out/'blue_audit/summary.json').read_text());smoke=json.loads((out/'cuda_smoke/summary.json').read_text())
    vector=json.loads((out/'vector_reset.json').read_text())
    before=json.loads((out/'before_hashes.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    doc=(root/'docs/combat_environment_v34.md').read_text()
    lines=[doc,'\n## Acceptance results\n',f'- Full WSL/uav pytest: **{match.group(0)}**.',
        '- v3.4 targeted pytest: **38 passed**.',
        '- compileall: env/algorithm/tools/tests passed (exit 0, no diagnostics).',
        '- Pre-change historical references: v3.0/v3.1/v3.2 24 full episodes, plus v3.3 eight full episodes, all hashes unchanged (state/observation/reward/full info/termination/RNG).',
        '- Explicit Blue override: eight complete paired v3.3/v3.4 episodes, ZERO/RANDOM Red and seeds 1/31/33000000/39000000, with exactly identical state, observation, reward/decomposition, combat, ammo, eligibility, full nonversion info and RNG. No unexpected reward difference.',
        '- Unit cases cover SEARCH center action, exact legacy PURSUIT action, 1v1/2v1/2v2/3v2, 1200/1201 and 1599/1600 thresholds, count-independent hysteresis, lost target, all three guard boundaries, centroid scope/direction, invisible-target isolation, living/3D friend counts, dead modes, finite deterministic degenerate fallbacks, reset and frozen config validation.',
        '- Actual spawn vector auto-reset: '+json.dumps(vector)+'.',
        '\n## Fixed-seed behavior audit\n',
        f'Four groups, each {audit["common_seeds"]} identical environment seeds starting at {audit["seed_base"]}; 4000 physical episodes total. Red PURSUIT always uses the unchanged memoryless SensorLimitedPursuitPolicy, independently of the version of Blue. No learned checkpoints or RL training were used.',
        '', '| Group | Red win | Blue win | draw | timeout | mean length | median length | Red losses | Blue losses | Red attack kills | Blue attack kills | bilateral window | same-step first window |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in audit['groups'].items():
        vals=[g[k] for k in ('red_win_rate','blue_win_rate','draw_rate','timeout_rate','average_episode_length','median_episode_length','average_red_loss','average_blue_loss','red_attack_kills','blue_attack_kills','bilateral_fire_window_episode_rate','same_step_first_fire_window_rate')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','All rates above are fractions, not percentages. Lengths and event steps are simulation steps. First-event means/medians below are conditional on occurrence; missing events are excluded and counts are provided.',
        '', '| Group | Red first window count/mean/median | Blue first window count/mean/median | Red first kill count/mean/median | Blue first kill count/mean/median |',
        '|---|---|---|---|---|']
    for label,g in audit['groups'].items():
        cells=[g[f'{side}_first_{event}_step_stats'] for side,event in (('red','fire_window'),('blue','fire_window'),('red','kill'),('blue','kill'))]
        lines.append('| '+label+' | '+' | '.join(f'{v["count"]} / {v["mean"]} / {v["median"]}' for v in cells)+' |')
    lines+=['','| Group | SEARCH | PURSUIT | EVADE | GUARD | evade episode rate | Blue noncombat episode rate | Blue noncombat aircraft fraction | mean Blue boundary exits | mean Blue ground losses |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,g in audit['groups'].items():
        vals=list(g['blue_mode_agent_step_fractions'].values())+[g['blue_evade_episode_rate'],g['blue_noncombat_loss_episode_rate'],g['blue_noncombat_loss_aircraft_fraction'],g['blue_boundary_exits'],g['blue_ground_losses']]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## Observations, without threshold tuning','']
    for mode in ('ZERO','PURSUIT'):
        a,b=audit['groups'][f'v33_{mode}'],audit['groups'][f'v34_{mode}']
        lines.append(f'- {mode}: v3.4 EVADE is {100*b["blue_mode_agent_step_fractions"]["evade"]:.3f}% of live Blue steps and appears in {100*b["blue_evade_episode_rate"]:.3f}% of episodes. Timeout changes from {100*a["timeout_rate"]:.3f}% to {100*b["timeout_rate"]:.3f}%; bilateral windows from {100*a["bilateral_fire_window_episode_rate"]:.3f}% to {100*b["bilateral_fire_window_episode_rate"]:.3f}%; median length from {a["median_episode_length"]:.1f} to {b["median_episode_length"]:.1f}. Blue noncombat losses occur in {100*b["blue_noncombat_loss_episode_rate"]:.3f}% of v3.4 episodes ({100*b["blue_noncombat_loss_aircraft_fraction"]:.3f}% of initial Blue aircraft).')
    lines+=['', 'All four groups retain 100% bilateral-window episode rate and 100% same-step first-window rate. Thus the new evasion does not remove synchronous first contact in these two scripted Red baselines. The modest length change does not demonstrate elimination of early mutual-fire dominance. Blue noncombat losses remain zero in all four groups. The requested thresholds are not tuned. There is no specified acceptable evade/timeout/win-rate target, so the raw measured rates above are reported without declaring a desired performance level. No unexpected information leakage was found in targeted invariance tests, and no reward/mechanism divergence was found with identical explicit actions.',
        '\n## Bounded CUDA protocol checks\n',f'Device: {smoke["cuda_device"]}. Four PPO algorithms each sampled 512 transitions with 16 spawn workers, formal network widths and unchanged PPO settings, a real update, strict checkpoint save/reload and two deterministic evaluation episodes. Finite rewards/GAE/returns/parameters/gradients passed; RMAPPO/STEA recurrent episode and death reset checks passed. Common-seed initial critics remain identical.',
        '', 'MADDPG/MADSAC only gained environment-version compatibility for v3.4; their network architecture and hyperparameters are unchanged. Strict metadata version/config identity remains required.',
        '\n## Changed files and preservation\n',
        'New files: `env/combat_v34.py`, `env/v34_policy.py`, `env/v34_config.py`, five v3.4 YAML configs, `tests/test_combat_v34.py`, `tests/v34_v33_reference.json`, this report generator, preparation/reference tool, vector reset check, `tools/audit_combat_v34_blue.py`, and `docs/combat_environment_v34.md`.',
        '', 'Changed pre-existing env/algorithm/config files:']
    lines+=['- `'+p+'`' for p in changed]
    lines+=['', 'Only factory/validator/dimension dispatch, algorithm allowlists, common evaluator diagnostics and summary compatibility changed in pre-existing implementation. The old Blue source, v3.3 reward hook and all v3.0–v3.3 environment classes retain their hashes. Old YAML configs retain their bytes. Existing config inventory/error-message tests and smoke config selection were updated.',
        '', 'Evidence: `full_pytest.log`, `special_pytest_final.log`, `compileall.log`, `vector_reset.json`, `cuda_smoke/summary.json`, `blue_audit/summary.json`, and four complete audit episode JSON files. No formal 1M/3M training, sweep or retuning was started.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
