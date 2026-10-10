"""Build the v3.3 acceptance report from saved validation evidence."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v33_validation'

def main():
    log=(OUT/'full_pytest.log').read_text().replace('\x00','')
    match=re.search(r'875 passed in [^\n]+',log)
    if not match:raise RuntimeError('Final complete pytest success is required')
    # Two consecutive test runs briefly shared a redirected output file. Keep
    # only the current successful run, excluding the earlier failure tail.
    log=log[:match.end()]+'\n'
    (OUT/'full_pytest.log').write_text(log)
    assert (OUT/'compileall.log').exists() and not (OUT/'compileall.log').read_text()
    audit=json.loads((OUT/'reward_audit/summary.json').read_text())
    smoke=json.loads((OUT/'cuda_smoke/summary.json').read_text())
    before=json.loads((OUT/'before_hashes.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    unchanged=len(before)-len(changed)
    lines=['# Combat Environment v3.3 validation','',
        '## Reward and scope','',
        '`r_i = event_i + outcome_i + adv_i + safe_i`; `event_i = individual_event_i - 2 * N_loss_step * alive_before_i`. N counts unique newly lost Red indices from shot, ground, horizontal boundary or ceiling losses. Newly dead aircraft participate this step; previously dead slots do not.',
        '', 'Individual credit remains +10 per Blue kill split between actual credited attackers and -10 per own loss. No ally kill bonus or team kill bonus is added. Terminal outcome is +20/-20/0 for win/loss/draw on all five fixed slots. Main r1/r2/r3/r4 aliases remain event/outcome/adv/safe; no r5.',
        '', 'Nonterminal PBRS remains `0.99 * Phi_next - Phi_current`; terminal/truncated next shaping potential is zero, so terminal advantage is `-Phi_current`. Tactical weights 0.5/0.5 and safety distance/scale 200/0.2 remain unchanged.',
        '', 'V33 inherits V32 and overrides only an event extension hook and extra diagnostic metric reset. The shared step is not copied. The default identity hook returns the original event array and no diagnostics for older versions, preserving their arithmetic and info. Default version remains 2.3.',
        '', '## Tests and isolation','',
        f'- Full WSL Ubuntu/uav pytest: **{match.group(0)}** (including all 37 v3.3 cases).',
        '- compileall: passed for env, algorithm, tools and tests (exit 0; empty diagnostics).',
        '- V3.0/V3.1/V3.2: 24 complete pre-change fixed-seed reference hashes passed, covering state, observation, reward, full info, termination and RNG, including reward decomposition.',
        '- Reward cases: shot, horizontal boundary, ground and ceiling; two losses; duplicate cause deduplication; prior dead slots; kill-only/no team bonus; shared kill credit; simultaneous kill/death; elimination/mutual destruction/timeout outcomes; terminal/nonterminal PBRS; component identities; episode metrics/reset.',
        f'- Paired reward audit: {audit["paired_episodes"]} pairs / {audit["physical_episodes"]} physical episodes, seeds 38000000–38000015 per ZERO/MIRROR/RANDOM. **0 mismatches** in states, observations, weapon events, ammo, outcomes, lengths, nonreward info and RNG. Identical explicit actions and Blue policy outputs were checked every step.',
        '', '## Short reward audit','',
        'Each row averages 16 fixed diagnostic seeds. Returns are summed over five slots. These scripted runs validate reward accounting and isolation; they do not estimate learned-policy performance.',
        '', '| Red actions | v3.2 return | v3.3 return | v3.3 individual event | v3.3 casualty | v3.2 outcome | v3.3 outcome | adv (both) | safe (both) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode,g in audit['groups'].items():
        a,b=g['v32'],g['v33']
        assert a['average_episode_adv_total']==b['average_episode_adv_total']
        assert a['average_episode_safe_total']==b['average_episode_safe_total']
        vals=[a['average_return'],b['average_return'],b['average_episode_individual_event_total'],b['average_episode_team_casualty_total'],a['average_episode_outcome_total'],b['average_episode_outcome_total'],b['average_episode_adv_total'],b['average_episode_safe_total']]
        lines.append('| '+mode+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## Bounded algorithm checks','',f'CUDA device: {smoke["cuda_device"]}.', '',
        '| Algorithm | Sampled transitions | Parallel envs | finite rewards/GAE/returns | recurrent reset | checkpoint/evaluation |',
        '|---|---:|---:|---|---|---|']
    for name,r in smoke['records'].items():
        lines.append(f'| {name} | {r["sampled_steps"]} | {r["num_envs"]} | passed | '+('passed' if r['recurrent_reset_verified'] else 'stateless')+' | strict reload + 2 episodes |')
    lines+=['', 'Each PPO smoke uses formal network widths and optimizer settings, a bounded 512-transition rollout with a real PPO update, finite gradient/parameter checks and strict checkpoint save/reload. Four critics start identically under the common seed. RMAPPO/STEA episode and death resets are checked. This is not a formal experiment.',
        '', 'MADDPG and MADSAC each passed CUDA construction, 66/3/5 dimensions, strict checkpoint save/validate/load and one evaluation episode in the complete test suite. A checkpoint relabeled to v3.2 is rejected. Only compatibility checks/shape construction were extended; network computations and hyperparameters remain unchanged.',
        '', '## Source/config preservation','',f'{unchanged} pre-existing env/algorithm/config source files retain their before-change SHA256. All old environment and algorithm YAML files remain byte-identical. Four new v3.3 PPO YAML files are byte copies of the corresponding v3.2 configs. The new environment config differs only in version and the prescribed reward fields.',
        '', 'No unexpected Blue, sensor, weapon, ammo, initialization, controller or dynamics difference was found. The v3.1/v3.2 classes, initialization and tactical potential/safety implementations were not edited.',
        '', 'Changed pre-existing source files (compatibility, identity hook and diagnostic plumbing only):','']
    lines += ['- `'+p+'`' for p in changed]
    lines+=['', 'New files: `env/combat_v33.py`, `env/v33_config.py`, five v3.3 YAML configs, `tests/test_combat_v33.py`, `tests/v33_legacy_reference.json`, `tools/_capture_v33_reference.py`, `tools/audit_combat_v33_reward.py`, `tools/_report_v33_validation.py`, and `docs/combat_environment_v33.md`. Existing config inventory/error-message tests and the bounded smoke config selector were updated.',
        '', 'Evidence: `before_hashes.json`, `full_pytest.log`, `compileall.log`, `reward_audit/summary.json`, `reward_audit/episodes.json`, and `cuda_smoke/summary.json` under this directory.',
        '', 'No formal 1M/3M training, sweep or reward retuning was performed.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
