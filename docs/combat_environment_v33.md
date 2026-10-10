# Combat Environment v3.3

v3.3 inherits `CombatEnvironmentV32`. Its only environment semantic changes are the team casualty event penalty and terminal outcome reward values. The default remains v2.3.

For Red aircraft i, let A_i denote whether it was alive at the start of this transition. Let N be the number of unique, newly lost Red aircraft indices across Blue kills, ground losses, horizontal exits and ceiling exits. Then:

```
team_casualty_i = -2 * N * A_i
event_i = individual_event_i + team_casualty_i
r_i = event_i + outcome_i + adv_i + safe_i
```

An aircraft dying in this transition receives the casualty penalty; an aircraft already dead before this transition does not. Multiple loss causes for one aircraft count once. Individual event credit is unchanged: each Blue kill contributes +10 divided among actual credited attackers; each own loss contributes -10. There is no ally kill or team kill bonus.

Terminal outcome is +20 for a Red win, -20 for a Red loss, and zero for a draw, applied to all five fixed reward slots, including dead slots. Elimination, mutual destruction and timeout survivor comparisons retain v3.2 semantics.

Advantage and safety are inherited unchanged. Nonterminal advantage is `0.99 * Phi_next - Phi_current`; terminated and truncated transitions use a zero next shaping potential and therefore `-Phi_current`. The distance/angle weights remain 0.5/0.5, and true 3D nearest-aircraft safety remains 200 m with scale 0.2. No height or speed shaping is added.

The shared step has an identity event extension hook for v3.0–v3.2. It returns the original event array directly and no additional diagnostics, preserving old floating point arithmetic and info fields. Only v3.3 overrides it. Combat, Blue policy, initialization, sensors, ammo, weapon, controller, dynamics, observations and termination are untouched.

Main diagnostics remain event/outcome/adv/safe and aliases r1/r2/r3/r4 respectively. v3.3 additionally emits `individual_event_rewards`, `team_casualty_rewards`, their per-slot episode arrays and episode totals. The common evaluator and four PPO training summaries expose `average_episode_individual_event_total` and `average_episode_team_casualty_total`. No r5 exists. Component identities are exact at component precision; returned rewards use the existing float32 conversion.

The four v3.3 PPO configuration files are byte copies of their v3.2 counterparts. Use them with `configs/combat_environment_v33.yaml`. Strict checkpoint contracts still require the actual environment version and config hashes; a v3.2 checkpoint is not relabeled v3.3. MADDPG and MADSAC accept v3.3 with correctly specified 66-observation/3-action/5-agent configurations; older dimension configs remain incompatible. Their network computations and training hyperparameters are unchanged.

Validation and bounded CUDA smoke results are recorded in `outputs/v33_validation/report.md`. No formal training or parameter tuning is performed for this change.
