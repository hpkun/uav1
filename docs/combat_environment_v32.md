# Combat Environment v3.2

v3.2 = v3.1 diversified initialization + terminal-safe potential-based reward shaping (PBRS). `CombatEnvironmentV32` inherits `CombatEnvironmentV31`; it overrides only `shaping_next_potential`. The full step, reset, sensor, weapon, ammunition, hit probability, Blue policy, outcome/timeout rules, boundary handling, dynamics, controller, observation and tactical potential are shared with v3.1.

## Reward protocol

For nonterminal transitions:

\[
r_{adv}=potential\_scale\,[potential\_gamma\,\Phi(s')-\Phi(s)].
\]

If either `terminated` or `truncated` is true:

\[
\Phi_{next,shaping}=0,\qquad r_{adv}=-potential\_scale\,\Phi(s).
\]

This covers all six elimination/mutual-destruction/timeout outcome reasons. It removes the residual `gamma * Phi(s_T)` contribution associated with the final physical geometry, including timeout. It does not change outcome classification or any other reward component. The existing tactical potential function and reward weights remain unchanged; there is no new terminal-potential hyperparameter.

The shared v3.0 parent has an identity hook: v3.0/v3.1 continue to use actual next potential even at terminal transitions. Historical behavior is covered by complete fixed-seed trajectory/reward references captured before the hook was introduced.

## Diagnostic field meanings

`phi_current` is the actual potential before the transition. `phi_next` retains its historical meaning: **actual** next-state potential. `phi_next_actual` explicitly aliases that value. `phi_next_for_shaping` is the potential used in `adv_rewards`: actual on nonterminal transitions and zero on v3.2 terminal transitions. `mean_potential_next` and the next distance/angle diagnostic components continue to describe actual geometry. Terminal geometry information is retained.

The emitted total reward is the float32 cast of the sum of the four float64 component arrays: event + outcome + adv + safe. Comparing different orders of episode-level floating-point summation uses roundoff tolerance; the per-agent total/cast contract is tested exactly.

## Version and algorithm contracts

The environment YAML differs from v3.1 only in `environment_version: '3.2'`. All initialization parameters remain frozen: 5000 m reference-center distance, independent lateral proposals in ±2500 m, 400 m same-team separation, independent index permutations and ±20° heading perturbations. All four algorithm YAML files are byte-identical copies of the v3.1 files. Dimensions remain 66/3/5 and EA/STEA retain self8/ally7/enemy6.

Version allowlists and the bounded CUDA validation tool recognize 3.2. Checkpoint version, dimensions, configuration hashes and algorithm identity remain strict. MADSAC/MADDPG support is unchanged. The default environment remains v2.3.

## Timeout statistics repair

`algorithm/common/metrics.py::episode_is_timeout` uses an explicit `timeout` field as the authoritative source, including explicit false. Otherwise it recognizes the known legacy `red_failure_timeout` reason and the three v3.x timeout reasons. Unknown reason strings are not classified by substring matching.

MAPPO step metrics and final summaries use this helper. RMAPPO, EA and STEA inherit those reporting methods. The common evaluator/aggregator and the legacy validation/audit writers use the same helper. Win/draw/loss definitions, episode termination, evaluation selection and checkpoint selection are untouched. Historical experiment JSON/CSV/log files are never rewritten; pre-fix timeout statistics remain in their original records.

## Acceptance and symmetry audit

Commands run in WSL Ubuntu after `conda activate uav`; CUDA is mandatory for smoke/checkpoint evaluation/audits. No CPU fallback is allowed. NumPy environment simulation still executes in workers as in normal training.

```bash
python tools/audit_combat_v32_symmetry.py \
  --workers 16 --output outputs/v32_validation/symmetry
python tools/smoke_formal_8v8.py \
  --env-config configs/combat_environment_v32.yaml \
  --output outputs/v32_validation/cuda_verified_smoke
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

The symmetry audit runs exactly 2000 MIRROR episodes in each block starting at 33M, 34M, 35M, 36M and 37M, totaling 10,000. Both sides call the same existing SensorLimitedPursuitPolicy. The environment RNG and its Red-before-Blue hit-draw order are unchanged. No 70M+ formal holdout is used.

Win proportions use Wilson 95% intervals. The signed Red−Blue gap uses an episode-level multinomial normal interval, accounting for mutually exclusive wins; it is not an interval formed by treating Red and Blue wins as independent samples. Empirical hit-rate Wilson intervals are nominal Bernoulli intervals over observed attempts, with adaptively varying attempt counts. Timing excludes absent events and reports counts. First-fire order uses first attempt, with separate first-window statistics. Ties and neither-side events are explicit.

Terminal reward stability is reported both for the first 64 sorted diagnostic episodes and for all 10,000 episodes. Each terminal vector is checked against `-phi_current`; the unchanged potential weights imply per-agent absolute value at most 1 for this configuration. No parameter is adjusted based on these results. The optional side-swap diagnostic is not performed; it is not needed to complete the requested five-block audit.

Four algorithms each run the existing bounded 512 sampled-step, 16-spawn-worker acceptance at formal network widths, including a real PPO update, checkpoint save/strict reload and two evaluation episodes. Additional read-only assertions inspect actual GAE outputs and recurrent episode/death resets. This diagnostic instrumentation does not modify trainer code or calculation outputs.

Complete implementation, regression, smoke and audit results are in `outputs/v32_validation/report.md`, with raw episodes in `symmetry/episodes.jsonl.gz` and machine-readable statistics in `symmetry/summary.json`. No formal 1M/3M training or parameter sweep is started.

<!-- SYMMETRY_RESULTS -->

## 10000局结果

|block（各2000局）|Red WR [95% CI]|Blue WR|draw|signed gap pp [95% CI]|absolute gap pp|length mean/median|timeout|
|---|---:|---:|---:|---:|---:|---:|---:|
|33000000|39.20% [37.08,41.36]|41.60%|19.20%|-2.40 [-6.34,+1.54]|2.40|126.098/104.0|0.40%|
|34000000|41.90% [39.76,44.08]|41.05%|17.05%|+0.85 [-3.14,+4.84]|0.85|127.395/104.0|0.50%|
|35000000|42.90% [40.75,45.08]|41.65%|15.45%|+1.25 [-2.78,+5.28]|1.25|124.023/103.0|0.45%|
|36000000|41.95% [39.80,44.13]|41.20%|16.85%|+0.75 [-3.25,+4.75]|0.75|127.116/105.0|0.40%|
|37000000|40.55% [38.42,42.72]|40.80%|18.65%|-0.25 [-4.20,+3.70]|0.25|128.784/104.0|0.65%|
|overall 10000|41.30% [40.34,42.27]|41.26%|17.44%|+0.04 [-1.74,+1.82]|0.04|126.683/104.0|0.48%|

未观察到此前−5.8pp在五个独立block中持续存在的稳定side bias；不能据此证明精确零偏差。
