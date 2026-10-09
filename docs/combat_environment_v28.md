# Combat environment v2.8

v2.8 is a single-variable difficulty reduction experiment based on v2.7.
The only substantive change is `blue_policy.conversion_range`: 2500 m to
2000 m. The environment version is 2.8. Every other configuration field is
identical to v2.7, including the 5v5 formation, 65/3/5 dimensions, weapon,
simulation, actions, aircraft, arena, observations and rewards.

The intended intervention lets Blue switch from OFFSET_RECIPROCAL to CONVERT
only at a smaller range, reducing the distance margin available for stern
conversion. It is used to evaluate task difficulty changes when the fixed
Stern opponent's conversion capability is reduced, not to target an algorithm
win rate. Depending on geometry, both range thresholds can already be satisfied
when Blue reaches the rear hemisphere, so this change need not delay every
conversion. Geometric results are reported without parameter tuning.

SternConversionPolicy source is unchanged. The same four phases, target locks,
outward candidate comparison, parity tie-break, entry bearing, reciprocal
heading, LOS conversion, 250 m/s and altitude controller remain. No Blue
capability, randomness, delay or cooldown is added. Reward source, reward target
selection and every reward config field are unchanged. v2.3-v2.7 configs and
the default v2.3 environment are preserved.

v2.8 uses RearAspectWeaponEnvelope and 5x5 PairFireState. Stern schema validation
is shared with v2.6/v2.7. Existing four formal 5v5 algorithm configs are reused;
RMAPPO/EA-MAPPO allow-lists add 2.8, while MAPPO/STEA retain their existing
dimension and checkpoint-fingerprint contracts. 8v8 configs remain incompatible.
MADSAC and MADDPG support ranges are unchanged. No network/hyperparameter changes.

## Changed files

New: `configs/combat_environment_v28.yaml`, `tests/test_combat_v28.py`,
`docs/combat_environment_v28.md`.

Modified: `env/config.py`, `env/combat_env.py`,
`algorithm/rmappo/runner.py`, `algorithm/rmappo/protocol.py`,
`algorithm/ea_mappo/runner.py`, `algorithm/ea_mappo/protocol.py`,
`tests/test_combat_protocol.py`, `tests/test_formal_8v8.py`,
`tools/validate_stern_conversion.py`.

The geometry tool only adds version support and mean/median CONVERT entry-step
statistics from already recorded transitions. Its trial algorithm is unchanged.
The existing smoke tool is reused unchanged.

## Acceptance commands

Run in WSL Ubuntu after `conda activate uav`:

```bash
python tools/validate_stern_conversion.py \
  --env-config configs/combat_environment_v28.yaml --trials 200 --seed-base 260000 \
  --output outputs/v28_validation/geometry_v28.json
python tools/smoke_formal_8v8.py \
  --env-config configs/combat_environment_v28.yaml \
  --output outputs/v28_validation/cuda_smoke
python -m pytest -q
python -m compileall -q algorithm env tools tests
```

The synthetic threshold test distinguishes a rear-hemisphere state at 2200 m:
v2.7 enters CONVERT while v2.8 stays OFFSET_RECIPROCAL. At 2000 m v2.8 converts.
A separate altitude offset verifies that the condition uses true 3-D distance.
Old Stern regressions, reset, explicit Blue overrides, rewards, termination,
formal config acceptance and incompatible algorithm support are also tested.

Reports are under `outputs/v28_validation/`. The smoke caps each algorithm at
512 sampled steps, checks a real CUDA update and two checkpoint-reload episodes.
It is not a performance estimate. No 3M training or parameter sweep is started.

## Verified results

- Targeted v2.6/v2.7/v2.8 tests: 89 passed, including 14 new v2.8 cases.
- Full pytest: 704 passed in 122.33 s; compileall passed.
- v2.8 trials 260000..260199: 100% fire-window success and phase completion,
  median/p90 first-window steps 268/271; mean/median CONVERT entry 170.45/171.
  Blue exits, ground losses and boundary-before-window failures are all zero.
  All 200 straight targets leave the arena after successful conversion.
- Same-seed comparison reuses the saved v2.7 report from the unchanged config
  and policy. All trial records are identical. The maximum v2.7 CONVERT entry
  range was 719.93 m, below both thresholds: the 2000 m condition did not bind
  in this block. This 1v1 result does not demonstrate reduced task difficulty.
- All four algorithms completed 512 CUDA sampled steps, one real PPO update,
  16 spawn workers and two checkpoint-reload evaluation episodes each. Finite
  gradients/parameters/values and recurrent pre-update ratios passed.
- SHA-256 checks confirm policy/reward source, all old environment YAMLs and
  all formal 5v5 algorithm YAMLs unchanged. No training budget was altered.
- No formal 3M training, parameter tuning or parameter sweep was performed.

See `geometry_v28.json`, `comparison.json`, `cuda_smoke/summary.json`,
`frozen_verified.json`, `pytest.log`, `compileall.log` and
`validation_report.json` in `outputs/v28_validation/`.
